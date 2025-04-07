import h5py as h5

from PIL import Image, ImageOps
import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity as compute_ssim
from skimage.metrics import peak_signal_noise_ratio as compute_psnr

from dataclasses import dataclass
import matplotlib.pyplot as plt
import sys
import cv2
import os

@dataclass
class ImageMetricInput:
    image_pred: Image
    image_gt: Image
    image_first_frame: Image
    pts_0: np.ndarray
    pts_1: np.ndarray
    save_path: str = None


@dataclass
class ImageMetricOutput:
    mse: float
    psnr: float
    ssim: float
    lpips: float
    flow_epe: float
    depth_rmse: float
    segment_iou: float
    warning: str



class ImageMetricCalculator:
    def __init__(self, device="cuda", lpips_model_type="alex", eval_resolution=(256, 256)):
        self.device = device
        self.lpips_model_type = lpips_model_type
        self.lpips_model = lpips.LPIPS(net=lpips_model_type).to(device)
        self.eval_resolution = (eval_resolution[1], eval_resolution[0])  # ImageOps.fit use (width, height) format

        # load depth model
        try:
            sys.path.insert(0, '/ccn2/u/wanhee/depth_anythingv2/metric_depth')
            from depth_anything_v2.dpt import DepthAnythingV2
        except:
            print("Depth model not found. Cloning the repository and installing the requirements.")
            print("git clone https://github.com/DepthAnything/Depth-Anything-V2")
            print("cd Depth-Anything-V2/metric_depth")
            print("pip install -r requirements.txt")
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }
        encoder = 'vitl'  # or 'vits', 'vitb'
        dataset = 'hypersim'  # 'hypersim' for indoor model, 'vkitti' for outdoor model
        max_depth = 20  # 20 for indoor model, 80 for outdoor model
        depth_model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
        depth_model.load_state_dict(torch.load(
            f'/ccn2/u/wanhee/depth_anythingv2/metric_depth/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth',
            map_location='cpu'))
        depth_model.eval()
        self.depth_model = depth_model.to("cuda")

        try:
            from segment_anything import SamPredictor, sam_model_registry
        except:
            print("Segmentation model not found. Cloning the repository and installing the requirements.")
            print("pip install git+https://github.com/facebookresearch/segment-anything.git")
            from segment_anything import SamPredictor, sam_model_registry

        sam = sam_model_registry["vit_h"](checkpoint='/ccn2/u/wanhee/segment-anything/checkpoints/sam_vit_h_4b8939.pth')
        sam.to(device="cuda")
        sam.eval()
        self.segmentation_model = SamPredictor(sam)

        self.flow_model = RAFTInterface().cuda().eval()

    @torch.no_grad()
    def predict_segmentation0_from_rgb0(self, rgb0_numpy_0to1,
                                        prompt={"input_point": [[500, 375]], "input_label": [[1]]}):

        rgb0_numpy = (rgb0_numpy_0to1 * 255).astype(np.uint8)
        self.segmentation_model.set_image(rgb0_numpy)
        input_point = prompt["input_point"]
        input_label = np.array(prompt["input_label"][0])
        masks, scores, logits = self.segmentation_model.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        best_mask_index = np.argmax(scores)
        segmentation0_numpy = masks[best_mask_index]  # HxW bool mask in numpy
        segmentation0_tensor = torch.tensor(segmentation0_numpy).unsqueeze(0).unsqueeze(0)  # 1x1xHxW
        return {"segmentation0_tensor": segmentation0_tensor, "segmentation0_numpy": segmentation0_numpy,
                "masks": masks, "scores": scores, "logits": logits}

    @torch.no_grad()
    def predict_depth0_from_rgb0(self, rgb0_numpy_0to1):
        rgb0_numpy = (rgb0_numpy_0to1 * 255).astype(np.uint8)
        rgb0_numpy_bgr = cv2.cvtColor(rgb0_numpy, cv2.COLOR_RGB2BGR)
        depth0_numpy = self.depth_model.infer_image(rgb0_numpy_bgr, device="cuda")
        depth0_tensor = torch.tensor(depth0_numpy).unsqueeze(0).unsqueeze(0)  # 1x1xHxW
        return {"depth0_tensor": depth0_tensor, "depth0_numpy": depth0_numpy}

    def get_segment_from_points(self, image, points):
        '''
        image: [H, W, 3] in (0, 255)
        points: [N, 2]
        '''

        segmentation_prompt = {"input_point": points, "input_label": [[1] * len(points)]}

        segment_dict = self.predict_segmentation0_from_rgb0(image, segmentation_prompt)

        segment_map = segment_dict['segmentation0_numpy']

        return segment_map

    def compute_iou(self, seg1, seg2):
        """
        Compute the Intersection over Union (IoU) between two segmentation maps.

        Args:
            seg1 (np.ndarray): First segmentation map of shape (H, W) with binary values.
            seg2 (np.ndarray): Second segmentation map of shape (H, W) with binary values.

        Returns:
            float: The IoU value.
        """
        if seg1.shape != seg2.shape:
            raise ValueError("Both segmentation maps must have the same shape.")

        # Calculate the intersection and union areas.
        intersection = np.logical_and(seg1, seg2)
        union = np.logical_or(seg1, seg2)

        # Avoid division by zero: if union is empty, we consider IoU to be 1.
        iou = np.sum(intersection) / np.sum(union) if np.sum(union) != 0 else 1.0
        return iou

    def get_iou_score(self, input: ImageMetricInput):

        image_pred = np.array(input.image_pred)
        image_gt = np.array(input.image_gt)
        image_frame0 = np.array(input.image_first_frame)

        pts_0 = input.pts_0
        pts_1 = input.pts_1

        frame0 = torch.from_numpy(image_frame0).permute(2, 0, 1).unsqueeze(0).cuda().to(torch.float32)
        frame1 = torch.from_numpy(image_pred).permute(2, 0, 1).unsqueeze(0).cuda().to(torch.float32)

        # get GT flow
        flow_map_pred = self.flow_model(
            frame0,
            frame1,
            unnormalize=False
        )

        flow_map_pred = flow_map_pred[0].permute(1, 2, 0).cpu().numpy()

        segment_gt = self.get_segment_from_points(image_gt, pts_1 / 4)

        segment_gt_frame0 = self.get_segment_from_points(image_frame0, pts_0 / 4)

        pts_0 = pts_0 // 4

        coords_frame1 = pts_0 + flow_map_pred[pts_0[:, 1], pts_0[:, 0]]

        segment_pred_frame1 = self.get_segment_from_points(image_pred, coords_frame1)

        iou = self.compute_iou(segment_gt, segment_pred_frame1)

        self.segment_gt = segment_gt

        self.segment_pred = segment_pred_frame1

        return iou

    def get_flow_epe(self, input: ImageMetricInput):

        image_pred = np.array(input.image_pred)
        image_gt = np.array(input.image_gt)
        image_frame0 = np.array(input.image_first_frame)

        frame0 = torch.from_numpy(image_frame0).permute(2, 0, 1).unsqueeze(0).cuda().to(torch.float32)
        frame1 = torch.from_numpy(image_pred).permute(2, 0, 1).unsqueeze(0).cuda().to(torch.float32)

        # get GT flow
        flow_map_pred = self.flow_model(
            frame0,
            frame1,
            unnormalize=False
        )

        frame0 = torch.from_numpy(image_frame0).permute(2, 0, 1).unsqueeze(0).cuda().to(torch.float32)
        frame1 = torch.from_numpy(image_gt).permute(2, 0, 1).unsqueeze(0).cuda().to(torch.float32)

        # get GT flow
        flow_map_gt = self.flow_model(
            frame0,
            frame1,
            unnormalize=False
        )

        flow_map_pred = flow_map_pred[0].permute(1, 2, 0).cpu().numpy()

        flow_map_gt = flow_map_gt[0].permute(1, 2, 0).cpu().numpy()

        epe = self.compute_epe(flow_map_pred, flow_map_gt)

        return epe[1]

    def compute_rmse_depth(self, input: ImageMetricInput):
        """
        Compute the Root Mean Square Error (RMSE) between two depth maps.

        Args:
            depth1 (np.ndarray): First depth map.
            depth2 (np.ndarray): Second depth map.

        Returns:
            float: The RMSE value.
        """
        # Ensure both depth maps have the same shape

        image_pred = np.array(input.image_pred)
        image_gt = np.array(input.image_gt)

        depth1 = self.predict_depth0_from_rgb0(image_pred / 255)['depth0_numpy']

        depth2 = self.predict_depth0_from_rgb0(image_gt / 255)['depth0_numpy']

        if depth1.shape != depth2.shape:
            raise ValueError("Depth maps must have the same shape.")

        # Compute the difference between the depth maps
        diff = depth1 - depth2

        # Compute the Mean Squared Error (MSE)
        mse = np.mean(diff ** 2)

        # Compute the RMSE
        rmse = np.sqrt(mse)
        return rmse

    def compute_epe(self, flow_est, flow_gt):
        """
        Compute the Endpoint Error (EPE) between two optical flow maps.

        Args:
            flow_est (np.ndarray): Estimated flow, shape (H, W, 2).
            flow_gt (np.ndarray): Ground truth flow, shape (H, W, 2).

        Returns:
            tuple: (epe_map, avg_epe)
                - epe_map (np.ndarray): Per-pixel EPE, shape (H, W).
                - avg_epe (float): Average EPE over all pixels.
        """
        # Compute the difference between estimated and ground truth flows.
        diff = flow_est - flow_gt

        # Compute the Euclidean distance (L2 norm) for each pixel.
        epe_map = np.sqrt(np.sum(diff ** 2, axis=2))

        # Average the endpoint error over all pixels.
        avg_epe = np.mean(epe_map)

        return epe_map, avg_epe

    def calculate_metrics(self, input: ImageMetricInput) -> ImageMetricOutput:
        image_pred = input.image_pred.convert("RGB")
        image_gt = input.image_gt.convert("RGB")
        if input.save_path is not None:
            image_pred.save(input.save_path.replace(".png", "_pred.png"))
            image_gt.save(input.save_path.replace(".png", "_gt.png"))

        image_pred = ImageOps.fit(image_pred, self.eval_resolution)
        image_gt = ImageOps.fit(image_gt, self.eval_resolution)
        if input.save_path is not None:
            image_pred.save(input.save_path.replace(".png", "_pred_metric.png"))
            image_gt.save(input.save_path.replace(".png", "_gt_metric.png"))

        mse = self.calculate_mse(image_pred, image_gt)
        psnr = self.calculate_psnr(image_pred, image_gt)
        ssim = self.calculate_ssim(image_pred, image_gt)
        lpips = self.calculate_lpips(image_pred, image_gt)

        flow_epe = self.get_flow_epe(input)

        depth_rmse = self.compute_rmse_depth(input)

        segment_iou = self.get_iou_score(input)

        warning = f"You are using {self.lpips_model_type} model for LPIPS calculation."

        # # write the metrics to a image as text and plot it with matplotlib
        # text = f"MSE: {mse:.4f}\nPSNR: {psnr:.4f}\nSSIM: {ssim:.4f}\nLPIPS: {lpips:.4f}"
        # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        # ax.imshow(image_pred)
        # ax.text(0, 0, text, color='black', fontsize=12, ha='left', va='bottom', wrap=True)
        # ax.axis("off")
        # plt.tight_layout()
        # plt.savefig(input.save_path.replace(".png", "_pred_metric_text.png"))
        # plt.close()

        return ImageMetricOutput(mse, psnr, ssim, lpips, flow_epe, depth_rmse, segment_iou, warning)

    def calculate_mse(self, image_pred: Image, image_gt: Image) -> float:
        image_pred_numpy_float = np.array(image_pred) / 255
        image_gt_numpy_float = np.array(image_gt) / 255
        return np.mean((image_pred_numpy_float - image_gt_numpy_float) ** 2)

    def calculate_psnr(self, image_pred: Image, image_gt: Image) -> float:
        image_pred_numpy_float = np.array(image_pred) / 255
        image_gt_numpy_float = np.array(image_gt) / 255
        mse = np.mean((image_pred_numpy_float - image_gt_numpy_float) ** 2)
        psnr = 10 * np.log10(1 / mse) if mse > 0 else 100
        return psnr

    def calculate_ssim(self, image_pred: Image, image_gt: Image) -> float:
        image_pred_numpy_float = np.array(image_pred) / 255
        image_gt_numpy_float = np.array(image_gt) / 255
        return compute_ssim(image_pred_numpy_float, image_gt_numpy_float, channel_axis=2, data_range=1)

    def calculate_lpips(self, image_pred: Image, image_gt: Image) -> float:
        image_pred_tensor = self.image_to_tensor(image_pred)
        image_gt_tensor = self.image_to_tensor(image_gt)
        image_pred_tensor_m1_1 = image_pred_tensor * 2 - 1
        image_gt_tensor_m1_1 = image_gt_tensor * 2 - 1
        return self.lpips_model(image_pred_tensor_m1_1, image_gt_tensor_m1_1).item()

    def image_to_tensor(self, image: Image) -> torch.Tensor:
        image_numpy = np.array(image)
        image_tensor = torch.tensor(image_numpy).permute(2, 0, 1).unsqueeze(0).float() / 255
        return image_tensor.to(self.device)


def save_metrics_in_h5(h5_path, image0_downsampled, image1_downsampled, rgb1_pred, pts_0, pts_1, metrics):

    with h5.File(h5_path, 'w') as f:
        f.create_dataset('image0', data=image0_downsampled)
        f.create_dataset('image1', data=image1_downsampled)
        f.create_dataset('counterfactual_image', data=rgb1_pred)
        f.create_dataset('pts_0', data=pts_0)
        f.create_dataset('pts_1', data=pts_1)
        # save metrics in new field
        new_field = f.create_group('metrics')
        new_field.create_dataset('mse', data=metrics.mse)
        new_field.create_dataset('psnr', data=metrics.psnr)
        new_field.create_dataset('ssim', data=metrics.ssim)
        new_field.create_dataset('lpips', data=metrics.lpips)
        new_field.create_dataset('flow_epe', data=metrics.flow_epe)
        new_field.create_dataset('depth_rmse', data=metrics.depth_rmse)
        new_field.create_dataset('segment_iou', data=metrics.segment_iou)


def load_data(h5_file, return_orig=False):
    with h5.File(h5_file, 'r') as f:
        image0 = f['image1'][:]

        image1 = f['image2'][:]

        pts_0_orig = f['points_image1'][:]
        pts_0 = pts_0_orig[:-6].reshape(-1, 4, 2)

        pts_1_orig = f['points_image2'][:]
        pts_1 = pts_1_orig[:-6].reshape(-1, 4, 2)

        K = f['K'][:]

    if return_orig:
        return image0, image1, pts_0, pts_1, K, pts_0_orig, pts_1_orig

    return image0, image1, pts_0, pts_1, K,