import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lras_3d.predictor.lras_3d_predictor import LRASPredictor
from lras_3d.task.obj_motion_utils import get_RT, get_dense_flow_from_segment_depth_RT, downsample_and_scale_flow, \
    downsample_flow, combine_dilated_bounding_boxes, get_true_pixel_coords, get_flattened_index_from_2d_index, \
    get_unmask_indices_from_flow_map, downsample_and_scale_flow_gt, downsample_and_scale_flow_pred, get_decoding_order, render_point_cloud, get_unmask_indices_from_rgb, to_tensor_segment, convert_segment_map_to_3d_coords, project_pixels
from lras_3d.utils.flow import compute_quantize_flow
from lras_3d.utils.object_motion_eval.object_motion_counterfactual import ObjectMotionCounterfactual
from lras_3d.utils.camera import pose_list_to_matrix, get_camera_orientation_dict_from_threepoints_depth_intrinsics

import os


def find_repo_root(path=None):
    if path is None:
        path = os.path.abspath(__file__)
    while path != os.path.dirname(path):
        if os.path.isdir(os.path.join(path, '.git')):
            return path
        path = os.path.dirname(path)
    raise RuntimeError("Repo root with .git not found.")


class ObjectMotionCounterfactualLRAS(ObjectMotionCounterfactual):

    def __init__(self, rollout_config, viz=False, special_decode_order=False, learnt_quantizer_frgb=False):
        super().__init__()

        repo_root = find_repo_root()

        # load depth model
        sys.path.insert(0, os.path.join(repo_root, 'external/depth_anything_v2/metric_depth'))
        from depth_anything_v2.dpt import DepthAnythingV2
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
            os.path.join(repo_root, f'checkpoints/depth_anything_v2/depth_anything_v2_metric_{dataset}_{encoder}.pth'),
            map_location='cpu'))
        depth_model.eval()
        self.depth_model = depth_model.to("cuda")

        from segment_anything import SamPredictor, sam_model_registry

        sam = sam_model_registry["vit_h"](checkpoint=os.path.join(repo_root, 'checkpoints/sam/sam_vit_h_4b8939.pth'))
        sam.to(device="cuda")
        sam.eval()
        self.segmentation_model = SamPredictor(sam)

        quantizer_name = "rgb_quantizer/rgb_quantizer.pt"
        rgb_model_name = "rgb_predictor/rgb_predictor.pt"
        fqn = None

        if not viz:
            self.lras_3d_rgb_predictor = LRASPredictor(rgb_model_name, quantizer_name, flow_quantizer_name=fqn, device="cuda")

        self.set_rollout_config(rollout_config)

        self.viz = viz

        self.special_decode_order = special_decode_order

        # self.flow_predictor = LRASPredictor('LRAS1B_RGB_C_F_quantized_mp_tpu_datav3/model_00180000.pt', 'LPQ_ps-4_vs-65536_nc-1_eb-1_db-11-medium-all_data/model_best.pt',
        #                                flow_quantizer_name='LPQ_ps-4_vs-32768_nc-1_eb-1_db-11-flow-10framegap/model_best.pt',  device="cuda")

        self.learnt_quantizer_frgb = learnt_quantizer_frgb

    def set_rollout_config(self, rollout_config):

        self.temperature = rollout_config["temperature"]
        self.top_k = rollout_config["top_k"]
        self.top_p = rollout_config["top_p"]
        self.rollout_mode = rollout_config["rollout_mode"]
        self.seed = rollout_config["seed"]

    @torch.no_grad()
    def predict_depth0_from_rgb0(self, rgb0_numpy_0to1):
        rgb0_numpy = (rgb0_numpy_0to1 * 255).astype(np.uint8)
        rgb0_numpy_bgr = cv2.cvtColor(rgb0_numpy, cv2.COLOR_RGB2BGR)
        depth0_numpy = self.depth_model.infer_image(rgb0_numpy_bgr)#, device="cuda")
        depth0_tensor = torch.tensor(depth0_numpy).unsqueeze(0).unsqueeze(0)  # 1x1xHxW
        return {"depth0_tensor": depth0_tensor, "depth0_numpy": depth0_numpy}

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

    def get_dense_flow_from_sparse_correspondences(self, image, depth_map0, depth_map1, points0, points1, K, use_full_segmentation=False):
        # estimate R and T from 2.5D correspondences
        R, T = get_RT(depth_map0, depth_map1, points0, points1, K)

        segment_map = self.get_segment_from_points(image, points0)

        #HARDCODED
        if use_full_segmentation:
            segment_map = np.ones_like(segment_map).astype(np.uint8)

        # segment the object and densify flow by warping the depth with R and T
        flow_map, segment_coords_in_pixels_img0, segment_coords_in_3d_img1, segment_map_frame1  = get_dense_flow_from_segment_depth_RT(segment_map, depth_map0, R, T, K)

        colors = image[segment_coords_in_pixels_img0[:, 0], segment_coords_in_pixels_img0[:, 1]]

        return flow_map, R, T, segment_map, segment_coords_in_3d_img1, colors, segment_map_frame1


    def get_segment_from_points(self, image, points):
        '''
        image: [H, W, 3] in (0, 255)
        points: [N, 2]
        '''

        segmentation_prompt = {"input_point": points, "input_label": [[1]*len(points)]}

        segment_dict = self.predict_segmentation0_from_rgb0(image / 255, segmentation_prompt)

        segment_map = segment_dict['segmentation0_numpy']

        return segment_map

    def cumulative_probability(self, logits, tokens):
        """
        Computes the cumulative probability of a sequence of tokens given logits.

        Args:
            logits (torch.Tensor): Tensor of shape (N, K) where N is the sequence length and K is the vocabulary size.
            tokens (torch.Tensor): Tensor of shape (N,) containing the predicted token indices.

        Returns:
            cum_prob (torch.Tensor): The cumulative probability of the sequence.
            cum_log_prob (torch.Tensor): The cumulative log probability of the sequence.
        """
        # Compute probabilities using softmax along the token dimension

        #reshape logits to (N, K)
        logits = logits.view(-1, logits.size(-1))
        #reshape tokens to (N,)
        tokens = tokens.view(-1)

        probs = F.softmax(logits, dim=1)

        # Select the probability corresponding to the predicted tokens at each time step
        selected_probs = probs[torch.arange(logits.shape[0]), tokens]

        # Cumulative probability: product of individual token probabilities
        cum_prob = torch.prod(selected_probs)

        # Alternatively, compute in log space to avoid numerical underflow
        log_probs = torch.log(selected_probs)
        cum_log_prob = torch.sum(log_probs)

        return cum_prob, cum_log_prob

    def predict_rgb1_from_flow(self, rgb0_numpy, flow_tensor_cuda, unmask_indices, decode_order=None, rgb1_numpy=None, unmask_indices_img1=None):
        '''
        rgb0_numpy: [H, W, 3]: (0, 255)
        flow_tensor_cuda: max pooled flow: [1, 2, 64, 64]
        unmask_indices: List of indices of where to reveal flow
        '''

        rgb0_numpy_0to1 = rgb0_numpy.astype(np.float32) / 255

        if rgb1_numpy is not None:
            rgb1_numpy_0to1 = rgb1_numpy.astype(np.float32) / 255
        else:
            rgb1_numpy_0to1 = rgb0_numpy_0to1

        flow_codes = compute_quantize_flow(flow_tensor_cuda, input_size=256, num_bins=512)

        # patchif
        # Quantize rgb
        rgb_prediction = self.lras_3d_rgb_predictor.flow_factual_prediction(
            rgb0_numpy_0to1, rgb1_numpy_0to1, flow=flow_codes, unmask_indices=unmask_indices,
            mode=self.rollout_mode, seed=self.seed,
            temperature=self.temperature, top_k=self.top_k, top_p=self.top_p, decoding_order=decode_order, unmask_indices_img1=unmask_indices_img1
        )

        logits = rgb_prediction['rgb_logits']

        frame1_pred_codes = rgb_prediction['frame1_pred_codes']

        logits = logits.view(32, 32, 2, 2, logits.shape[-1])

        logits = logits.permute([0, 2, 1, 3, 4])

        logits = logits.contiguous().view(64, 64, -1)

        prob, cum_log_prob = self.cumulative_probability(logits, frame1_pred_codes)

        return rgb_prediction['frame1_pred_pil'], cum_log_prob

    def get_inputs_for_forward(self, image, start_points, end_points, K, depth_img0, depth_img1, new_segment_sampling=False, use_full_segmentation=False, num_fg_flows=80, num_bg_flows=20):

        combined_flow_map = None
        segment_map = None
        point_cloud = None
        seg_map_frame1 = None
        for points0, points1 in zip(start_points, end_points):
            flow_map, R, T, seg_map, coords_3d, colors, frame1_estimated_segment_map = self.get_dense_flow_from_sparse_correspondences(
                image, depth_img0, depth_img1, points0, points1, K, use_full_segmentation=use_full_segmentation)

            pcl_img = render_point_cloud(coords_3d, colors, K, image.shape[:2])

            pcl_img = pcl_img.detach().cpu().numpy().astype('uint8')

            if combined_flow_map is None:
                segment_map = seg_map
                combined_flow_map = flow_map
                point_cloud = pcl_img
                seg_map_frame1 = frame1_estimated_segment_map
            else:
                segment_map = segment_map + seg_map
                combined_flow_map += flow_map
                point_cloud += pcl_img
                seg_map_frame1 = seg_map_frame1 + frame1_estimated_segment_map

        flow_map, indices_flow_in_256 = downsample_and_scale_flow(combined_flow_map)

        if self.special_decode_order:
            decode_order = get_decoding_order(flow_map)
        else:
            decode_order = None

        unmask_indices = get_unmask_indices_from_flow_map(flow_map, num_fg_flows=num_fg_flows, num_bg_flows=num_bg_flows,
                                                          new_sampling_method=new_segment_sampling)

        return flow_map, segment_map, point_cloud, seg_map_frame1, indices_flow_in_256, unmask_indices, decode_order

    def downsamle_segment_map(self, segment_map, kernel_size=32):

        segment_map_tensor = to_tensor_segment(segment_map)

        segment_map_tensor_fg = -torch.nn.functional.max_pool2d(-segment_map_tensor.float(), kernel_size=kernel_size, stride=kernel_size)[
            0, 0]
        segment_map_tensor_fg = segment_map_tensor_fg.bool().cpu().numpy()

        return segment_map_tensor_fg

    def get_unmask_inds_fromseg_map(self, segment_map):

        unmask_indices_rgb1 = get_flattened_index_from_2d_index(get_true_pixel_coords(~segment_map),
                                                                segment_map.shape[0])
        unmask_indices_rgb1 = unmask_indices_rgb1.tolist()
        np.random.shuffle(unmask_indices_rgb1)

        return unmask_indices_rgb1

    def prepare_inputs_world_method(self, threepoints_on_ground, image, start_points, K, depth_img0, R_world, T_world, new_segment_sampling=False, full_segment_map=False):

        cam_orientation = get_camera_orientation_dict_from_threepoints_depth_intrinsics(threepoints_on_ground, K[None],
                                                                                        depth_img0)
        cam_to_world = np.array(cam_orientation['transform_world_from_camera'][0][:3, :3])
        world_to_cam = np.array(cam_orientation['transform_camera_from_world'][:3, :3])

        # get centroid of object in camera coordinate system
        segment_map = self.get_segment_from_points(image, start_points)
        object_coords_in_3d_img0, _ = convert_segment_map_to_3d_coords(segment_map,
                                                                                                    depth_img0, K)
        centroid = np.mean(object_coords_in_3d_img0, axis=0)

        # make point cloud from segment map
        if full_segment_map:
            segment_map_for_flow = np.ones_like(segment_map)
        else:
            segment_map_for_flow = segment_map

        segment_coords_in_cam_3d_img0, segment_coords_in_pixels_img0 = convert_segment_map_to_3d_coords(
            segment_map_for_flow,
            depth_img0, K)

        # covert point cloud to new coordinate system centered at the centroid of the object

        segment_coords_in_cam_3d_img0 = segment_coords_in_cam_3d_img0 - centroid[None, :]
        segment_coords_in_world_3d_img0 = np.matmul(cam_to_world, segment_coords_in_cam_3d_img0.T).T

        # rotate the scene about the new coordinate system and then translate
        segment_coords_in_world_3d_img0 = np.matmul(R_world, segment_coords_in_world_3d_img0.T).T + T_world

        # bring back to camera system and untranslate
        segment_coords_in_cam_3d_img1 = np.matmul(world_to_cam, segment_coords_in_world_3d_img0.T).T + centroid[None, :]

        # project points to get flow
        segment_coords_in_pixels_img1 = project_pixels(segment_coords_in_cam_3d_img1, K)

        # get segment map: TODO need to make it from the segmented image
        segment_coords_in_pixels_img1 = segment_coords_in_pixels_img1.astype(int)
        segment_coords_in_pixels_img1 = np.clip(segment_coords_in_pixels_img1, 0, segment_map.shape[0] - 1)
        segment_map_img1_estimated = np.zeros([segment_map.shape[0], segment_map.shape[1]])
        segment_map_img1_estimated[segment_coords_in_pixels_img1[:, 0], segment_coords_in_pixels_img1[:, 1]] = 1

        # compute flow vectors
        flow_ = segment_coords_in_pixels_img1[:, :2] - segment_coords_in_pixels_img0

        # make flow map
        flow_map = np.zeros([segment_map.shape[0], segment_map.shape[1], 2])
        flow_map[segment_coords_in_pixels_img0[:, 0], segment_coords_in_pixels_img0[:, 1]] = flow_

        flow_map, indices_flow_in_256 = downsample_and_scale_flow(flow_map)

        if full_segment_map:
            num_fg_flows = 200
            num_bg_flows = 0
        else:
            num_fg_flows = 140
            num_bg_flows = 60

        unmask_indices_nvs = get_unmask_indices_from_flow_map(flow_map, num_fg_flows=num_fg_flows, num_bg_flows=num_bg_flows,
                                                              new_sampling_method=new_segment_sampling)

        return flow_map, unmask_indices_nvs, indices_flow_in_256, segment_map, segment_coords_in_pixels_img0, segment_coords_in_pixels_img1, segment_map_img1_estimated

    def run_forward_with_RT(self, image, start_points, threepoints_on_ground, R_world, T_world, K, condition_rgb=False, new_segment_sampling=False,
                    condition_from_nvs=False, full_segment_map=False):

        '''
        :param image: [H, W, 3] in [0, 1] range
        :param start_points: [N, 2] a list of K points for each mask, K >=3
        :param end_points: [N, 2] a list of K points for each mask, K >=3
        :return:
            counterfactual_image: [H, W, 3]
        '''

        image0_downsampled = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

        depth_img0 = self.predict_depth0_from_rgb0(image / 255)['depth0_numpy']

        flow_map, unmask_indices, indices_flow_in_256, segment_map, _, _, _ = \
            self.prepare_inputs_world_method(threepoints_on_ground, image, start_points, K, depth_img0, R_world, T_world, new_segment_sampling, full_segment_map)

        if self.viz:
            rgb1_pred = image0_downsampled
            cum_log_prob = 0
        else:
            rgb1_pred, cum_log_prob = self.predict_rgb1_from_flow(image0_downsampled, flow_map[:, [1, 0]], unmask_indices)

        if full_segment_map or (not(condition_from_nvs) and not(condition_rgb)):
            return rgb1_pred, flow_map, unmask_indices, indices_flow_in_256, segment_map, cum_log_prob
        else:
            flow_map, unmask_indices_obj_motion, indices_flow_in_256, _, _, _, segment_map_img1_estimated = \
                self.prepare_inputs_world_method(threepoints_on_ground, image, start_points, K, depth_img0, R_world,
                                                 T_world, new_segment_sampling, full_segment_map=False)

            image_1_gt = None

            unmask_indices_rgb1 = None

            if condition_from_nvs:

                segment_map_32 = self.downsamle_segment_map(segment_map_img1_estimated, kernel_size=32)
                unmask_indices_rgb1 = self.get_unmask_inds_fromseg_map(segment_map_32)
                unmask_indices_rgb1 = unmask_indices_rgb1[:2]
                image_1_gt = np.array(rgb1_pred)

            if condition_rgb:
                combined_seg_map = combine_dilated_bounding_boxes(segment_map.astype('uint8') * 255,
                                                                  segment_map_img1_estimated.astype('uint8') * 255, kernel_size=40)

                combined_seg_map_256 = cv2.resize(combined_seg_map, (256, 256), interpolation=cv2.INTER_AREA)
                combined_seg_map_256 = combined_seg_map_256.astype('bool')

                combined_seg_map = cv2.resize(combined_seg_map, (32, 32), interpolation=cv2.INTER_AREA)
                combined_seg_map = combined_seg_map.astype('bool')
                unmask_indices_rgb1_combined = get_flattened_index_from_2d_index(
                    get_true_pixel_coords(~combined_seg_map),
                    combined_seg_map.shape[0])
                unmask_indices_rgb1_combined = unmask_indices_rgb1_combined.tolist()
                np.random.shuffle(unmask_indices_rgb1_combined)

                unmask_indices_rgb1_combined = unmask_indices_rgb1_combined[:2]

                if unmask_indices_rgb1 is not None:
                    unmask_indices_rgb1 = unmask_indices_rgb1 + unmask_indices_rgb1_combined

                if image_1_gt is not None:
                    image_1_gt[~combined_seg_map_256] = image0_downsampled[~combined_seg_map_256]
                else:
                    image_1_gt = image0_downsampled

            if self.viz:
                rgb1_pred = image0_downsampled
                cum_log_prob = 0
            else:
                rgb1_pred, cum_log_prob = self.predict_rgb1_from_flow(image0_downsampled, flow_map[:, [1, 0]],
                                                                  unmask_indices_obj_motion, None, image_1_gt,
                                                                  unmask_indices_rgb1)

            return rgb1_pred, flow_map, unmask_indices_obj_motion, indices_flow_in_256, segment_map, cum_log_prob

    def run_forward(self, image, image_gt, start_points, end_points, K, condition_rgb=False, new_segment_sampling=False, condition_from_nvs=False):
        '''
        :param image: [H, W, 3] in [0, 1] range
        :param start_points: [N, K, 2] a list of K points for each mask, K >=3
        :param end_points: [N, K, 2] a list of K points for each mask, K >=3
        :return:
            counterfactual_image: [H, W, 3]
        '''


        image0_downsampled = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

        image1_downsampled = cv2.resize(image_gt, (256, 256), interpolation=cv2.INTER_AREA)

        depth_img0 = self.predict_depth0_from_rgb0(image / 255)['depth0_numpy']

        depth_img1 = self.predict_depth0_from_rgb0(image_gt / 255)['depth0_numpy']

        image_1_gt = None

        unmask_indices_rgb1 = None

        flow_map, segment_map, point_cloud, seg_map_frame1, indices_flow_in_256, unmask_indices, decode_order = self.get_inputs_for_forward(image, start_points, end_points, K, depth_img0, depth_img1, new_segment_sampling, use_full_segmentation=False, num_fg_flows=140, num_bg_flows=60)

        if condition_from_nvs:

            flow_map_nvs,  _, _, seg_map_frame1_nvs, _, unmask_indices_nvs, _ = self.get_inputs_for_forward(image, start_points, end_points, K, depth_img0, depth_img1, new_segment_sampling, use_full_segmentation=True, num_fg_flows=200, num_bg_flows=0)

            if self.viz:
                rgb1_pred_nvs = image0_downsampled
            else:
                rgb1_pred_nvs, _ = self.predict_rgb1_from_flow(image0_downsampled, flow_map_nvs[:, [1, 0]], unmask_indices_nvs)

            segment_map_32 = self.downsamle_segment_map(seg_map_frame1, kernel_size=32)

            unmask_indices_rgb1 = self.get_unmask_inds_fromseg_map(segment_map_32)

            unmask_indices_rgb1 = unmask_indices_rgb1[:2]

            image_1_gt = np.array(rgb1_pred_nvs)

        if condition_rgb:

            combined_seg_map = combine_dilated_bounding_boxes(segment_map.astype('uint8') * 255, seg_map_frame1.astype('uint8') * 255, kernel_size=40)

            combined_seg_map_256 = cv2.resize(combined_seg_map, (256, 256), interpolation=cv2.INTER_AREA)
            combined_seg_map_256 = combined_seg_map_256.astype('bool')

            combined_seg_map = cv2.resize(combined_seg_map, (32, 32), interpolation=cv2.INTER_AREA)
            combined_seg_map = combined_seg_map.astype('bool')
            unmask_indices_rgb1_combined = get_flattened_index_from_2d_index(get_true_pixel_coords(~combined_seg_map), combined_seg_map.shape[0])
            unmask_indices_rgb1_combined = unmask_indices_rgb1_combined.tolist()
            np.random.shuffle(unmask_indices_rgb1_combined)

            unmask_indices_rgb1_combined = unmask_indices_rgb1_combined[:2]

            if unmask_indices_rgb1 is not None:
                unmask_indices_rgb1 = unmask_indices_rgb1 + unmask_indices_rgb1_combined

            if image_1_gt is not None:
                image_1_gt[~combined_seg_map_256] = image0_downsampled[~combined_seg_map_256]
            else:
                image_1_gt = image0_downsampled

        if self.viz:
            rgb1_pred = image0_downsampled
            cum_log_prob = 0
        else:
            rgb1_pred, cum_log_prob = self.predict_rgb1_from_flow(image0_downsampled, flow_map[:, [1, 0]], unmask_indices, decode_order, image_1_gt, unmask_indices_rgb1)

        return rgb1_pred, flow_map, unmask_indices, indices_flow_in_256, segment_map, cum_log_prob

