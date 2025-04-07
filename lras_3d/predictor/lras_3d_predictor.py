import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from PIL import Image
from typing import Tuple, Union, List, Dict
import tqdm
import random

from lras_3d.utils.model_wrapper import ModelFactory
from lras_3d.utils.sequence_construction import (
    add_patch_indexes, get_pos_idxs, shuffle_and_trim_values_and_positions, supress_targets
)
from lras_3d.utils.image_processing import patchify, patchify_logits, unpatchify
# from lras_3d.utils.viz import mask_out_image
from lras_3d.utils.flow import decode_flow_code, sample_flow_values_and_positions
from lras_3d.utils.camera import transform_matrix_to_six_dof_axis_angle, quantize_6dof_campose


class LRASPredictor:

    def __init__(self, model_name: str, quantizer_name: str, flow_quantizer_name: str = None, device: str = 'cpu'):
        
        # Load the model and quantizer
        try:
            self.model = ModelFactory().load_model(model_name).to(torch.bfloat16).to(device).eval()
        except:
            self.model = ModelFactory().load_model_from_checkpoint(model_name).to(torch.bfloat16).to(device).eval()
        try:
            self.quantizer = ModelFactory().load_model(quantizer_name).to(device).to(torch.float32).eval()
        except:
            self.quantizer = ModelFactory().load_model_from_checkpoint(quantizer_name).to(device).to(torch.float32).eval()
        if flow_quantizer_name is not None:
            try:
                self.flow_quantizer = ModelFactory().load_model(flow_quantizer_name).to(device).to(torch.float32).eval()
            except:
                self.flow_quantizer = ModelFactory().load_model_from_checkpoint(flow_quantizer_name).to(device).to(torch.float32).eval()

        self.ctx = torch.amp.autocast(device_type='cuda' if 'cuda' in device else 'cpu', dtype=torch.bfloat16)

        # Set parameters
        self.device = device

        # Set transforms
        self.in_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.inv_in_transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
            torchvision.transforms.Lambda(lambda x: torch.clamp(x, 0, 1)), 
            torchvision.transforms.ToPILImage()
        ])

        self.flow_transform = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: torch.tensor(x).permute(2,0,1) if len(x.shape) == 3 else torch.tensor(x).permute(0,3,1,2)),
            torchvision.transforms.Normalize(mean=[0.0, 0.0], std=[20.0, 20.0]),
        ])
        self.inv_flow_transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.0, 0.0], std=[1/20.0, 1/20.0]),
            torchvision.transforms.Lambda(lambda x: x.permute(1,2,0).cpu().numpy() if len(x.shape) == 3 else x.permute(0,2,3,1).numpy()),
        ])

        self.resize_crop_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(256),
        ])

    @torch.no_grad()
    def flow_prediction(
            self, 
            frame0: Union[Image.Image, np.ndarray, torch.Tensor], 
            campose: torch.FloatTensor = None,
            num_flow_patches_to_predict: int = 100,
            flow_cond: List[List[int]] = None,
            mask_ratio: float = None,
            mask_indices: List[int] = None,
            mode: str = 'sequential',
            temperature: float = 1.0,
            top_p: float = 0.9,
            top_k: int = 1000,
            seed : int = 0,
            mask_out: bool = True,
        ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        
        """
        Perform a forward pass through the model using all of frame0 and part of frame1.

        Parameters:
            frame0: Image.Image, the first frame
            campose: torch.FloatTensor, the camera pose represented as a 4x4 transformation matrix
            mask_ratio: float, the ratio of patches to mask out
            mask_indices: List[int], the indices of the patches to mask out
            mode: str, the mode to use for the forward pass (sequential or parallel)
            seed: int, the random seed to use
            mask_out: bool, whether to black out the unmasked patches in the predicted frame
        
        Returns:
            frame0: Image.Image, the first frame
            frame1_pred: Image.Image, the predicted second frame
            frame1: Image.Image, the actual second frame
            rgb_logits: torch.FloatTensor, the logits for the predicted image codes
        """

        assert mode in ['sequential', 'parallel'], "Mode must be one of ['sequential', 'parallel']"
        self._set_seed(seed)
        
        # Transform the input frames and quantize them if they are provided as PIL images
        if isinstance(frame0, Image.Image) or isinstance(frame0, np.ndarray):
            frame0_codes = self.quantizer.quantize(self.in_transform(frame0).unsqueeze(0).to(self.device))
        else:
            frame0_codes = frame0

        # Transform the 4x4 campose matrix to a 6-DOF quantized vector
        if campose is not None:
            six_dof_campose = transform_matrix_to_six_dof_axis_angle(campose.cpu().numpy(), scale=True)
            campose_codes = torch.tensor(quantize_6dof_campose(six_dof_campose), dtype=torch.long)

        # Create flow conditioning
        if flow_cond is not None:
            flow_cond_seq = []
            flow_cond_pos = []
            # grab x, y indexes in each flow cond entry and the flow at that index
            for x, y, dx, dy in flow_cond:
                # convert x, y to patch index
                flow_patch_idx = x + y * 64
                flow_patch_idx_token = flow_patch_idx + self.model.config.flow_patch_idx_range[0]
                # grab dx and dy flow
                flow_midpoint = (self.model.config.flow_range[1] - self.model.config.flow_range[0]) // 2
                flow_patch_dx = dx + flow_midpoint + self.model.config.flow_range[0]
                flow_patch_dy = dy + flow_midpoint + self.model.config.flow_range[0]
                # grab positions
                flow_pos_idx = (3 * flow_patch_idx) + self.model.config.flow_pos_range[0]
                flow_pos_dx = flow_pos_idx + 1
                flow_pos_dy = flow_pos_idx + 2
                # append to flow cond seq
                flow_cond_seq.append(
                    torch.tensor([int(flow_patch_idx_token), int(flow_patch_dx), int(flow_patch_dy)], dtype=torch.long)
                )
                flow_cond_pos.append(
                    torch.tensor([int(flow_pos_idx), int(flow_pos_dx), int(flow_pos_dy)], dtype=torch.long)
                )
            # convert to tensor
            flow_cond_seq = torch.stack(flow_cond_seq)
            flow_cond_pos = torch.stack(flow_cond_pos)
        else:
            flow_cond_seq = None
            flow_cond_pos = None

                
        if mode == 'sequential':
            flow_pred_codes, flow_valid_mask = self.one_frame_flow_forward(
                frame0_codes.clone(), num_new_patches=num_flow_patches_to_predict, tokens_per_patch=3,
                flow_cond_seq=flow_cond_seq, flow_cond_pos=flow_cond_pos,
                campose_codes=campose_codes if campose is not None else None,
                temperature=temperature, top_p=top_p, top_k=top_k,
            )
        else:
            raise NotImplementedError("Parallel mode not implemented for flow prediction")
            # frame1_pred_codes, rgb_logits = self.two_frame_patchwise_parallel_forward(
            #     frame0_codes.clone(), unmask_idxs,
            #     temperature=temperature, top_p=top_p, top_k=top_k
            # )
        
        # Decode cond frame
        frame0 = self.quantizer.decode(frame0_codes.to(self.device))

        # Un-normalize image and convert to PIL
        frame0_pil = self.inv_in_transform(frame0[0])

        # Decode the predicted frame
        flow_pred = decode_flow_code(flow_pred_codes, input_size=256, num_bins=512)
        flow_pred_np = flow_pred[0].cpu().permute(1,2,0).numpy()

        return {
            "frame0_rgb": frame0,
            "frame0_pil": frame0_pil,
            "frame0_codes": frame0_codes[0],
            "flow_pred_np": flow_pred_np,
            "flow_pred_codes": flow_pred_codes[0],  
            "flow_valid_mask": flow_valid_mask[0],
        }

    @torch.no_grad()
    def flow_factual_prediction(
            self, 
            frame0: Union[Image.Image, np.ndarray, torch.Tensor], 
            frame1: Union[Image.Image, np.ndarray, torch.Tensor], 
            flow: torch.FloatTensor = None,
            unmask_indices: List[int] = None,
            mode: str = 'sequential',
            temperature: float = 1.0,
            top_p: float = 0.9,
            top_k: int = 1000,
            seed : int = 0,
            decoding_order: List[int] = None,
            unmask_indices_img1: List[int] = None,
        ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        
        """
        Perform a forward pass through the model using all of frame0 and part of frame1.

        Parameters:
            frame0: Image.Image, the first frame
            frame1: Image.Image, the second frame
            campose: torch.FloatTensor, the camera pose represented as a 4x4 transformation matrix
            mask_ratio: float, the ratio of patches to mask out
            mask_indices: List[int], the indices of the patches to mask out
            mode: str, the mode to use for the forward pass (sequential or parallel)
            seed: int, the random seed to use
            mask_out: bool, whether to black out the unmasked patches in the predicted frame
        
        Returns:
            frame0: Image.Image, the first frame
            frame1_pred: Image.Image, the predicted second frame
            frame1: Image.Image, the actual second frame
            rgb_logits: torch.FloatTensor, the logits for the predicted image codes
        """

        assert mode in ['sequential', 'parallel'], "Mode must be one of ['sequential', 'parallel']"

        self._set_seed(seed)
        
        # Transform the input frames and quantize them if they are provided as PIL images
        if isinstance(frame0, Image.Image) or isinstance(frame0, np.ndarray):
            frame0_codes = self.quantizer.quantize(self.in_transform(frame0).unsqueeze(0).to(self.device))
        else:
            frame0_codes = frame0
        if isinstance(frame1, Image.Image) or isinstance(frame1, np.ndarray):
            frame1_codes = self.quantizer.quantize(self.in_transform(frame1).unsqueeze(0).to(self.device))
        else:
            frame1_codes = frame1

        if unmask_indices is None:
            unmask_idxs = [x for x in range(1024)]
            random.shuffle(unmask_idxs)
            unmask_idxs = unmask_idxs[:100]
        else:
            unmask_idxs = unmask_indices

        if mode == 'sequential':
            frame1_pred_codes, rgb_logits, _ = self.frame0_flow_frame1_sequential_forward(
                frame0_codes.clone(),
                flow,
                frame1_codes.clone(),
                unmask_idxs=unmask_idxs,
                unmask_idxs_img1=unmask_indices_img1,
                temperature=temperature, top_p=top_p, top_k=top_k, decoding_order=decoding_order
            )
        else:
            raise NotImplementedError("Parallel mode not implemented for flow prediction")
            # frame1_pred_codes, rgb_logits = self.two_frame_patchwise_parallel_forward(
            #     frame0_codes.clone(), frame1_codes.clone(), unmask_idxs,
            #     temperature=temperature, top_p=top_p, top_k=top_k
            # )

        # Compute grid entropy and varentropy
        if unmask_indices_img1 is None:
            rgb_grid_entropy = self._compute_rgb_grid_entropy(rgb_logits, unmask_idxs).detach().cpu().float()
            rgb_grid_varentropy = self._compute_rgb_grid_varentropy(rgb_logits, unmask_idxs).detach().cpu().float()
            ce_error = self._compute_rgb_grid_ce_error(rgb_logits, frame1_codes).detach().cpu().float()
        else:
            rgb_grid_entropy = None
            rgb_grid_varentropy = None
            ce_error = None
            
        # Decode the predicted frame
        frame0 = self.quantizer.decode(frame0_codes.to(self.device))
        frame1_pred = self.quantizer.decode(frame1_pred_codes)
        frame1 = self.quantizer.decode(frame1_codes.to(self.device))

        # Un-normalize and convert to PIL
        frame0_pil = self.inv_in_transform(frame0[0])
        frame1_pred_pil = self.inv_in_transform(frame1_pred[0])
        frame1_pil = self.inv_in_transform(frame1[0])

        # Compute CE and MSE errors

        l1_error = self._compute_rgb_grid_l1_error(frame1_pred, frame1).detach().cpu().float()
        mse_error = self._compute_rgb_grid_mse_error(frame1_pred, frame1).detach().cpu().float()

    
        return {
            "frame0_rgb": frame0,
            "frame0_pil": frame0_pil,
            "frame0_codes": frame0_codes[0],
            "frame1_pred_rgb": frame1_pred,
            "frame1_pred_pil": frame1_pred_pil,
            "frame1_pred_codes": frame1_pred_codes[0],
            "frame1_rgb": frame1,
            "frame1_pil": frame1_pil,
            "frame1_codes": frame1_codes[0],
            "rgb_logits": rgb_logits,
            "rgb_grid_entropy": rgb_grid_entropy,
            "rgb_grid_varentropy": rgb_grid_varentropy,
            "ce_grid_error": ce_error,
            "l1_grid_error": l1_error,
            "mse_grid_error": mse_error,
        }
    
    @torch.no_grad()
    def quantized_flow_prediction(
            self, 
            frame0: Union[Image.Image, np.ndarray, torch.Tensor], 
            flow_cond: Union[np.ndarray, torch.Tensor] = None, 
            motion_indices: List[int] = None,
            campose: torch.FloatTensor = None,
            mask_ratio: float = 1.0,
            unmask_indices: List[int] = None,
            mode: str = 'sequential',
            temperature: float = 1.0,
            top_p: float = 0.9,
            top_k: int = 1000,
            seed : int = 0,
            num_seq_patches: int = 32,
            mask_out: bool = True,
            segment_map: torch.Tensor = None,
            rmi=None,
        ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        
        """
        Perform a forward pass through the model using all of frame0 and part of frame1.

        Parameters:
            frame0: Image.Image or np.ndarray or torch.Tensor, the first frame
            flow_cond: first value is horizontal index, second value is vertical index
            flow: torch.Tensor or np.ndarray, the flow represented as a 2-channel image
            campose: torch.FloatTensor, the camera pose represented as a 4x4 transformation matrix
            mask_ratio: float, the ratio of patches to mask out
            mask_indices: List[int], the indices of the patches to mask out
            mode: str, the mode to use for the forward pass (sequential or parallel)
            seed: int, the random seed to use
            mask_out: bool, whether to black out the unmasked patches in the predicted frame
        
        Returns:
            frame0: Image.Image, the first frame
            frame1_pred: Image.Image, the predicted second frame
            frame1: Image.Image, the actual second frame
            rgb_logits: torch.FloatTensor, the logits for the predicted image codes
        """

        assert mode in ['sequential', 'patchwise_parallel', 'parallel', 'seq2par'], "Mode must be one of ['sequential', 'patchwise_parallel', 'parallel']"
        # assert mask_ratio is not None or unmask_indices is not None, "Either mask_ratio or mask_indices must be provided"
        # assert mask_ratio is None or unmask_indices is None, "Only one of mask_ratio or mask_indices can be provided"
        
        # Transform the input frames and quantize them if they are provided as PIL images
        if isinstance(frame0, Image.Image) or isinstance(frame0, np.ndarray):
            frame0_codes = self.quantizer.quantize(self.in_transform(frame0).unsqueeze(0).to(self.device))
        else:
            frame0_codes = frame0


        flow = None
        if flow_cond is not None:
            # make a flow map depending on the flow conditioning
            flow = np.zeros((256, 256, 2), dtype=np.float32)
            # make a list of unmask indices
            unmask_indices = []

            for y, x, dy, dx in flow_cond:
                # divide dx and dy by 2 to convert from 512x512 to 256x256 grid
                dx = (dx-x) #/ 2
                dy = (dy-y) #/ 2
                # round x and y indices down to nearest multiple of 16
                x = int(x // 8)
                y = int(y // 8)
                # set the part of the flow map to the dx and dy values
                flow[8*x:8*(x+1), 8*y:8*(y+1), 0] = dy
                flow[8*x:8*(x+1), 8*y:8*(y+1), 1] = dx
                # convert x and y to patch index on a 32x32 grid
                patch_idx = y + x * 32
                unmask_indices.append(patch_idx)

        if segment_map is not None:
            flow = flow * segment_map[:, :, None]

        print("unmask_indices", unmask_indices)
                
        # If flow is None, make it a tensor of zeros like frame0_codes
        if flow is None:
            flow_codes = torch.zeros_like(frame0_codes)
        # If flow is a numpy array, convert it to a tensor with the flow quantizer
        if isinstance(flow, np.ndarray):
            flow_codes = self.flow_quantizer.quantize(self.flow_transform(torch.tensor(flow).unsqueeze(0).to(self.device)))
        # If flow is a tensor, assume it is already quantized codes
        if isinstance(flow, torch.Tensor):
            flow_codes = flow

        flow_codes = flow_codes + self.model.config.flow_range[0]
        # Transform the 4x4 campose matrix to a 6-DOF quantized vector
        if campose is not None:
            six_dof_campose = transform_matrix_to_six_dof_axis_angle(campose.cpu().numpy(), scale=True)
            campose_codes = torch.tensor(quantize_6dof_campose(six_dof_campose), dtype=torch.long)

        # Generate random list of unmasked indexes
        if unmask_indices is None:
            unmask_indices = random.sample(range(self.model.config.flow_patch_idx_range[1] - self.model.config.flow_patch_idx_range[0]), 
                int((self.model.config.flow_patch_idx_range[1] - self.model.config.flow_patch_idx_range[0]) * (1.0 - mask_ratio)))

        self._set_seed(seed)

        if unmask_indices is None:
            unmask_indices = []

        if mode == 'sequential':
            flow_pred_codes, flow_logits, decoding_order = self.two_frame_sequential_forward(
                frame0_codes.clone(), flow_codes.clone(), unmask_indices,
                frame0_patch_offset=None, frame0_seq_offset=None,
                frame1_patch_offset=self.model.config.flow_patch_idx_range[0],
                frame1_seq_offset=self.model.config.flow_pos_range[0],
                campose_codes=campose_codes if campose is not None else None,
                # flow_codes=flow if flow is not None else None,
                temperature=temperature, top_p=top_p, top_k=top_k,
            )
        elif mode == 'seq2par':
            flow_pred_codes, flow_logits, decoding_order = self.two_frame_seq2par_forward(
                frame0_codes.clone(), flow_codes.clone(), unmask_indices,
                frame0_patch_offset=None, frame0_seq_offset=None,
                frame1_patch_offset=self.model.config.flow_patch_idx_range[0],
                frame1_seq_offset=self.model.config.flow_pos_range[0],
                motion_indices=motion_indices,
                rmi=rmi,
                campose_codes=campose_codes if campose is not None else None,
                num_seq_patches=num_seq_patches, temperature=temperature, top_p=top_p, top_k=top_k,
            )

        # Compute grid entropy and varentropy
        flow_logits = flow_logits[-1024:]
        flow_grid_entropy = self._compute_flow_grid_entropy(flow_logits.cpu(), unmask_indices).detach().cpu().float()
        prob_no_motion = self._compute_flow_grid_cumulative_probability(
            flow_logits.cpu(),
            unmask_indices,
            [self.model.config.flow_range[0] + 11646, self.model.config.flow_range[0] + 11582],
            self.model.config.flow_range
        ).detach().cpu().float()
        # flow_rgb_grid_varentropy = self._compute_rgb_grid_varentropy(flow_logits.cpu(), unmask_indices).detach().cpu().float()

        # Decode the predicted frame
        flow_pred_codes = flow_pred_codes - self.model.config.flow_range[0]
        flow_codes = flow_codes - self.model.config.flow_range[0]
        frame0 = self.quantizer.decode(frame0_codes.to(self.device))
        flow_pred = self.flow_quantizer.decode(flow_pred_codes.to(self.device))
        flow = self.flow_quantizer.decode(flow_codes.to(self.device))

        # Un-normalize and convert to PIL
        frame0_pil = self.inv_in_transform(frame0[0])
        flow_pred_np = self.inv_flow_transform(flow_pred[0])
        flow_np = self.inv_flow_transform(flow[0])

        # Compute CE and MSE errors
        # ce_error = self._compute_rgb_grid_ce_error(rgb_logits.cpu(), frame1_codes.cpu()).detach().cpu().float()
        # l1_error = self._compute_rgb_grid_l1_error(frame1_pred.cpu(), frame1.cpu()).detach().cpu().float()
        # mse_error = self._compute_rgb_grid_mse_error(frame1_pred.cpu(), frame1.cpu()).detach().cpu().float()

        # Black out the unmasked patches if mask_out is True
        # if mask_out:
        #     flow_pred_np = mask_out_image(flow_pred_np, unmask_indices, color=200, patch_size=self.model.config.patch_size*4)

        return {
            "frame0_rgb": frame0,
            "frame0_pil": frame0_pil,
            "frame0_codes": frame0_codes[0],
            "flow_pred_rgb": flow_pred,
            "flow_pred_np": flow_pred_np,
            "flow_pred_codes": flow_pred_codes[0],
            "flow_rgb": flow,
            "flow_np": flow_np,
            "flow_logits": flow_logits.cpu(),
            # "frame1_codes": frame1_codes[0],
            # "rgb_logits": rgb_logits,
            "flow_grid_entropy": flow_grid_entropy,
            "prob_no_motion": prob_no_motion,
            # "flow_grid_varentropy": flow_rgb_grid_varentropy,
            # "ce_grid_error": ce_error,
            # "l1_grid_error": l1_error,
            # "mse_grid_error": mse_error,
            "decoding_order": decoding_order if mode == 'sequential' else []
        }

    def quantized_flow_prediction_with_flow_map(
            self,
            frame0: Union[Image.Image, np.ndarray, torch.Tensor],
            flow: Union[np.ndarray, torch.Tensor] = None,
            campose: torch.FloatTensor = None,
            mask_ratio: float = 1.0,
            unmask_indices: List[int] = None,
            mode: str = 'sequential',
            temperature: float = 1.0,
            top_p: float = 0.9,
            top_k: int = 1000,
            seed: int = 0,
            num_seq_patches: int = 32,
            mask_out: bool = True,
            segment_map: torch.Tensor = None,
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:

        """
        Perform a forward pass through the model using all of frame0 and part of frame1.

        Parameters:
            frame0: Image.Image or np.ndarray or torch.Tensor, the first frame
            flow_cond: flow map of size (H, W, 2)
            flow: torch.Tensor or np.ndarray, the flow represented as a 2-channel image
            campose: torch.FloatTensor, the camera pose represented as a 4x4 transformation matrix
            mask_ratio: float, the ratio of patches to mask out
            mask_indices: List[int], the indices of the patches to mask out
            mode: str, the mode to use for the forward pass (sequential or parallel)
            seed: int, the random seed to use
            mask_out: bool, whether to black out the unmasked patches in the predicted frame

        Returns:
            frame0: Image.Image, the first frame
            frame1_pred: Image.Image, the predicted second frame
            frame1: Image.Image, the actual second frame
            rgb_logits: torch.FloatTensor, the logits for the predicted image codes
        """

        assert mode in ['sequential', 'patchwise_parallel', 'parallel',
                        'seq2par'], "Mode must be one of ['sequential', 'patchwise_parallel', 'parallel']"
        # assert mask_ratio is not None or unmask_indices is not None, "Either mask_ratio or mask_indices must be provided"
        # assert mask_ratio is None or unmask_indices is None, "Only one of mask_ratio or mask_indices can be provided"

        # Transform the input frames and quantize them if they are provided as PIL images
        if isinstance(frame0, Image.Image) or isinstance(frame0, np.ndarray):
            frame0_codes = self.quantizer.quantize(self.in_transform(frame0).unsqueeze(0).to(self.device))
        else:
            frame0_codes = frame0

        if segment_map is not None:
            flow = flow * segment_map[:, :, None]

        print("unmask_indices", unmask_indices)

        # If flow is None, make it a tensor of zeros like frame0_codes
        if flow is None:
            flow_codes = torch.zeros_like(frame0_codes)
        # If flow is a numpy array, convert it to a tensor with the flow quantizer
        if isinstance(flow, np.ndarray):
            flow_codes = self.flow_quantizer.quantize(
                self.flow_transform(torch.tensor(flow).unsqueeze(0).to(self.device)))
        # If flow is a tensor, assume it is already quantized codes
        if isinstance(flow, torch.Tensor):
            flow_codes = flow

        flow_codes = flow_codes + self.model.config.flow_range[0]
        # Transform the 4x4 campose matrix to a 6-DOF quantized vector
        if campose is not None:
            six_dof_campose = transform_matrix_to_six_dof_axis_angle(campose.cpu().numpy(), scale=True)
            campose_codes = torch.tensor(quantize_6dof_campose(six_dof_campose), dtype=torch.long)

        # Generate random list of unmasked indexes
        if unmask_indices is None:
            unmask_indices = random.sample(
                range(self.model.config.flow_patch_idx_range[1] - self.model.config.flow_patch_idx_range[0]),
                int((self.model.config.flow_patch_idx_range[1] - self.model.config.flow_patch_idx_range[0]) * (
                            1.0 - mask_ratio)))

        self._set_seed(seed)

        if unmask_indices is None:
            unmask_indices = []

        if mode == 'sequential':
            flow_pred_codes, flow_logits, decoding_order = self.two_frame_sequential_forward(
                frame0_codes.clone(), flow_codes.clone(), unmask_indices,
                frame0_patch_offset=None, frame0_seq_offset=None,
                frame1_patch_offset=self.model.config.flow_patch_idx_range[0],
                frame1_seq_offset=self.model.config.flow_pos_range[0],
                campose_codes=campose_codes if campose is not None else None,
                # flow_codes=flow if flow is not None else None,
                temperature=temperature, top_p=top_p, top_k=top_k,
            )
        elif mode == 'seq2par':
            flow_pred_codes, flow_logits, decoding_order = self.two_frame_seq2par_forward(
                frame0_codes.clone(), flow_codes.clone(), unmask_indices,
                frame0_patch_offset=None, frame0_seq_offset=None,
                frame1_patch_offset=self.model.config.flow_patch_idx_range[0],
                frame1_seq_offset=self.model.config.flow_pos_range[0],
                campose_codes=campose_codes if campose is not None else None,
                num_seq_patches=num_seq_patches,
            )


        # Compute grid entropy and varentropy
        flow_logits = flow_logits[-1024:]
        flow_grid_entropy = self._compute_flow_grid_entropy(flow_logits.cpu(), unmask_indices).detach().cpu().float()
        prob_no_motion = self._compute_flow_grid_cumulative_probability(
            flow_logits.cpu(),
            unmask_indices,
            [self.model.config.flow_range[0] + 11646, self.model.config.flow_range[0] + 11582],
            self.model.config.flow_range
        ).detach().cpu().float()
        # flow_rgb_grid_varentropy = self._compute_rgb_grid_varentropy(flow_logits.cpu(), unmask_indices).detach().cpu().float()

        # Decode the predicted frame
        flow_pred_codes = flow_pred_codes - self.model.config.flow_range[0]
        flow_codes = flow_codes - self.model.config.flow_range[0]
        frame0 = self.quantizer.decode(frame0_codes.to(self.device))
        flow_pred = self.flow_quantizer.decode(flow_pred_codes.to(self.device))
        flow = self.flow_quantizer.decode(flow_codes.to(self.device))

        # Un-normalize and convert to PIL
        frame0_pil = self.inv_in_transform(frame0[0])
        flow_pred_np = self.inv_flow_transform(flow_pred[0])
        flow_np = self.inv_flow_transform(flow[0])

        # Compute CE and MSE errors
        # ce_error = self._compute_rgb_grid_ce_error(rgb_logits.cpu(), frame1_codes.cpu()).detach().cpu().float()
        # l1_error = self._compute_rgb_grid_l1_error(frame1_pred.cpu(), frame1.cpu()).detach().cpu().float()
        # mse_error = self._compute_rgb_grid_mse_error(frame1_pred.cpu(), frame1.cpu()).detach().cpu().float()

        # Black out the unmasked patches if mask_out is True
        # if mask_out:
        #     flow_pred_np = mask_out_image(flow_pred_np, unmask_indices, color=200, patch_size=self.model.config.patch_size*4)

        return {
            "frame0_rgb": frame0,
            "frame0_pil": frame0_pil,
            "frame0_codes": frame0_codes[0],
            "flow_pred_rgb": flow_pred,
            "flow_pred_np": flow_pred_np,
            "flow_pred_codes": flow_pred_codes[0],
            "flow_rgb": flow,
            "flow_np": flow_np,
            "flow_logits": flow_logits.cpu(),
            # "frame1_codes": frame1_codes[0],
            # "rgb_logits": rgb_logits,
            "flow_grid_entropy": flow_grid_entropy,
            "prob_no_motion": prob_no_motion,
            # "flow_grid_varentropy": flow_rgb_grid_varentropy,
            # "ce_grid_error": ce_error,
            # "l1_grid_error": l1_error,
            # "mse_grid_error": mse_error,
            "decoding_order": decoding_order if mode == 'sequential' else []
        }
      




    @torch.no_grad()
    def quantized_flow_conditioned_prediction(
            self, 
            frame0: Union[Image.Image, np.ndarray, torch.Tensor], 
            flow_cond: Union[np.ndarray, torch.Tensor] = None, 
            mask_ratio: float = 1.0,
            unmask_indices: List[int] = None,
            flow_unmask_indices: List[int] = None,
            mode: str = 'sequential',
            temperature: float = 1.0,
            top_p: float = 0.9,
            top_k: int = 1000,
            seed : int = 0,
            num_seq_patches: int = 32,
            mask_out: bool = True,
        ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        
        """
        Perform a forward pass through the model using all of frame0 and part of frame1.

        Parameters:
            frame0: Image.Image or np.ndarray or torch.Tensor, the first frame
            flow: torch.Tensor or np.ndarray, the flow represented as a 2-channel image
            campose: torch.FloatTensor, the camera pose represented as a 4x4 transformation matrix
            mask_ratio: float, the ratio of patches to mask out
            mask_indices: List[int], the indices of the patches to mask out
            mode: str, the mode to use for the forward pass (sequential or parallel)
            seed: int, the random seed to use
            mask_out: bool, whether to black out the unmasked patches in the predicted frame
        
        Returns:
            frame0: Image.Image, the first frame
            frame1_pred: Image.Image, the predicted second frame
            frame1: Image.Image, the actual second frame
            rgb_logits: torch.FloatTensor, the logits for the predicted image codes
        """

        assert mode in ['sequential', 'patchwise_parallel', 'parallel', 'seq2par'], "Mode must be one of ['sequential', 'patchwise_parallel', 'parallel']"
        # assert mask_ratio is not None or unmask_indices is not None, "Either mask_ratio or mask_indices must be provided"
        # assert mask_ratio is None or unmask_indices is None, "Only one of mask_ratio or mask_indices can be provided"
        
        # Transform the input frames and quantize them if they are provided as PIL images
        if isinstance(frame0, Image.Image) or isinstance(frame0, np.ndarray):
            frame0_codes = self.quantizer.quantize(self.in_transform(frame0).unsqueeze(0).to(self.device))
        else:
            frame0_codes = frame0

        print("unmask_indices", unmask_indices)
                
        flow_codes = flow_cond

        flow_codes = flow_codes + self.model.config.flow_range[0]

        self._set_seed(seed)

        if unmask_indices is None:
            unmask_indices = []

        if mode == 'sequential':
            flow_pred_codes, flow_logits, decoding_order = self.two_frame_sequential_forward(
                frame0_codes.clone(), frame0_codes.clone(), unmask_indices,
                frame0_patch_offset=None, frame0_seq_offset=None,
                frame1_patch_offset=self.model.config.flow_patch_idx_range[0],
                frame1_seq_offset=self.model.config.flow_pos_range[0],
                campose_codes=None,
                # flow_codes=flow if flow is not None else None,
                temperature=temperature, top_p=top_p, top_k=top_k,
            )
        elif mode == 'seq2par':
            frame1_pred_codes, rgb_logits, decoding_order = self.two_frame_seq2par_forward(
                frame0_codes.clone(), frame0_codes.clone(), unmask_indices, flow_unmask_indices,
                frame0_patch_offset=None, frame0_seq_offset=None,
                frame1_patch_offset=None, frame1_seq_offset=None,
                campose_codes=None,
                num_seq_patches=num_seq_patches,
                temperature=temperature, top_p=top_p, top_k=top_k,
                flow_codes=flow_codes.clone(),
            )

        # Compute rgb grid entropy and varentropy
        rgb_grid_entropy = self._compute_rgb_grid_entropy(rgb_logits.cpu(), unmask_indices).detach().cpu().float()
        rgb_grid_varentropy = self._compute_rgb_grid_varentropy(rgb_logits.cpu(), unmask_indices).detach().cpu().float()

        # Decode the predicted frame
        frame0 = self.quantizer.decode(frame0_codes.to(self.device))
        frame1_pred = self.quantizer.decode(frame1_pred_codes.to(self.device))
        frame1 = self.quantizer.decode(frame0_codes.to(self.device))

        # Un-normalize and convert to PIL
        frame0_pil = self.inv_in_transform(frame0[0])
        frame1_pred_pil = self.inv_in_transform(frame1_pred[0])
        frame1_pil = self.inv_in_transform(frame1[0])

        # Compute CE and MSE errors
        ce_error = self._compute_rgb_grid_ce_error(rgb_logits.cpu(), frame0_codes.cpu()).detach().cpu().float()
        l1_error = self._compute_rgb_grid_l1_error(frame1_pred.cpu(), frame1.cpu()).detach().cpu().float()
        mse_error = self._compute_rgb_grid_mse_error(frame1_pred.cpu(), frame1.cpu()).detach().cpu().float()

        # Black out the unmasked patches if mask_out is True
        if mask_out:
            frame1_pred_pil = mask_out_image(frame1_pred_pil, unmask_indices, color=200, patch_size=self.model.config.patch_size*4)

        return {
            "frame0_rgb": frame0,
            "frame0_pil": frame0_pil,
            "frame0_codes": frame0_codes[0],
            "frame1_pred_rgb": frame1_pred,
            "frame1_pred_pil": frame1_pred_pil,
            "frame1_pred_codes": frame1_pred_codes[0],
            "frame1_rgb": frame1,
            "frame1_pil": frame1_pil,
            "frame1_codes": frame0_codes[0],
            "rgb_logits": rgb_logits,
            "rgb_grid_entropy": rgb_grid_entropy,
            "rgb_grid_varentropy": rgb_grid_varentropy,
            "ce_grid_error": ce_error,
            "l1_grid_error": l1_error,
            "mse_grid_error": mse_error,
            "decoding_order": decoding_order if mode == 'sequential' else []
        }
        

    @torch.no_grad()
    def quantized_flow_conditioned_rgb_model_prediction(
        self, 
        frame0: Union[Image.Image, np.ndarray, torch.Tensor], 
        flow_cond: Union[np.ndarray, torch.Tensor] = None, 
        mask_ratio: float = 1.0,
        unmask_indices: List[int] = None,
        mode: str = 'sequential',
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 1000,
        seed : int = 0,
        num_seq_patches: int = 32,
        mask_out: bool = True,
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        
        """
        Copies patches from frame0 to a new location in frame1, based on flow_cond quantized tokens
        """
        
        flow_pred = self.flow_quantizer.decode(flow_cond.to('cuda'))
        flow_pred_np = self.inv_flow_transform(flow_pred[0])
        flow_pred_np = self.resize_flow(torch.tensor(flow_pred_np), 32, 32).numpy()
        points = self.create_points_list_from_flow(flow_pred_np)
        points_list = points
        points = np.array(points).reshape(-1, 4)
        print("points:", points.shape)
        
        # points: (N, 4) where N is the number of points and the 4 values are x1, y1, x2, y2
        fix_idxs = []
        move_src_idxs = []
        move_dst_idxs = []
        for x1, y1, x2, y2 in points:
            if x1 == x2 and y1 == y2:
                fix_idxs.append(y1 * 32 + x1)
            else:
                move_src_idxs.append(y1 * 32 + x1)
                move_dst_idxs.append(y2 * 32 + x2)
                
        num_idxs = len(fix_idxs) + len(move_src_idxs)
        if num_idxs >= num_seq_patches:
            num_seq_patches = 1
        else:
            num_seq_patches = num_seq_patches - num_idxs
            num_seq_patches = max(num_seq_patches, 1)
        
        rgb_prediction = self.counterfactual_prediction(
            frame0,
            move_src_idxs,
            move_dst_idxs,
            fix_idxs,
            seed=seed,
            mask_out=False,
            method="seq2par",
            num_seq_patches=num_seq_patches
        )
        
        return rgb_prediction, points_list












    @torch.no_grad()
    def factual_prediction(
            self, 
            frame0: Union[Image.Image, np.ndarray, torch.Tensor], 
            frame1: Union[Image.Image, np.ndarray, torch.Tensor], 
            campose: torch.FloatTensor = None,
            flow: torch.FloatTensor = None,
            mask_ratio: float = None,
            unmask_indices: List[int] = None,
            mode: str = 'sequential',
            temperature: float = 1.0,
            top_p: float = 0.9,
            top_k: int = 1000,
            seed : int = 0,
            num_seq_patches: int = 32,
            mask_out: bool = True,
        ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        
        """
        Perform a forward pass through the model using all of frame0 and part of frame1.

        Parameters:
            frame0: Image.Image, the first frame
            frame1: Image.Image, the second frame
            campose: torch.FloatTensor, the camera pose represented as a 4x4 transformation matrix
            mask_ratio: float, the ratio of patches to mask out
            mask_indices: List[int], the indices of the patches to mask out
            mode: str, the mode to use for the forward pass (sequential or parallel)
            seed: int, the random seed to use
            mask_out: bool, whether to black out the unmasked patches in the predicted frame
        
        Returns:
            frame0: Image.Image, the first frame
            frame1_pred: Image.Image, the predicted second frame
            frame1: Image.Image, the actual second frame
            rgb_logits: torch.FloatTensor, the logits for the predicted image codes
        """

        assert mode in ['sequential', 'patchwise_parallel', 'seq2par', 'parallel'], "Mode must be one of ['sequential', 'patchwise_parallel', 'parallel']"
        assert mask_ratio is not None or unmask_indices is not None, "Either mask_ratio or mask_indices must be provided"
        assert mask_ratio is None or unmask_indices is None, "Only one of mask_ratio or mask_indices can be provided"

        self._set_seed(seed)
        
        # Transform the input frames and quantize them if they are provided as PIL images
        if isinstance(frame0, Image.Image) or isinstance(frame0, np.ndarray):
            frame0_codes = self.quantizer.quantize(self.in_transform(frame0).unsqueeze(0).to(self.device))
        else:
            frame0_codes = frame0
        if isinstance(frame1, Image.Image) or isinstance(frame1, np.ndarray):
            frame1_codes = self.quantizer.quantize(self.in_transform(frame1).unsqueeze(0).to(self.device))
        else:
            frame1_codes = frame1

        # Transform the 4x4 campose matrix to a 6-DOF quantized vector
        if campose is not None:
            six_dof_campose = transform_matrix_to_six_dof_axis_angle(campose.cpu().numpy(), scale=True)
            campose_codes = torch.tensor(quantize_6dof_campose(six_dof_campose), dtype=torch.long)

        # Generate random list of unmasked indexes
        if unmask_indices is None:
            unmask_indices = random.sample(range(self.model.config.rgb_patch_1_idx_range[1] - self.model.config.rgb_patch_1_idx_range[0]), 
                int((self.model.config.rgb_patch_1_idx_range[1] - self.model.config.rgb_patch_1_idx_range[0]) * (1.0 -mask_ratio)))

        self._set_seed(seed)

        if mode == 'sequential':
            frame1_pred_codes, rgb_logits, decoding_order = self.two_frame_sequential_forward(
                frame0_codes.clone(), frame1_codes.clone(), unmask_indices,
                campose_codes=campose_codes if campose is not None else None,
                flow_codes=flow if flow is not None else None,
                temperature=temperature, top_p=top_p, top_k=top_k,
            )
        elif mode == 'patchwise_parallel':
            frame1_pred_codes, rgb_logits = self.two_frame_patchwise_parallel_forward(
                frame0_codes.clone(), frame1_codes.clone(), unmask_indices,
                temperature=temperature, top_p=top_p, top_k=top_k,
                campose_codes=campose_codes if campose is not None else None,
                flow_codes=flow if flow is not None else None,
            )
        elif mode == 'seq2par':
            frame1_pred_codes, rgb_logits, decoding_order = self.two_frame_seq2par_forward(
                frame0_codes.clone(), frame1_codes.clone(), unmask_indices,
                campose_codes=campose_codes if campose is not None else None,
                temperature=temperature, top_p=top_p, top_k=top_k,
                num_seq_patches=num_seq_patches,
            )
        elif mode == 'parallel':
            frame1_pred_codes, rgb_logits = self.two_frame_parallel_forward(
                frame0_codes.clone(), frame1_codes.clone(), unmask_indices,
                campose_codes=campose_codes if campose is not None else None,
                flow_codes=flow if flow is not None else None,
            )

        # Compute grid entropy and varentropy
        rgb_grid_entropy = self._compute_rgb_grid_entropy(rgb_logits.cpu(), unmask_indices).detach().cpu().float()
        rgb_grid_varentropy = self._compute_rgb_grid_varentropy(rgb_logits.cpu(), unmask_indices).detach().cpu().float()
            
        # Decode the predicted frame
        frame0 = self.quantizer.decode(frame0_codes.to(self.device))
        frame1_pred = self.quantizer.decode(frame1_pred_codes.to(self.device))
        frame1 = self.quantizer.decode(frame1_codes.to(self.device))

        # Un-normalize and convert to PIL
        frame0_pil = self.inv_in_transform(frame0[0])
        frame1_pred_pil = self.inv_in_transform(frame1_pred[0])
        frame1_pil = self.inv_in_transform(frame1[0])

        # Compute CE and MSE errors
        ce_error = self._compute_rgb_grid_ce_error(rgb_logits.cpu(), frame1_codes.cpu()).detach().cpu().float()
        l1_error = self._compute_rgb_grid_l1_error(frame1_pred.cpu(), frame1.cpu()).detach().cpu().float()
        mse_error = self._compute_rgb_grid_mse_error(frame1_pred.cpu(), frame1.cpu()).detach().cpu().float()

        # Black out the unmasked patches if mask_out is True
        if mask_out:
            frame1_pred_pil = mask_out_image(frame1_pred_pil, unmask_indices, color=200, patch_size=self.model.config.patch_size*4)

        return {
            "frame0_rgb": frame0,
            "frame0_pil": frame0_pil,
            "frame0_codes": frame0_codes[0],
            "frame1_pred_rgb": frame1_pred,
            "frame1_pred_pil": frame1_pred_pil,
            "frame1_pred_codes": frame1_pred_codes[0],
            "frame1_rgb": frame1,
            "frame1_pil": frame1_pil,
            "frame1_codes": frame1_codes[0],
            "rgb_logits": rgb_logits,
            "rgb_grid_entropy": rgb_grid_entropy,
            "rgb_grid_varentropy": rgb_grid_varentropy,
            "ce_grid_error": ce_error,
            "l1_grid_error": l1_error,
            "mse_grid_error": mse_error,
            "decoding_order": decoding_order if mode == 'sequential' else []
        }
    
    @torch.no_grad()
    def factual_prediction_multiple_inputs(
            self,
            frame0_list: List[Union[Image.Image, np.ndarray, torch.Tensor]],
            frame1: Union[Image.Image, np.ndarray, torch.Tensor],
            campose_list: List[torch.FloatTensor] = None,
            flow_list: List[torch.FloatTensor] = None,
            input_rgb_mask_ratio: float = None,
            input_rgb_unmask_indices: List[int] = None,
            input_flow_mask_ratio: float = None,
            input_flow_unmask_indices_list: List[List[int]] = None,
            output_rgb_mask_ratio: float = None,
            output_rgb_indices: List[int] = None,
            mode: str = 'sequential',
            temperature: float = 1.0,
            top_p: float = 0.9,
            top_k: int = 1000,
            seed : int = 0,
            mask_out: bool = True,
            scaling_factor_list: List[float] = None,
            decoding_strategy: str = None,
        ) -> List[Dict[str, Union[Image.Image, torch.Tensor]]]:
            
        """
        This function quantize input and give it to the next function
        Then, take the output prediction and compute error after decoding the output.
        """
        
        assert mode in ['sequential'], "Mode must be one of ['sequential']. Other modes ('patchwise_parallel', 'parallel') are not implemented yet"

        self._set_seed(seed)

        num_inputs = len(frame0_list)
        if campose_list is not None:
            assert len(campose_list) == num_inputs, "The number of campose matrices must match the number of input frames"
        if flow_list is not None:
            assert len(flow_list) == num_inputs, "The number of flow matrices must match the number of input frames"

        # Transform input frames and quantize them if they are provided as PIL images
        frame0_codes_list = []
        for frame0 in frame0_list:
            if isinstance(frame0, Image.Image) or isinstance(frame0, np.ndarray):
                frame0_codes = self.quantizer.quantize(self.in_transform(frame0).unsqueeze(0).to(self.device))
            else:
                frame0_codes = frame0
            frame0_codes_list.append(frame0_codes)

        # We assume there will be only one output frame
        if isinstance(frame1, Image.Image) or isinstance(frame1, np.ndarray):
            frame1_codes = self.quantizer.quantize(self.in_transform(frame1).unsqueeze(0).to(self.device))
        else:
            frame1_codes = frame1

        # Transform the 4x4 campose matrix to a 6-DOF quantized vector
        if campose_list is not None:
            six_dof_campose_list = [transform_matrix_to_six_dof_axis_angle(campose.cpu().numpy(), scale=True) for campose in campose_list]
            campose_codes_list = [torch.tensor(quantize_6dof_campose(six_dof_campose), dtype=torch.long) for six_dof_campose in six_dof_campose_list]

        # Generate random list of unmasked indexes
        if input_rgb_unmask_indices is None:
            input_rgb_unmask_indices = random.sample(range(self.model.config.rgb_patch_1_idx_range[1] - self.model.config.rgb_patch_1_idx_range[0]), 
                int((self.model.config.rgb_patch_1_idx_range[1] - self.model.config.rgb_patch_1_idx_range[0]) * (1.0 - input_rgb_mask_ratio)))
            
        # Generate random list of unmasked indexes for flow
        if input_flow_unmask_indices_list is None and input_flow_mask_ratio is not None:
            input_flow_unmask_indices_list = []
            for _ in range(num_inputs):
                input_flow_unmask_indices_list.append(random.sample(range(self.model.config.flow_patch_1_idx_range[1] - self.model.config.flow_patch_1_idx_range[0]), 
                    int((self.model.config.flow_patch_1_idx_range[1] - self.model.config.flow_patch_1_idx_range[0]) * (1.0 - input_flow_mask_ratio))))

        # Get entropy map using parallel forward
        frame1_pred_codes, rgb_logits = self.two_frame_patchwise_parallel_forward(
                frame0_codes.clone(), frame1_codes.clone(), input_rgb_unmask_indices,
                temperature=temperature, top_p=top_p, top_k=top_k,
                flow_codes=flow_list[0] if flow_list is not None else None,
                flow_unmask_idxs=input_flow_unmask_indices_list[0] if input_flow_unmask_indices_list is not None else None,
            )

        # Get entropy map from rgb_logits in parallel forward
        rgb_grid_entropy = self._compute_rgb_grid_entropy(rgb_logits, input_rgb_unmask_indices).detach().cpu().float()
        rgb_grid_varentropy = self._compute_rgb_grid_varentropy(rgb_logits, input_rgb_unmask_indices).detach().cpu().float()

        # Generate random list of unmasked indexes for output frame
        if output_rgb_indices is None and decoding_strategy is None:
            if output_rgb_mask_ratio is None:
                output_rgb_mask_ratio = 0.0
            all_indices = list(range(self.model.config.rgb_patch_1_idx_range[1] - self.model.config.rgb_patch_1_idx_range[0]))
            all_indices_except_input_indices = list(set(all_indices) - set(input_rgb_unmask_indices))
            random.shuffle(all_indices_except_input_indices)
            output_rgb_indices = all_indices_except_input_indices[:int(len(all_indices) * (1.0 - output_rgb_mask_ratio))]
        elif output_rgb_indices is None and decoding_strategy == 'entropy_low_to_high':
            if output_rgb_mask_ratio is None:
                output_rgb_mask_ratio = 0.0
            # Flatten the tensor
            flat_tensor = rgb_grid_entropy.flatten()
            # Get the indices that would sort the array
            sorted_indices = flat_tensor.argsort(descending=False)
            sorted_indices_except_input_indices = [idx.item() for idx in sorted_indices if idx not in input_rgb_unmask_indices]
            output_rgb_indices = sorted_indices_except_input_indices[:int(len(sorted_indices_except_input_indices) * (1.0 - output_rgb_mask_ratio))]
        elif output_rgb_indices is None and decoding_strategy == 'entropy_high_to_low':
            if output_rgb_mask_ratio is None:
                output_rgb_mask_ratio = 0.0
            # Flatten the tensor
            flat_tensor = rgb_grid_entropy.flatten()
            # Get the indices that would sort the array
            sorted_indices = flat_tensor.argsort(descending=True)
            sorted_indices_except_input_indices = [idx.item() for idx in sorted_indices if idx not in input_rgb_unmask_indices]
            output_rgb_indices = sorted_indices_except_input_indices[:int(len(sorted_indices_except_input_indices) * (1.0 - output_rgb_mask_ratio))]

        # Run the sequential forward pass
        frame1_pred_codes, rgb_logits, decoding_order = self.two_frame_sequential_forward_multiple_inputs(
            frame0_codes_list, frame1_codes, 
            input_rgb_unmask_indices=input_rgb_unmask_indices, 
            input_flow_unmask_indices_list=input_flow_unmask_indices_list,
            output_rgb_indices=output_rgb_indices,
            campose_codes_list=campose_codes_list if campose_list is not None else None,
            flow_codes_list=flow_list if flow_list is not None else None,
            temperature=temperature, top_p=top_p, top_k=top_k,
            scaling_factor_list=scaling_factor_list,
        )

        # Decode the predicted frames
        frame0_list = [self.quantizer.decode(frame0_codes.to(self.device)) for frame0_codes in frame0_codes_list]
        frame1_pred = self.quantizer.decode(frame1_pred_codes.to(self.device))
        frame1 = self.quantizer.decode(frame1_codes.to(self.device))

        # Un-normalize and convert to PIL
        frame0_pil_list = [self.inv_in_transform(frame0[0]) for frame0 in frame0_list]
        frame1_pred_pil = self.inv_in_transform(frame1_pred[0])
        frame1_pil = self.inv_in_transform(frame1[0])

        # Compute CE and MSE errors
        ce_error = self._compute_rgb_grid_ce_error(rgb_logits.cpu(), frame1_codes.cpu()).detach().cpu().float()
        l1_error = self._compute_rgb_grid_l1_error(frame1_pred.cpu(), frame1.cpu()).detach().cpu().float()
        mse_error = self._compute_rgb_grid_mse_error(frame1_pred.cpu(), frame1.cpu()).detach().cpu().float()

        # Black out the unmasked patches if mask_out is True
        if mask_out:
            frame1_pred_pil = mask_out_image(frame1_pred_pil, input_rgb_unmask_indices, color=200, patch_size=self.model.config.patch_size*4)
            remaining_indices = list(set(range(1024)) - set(output_rgb_indices) - set(input_rgb_unmask_indices))
            if len(remaining_indices) > 0:
                frame1_pred_pil = mask_out_image(frame1_pred_pil, remaining_indices, color=200, patch_size=self.model.config.patch_size*4)

        return {
            "frame0_rgb_list": frame0_list,
            "frame0_pil_list": frame0_pil_list,
            "frame0_codes_list": frame0_codes_list,
            "frame1_pred_rgb": frame1_pred,
            "frame1_pred_pil": frame1_pred_pil,
            "frame1_pred_codes": frame1_pred_codes,
            "frame1_rgb": frame1,
            "frame1_pil": frame1_pil,
            "rgb_logits": rgb_logits,
            "ce_grid_error": ce_error,
            "l1_grid_error": l1_error,
            "mse_grid_error": mse_error,
            "decoding_order": decoding_order,
            "rgb_grid_entropy": rgb_grid_entropy,
            "rgb_grid_varentropy": rgb_grid_varentropy,
        }

    
    @torch.no_grad()
    def counterfactual_prediction(
        self,
        frame0: Image.Image, 
        move_src_idxs: List[int] = [], 
        move_dst_indxs: List[int] = [], 
        fix_idxs: List[int] = [],
        campose: torch.FloatTensor = None,
        seed: int = 0, 
        mask_out: bool = True, 
        method: str = 'sequential',
        num_seq_patches: int = 32,
        frame0_for_move_counterfactual_list: List[Image.Image] = None,
        frame0_for_fix_counterfactual_list: List[Image.Image] = None,
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:

        assert len(move_src_idxs) == len(move_dst_indxs)

        self._set_seed(seed)
        
        # Transform the input frame and quantize them
        frame0_codes = self.quantizer.quantize(self.in_transform(frame0).unsqueeze(0).to(self.device))

        # Flatten the frame codes and make a copy for the second frame
        flat_frame0_codes = patchify(frame0_codes.clone(), patch_size=self.model.config.patch_size)
        flat_frame1_codes = flat_frame0_codes.clone()

        # Transform the 4x4 campose matrix to a 6-DOF quantized vector
        if campose is not None:
            six_dof_campose = transform_matrix_to_six_dof_axis_angle(campose.cpu().numpy(), scale=True)
            campose_codes = torch.tensor(quantize_6dof_campose(six_dof_campose), dtype=torch.long)

        print(f"Counterfactual motion patches from frame0: {len(move_src_idxs)}")
        print(f"Counterfactual motion patches to frame1: {len(move_dst_indxs)}")
        print(f"Counterfactual fix patches: {len(fix_idxs)}")

        # Insert counterfactual motion patches from frame0 codes into frame1
        if frame0_for_move_counterfactual_list is None:
            for src, dst in zip(move_src_idxs, move_dst_indxs):
                # Convert the source indexes into 2D indexes
                flat_frame1_codes[:, dst] = flat_frame0_codes[:, src]
        else:
            for idx, (src, dst) in enumerate(zip(move_src_idxs, move_dst_indxs)):
                # print(f"Counterfactual patch from transformed image {idx//4}/{len(move_src_idxs)//4}")
                # Convert the source indexes into 2D indexes
                flat_frame0_codes = patchify(self.quantizer.quantize(self.in_transform(frame0_for_move_counterfactual_list[idx//4]).unsqueeze(0).to(self.device)).clone(), patch_size=self.model.config.patch_size)
                flat_frame1_codes[:, dst] = flat_frame0_codes[:, src]
            for idx, fix in enumerate(fix_idxs):
                flat_frame0_codes = patchify(self.quantizer.quantize(self.in_transform(frame0_for_fix_counterfactual_list[idx//4]).unsqueeze(0).to(self.device)).clone(), patch_size=self.model.config.patch_size)
                flat_frame1_codes[:, fix] = flat_frame0_codes[:, fix]

        frame1_codes = unpatchify(flat_frame1_codes)
        unmask_idxs = fix_idxs + move_dst_indxs

        if method == 'sequential':
            frame1_pred_codes, rgb_logits, decoding_order = self.two_frame_sequential_forward(
                frame0_codes.clone(), frame1_codes.clone(), unmask_idxs,
                campose_codes=campose_codes if campose is not None else None
            )
        elif method == 'seq2par':
            frame1_pred_codes, rgb_logits, decoding_order = self.two_frame_seq2par_forward(
                frame0_codes.clone(), frame1_codes.clone(), unmask_idxs,
                campose_codes=campose_codes if campose is not None else None,
                num_seq_patches=num_seq_patches,
            )
        else:
            frame1_pred_codes, rgb_logits = self.two_frame_patchwise_parallel_forward(
                frame0_codes.clone(), frame1_codes.clone(), unmask_idxs,
                campose_codes=campose_codes if campose is not None else None
            )

        # Compute grid entropy and varentropy
        rgb_grid_entropy = self._compute_rgb_grid_entropy(rgb_logits, unmask_idxs).detach().cpu().float()
        rgb_grid_varentropy = self._compute_rgb_grid_varentropy(rgb_logits, unmask_idxs).detach().cpu().float()

        # Decode the predicted frame
        frame0 = self.quantizer.decode(frame0_codes.to(self.device))
        frame1_pred = self.quantizer.decode(frame1_pred_codes.to(self.device))
        frame1 = self.quantizer.decode(frame1_codes.to(self.device))
        
        # Compute CE and MSE errors
        ce_error = self._compute_rgb_grid_ce_error(rgb_logits, frame1_codes).detach().cpu().float()
        l1_error = self._compute_rgb_grid_l1_error(frame1_pred, frame1).detach().cpu().float()
        mse_error = self._compute_rgb_grid_mse_error(frame1_pred, frame1).detach().cpu().float()

        # Un-normalize and convert to PIL
        frame0_pil = self.inv_in_transform(frame0[0])
        frame1_pred_pil = self.inv_in_transform(frame1_pred[0])
        frame1_pil = self.inv_in_transform(frame1[0])

        # Black out the unmasked patches if mask_out is True
        if mask_out:
            frame1_pred = mask_out_image(frame1_pred_pil, move_dst_indxs, color=(0, 255, 0), patch_size=self.model.config.patch_size*4)
            frame1_pred = mask_out_image(frame1_pred_pil, move_src_idxs, color=(255, 0, 0), patch_size=self.model.config.patch_size*4)
            frame0 = mask_out_image(frame0_pil, move_dst_indxs, color=(0, 255, 0), patch_size=self.model.config.patch_size*4)
            frame0 = mask_out_image(frame0_pil, move_src_idxs, color=(255, 0, 0), patch_size=self.model.config.patch_size*4)
            frame0 = mask_out_image(frame0_pil, fix_idxs, color=200, patch_size=self.model.config.patch_size*4)

        return {
            "frame0_rgb": frame0,
            "frame0_pil": frame0_pil,
            "frame0_codes": frame0_codes[0].detach().cpu(),
            "frame1_pred_rgb": frame1_pred,
            "frame1_pred_pil": frame1_pred_pil,
            "frame1_pred_codes": frame1_pred_codes[0].detach().cpu(),
            "frame1_rgb": frame1,
            "frame1_pil": frame1_pil,
            "frame1_codes": frame1_codes[0].detach().cpu(),
            "rgb_logits": rgb_logits,
            "rgb_grid_entropy": rgb_grid_entropy,
            "rgb_grid_varentropy": rgb_grid_varentropy,
            "ce_grid_error": ce_error,
            "l1_grid_error": l1_error,
            "mse_grid_error": mse_error,
            "decoding_order": decoding_order
        }

    @torch.no_grad()
    def compute_probability_second_frame(
        self, 
        frame0: Union[Image.Image, np.ndarray, torch.Tensor],
        frame1: Union[Image.Image, np.ndarray, torch.Tensor], 
    ) -> float:
        """
        Compute the probability of second frame given the first frame.

        Parameters:
            frame0: Image.Image or np.ndarray or torch.Tensor, the first frame
            frame1: Image.Image or np.ndarray or torch.Tensor, the second frame

        Returns:
            probability: float, the probability of the second frame given the first frame
        """
        
        # Transform the input frames and quantize them if they are provided as PIL images
        if isinstance(frame0, Image.Image) or isinstance(frame0, np.ndarray):
            frame0 = self.resize_crop_transform(frame0)
            frame0_codes = self.quantizer.quantize(self.in_transform(frame0).unsqueeze(0).to(self.device))
        else:
            frame0_codes = frame0
        if isinstance(frame1, Image.Image) or isinstance(frame1, np.ndarray):
            frame1 = self.resize_crop_transform(frame1)
            frame1_codes = self.quantizer.quantize(self.in_transform(frame1).unsqueeze(0).to(self.device))
        else:
            frame1_codes = frame1

        # Pack the images into sequences
        # im0_seq, im0_pos, im1_seq, im1_pos: [1, 1024, 5]
        im0_seq, im0_pos = self._pack_image_codes_into_sequence(
            frame0_codes.cpu(), mask=0.0,
            patch_offset=self.model.config.rgb_patch_0_idx_range[0], 
            seq_offset=self.model.config.frame_0_rgb_pos_range[0])
        im1_seq, im1_pos = self._pack_image_codes_into_sequence(
            frame1_codes.cpu(), mask=0.0,
            patch_offset=self.model.config.rgb_patch_1_idx_range[0], 
            seq_offset=self.model.config.frame_1_rgb_pos_range[0])

        # create seq, pos: [10240]
        seq = torch.cat([im0_seq.view(-1), im1_seq.view(-1)])
        pos = torch.cat([im0_pos.view(-1), im1_pos.view(-1)])
        
        # create tgt: [10240]: [   -1,    -1,    -1,  ..., 15635, 48403, 15635]
        tgt = seq.clone()
        tgt[:im0_seq.numel()] = -1
        tgt = supress_targets(tgt, self.model.config.rgb_patch_1_idx_range)

        # create mask: [10240]: [True, True, True,  ..., True, True, True]
        # WARNING: this mask does not matter; a causal mask is created downstream to replace this
        mask = torch.ones_like(seq).bool()

        # Use seq, pos to predict tgt
        seq, pos, tgt, mask = seq[:-1], pos[:-1], tgt[1:], mask

        # Perform the prediction
        with self.ctx:
            logits, loss = self.model(
                seq.unsqueeze(0).to(self.device).long(), 
                pos=pos.unsqueeze(0).to(self.device).long(),
                mask=mask.unsqueeze(0).to(self.device).bool(),
                tgt=tgt.unsqueeze(0).to(self.device).long(),
            )
        
        # Compute the probability of the second frame given the first frame
        prob = torch.exp(-loss).item()    
        return prob, logits

    @torch.no_grad()
    def two_frame_parallel_forward(
        self, 
        frame0_codes: torch.LongTensor, 
        frame1_codes: torch.LongTensor, 
        unmask_idxs: List[int],
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Perform a forward pass through the model using the "patchwise parallel" method.

        Parameters:
            frame0_codes: torch.LongTensor, shape (B, H, W), the quantized image codes for frame 0
            frame1_codes: torch.LongTensor, shape (B, H, W)), the quantized image codes for frame 1
            unmask_idxs: List[int], the indexes of the patches to reveal
            temperature: float, the temperature value for sampling
            top_p: float, the top_p value for sampling
            top_k: int, the top_k value for sampling

        Returns:
            frame1_pred: torch.LongTensor, shape (B, H, W), the predicted image codes for frame 1
            logits: torch.FloatTensor, shape (B, H, W, C), the logits for the predicted image codes
        """

        # Pack the images into sequences
        im0_seq, im0_pos = self._pack_image_codes_into_sequence(
            frame0_codes.cpu(), mask=0.0,
            patch_offset=self.model.config.rgb_patch_0_idx_range[0], 
            seq_offset=self.model.config.frame_0_rgb_pos_range[0])
        im1_seq, im1_pos = self._pack_image_codes_into_sequence(
            frame1_codes.cpu(), mask=0.0,
            patch_offset=self.model.config.rgb_patch_1_idx_range[0], 
            seq_offset=self.model.config.frame_1_rgb_pos_range[0])

        # Brind the revealed patches to the front
        im1_seq, im1_pos = self._bring_patches_to_front(im1_seq, im1_pos, unmask_idxs)

        # Concatenate the two image sequences into a single sequence
        seq = torch.cat([im0_seq.view(-1), im1_seq.view(-1)])
        pos = torch.cat([im0_pos.view(-1), im1_pos.view(-1)])

        # Mask out part of frame 1
        seq_delim_idx = im0_seq.numel() + len(unmask_idxs) * im1_seq.shape[2]
        # Set masked part of the seuqence to mask value
        cond_seq = seq.clone()
        cond_seq[seq_delim_idx:] = self.model.config.mask_token_range[0]

        # Make Target sequence
        tgt_seq = seq[seq_delim_idx:].clone()

        # Make mask for the sequence
        mask = torch.ones_like(seq).bool()

        # Perform the prediction
        with self.ctx:
            logits, loss = self.model(
                cond_seq.unsqueeze(0).to(self.device).long(), 
                pos=pos.unsqueeze(0).to(self.device).long(),
                mask=mask.unsqueeze(0).to(self.device).bool(),
                tgt=tgt_seq.unsqueeze(0).to(self.device).long(),
                exotic_mask="no_mask"
            )
        # sampled_tokens = self.model.sample_logits(logits, temp=temperature, top_k=top_k, top_p=top_p)[0]
        logits = logits.cpu()
        sampled_tokens = self.model.sample_logits(logits, temp=0.0)[0]

        # reshape the predicted tokens into: (num_patches, num_tokens_per_patch)
        n_tokens_per_patch = im1_seq.shape[2]
        sampled_tokens = sampled_tokens.view(-1, n_tokens_per_patch)
        # insert the GT positional tokens from sequence 1 into the sampled predited sequence
        sampled_tokens[:, 0] = im1_seq[:, len(unmask_idxs):, 0].reshape(-1)
        sampled_tokens = sampled_tokens.reshape(-1)

        sort_order = im1_pos[:, len(unmask_idxs):, 0].argsort().cpu()
        logits = logits.view(-1, sort_order.shape[1], logits.shape[-1])
        rgb_logits = logits[:, sort_order[0]]

        # insert a dummy patch at any missing patch indexes:
        missing_patch_indexes = [i for i in range(im1_seq.shape[1]) if i not in 
            (im1_seq[:, len(unmask_idxs):, 0] - self.model.config.rgb_patch_1_idx_range[0]).tolist()[0]]
        for i in missing_patch_indexes:
            rgb_logits = torch.cat([rgb_logits[:, :i], torch.zeros_like(rgb_logits[:, 0:1]), rgb_logits[:, i:]], dim=1)
            # rgb_logits = torch.cat([rgb_logits[:i], torch.zeros_like(rgb_logits[[0]]), rgb_logits[i:]], dim=1)

        # unpack frame 1 sequence
        frame1_pred_seq = torch.cat([seq[im0_seq.numel():seq_delim_idx], sampled_tokens.reshape(-1)])
        frame1_pred =  self.model.unpack_and_sort_img_seq(frame1_pred_seq.reshape(1, -1))

        return frame1_pred, rgb_logits[1:].permute(1, 0, 2)

    @torch.no_grad()
    def two_frame_patchwise_parallel_forward(
        self, frame0_codes: torch.LongTensor, 
        frame1_codes: torch.LongTensor, 
        unmask_idxs: List[int],
        campose_codes: torch.LongTensor = None,
        flow_codes: torch.LongTensor = None,
        flow_unmask_idxs: List[int] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 1000,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Perform a forward pass through the model using the "patchwise parallel" method.

        Parameters:
            frame0_codes: torch.LongTensor, shape (B, H, W), the quantized image codes for frame 0
            frame1_codes: torch.LongTensor, shape (B, H, W)), the quantized image codes for frame 1
            unmask_idxs: List[int], the indexes of the patches to reveal
            temperature: float, the temperature value for sampling
            top_p: float, the top_p value for sampling
            top_k: int, the top_k value for sampling

        Returns:
            frame1_pred: torch.LongTensor, shape (B, H, W), the predicted image codes for frame 1
            logits: torch.FloatTensor, shape (B, H, W, C), the logits for the predicted image codes
        """

        # Pack the images into sequences
        im0_seq, im0_pos = self._pack_image_codes_into_sequence(
            frame0_codes.cpu(), mask=0.0,
            patch_offset=self.model.config.rgb_patch_0_idx_range[0], 
            seq_offset=self.model.config.frame_0_rgb_pos_range[0])
        im1_seq, im1_pos = self._pack_image_codes_into_sequence(
            frame1_codes.cpu(), mask=0.0,
            patch_offset=self.model.config.rgb_patch_1_idx_range[0], 
            seq_offset=self.model.config.frame_1_rgb_pos_range[0])
        
        # Bring the revealed patches to the front
        im1_seq, im1_pos = self._bring_patches_to_front(im1_seq, im1_pos, unmask_idxs)

        # Create the sequence
        seq = im0_seq.view(-1)
        pos = im0_pos.view(-1)
        seq_delim_idx = im0_seq.numel()
        
        # If flow is provided, pack it into a sequence
        if flow_codes is not None:
            flow_seq, flow_pos = self._pack_flow_codes_into_sequence(
                flow_codes.cpu(), mask=0.0,
                patch_offset=self.model.config.flow_patch_idx_range[0],
                seq_offset=self.model.config.flow_pos_range[0])
            flow_seq, flow_pos = self._bring_flow_patches_to_front(flow_seq, flow_pos, flow_unmask_idxs, discard=True)
            seq = torch.cat([seq, flow_seq.view(-1)])
            pos = torch.cat([pos, flow_pos.view(-1)])
            seq_delim_idx += flow_seq.numel()

        # Concatenate the two image sequences into a single sequence
        seq = torch.cat([seq, im1_seq.view(-1)])
        pos = torch.cat([pos, im1_pos.view(-1)])
        seq_delim_idx += len(unmask_idxs) * im1_seq.shape[2]

        # Mask out part of frame 1
        im0_delim_idx = im0_seq.numel()

        cond_seq = seq[:seq_delim_idx].clone()
        cond_pos = pos[:seq_delim_idx].clone()

        pred_seq = seq[seq_delim_idx:].clone().view(-1, im1_seq.shape[2])
        pred_pos = pos[seq_delim_idx:].clone().view(-1, im1_pos.shape[2])

        step_seq = torch.cat([cond_seq, pred_seq[:, 0].view(-1)])
        step_pos = cond_pos.clone()

        all_logits = []

        # Iterate over the patches once for each patch in the second frame
        for it in range(im1_seq.shape[2] - 1):

            step_pos = torch.cat([step_pos, pred_pos[:, it].view(-1)])

            step_mask = torch.zeros(step_pos.shape[0], step_pos.shape[0]).to(step_pos.device)


            # attention mask for frame 0 + unmasked part of frame 1
            step_mask[:, :seq_delim_idx] = 1
            step_mask[:seq_delim_idx, :seq_delim_idx].tril_()

            # attention mask for rest of frame 1
            step_mask[seq_delim_idx:, seq_delim_idx:] = 1


            # # attention mask for frame 0
            # step_mask[:, :im0_delim_idx] = 1
            # step_mask[:im0_delim_idx, :im0_delim_idx].tril_()

            # # attention mask for unmask idx
            # if len(unmask_idxs) > 0:
            #     step_mask[im0_delim_idx:seq_delim_idx, im0_delim_idx:seq_delim_idx] = 1
            #     step_mask[im0_delim_idx:seq_delim_idx, im0_delim_idx:seq_delim_idx].tril_()
            #     step_mask[seq_delim_idx:, im0_delim_idx:seq_delim_idx] = 1

            # # attention mask for predicted
            # for k in range(it + 1):
            #     pred_len = pred_pos.shape[0] * (it + 1 - k)
            #     row_start_idx = seq_delim_idx + pred_pos.shape[0] * k
            #     col_start_idx = seq_delim_idx
            #     row_end_idx = row_start_idx + pred_len
            #     col_end_idx = col_start_idx + pred_len
            #     step_mask[row_start_idx:row_end_idx, col_start_idx:col_end_idx].fill_diagonal_(1)


            step_mask = step_mask.unsqueeze(0) # Add a batch dimension to the mask
            step_mask = 1 - step_mask # the forward function assumes 0 to participate in attention, 1 otherwise

            step_tgt = pred_seq[:, it+1].view(-1)

            # Perform the prediction
            with self.ctx:
                logits, loss = self.model(
                    step_seq.unsqueeze(0).to(self.device).long(), 
                    pos=step_pos.unsqueeze(0).to(self.device).long(),
                    mask=step_mask.unsqueeze(0).to(self.device).bool(),
                    tgt=step_tgt.unsqueeze(0).to(self.device).long(),
                )
            # sampled_tokens = self.model.sample_logits(logits, temp=temperature, top_k=top_k, top_p=top_p)[0]
            sampled_tokens = self.model.sample_logits(logits, temp=0.0)[0]
            step_seq = torch.cat([step_seq, sampled_tokens.cpu()])

            all_logits.append(logits[0].clone())

        all_logits = torch.stack(all_logits, dim=1)
        sort_order = im1_pos[:, len(unmask_idxs):, 0].argsort()
        rgb_logits = all_logits[sort_order][0]

        # insert a dummy patch at any missing patch indexes:
        missing_patch_indexes = [i for i in range(im1_seq.shape[1]) if i not in 
            (im1_seq[:, len(unmask_idxs):, 0] - self.model.config.rgb_patch_1_idx_range[0]).tolist()[0]]
        for i in missing_patch_indexes:
            rgb_logits = torch.cat([rgb_logits[:i], torch.zeros_like(rgb_logits[[0]]), rgb_logits[i:]], dim=0)

        pred_seq = step_seq[cond_seq.numel():].reshape(im1_seq.shape[2], -1).permute(1, 0)
        frame1_seq = torch.cat([cond_seq[:-len(unmask_idxs)*im1_seq.shape[2]], pred_seq.reshape(-1)])

        # frame1_pred = step_seq[im0_seq.numel():].reshape(im1_seq.shape[2], im1_seq.shape[1]).permute(1, 0)
        frame1_pred =  self.model.unpack_and_sort_img_seq(frame1_seq.reshape(1, -1))

        return frame1_pred, rgb_logits

    @torch.no_grad()
    def two_frame_seq2par_forward(
        self,
        frame0_codes: torch.LongTensor,
        frame1_codes: torch.LongTensor,
        unmask_idxs: List[int],
        flow_unmask_indices: List[int] = None,
        motion_indices: List[int] = None,
        frame0_patch_offset: int = None,
        frame0_seq_offset: int = None,
        frame1_patch_offset: int = None,
        frame1_seq_offset: int = None,
        campose_codes: torch.LongTensor = None,
        flow_codes: torch.LongTensor = None,
        top_p: Union[float, List[float]] = 0.9,
        top_k: Union[int, List[int]] = 1000,
        temperature: Union[float, List[float]] = 1.0,
        num_seq_patches: int = 32,
        rmi=None,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Perform a forward pass through the model using the "sequential" method.

        Parameters:
            frame0_codes: torch.LongTensor, shape (B, H, W), the quantized image codes for frame 0
            frame1_codes: torch.LongTensor, shape (B, H, W)), the quantized image codes for frame 1
            campose_codes: torch.LongTensor, shape (B, 8), the quantized camera pose codes
            unmask_idxs: List[int], the indexes of the patches to reveal
            top_p: Union[float, List[float]], the top_p value for sampling (optionally per token)
            top_k: Union[int, List[int]], the top_k value for sampling (optionally per token)
            temperature: Union[float, List[float]], the temperature value for sampling (optionally per token)
        
        Returns:
            frame1_pred: torch.LongTensor, shape (B, H, W), the predicted image codes for frame 1
            logits: torch.FloatTensor, shape (B, H, W, C), the logits for the predicted image codes
            decoding_order: torch.LongTensor, shape (B, H), the order in which the patches were decoded
        """

        # Grab default RGB pos and patch idx ranges
        if frame0_patch_offset is None:
            frame0_patch_offset = self.model.config.rgb_patch_0_idx_range[0]
        if frame0_seq_offset is None:
            frame0_seq_offset = self.model.config.frame_0_rgb_pos_range[0]
        if frame1_patch_offset is None:
            frame1_patch_offset = self.model.config.rgb_patch_1_idx_range[0]
        if frame1_seq_offset is None:
            frame1_seq_offset = self.model.config.frame_1_rgb_pos_range[0]
        
        # Pack the images into sequences
        im0_seq, im0_pos = self._pack_image_codes_into_sequence(
            frame0_codes.cpu(), mask=0.0,
            patch_offset=frame0_patch_offset, seq_offset=frame0_seq_offset)
        im1_seq, im1_pos = self._pack_image_codes_into_sequence(
            frame1_codes.cpu(), mask=0.0,
            shuffle_order=rmi,
            patch_offset=frame1_patch_offset, seq_offset=frame1_seq_offset)

        # Pack the camera pose into a sequence if provided
        if campose_codes is not None:
            campose_seq, campose_pos = self._pack_camera_pose_codes_into_sequence(
                campose_codes.cpu(),
                campose_offse=self.model.config.campose_range[0],
                patch_idx_offset=self.model.config.campose_patch_idx_range[0],
                seq_offset=self.model.config.campose_pos_range[0])
        
        # Bring the revealed patches to the front
        if unmask_idxs is not None and len(unmask_idxs) > 0:
            im1_seq, im1_pos = self._bring_patches_to_front(
                im1_seq, im1_pos, unmask_idxs, patch_idx_offset=frame1_patch_offset)
    
        if flow_codes is not None:
            flows = self.flow_quantizer.decode(flow_codes.to(self.device))
            flow_norm = torch.norm(flows[0], dim=0, keepdim=False, p=2)
            # apply 2d 8x8 average pooling to the flow with stride 8
            pooled_flow = F.avg_pool2d(flow_norm.unsqueeze(0).unsqueeze(0), kernel_size=8, stride=8)[0,0]
            values, indices = torch.sort(pooled_flow.view(-1), descending=True)
            # indices = indices[:int((1.0-0.75)*indices.numel())]

            flow_seq, flow_pos = self._pack_image_codes_into_sequence(
                flow_codes.cpu(), mask=0.0, shuffle=True,
                # shuffle_order=indices.cpu(),
                patch_offset=self.model.config.flow_patch_idx_range[0], 
                seq_offset=self.model.config.flow_pos_range[0])

            # Bring the revealed patches to the front
            if flow_unmask_indices is not None and len(flow_unmask_indices) > 0:
                flow_seq, flow_pos = self._bring_flow_patches_to_front(
                    flow_seq, flow_pos, flow_unmask_indices, discard=True)
            
            if campose_codes is not None:
                seq = torch.cat([im0_seq.view(-1), flow_seq.view(-1), campose_seq.view(-1), im1_seq.view(-1)])
                pos = torch.cat([im0_pos.view(-1), flow_pos.view(-1), campose_pos.view(-1), im1_pos.view(-1)])
            else:
                seq = torch.cat([im0_seq.view(-1), flow_seq.view(-1), im1_seq.view(-1)])
                pos = torch.cat([im0_pos.view(-1), flow_pos.view(-1), im1_pos.view(-1)])
        else:
            if campose_codes is not None:
                seq = torch.cat([im0_seq.view(-1), campose_seq.view(-1), im1_seq.view(-1)])
                pos = torch.cat([im0_pos.view(-1), campose_pos.view(-1), im1_pos.view(-1)])
            else:
                seq = torch.cat([im0_seq.view(-1), im1_seq.view(-1)])
                pos = torch.cat([im0_pos.view(-1), im1_pos.view(-1)])

        # Mask out part of frame 1
        seq_delim_idx = im0_seq.numel()

        if unmask_idxs is not None and len(unmask_idxs) > 0:
            seq_delim_idx += len(unmask_idxs) * im1_seq.shape[2]

        if flow_codes is not None:
            seq_delim_idx += flow_seq.numel()
        if campose_codes is not None:
            seq_delim_idx += campose_seq.numel()

        cond_seq = seq[:seq_delim_idx]
        cond_pos = pos[:seq_delim_idx]

        # Grab the number of tokens to generate by subtracting the number of 
        # tokens in the conditional sequence from the total number of tokens
        num_total_tokens = seq.numel() - cond_seq.numel()

        num_seq_tokens = num_seq_patches * im1_seq.shape[2]
        num_par_tokens = num_total_tokens - num_seq_tokens

        # Move motion indixes of seq1 tokens to the front
        if motion_indices is not None:
            im1_seq, im1_pos = self._bring_patches_to_front(
                im1_seq, im1_pos, motion_indices, patch_idx_offset=frame1_patch_offset) 

        # # Grab the indexes of the patches of frame 1 to reveal
        # if rmi is None:
        #     rmi = im1_seq[:, :, 0].view(-1)[len(unmask_idxs):]
        # else:
        #     # if rmi is list convert it to tensor
        #     if isinstance(rmi, list):
        #         rmi = torch.tensor(rmi, device=im1_seq.device, dtype=torch.long)

        rmi = im1_seq[:, :, 0].view(-1)[len(unmask_idxs):]

        # Make sampling blacklist
        if motion_indices is not None:
            sampling_blacklist = [[self.model.config.flow_range[0] + 11646, 
                                         self.model.config.flow_range[0] + 11582
                                         ]]*(im1_seq.shape[2] * 1) 
        else:
            sampling_blacklist = []

        if num_seq_patches > 0:

            # Perform the sequential prediction
            with self.ctx:
                frame1_pred, logits = self.model.rollout(
                    cond_seq.unsqueeze(0).to(self.device).long().clone(), 
                    temperature=temperature,
                    random_masked_indices=rmi.unsqueeze(0).to(self.device).long(),
                    sampling_blacklist=sampling_blacklist,
                    pos=pos.unsqueeze(0).to(self.device).long(),
                    num_new_tokens=num_seq_tokens,
                    top_k=top_k,
                    top_p=top_p,
                    # causal_mask_length=(cond_seq.numel()-im0_seq.numel()),
                    causal_mask_length=cond_seq.numel(),
                )

            seq_delim_idx += num_seq_tokens

            cond_seq = frame1_pred.cpu().clone().reshape(-1)
            cond_pos = pos[:seq_delim_idx].clone()

            pred_seq = seq[seq_delim_idx:].clone().view(-1, im1_seq.shape[2])
            pred_pos = pos[seq_delim_idx:].clone().view(-1, im1_pos.shape[2])

            step_seq = torch.cat([cond_seq, pred_seq[:, 0].view(-1)])
            step_pos = cond_pos.clone()
        else:

            frame1_pred = cond_seq.clone()

            seq_delim_idx += num_seq_tokens

            cond_seq = frame1_pred.cpu().clone().reshape(-1)
            cond_pos = pos[:seq_delim_idx].clone()

            pred_seq = seq[seq_delim_idx:].clone().view(-1, im1_seq.shape[2])
            pred_pos = pos[seq_delim_idx:].clone().view(-1, im1_pos.shape[2])

            step_seq = torch.cat([cond_seq, pred_seq[:, 0].view(-1)])
            step_pos = cond_pos.clone()

        all_logits = []

        # Iterate over the patches once for each patch in the second frame
        for it in range(im1_seq.shape[2] - 1):

            step_pos = torch.cat([step_pos, pred_pos[:, it].view(-1)])
            step_mask = torch.zeros(step_pos.shape[0], step_pos.shape[0]).to(step_pos.device)

            # attention mask for frame 0 + unmasked part of frame 1
            step_mask[:, :seq_delim_idx] = 1
            step_mask[:seq_delim_idx, :seq_delim_idx].tril_()

            # attention mask for rest of frame 1
            step_mask[seq_delim_idx:, seq_delim_idx:] = 1

            step_mask = step_mask.unsqueeze(0) # Add a batch dimension to the mask
            step_mask = 1 - step_mask # the forward function assumes 0 to participate in attention, 1 otherwise

            step_tgt = pred_seq[:, it+1].view(-1)

            # Perform the prediction
            with self.ctx:
                logits, loss = self.model(
                    step_seq.unsqueeze(0).to(self.device).long(), 
                    pos=step_pos.unsqueeze(0).to(self.device).long(),
                    mask=step_mask.unsqueeze(0).to(self.device).bool(),
                    tgt=step_tgt.unsqueeze(0).to(self.device).long(),
                )
            # sampled_tokens = self.model.sample_logits(logits, temp=temperature, top_k=top_k, top_p=top_p)[0]
            sampled_tokens = self.model.sample_logits(logits.cpu(), temp=0.0)[0]
            step_seq = torch.cat([step_seq, sampled_tokens])

            all_logits.append(logits[0].clone())

        all_logits = torch.stack(all_logits, dim=1)
        sort_order = im1_pos[:, len(unmask_idxs) + num_seq_patches:, 0].argsort()
        rgb_logits = all_logits[sort_order][0]

        # insert a dummy patch at any missing patch indexes:
        missing_patch_indexes = [i for i in range(im1_seq.shape[1]) if i not in 
            (pred_seq.unsqueeze(0)[:, :, 0] - frame1_patch_offset).tolist()[0]]
        for i in missing_patch_indexes:
            rgb_logits = torch.cat([rgb_logits[:i], torch.zeros_like(rgb_logits[[0]]), rgb_logits[i:]], dim=0)

        pred_seq = step_seq[cond_seq.numel():].reshape(im1_seq.shape[2], -1).permute(1, 0)

        frame1_delim_idx = im1_seq.numel()
        if campose_codes is not None:
            frame1_delim_idx += campose_seq.numel()
        frame1_seq = torch.cat([cond_seq, pred_seq.reshape(-1)])[-im1_seq.numel():]

        # frame1_pred = step_seq[im0_seq.numel():].reshape(im1_seq.shape[2], im1_seq.shape[1]).permute(1, 0)
        frame1_pred = self.model.unpack_and_sort_img_seq(frame1_seq.reshape(1, -1))

        return frame1_pred, rgb_logits, rmi - frame1_patch_offset

    def get_logits_from_forward(
            self,
            frame0_codes: torch.LongTensor,
            flow_codes: torch.LongTensor,
            unmask_idxs: List[int],
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Perform a forward pass through the model using the "sequential" method.

        Parameters:
            frame0_codes: torch.LongTensor, shape (B, H, W), the quantized image codes for frame 0
            frame1_codes: torch.LongTensor, shape (B, H, W)), the quantized image codes for frame 1
            campose_codes: torch.LongTensor, shape (B, 8), the quantized camera pose codes
            unmask_idxs: List[int], the indexes of the patches to reveal
            top_p: Union[float, List[float]], the top_p value for sampling (optionally per token)
            top_k: Union[int, List[int]], the top_k value for sampling (optionally per token)
            temperature: Union[float, List[float]], the temperature value for sampling (optionally per token)

        Returns:
            frame1_pred: torch.LongTensor, shape (B, H, W), the predicted image codes for frame 1
            logits: torch.FloatTensor, shape (B, H, W, C), the logits for the predicted image codes
            decoding_order: torch.LongTensor, shape (B, H), the order in which the patches were decoded
        """

        # Grab default RGB pos and patch idx ranges
        # Pack the images into sequences


        frame0_patch_offset = self.model.config.rgb_patch_0_idx_range[0]

        frame0_seq_offset = self.model.config.frame_0_rgb_pos_range[0]

        im0_seq, im0_pos = self._pack_image_codes_into_sequence(
            frame0_codes.cpu(), mask=0.0,
            patch_offset=frame0_patch_offset, seq_offset=frame0_seq_offset)

        flow_seq, flow_pos = self._pack_image_codes_into_sequence(
            flow_codes.cpu(), mask=0.0,
            patch_offset=self.model.config.flow_patch_idx_range[0],
            seq_offset=self.model.config.flow_pos_range[0])

        # Bring the revealed flow patches to the front
        flow_seq, flow_pos = self._bring_flow_patches_to_front(flow_seq, flow_pos, unmask_idxs, discard=True)


        seq = torch.cat([im0_seq.view(-1), flow_seq.view(-1)])
        pos = torch.cat([im0_pos.view(-1), flow_pos.view(-1)])


        cond_seq = seq[:-1]
        cond_pos = pos[:-1]
        tgt = seq[1:]
        mask = torch.ones_like(seq).bool()

        with self.ctx:
            logits, loss = self.model(
                cond_seq.unsqueeze(0).to(self.device).long(),
                pos=cond_pos.unsqueeze(0).to(self.device).long(),
                mask=mask,
                tgt=tgt.unsqueeze(0).to(self.device).long(),
            )

        return logits, tgt


    def get_flow_surprisal_score(
            self,
            frame0: Union[Image.Image, np.ndarray, torch.Tensor],
            flow_map: np.ndarray,
            segment_map: np.ndarray,
            seed: int = 0,
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:

        """
        Perform a forward pass through the model using all of frame0 and part of frame1.

        Parameters:
            frame0: Image.Image or np.ndarray or torch.Tensor, the first frame
            flow_cond: first value is horizontal index, second value is vertical index
            flow: torch.Tensor or np.ndarray, the flow represented as a 2-channel image
            campose: torch.FloatTensor, the camera pose represented as a 4x4 transformation matrix
            mask_ratio: float, the ratio of patches to mask out
            mask_indices: List[int], the indices of the patches to mask out
            mode: str, the mode to use for the forward pass (sequential or parallel)
            seed: int, the random seed to use
            mask_out: bool, whether to black out the unmasked patches in the predicted frame

        Returns:
            frame0: Image.Image, the first frame
            frame1_pred: Image.Image, the predicted second frame
            frame1: Image.Image, the actual second frame
            rgb_logits: torch.FloatTensor, the logits for the predicted image codes
        """

        # assert mask_ratio is not None or unmask_indices is not None, "Either mask_ratio or mask_indices must be provided"
        # assert mask_ratio is None or unmask_indices is None, "Only one of mask_ratio or mask_indices can be provided"

        # Transform the input frames and quantize them if they are provided as PIL images
        if isinstance(frame0, Image.Image) or isinstance(frame0, np.ndarray):
            frame0_codes = self.quantizer.quantize(self.in_transform(frame0).unsqueeze(0).to(self.device))
        else:
            frame0_codes = frame0

        flow = flow_map

        #resize segment map into 32x32
        seg = torch.from_numpy(segment_map)[None, None].float().cuda()

        # downsample segment map to 32 x 32 resolution
        segment_map_tensor = torch.nn.functional.max_pool2d(seg, kernel_size=8, stride=8)[
            0, 0].cpu().numpy() > 0.5

        # Convert the segment map to a list of unmask indices
        unmask_indices_fg = np.where(segment_map_tensor.flatten() == 1)[0].tolist()
        unmask_indices_bg = np.where(segment_map_tensor.flatten() == 0)[0].tolist()
        #randomize the unmask indices
        random.shuffle(unmask_indices_fg)
        random.shuffle(unmask_indices_bg)
        unmask_indices_bg_start = unmask_indices_bg[:5]
        unmask_indices_bg_end = unmask_indices_bg[5:10]

        unmask_indices = unmask_indices_bg_start +  unmask_indices_fg[:40] + unmask_indices_bg_end

        # print("unmask_indices", unmask_indices)

        # If flow is a numpy array, convert it to a tensor with the flow quantizer
        flow_codes = self.flow_quantizer.quantize(
            self.flow_transform(torch.tensor(flow).unsqueeze(0).to(self.device)))

        flow_codes = flow_codes + self.model.config.flow_range[0]

        self._set_seed(seed)

        logits, tgt = self.get_logits_from_forward(
                frame0_codes.clone(), flow_codes.clone(), unmask_indices,
            )

        preds = logits[0, (32*32 + 7)*5 - 1:, :]
        gt = tgt[(32*32 + 7)*5 - 1:]

        #reshape preds
        preds = preds.reshape(-1, 5, logits.shape[-1])
        gt = gt.reshape(-1, 5)

        #remove index token
        preds = preds[:, 1:]
        preds = preds.reshape(-1, preds.shape[-1])
        gt = gt[:, 1:]
        gt = gt.reshape(-1).to(self.device)
        #get surprisal score
        #softmax preds
        preds = torch.nn.functional.softmax(preds, dim=-1)
        preds = preds[np.arange(len(preds)), gt] #torch.gather(preds, 1, gt.unsqueeze(1)).squeeze(1)
        preds = -torch.log(preds)
        surprisal = preds.mean()

        return surprisal


        

    @torch.no_grad()
    def two_frame_sequential_forward(
        self, 
        frame0_codes: torch.LongTensor, 
        frame1_codes: torch.LongTensor, 
        unmask_idxs: List[int],
        frame0_patch_offset: int = None,
        frame0_seq_offset: int = None,
        frame1_patch_offset: int = None,
        frame1_seq_offset: int = None,
        campose_codes: torch.LongTensor = None,
        flow_codes: torch.LongTensor = None,
        top_p: Union[float, List[float]] = 0.9,
        top_k: Union[int, List[int]] = 1000,
        temperature: Union[float, List[float]] = 1.0,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Perform a forward pass through the model using the "sequential" method.

        Parameters:
            frame0_codes: torch.LongTensor, shape (B, H, W), the quantized image codes for frame 0
            frame1_codes: torch.LongTensor, shape (B, H, W)), the quantized image codes for frame 1
            campose_codes: torch.LongTensor, shape (B, 8), the quantized camera pose codes
            unmask_idxs: List[int], the indexes of the patches to reveal
            top_p: Union[float, List[float]], the top_p value for sampling (optionally per token)
            top_k: Union[int, List[int]], the top_k value for sampling (optionally per token)
            temperature: Union[float, List[float]], the temperature value for sampling (optionally per token)
        
        Returns:
            frame1_pred: torch.LongTensor, shape (B, H, W), the predicted image codes for frame 1
            logits: torch.FloatTensor, shape (B, H, W, C), the logits for the predicted image codes
            decoding_order: torch.LongTensor, shape (B, H), the order in which the patches were decoded
        """

        # Grab default RGB pos and patch idx ranges
        if frame0_patch_offset is None:
            frame0_patch_offset = self.model.config.rgb_patch_0_idx_range[0]
        if frame0_seq_offset is None:
            frame0_seq_offset = self.model.config.frame_0_rgb_pos_range[0]
        if frame1_patch_offset is None:
            frame1_patch_offset = self.model.config.rgb_patch_1_idx_range[0]
        if frame1_seq_offset is None:
            frame1_seq_offset = self.model.config.frame_1_rgb_pos_range[0]
        
        # Pack the images into sequences
        im0_seq, im0_pos = self._pack_image_codes_into_sequence(
            frame0_codes.cpu(), mask=0.0,
            patch_offset=frame0_patch_offset, seq_offset=frame0_seq_offset)
        im1_seq, im1_pos = self._pack_image_codes_into_sequence(
            frame1_codes.cpu(), mask=0.0,
            patch_offset=frame1_patch_offset, seq_offset=frame1_seq_offset)
        
        # Pack the camera pose into a sequence if provided
        if campose_codes is not None:
            campose_seq, campose_pos = self._pack_camera_pose_codes_into_sequence(
                campose_codes.cpu(),
                campose_offse=self.model.config.campose_range[0],
                patch_idx_offset=self.model.config.campose_patch_idx_range[0],
                seq_offset=self.model.config.campose_pos_range[0])
        
        # Bring the revealed patches to the front
        im1_seq, im1_pos = self._bring_patches_to_front(
            im1_seq, im1_pos, unmask_idxs, patch_idx_offset=frame1_patch_offset)

        # Concatenate the two image sequences (and additional conditioning) into a single sequence
        if campose_codes is not None:
            seq = torch.cat([im0_seq.view(-1), campose_seq.view(-1), im1_seq.view(-1)])
            pos = torch.cat([im0_pos.view(-1), campose_pos.view(-1), im1_pos.view(-1)])
        else:
            seq = torch.cat([im0_seq.view(-1), im1_seq.view(-1)])
            pos = torch.cat([im0_pos.view(-1), im1_pos.view(-1)])
        
        if flow_codes is not None:
            flow_seq, flow_pos = self._pack_flow_codes_into_sequence(
                flow_codes.cpu(), mask=0.0,
                patch_offset=self.model.config.flow_patch_idx_range[0],
                seq_offset=self.model.config.flow_pos_range[0])
            seq = torch.cat([seq, flow_seq.view(-1)])
            pos = torch.cat([pos, flow_pos.view(-1)])

        # Mask out part of frame 1
        seq_delim_idx = im0_seq.numel() + len(unmask_idxs) * im1_seq.shape[2]
        # Add camppose sequence length to seq delim index if campose is provided
        if campose_codes is not None:
            seq_delim_idx += campose_seq.numel()

        cond_seq = seq[:seq_delim_idx]
        cond_pos = pos[:seq_delim_idx]

        # Grab the number of tokens to generate by subtracting the number of 
        # tokens in the conditional sequence from the total number of tokens
        num_seq_tokens = seq.numel() - cond_seq.numel()

        # Grab the indexes of the patches of frame 1 to reveal
        rmi = im1_seq[:, :, 0].view(-1)[len(unmask_idxs):]

        # Perform the sequential prediction
        with self.ctx:
            frame1_pred, logits = self.model.rollout(
                cond_seq.unsqueeze(0).to(self.device).long(), 
                temperature=temperature,
                random_masked_indices=rmi.unsqueeze(0).to(self.device).long(),
                pos=pos.unsqueeze(0).to(self.device).long(),
                num_new_tokens=num_seq_tokens,
                top_k=top_k,
                top_p=top_p,
                # causal_mask_length=(cond_seq.numel()-im0_seq.numel()),
                causal_mask_length=cond_seq.numel(),
            )
        
        frame1_pred = frame1_pred[0, -im1_seq.numel():]
        logits = logits[0].reshape(im1_seq.shape[1] - len(unmask_idxs), im1_seq.shape[2], -1)

        sort_order = im1_pos[:, len(unmask_idxs):, 0].argsort()
        # TODO: check if this is correct !!!!
        # Should it be rgb_logits = logits[sort_order][0, :, 1:] instead ????
        rgb_logits = logits[sort_order][0, :, :-1]

        # insert a dummy patch at any missing patch indexes:
        missing_patch_indexes = [i for i in range(im1_seq.shape[1]) if i not in 
            (im1_seq[:, len(unmask_idxs):, 0] - self.model.config.rgb_patch_1_idx_range[0]).tolist()[0]]
        for i in missing_patch_indexes:
            rgb_logits = torch.cat([rgb_logits[:i], torch.zeros_like(rgb_logits[[0]]), rgb_logits[i:]], dim=0)

        frame1_pred = self.model.unpack_and_sort_img_seq(frame1_pred.reshape(1, -1))
             
        return frame1_pred, rgb_logits, rmi - self.model.config.rgb_patch_1_idx_range[0]
    
    def two_frame_sequential_forward_multiple_inputs(
            self,
            frame0_codes_list: List[torch.LongTensor],
            frame1_codes: torch.LongTensor,
            campose_codes_list: List[torch.LongTensor] = None,
            flow_codes_list: List[torch.LongTensor] = None,
            input_rgb_unmask_indices: List[int] = None,
            input_flow_unmask_indices_list: List[List[int]] = None,
            output_rgb_indices: List[int] = None,
            temperature: float = 1.0,
            top_p: float = 0.9,
            top_k: int = 1000,
            scaling_factor_list: List[float] = None,
        ) -> List[Dict[str, Union[Image.Image, torch.Tensor]]]:
        
        """
        This function construct the sequences (shuffle, append) for multiple input conditioning and perform the prediction
        Then, change the predicted sequence back to image and return the results
        """
        
        # Pack the images into sequences
        im0_seq_list, im0_pos_list = [], []
        for frame0_codes in frame0_codes_list:
            im0_seq, im0_pos = self._pack_image_codes_into_sequence(
                frame0_codes.cpu(), mask=0.0,
                patch_offset=self.model.config.rgb_patch_0_idx_range[0],
                seq_offset=self.model.config.frame_0_rgb_pos_range[0])
            im0_seq_list.append(im0_seq)
            im0_pos_list.append(im0_pos)

        im1_seq, im1_pos = self._pack_image_codes_into_sequence(
            frame1_codes.cpu(), mask=0.0,
            patch_offset=self.model.config.rgb_patch_1_idx_range[0],
            seq_offset=self.model.config.frame_1_rgb_pos_range[0])

        # Pack the camera pose into a sequence if provided
        if campose_codes_list is not None:
            campose_seq_list, campose_pos_list = [], []
            for campose_codes in campose_codes_list:
                campose_seq, campose_pos = self._pack_camera_pose_codes_into_sequence(
                    campose_codes.cpu(),
                    campose_offse=self.model.config.campose_range[0],
                    patch_idx_offset=self.model.config.campose_patch_idx_range[0],
                    seq_offset=self.model.config.campose_pos_range[0])
                campose_seq_list.append(campose_seq)
                campose_pos_list.append(campose_pos)

        # Pack the flow into a sequence if provided
        if flow_codes_list is not None:
            flow_seq_list, flow_pos_list = [], []
            for flow_codes in flow_codes_list:
                flow_seq, flow_pos = self._pack_flow_codes_into_sequence(
                    flow_codes.cpu(), mask=0.0,
                    patch_offset=self.model.config.flow_patch_idx_range[0],
                    seq_offset=self.model.config.flow_pos_range[0])
                flow_seq_list.append(flow_seq)
                flow_pos_list.append(flow_pos)

        # If not specified, output_rgb_indices is all the indices except the input_rgb_unmask_indices
        if output_rgb_indices is None:
            all_indices = list(range(im1_seq.shape[1]))
            all_indices_except_input_indices = list(set(all_indices) - set(input_rgb_unmask_indices))
            random.shuffle(all_indices_except_input_indices)
            output_rgb_indices = all_indices_except_input_indices

        # Construct the remaining indices
        remaining_indices = list(set(range(1024)) - set(output_rgb_indices) - set(input_rgb_unmask_indices))
        random.shuffle(remaining_indices)

        # Bring the rgb1 revealed patches to the front
        # im1_seq_input, im1_pos_input = self._bring_patches_to_front(im1_seq, im1_pos, input_rgb_unmask_indices)
        # im1_seq_output, im1_pos_output = self._bring_patches_to_front(im1_seq, im1_pos, output_rgb_indices)
        im1_seq_input, im1_pos_input = self._reorder_patches(im1_seq, im1_pos, input_rgb_unmask_indices)
        im1_seq_output, im1_pos_output = self._reorder_patches(im1_seq, im1_pos, output_rgb_indices)
        im1_seq_remaining, im1_pos_remaining = self._reorder_patches(im1_seq, im1_pos, remaining_indices)

        # Bring the flow revealed patches to the front
        if input_flow_unmask_indices_list is not None:
            for flow_unmask_indices in input_flow_unmask_indices_list:
                for seq_idx, (flow_seq, flow_pos) in enumerate(zip(flow_seq_list, flow_pos_list)):
                    flow_seq, flow_pos = self._bring_flow_patches_to_front(flow_seq, flow_pos, flow_unmask_indices, discard=True)
                    flow_seq_list[seq_idx] = flow_seq
                    flow_pos_list[seq_idx] = flow_pos

        # Concatenate the two image sequences (and additional conditioning) into a single sequence
        seq_list, pos_list = [], []
        for seq_idx, (im0_seq, im0_pos) in enumerate(zip(im0_seq_list, im0_pos_list)):
            seq = im0_seq.view(-1)
            pos = im0_pos.view(-1)
            print(f"Adding frame0 to the sequence. Number of tokens: {im0_seq.numel()}")
            if campose_codes_list is not None:
                seq = torch.cat([seq, campose_seq_list[seq_idx].view(-1)])
                pos = torch.cat([pos, campose_pos_list[seq_idx].view(-1)])
                print(f"Adding campose to the sequence. Number of tokens: {campose_seq_list[seq_idx].numel()}")

            if flow_codes_list is not None:
                seq = torch.cat([seq, flow_seq_list[seq_idx].view(-1)])
                pos = torch.cat([pos, flow_pos_list[seq_idx].view(-1)])
                print(f"Adding flow to the sequence. Number of tokens: {flow_seq_list[seq_idx].numel()}")
            
            if im1_seq_input is not None:
                seq = torch.cat([seq, im1_seq_input.view(-1)])
                pos = torch.cat([pos, im1_pos_input.view(-1)])
                print(f"Adding frame1 input to the sequence. Number of tokens: {im1_seq_input.numel()}")

            seq_list.append(seq)
            pos_list.append(pos)

        # Mask out part of frame 1
        seq_delim_idx = im0_seq.numel()

        if im1_seq_input is not None:
            seq_delim_idx += len(input_rgb_unmask_indices) * im1_seq_input.shape[2]

        # Add camppose sequence length to seq delim index if campose is provided
        if campose_codes_list is not None:
            seq_delim_idx += campose_seq.numel()

        if flow_codes_list is not None:
            seq_delim_idx += flow_seq_list[0].numel() # assuming that the number of flow codes is the same for all inputs

        print("Conditioning sequence length:", seq_delim_idx)

        cond_seq_list, cond_pos_list = [], []
        for seq, pos in zip(seq_list, pos_list):
            cond_seq = seq[:seq_delim_idx]
            cond_pos = pos[:seq_delim_idx]
            cond_seq_list.append(cond_seq.unsqueeze(0).to(self.device).long())
            cond_pos_list.append(cond_pos.unsqueeze(0).to(self.device).long())

        # target construction
        target_pos = im1_pos_output.clone().view(-1)
        target_pos = target_pos.unsqueeze(0).to(self.device).long()
        target_pos = target_pos[:, :len(output_rgb_indices) * im1_seq_output.shape[2]]
        num_new_tokens = len(output_rgb_indices) * im1_seq_output.shape[2]
        rmi = im1_seq_output[:, :, 0].view(-1)[:len(output_rgb_indices)]
        rmi = rmi.unsqueeze(0).to(self.device).long()
        print("Decoding order", rmi - self.model.config.rgb_patch_1_idx_range[0])

        # Perform the prediction
        with self.ctx:
            generated_seq, logits = self.model.rollout_kv_cache_multiple_inputs(
                cond_seq_list, cond_pos_list, scaling_factor_list, rmi,
                target_pos, num_new_tokens=num_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p
            )

        # there are rgb1 input indices, rgb1 output indices, and remaining indices
        if im1_seq_input is not None:
            im1_seq_input = im1_seq_input.view(-1)[:len(input_rgb_unmask_indices) * im1_seq_input.shape[2]]
        im1_seq_output_pred = generated_seq[0].view(-1).cpu() # output_rgb_indices

        # frame1_pred construction
        if im1_seq_input is not None:
            frame1_pred = torch.cat([im1_seq_input, im1_seq_output_pred])
        else:
            frame1_pred = im1_seq_output_pred
        if len(remaining_indices) > 0:
            im1_seq_remaining[:, :, 1:] = torch.zeros_like(im1_seq_remaining[:, :, 1:])
            im1_seq_remaining = im1_seq_remaining.view(-1)
            frame1_pred = torch.cat([frame1_pred, im1_seq_remaining])
        frame1_pred =  self.model.unpack_and_sort_img_seq(frame1_pred.reshape(1, -1))

        # logits construction
        logits = logits.cpu()
        logits = logits[0].reshape(len(output_rgb_indices), im1_seq.shape[2], -1)
        sort_order = im1_pos_output[:, :len(output_rgb_indices), 0].argsort()
        rgb_logits = logits[sort_order][0, :, :-1]
        missing_patch_indexes = [i for i in range(im1_seq.shape[1]) if i not in output_rgb_indices]
        for i in missing_patch_indexes:
            rgb_logits = torch.cat([rgb_logits[:i], torch.zeros_like(rgb_logits[[0]]), rgb_logits[i:]], dim=0)

        return frame1_pred, rgb_logits, rmi - self.model.config.rgb_patch_1_idx_range[0]
    


    def frame0_flow_sequential_forward(
            self, frame0_codes: torch.LongTensor,
            flow_codes: torch.LongTensor,
            unmask_idxs: List[int],
            decode_idxs: List[int],
            top_p: Union[float, List[float]] = 0.9,
            top_k: Union[int, List[int]] = 1000,
            temperature: Union[float, List[float]] = 1.0,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Perform a forward pass through the model using the "sequential" method.

        Parameters:
            frame0_codes: torch.LongTensor, shape (B, H, W), the quantized image codes for frame 0
            flow_codes: torch.LongTensor, shape (B, 2, H, W)), the quantized xy codes for flow
            unmask_idxs: List[int], the indexes of the patches to reveal
            decode_idxs: List[int], the indexes of the patches to decode first 
            top_p: Union[float, List[float]], the top_p value for sampling (optionally per token)
            top_k: Union[int, List[int]], the top_k value for sampling (optionally per token)
            temperature: Union[float, List[float]], the temperature value for sampling (optionally per token)

        Returns:
            frame1_pred: torch.LongTensor, shape (B, H, W), the predicted image codes for frame 1
            logits: torch.FloatTensor, shape (B, H, W, C), the logits for the predicted image codes
        """

        # Pack the images into sequences
        im0_seq, im0_pos = self._pack_image_codes_into_sequence(
            frame0_codes.cpu(), mask=0.0,
            patch_offset=self.model.config.rgb_patch_0_idx_range[0],
            seq_offset=self.model.config.frame_0_rgb_pos_range[0])
        flow_seq, flow_pos = self._pack_flow_codes_into_sequence(
            flow_codes.cpu(), mask=0.0,
            patch_offset=self.model.config.flow_patch_idx_range[0],
            seq_offset=self.model.config.flow_pos_range[0])

        # Bring the revealed patches to the front
        flow_seq, flow_pos = self._bring_flow_patches_to_front(flow_seq, flow_pos, unmask_idxs + decode_idxs)

        # Concatenate the two image sequences into a single sequence
        seq = torch.cat([im0_seq.view(-1), flow_seq.view(-1)])
        pos = torch.cat([im0_pos.view(-1), flow_pos.view(-1)])

        # Mask out part of frame 1
        seq_delim_idx = im0_seq.numel() + len(unmask_idxs) * flow_seq.shape[2]

        cond_seq = seq[:seq_delim_idx]
        cond_pos = pos[:seq_delim_idx]

        # Grab the number of tokens to generate by subtracting the number of
        # tokens in the conditional sequence from the total number of tokens
        num_seq_tokens = seq.numel() - cond_seq.numel()

        # Grab the indexes of the patches of frame 1 to reveal
        rmi = flow_seq[:, :, 0].view(-1)[len(unmask_idxs):]

        # Perform the prediction
        with self.ctx:
            flow_pred, logits = self.model.rollout(
                cond_seq.unsqueeze(0).to(self.device).long(),
                temperature=temperature,
                random_masked_indices=rmi.unsqueeze(0).to(self.device).long(),
                pos=pos.unsqueeze(0).to(self.device).long(),
                num_new_tokens=num_seq_tokens,
                top_k=top_k,
                top_p=top_p,
                # causal_mask_length=(cond_seq.numel()-im0_seq.numel()),
                causal_mask_length=cond_seq.numel(),
                n_tokens_per_patch=3, # self.model.config.patch_size**2 * 2 + 1 # account for xy
            )

        flow_pred = flow_pred[0, im0_seq.numel():]
        logits = logits[0].reshape(flow_seq.shape[1] - len(unmask_idxs), flow_seq.shape[2], -1)

        sort_order = flow_pos[:, len(unmask_idxs):, 0].argsort()
        flow_logits = logits[sort_order][0, :, :-1]

        # insert a dummy patch at any missing patch indexes:
        missing_patch_indexes = [i for i in range(flow_seq.shape[1]) if i not in
                                 (flow_seq[:, len(unmask_idxs):, 0] - self.model.config.flow_patch_idx_range[
                                     0]).tolist()[0]]
        for i in missing_patch_indexes:
            flow_logits = torch.cat([flow_logits[:i], torch.zeros_like(flow_logits[[0]]), flow_logits[i:]], dim=0)
        flow_pred = self.model.unpack_and_sort_flow_seq(flow_pred.reshape(1, -1))

        return flow_pred, flow_logits
    
    def one_frame_flow_forward(
        self, 
        frame0_codes: torch.LongTensor, 
        num_new_patches: int,
        tokens_per_patch: int,
        campose_codes: torch.LongTensor = None,
        flow_cond_seq: torch.LongTensor = None,
        flow_cond_pos: torch.LongTensor = None,
        top_p: Union[float, List[float]] = 0.9,
        top_k: Union[int, List[int]] = 1000,
        temperature: Union[float, List[float]] = 1.0,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Perform sequential flow prediction given frame 0 and camera pose.
        
        Takes a sequence containing frame 0 and camera pose tokens, and predicts flow patches
        one at a time in a sequential manner.

        Args:
            frame0_codes: Input sequence containing frame 0 tokens
            num_new_patches: Number of flow patches to generate
            tokens_per_patch: Number of tokens per flow patch
            campose_codes: Input sequence containing camera pose tokens
            seq_delim_idx: Index separating conditioning tokens from generation
            top_p: Nucleus sampling threshold (per token or single value)
            top_k: Top-k sampling threshold (per token or single value) 
            temperature: Sampling temperature (per token or single value)

        Returns:
            Tuple containing:
            - Predicted flow tensor
            - None (logits not returned for this method)
        """

        # Pack the images into sequences
        im0_seq, im0_pos = self._pack_image_codes_into_sequence(
            frame0_codes.cpu(), mask=0.0,
            patch_offset=self.model.config.rgb_patch_0_idx_range[0], 
            seq_offset=self.model.config.frame_0_rgb_pos_range[0])
        
        # Pack the camera pose into a sequence if provided
        if campose_codes is not None:
            campose_seq, campose_pos = self._pack_camera_pose_codes_into_sequence(
                campose_codes.cpu(),
                campose_offse=self.model.config.campose_range[0],
                patch_idx_offset=self.model.config.campose_patch_idx_range[0],
                seq_offset=self.model.config.campose_pos_range[0])

        # Create a sequence
        seq = im0_seq.view(-1)
        pos = im0_pos.view(-1)
            
        # Concatenate the campose sequence if provided
        if campose_codes is not None:
            seq = torch.cat([seq, campose_seq.view(-1)])
            pos = torch.cat([pos, campose_pos.view(-1)])
        
        # Concatenate the flow condition sequence if provided
        if flow_cond_seq is not None:
            seq = torch.cat([seq, flow_cond_seq.view(-1)])
            pos = torch.cat([pos, flow_cond_pos.view(-1)])

        # Set the sequence delimiter index
        seq_delim_idx = seq.numel()

        # Get conditioning sequence up to delimiter
        cond_seq = seq[:seq_delim_idx]
        cond_pos = pos[:seq_delim_idx]

        # Generate random order for flow patch indices, exclude the patches included in the flow condition
        if flow_cond_seq is not None:
            cond_idxs = flow_cond_seq[:, 0].view(-1).tolist()
        else:
            cond_idxs = []
        rmi = [i for i in range(self.model.config.flow_patch_idx_range[0], 
            self.model.config.flow_patch_idx_range[1]) if i not in cond_idxs]
        random.shuffle(rmi)
        rmi = torch.tensor(rmi[:num_new_patches])


        # rmi = torch.randperm(self.model.config.flow_patch_idx_range[1] - self.model.config.flow_patch_idx_range[0]
        #                      )[:num_new_patches] + self.model.config.flow_patch_idx_range[0]

        # find the correspoding flow patch pos indices
        all_flow_pos = torch.arange(self.model.config.flow_pos_range[0], self.model.config.flow_pos_range[1])
        patchwise_flow_pos = all_flow_pos.reshape(-1, tokens_per_patch)
        selected_patchwise_flow_pos = patchwise_flow_pos[rmi - self.model.config.flow_patch_idx_range[0]]
        # add the selected flow patch pos indices to the sequence of pos indices
        pos = torch.cat([pos, selected_patchwise_flow_pos.reshape(-1)])

        num_new_tokens = num_new_patches * tokens_per_patch
        print(f"num_new_tokens: {num_new_tokens}")

        # Generate flow predictions
        with self.ctx:
            flow_pred, logits = self.model.rollout(
                cond_seq.unsqueeze(0).to(self.device).long(),
                random_masked_indices=rmi.unsqueeze(0).to(self.device).long(),
                pos=pos.unsqueeze(0).to(self.device).long(),
                num_new_tokens=num_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                causal_mask_length=cond_seq.numel(),
                n_tokens_per_patch=3, #self.model.config.patch_size**2 * 2 + 1  # Flow has x,y channels + index
            )
 
        # Extract and reshape predictions
        if flow_cond_seq is not None:
            flow_pred = flow_pred[0, -(num_new_tokens + flow_cond_seq.numel()):]
        else:
            flow_pred = flow_pred[0, -num_new_tokens:]

        # insert a dummy patch at any missing patch indexes:
        flow_patches = flow_pred.view(-1, tokens_per_patch)
        predicted_idxs = (flow_patches[:, 0] - self.model.config.flow_patch_idx_range[0]).cpu().tolist()

        for i in range(self.model.config.flow_patch_idx_range[1] - self.model.config.flow_patch_idx_range[0]):
            if i not in predicted_idxs:
                new_dummy_patch = torch.tensor([
                    i+self.model.config.flow_patch_idx_range[0], 
                    255+self.model.config.flow_range[0],
                    255+self.model.config.flow_range[0],
                ], dtype=torch.long, device=flow_patches.device)
                flow_patches = torch.cat([flow_patches, new_dummy_patch.view(1, -1)], dim=0)

        flow_pred = flow_patches.view(-1)
        flow_pred, valid_mask = self.model.unpack_and_sort_flow_seq(flow_pred.reshape(1, -1))

        return flow_pred, valid_mask

    def frame0_campose_flow_sequential_forward(
            self, 
            seq: torch.LongTensor,
            pos: torch.LongTensor,
            tgt: torch.LongTensor,
            seq_delim_idx: int,
            num_new_patches: int,
            tokens_per_patch: int,
            top_p: Union[float, List[float]] = 0.9,
            top_k: Union[int, List[int]] = 1000,
            temperature: Union[float, List[float]] = 1.0,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Perform sequential flow prediction given frame 0 and camera pose.
        
        Takes a sequence containing frame 0 and camera pose tokens, and predicts flow patches
        one at a time in a sequential manner.

        Args:
            seq: Input sequence containing frame 0 and camera pose tokens
            pos: Position embeddings for the input sequence
            tgt: Target sequence (not used in this method)
            seq_delim_idx: Index separating conditioning tokens from generation
            num_new_patches: Number of flow patches to generate
            tokens_per_patch: Number of tokens per flow patch
            top_p: Nucleus sampling threshold (per token or single value)
            top_k: Top-k sampling threshold (per token or single value) 
            temperature: Sampling temperature (per token or single value)

        Returns:
            Tuple containing:
            - Predicted flow tensor
            - None (logits not returned for this method)
        """
        # Get conditioning sequence up to delimiter
        cond_seq = seq[:seq_delim_idx]
        cond_pos = pos[:seq_delim_idx]

        # Generate random order for flow patch indices
        rmi = torch.tensor([seq[seq_delim_idx + i * tokens_per_patch] for i in range(num_new_patches)])
        num_new_tokens = num_new_patches * tokens_per_patch

        # Generate flow predictions
        with self.ctx:
            flow_pred, logits = self.model.rollout(
                cond_seq.unsqueeze(0).to(self.device).long(),
                random_masked_indices=rmi.unsqueeze(0).to(self.device).long(),
                pos=pos.unsqueeze(0).to(self.device).long(),
                num_new_tokens=num_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                causal_mask_length=cond_seq.numel(),
                n_tokens_per_patch=3, #self.model.config.patch_size**2 * 2 + 1  # Flow has x,y channels + index
            )

        # Extract and reshape predictions
        flow_pred = flow_pred[0, -num_new_tokens:]
        #logits = logits[0].reshape(num_new_patches, tokens_per_patch, -1)

        # Sort predictions back to original order
        #sort_order = rmi.argsort()
        #flow_logits = logits[sort_order][0, :, :-1]
        flow_pred = self.model.unpack_and_sort_flow_seq(flow_pred.reshape(1, -1))

        return flow_pred, None

    def frame0_flow_frame1_sequential_forward(
            self, frame0_codes: torch.LongTensor,
            flow_codes: torch.LongTensor,
            frame1_codes: torch.LongTensor,
            unmask_idxs: List[int],
            top_p: Union[float, List[float]] = 0.9,
            top_k: Union[int, List[int]] = 1000,
            temperature: Union[float, List[float]] = 1.0,
            decoding_order: torch.LongTensor = None,
            unmask_idxs_img1: List[int] = None,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Perform a forward pass through the model using the "sequential" method.

        Parameters:
            frame0_codes: torch.LongTensor, shape (B, H, W), the quantized image codes for frame 0
            flow_codes: torch.LongTensor, shape (B, 2, H, W), the quantized xy codes for flow
            unmask_idxs: List[int], the indexes of the patches to reveal
            top_p: Union[float, List[float]], the top_p value for sampling (optionally per token)
            top_k: Union[int, List[int]], the top_k value for sampling (optionally per token)
            temperature: Union[float, List[float]], the temperature value for sampling (optionally per token)
            decoding_order: torch.LongTensor, shape (B, H/patch_size*W/patch_size), the order in which to decode the patches

        Returns:
            frame1_pred: torch.LongTensor, shape (B, H, W), the predicted image codes for frame 1
            logits: torch.FloatTensor, shape (B, H, W, C), the logits for the predicted image codes
        """

        # Pack the images into sequences

        im0_seq, im0_pos = self._pack_image_codes_into_sequence(
            frame0_codes.cpu(), mask=0.25,
            patch_offset=self.model.config.rgb_patch_0_idx_range[0],
            seq_offset=self.model.config.frame_0_rgb_pos_range[0])
        flow_seq, flow_pos = self._pack_flow_codes_into_sequence(
            flow_codes.cpu(), mask=0.0,
            patch_offset=self.model.config.flow_patch_idx_range[0],
            seq_offset=self.model.config.flow_pos_range[0])

        im1_seq, im1_pos = self._pack_image_codes_into_sequence(
            frame1_codes.cpu(), mask=0.0,
            patch_offset=self.model.config.rgb_patch_1_idx_range[0],
            seq_offset=self.model.config.frame_1_rgb_pos_range[0], shuffle_order=decoding_order)

        # Bring the revealed patches to the front
        flow_seq, flow_pos = self._bring_flow_patches_to_front(flow_seq, flow_pos, unmask_idxs, discard=True)

        if unmask_idxs_img1 is not None:
            im1_seq, im1_pos = self._bring_patches_to_front(im1_seq, im1_pos, unmask_idxs_img1)

        # Concatenate the two image sequences into a single sequence
        seq = torch.cat([im0_seq.view(-1), flow_seq.view(-1), im1_seq.view(-1)])
        pos = torch.cat([im0_pos.view(-1), flow_pos.view(-1), im1_pos.view(-1)])
        tgt = seq.clone()

        # Mask out part of frame 1
        if unmask_idxs_img1 is not None:
            num_unmasked_tokens =  len(unmask_idxs_img1) * im1_seq.shape[2]
        else:
            num_unmasked_tokens = 0

        seq_delim_idx = im0_seq.numel() + len(unmask_idxs) * flow_seq.shape[2]

        cond_seq = seq[:seq_delim_idx]
        cond_pos = pos[:seq_delim_idx]

        # Grab the number of tokens to generate by subtracting the number of
        # tokens in the conditional sequence from the total number of tokens
        num_seq_tokens = seq.numel() - cond_seq.numel()

        # Grab the indexes of the patches of frame 1 to reveal
        rmi = flow_seq[:, :, 0].view(-1)[len(unmask_idxs):]
        # Perform the prediction
        rmi_cutoff = 0 #len(unmask_idxs_img1) if unmask_idxs_img1 is not None else 0
        with self.ctx:
            rmi = im1_seq[:, rmi_cutoff:, 0].view(-1)

            frame1_pred, logits = self.model.rollout(
                cond_seq.unsqueeze(0).to(self.device).long(),
                temperature=temperature,
                random_masked_indices=rmi.unsqueeze(0).to(self.device).long(),
                pos=pos.unsqueeze(0).to(self.device).long(),
                num_new_tokens=num_seq_tokens,
                top_k=top_k,
                top_p=top_p,
                # causal_mask_length=(cond_seq.numel()-im0_seq.numel()),
                causal_mask_length=cond_seq.numel(), #im0_seq.numel() + len(unmask_idxs) * flow_seq.shape[2]
                num_unmasked_tokens=num_unmasked_tokens,
                remaining_seq=seq[seq_delim_idx:].to(self.device).long()
            )


        unmask_idxs = []

        frame1_pred = frame1_pred[0, -im1_seq.numel():]

        # if unmask_idxs_img1 is not None:
        #     frame1_pred = self.model.unpack_and_sort_img_seq(frame1_pred.reshape(1, -1))
        #     return frame1_pred, None, tgt

        logits = logits[0].reshape(im1_seq.shape[1], im1_seq.shape[2], -1)

        sort_order = im1_pos[:, len(unmask_idxs):, 0].argsort()
        rgb_logits = logits[sort_order][0, :, :-1]

        # insert a dummy patch at any missing patch indexes:
        missing_patch_indexes = [i for i in range(im1_seq.shape[1]) if i not in
                                 (im1_seq[:, len(unmask_idxs):, 0] - self.model.config.rgb_patch_1_idx_range[
                                     0]).tolist()[0]]
        for i in missing_patch_indexes:
            rgb_logits = torch.cat([rgb_logits[:i], torch.zeros_like(rgb_logits[[0]]), rgb_logits[i:]], dim=0)

        frame1_pred = self.model.unpack_and_sort_img_seq(frame1_pred.reshape(1, -1))

        return frame1_pred, rgb_logits, tgt

    @torch.no_grad()
    def seq_to_par(
            self, frame0: Image.Image, frame1: Image.Image,
            seed: int = 0,
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:

        self._set_seed(seed)

        original_frame1 = frame1.copy()

        # Transform the input frames and quantize them
        frame0_codes = self.quantizer.quantize(self.in_transform(frame0).unsqueeze(0).to(self.device))
        frame1_codes = self.quantizer.quantize(self.in_transform(frame1).unsqueeze(0).to(self.device))

        unmask_idxs = []
        all_frame1_reseeded = []
        all_frame1_masked = []
        all_frame1_pred_par = []
        all_entropy_par = []
        all_error_par = []

        # create random order in which to reveal the patches
        idxs_to_unmask = torch.randperm(1024).tolist()


        for i in tqdm.tqdm(range(64)):

            # Compute parallel rollout
            par_frame1_pred, par_rgb_logits = self.two_frame_patchwise_parallel_forward(
                frame0_codes.clone(), frame1_codes.clone(), unmask_idxs)
            # print(i, unmask_idxs, frame0_codes.shape, frame1_codes.shape, par_rgb_logits.shape)
            # print("par_rgb_logits shape", i, par_rgb_logits.shape)
            par_grid_entropy = self._compute_rgb_grid_entropy(par_rgb_logits, unmask_idxs)

            par_grid_ce = self._compute_rgb_grid_ce_error(par_rgb_logits, frame1_codes)
            patch_grid_ce = patchify(par_grid_ce.unsqueeze(0), patch_size=self.model.config.patch_size)
            patch_grid_ce = patch_grid_ce.mean(dim=-1)

            # Append the max par entropy to the
            flat_par_entropy = par_grid_entropy.view(-1)

            # zero out the entropy of the unmasked patches
            flat_par_entropy[unmask_idxs] = 0.0

            flat_patch_ce_error = patch_grid_ce.view(-1)
            # zero out the entropy of the unmasked patches

            flat_patch_ce_error[unmask_idxs] = 0.0

            flat_par_entropy_sample = flat_par_entropy.clone()
            # flat_par_entropy_sample[unmask_idxs] = float('inf')
            # sample a random point from flat_par_entropy_sample from all the indexes which are not in unmask_idxs
            # idxs_to_sample = [i for i in range(1024) if i not in unmask_idxs]
            # next_sampled_keypoint = random.choice(idxs_to_sample)
            # next_sampled_keypoint = flat_par_entropy_sample.argmin().item()

            next_sampled_keypoint = flat_par_entropy_sample.argmax().item()
            # next_sampled_keypoint = flat_patch_ce_error.argmax().item()

            # par_grid_varentropy = self._compute_rgb_grid_varentropy(par_rgb_logits, unmask_idxs)
            # flat_par_varentropy = par_grid_varentropy.view(-1)
            # flat_par_varentropy[unmask_idxs] = 0.0
            # next_sampled_keypoint = flat_par_varentropy.argmax().item()

            # next_sampled_keypoint = idxs_to_unmask.pop(0)
            unmask_idxs.append(next_sampled_keypoint) 

            # Compute Cross Entropy Error
            flat_frame_1_codes = patchify(frame1_codes, patch_size=self.model.config.patch_size)
            rgb_error = F.cross_entropy(par_rgb_logits.view(par_rgb_logits.shape[0]*par_rgb_logits.shape[1], -1), 
                                        flat_frame_1_codes.reshape(-1).to(self.device), reduce=False)
            # rgb_error = F.cross_entropy(par_rgb_logits.view(par_rgb_logits.shape[0]*par_rgb_logits.shape[1], -1),
            #                             frame1_codes.reshape(-1).to(self.device), reduce=False)
            rgb_patch_error = rgb_error.view(par_rgb_logits.shape[0], par_rgb_logits.shape[1])
            # rgb_patch_error = rgb_patch_error.mean(dim=1).view(32, 32)
            rgb_patch_error = rgb_patch_error[:, 0].view(32, 32)

            # make next sampled keypoint as the minimum CE error keypoint
            # next_sampled_keypoint = rgb_patch_error.argmin().item()
            # unmask_idxs.append(next_sampled_keypoint)

            # Set the predicted keypoint rgb tokens as the gt tokens in frame1_codes

            # flat_frame1_codes = patchify(frame1_codes, patch_size=self.model.config.patch_size)
            # flat_par_frame1_pred = patchify(par_frame1_pred, patch_size=self.model.config.patch_size)
            # flat_frame1_codes[:, next_sampled_keypoint] = flat_par_frame1_pred[:, next_sampled_keypoint]
            # frame1_codes = unpatchify(flat_frame1_codes)

            # Decode the predicted frame
            frame0 = self.quantizer.decode(frame0_codes.to(self.device))
            frame1_pred_par = self.quantizer.decode(par_frame1_pred.to(self.device))
            frame1 = self.quantizer.decode(frame1_codes.to(self.device))

            # Un-normalize and convert to PIL
            frame0 = self.inv_in_transform(frame0[0])
            frame1_pred_par = self.inv_in_transform(frame1_pred_par[0])
            frame1 = self.inv_in_transform(frame1[0])

            masked_frame1 = mask_out_image(frame1, unmask_idxs, color=255, patch_size=self.model.config.patch_size * 4)

            all_entropy_par.append(par_grid_entropy.float().cpu())
            all_frame1_masked.append(masked_frame1)
            all_frame1_reseeded.append(frame1)
            all_frame1_pred_par.append(frame1_pred_par)
            all_error_par.append(rgb_patch_error.float().cpu())
           
        return (
            frame0,
            original_frame1,
            all_frame1_reseeded,
            all_frame1_masked,
            all_frame1_pred_par,
            all_entropy_par,
            all_error_par
        )

    def quantize_image(self, image: Image.Image) -> torch.Tensor:
        return self.quantizer.quantize(self.in_transform(image).unsqueeze(0).to(self.device))

    def decode_image_codes(self, codes: torch.Tensor) -> Image.Image:
        return self.inv_in_transform(self.quantizer.decode(codes.to(self.device))[0])

    def _compute_rgb_grid_entropy(self, logits: torch.Tensor, unmasked_idxs = []) -> torch.Tensor:
        """
        Compute the entropy of the RGB grid of the logits.

        Parameters:
            logits: torch.Tensor, shape (B, H * W, C), the logits of the RGB grid
            unmasked_idxs: List[int], the indexes of the unmasked patches
        
        Returns:
            rgb_grid_entropy: torch.Tensor, shape (H, W), the entropy of the RGB grid
        """
        # Extract RGB logits based on the range
        rgb_logits = logits[:, :, self.model.config.rgb_range[0]:self.model.config.rgb_range[1]]
        # Compute the per-patch entropy
        rgb_entropy = F.softmax(rgb_logits, dim=-1) * F.log_softmax(rgb_logits, dim=-1)
        rgb_entropy = -rgb_entropy.sum(dim=-1)
        rgb_patch_entropy = rgb_entropy.mean(dim=1)
        # rgb_patch_entropy = rgb_entropy[:, 0]#.mean(dim=1)
        # Set rgb patch entropy of the unmasked patches to 0
        rgb_patch_entropy[unmasked_idxs] = 0.0
        im_size = int(rgb_entropy.shape[0] ** 0.5)
        rgb_grid_entropy = rgb_entropy.mean(dim=1).view(im_size, im_size)

        return rgb_grid_entropy
    
    def _compute_flow_grid_entropy(self, logits: torch.Tensor, unmasked_idxs = []) -> torch.Tensor:
        """
        Compute the entropy of the RGB grid of the logits.

        Parameters:
            logits: torch.Tensor, shape (B, H * W, C), the logits of the RGB grid
            unmasked_idxs: List[int], the indexes of the unmasked patches
        
        Returns:
            rgb_grid_entropy: torch.Tensor, shape (H, W), the entropy of the RGB grid
        """
        # Extract RGB logits based on the range
        flow_logits = logits[:, :, self.model.config.flow_range[0]:self.model.config.flow_range[1]]
        # Compute the per-patch entropy
        flow_entropy = F.softmax(flow_logits, dim=-1) * F.log_softmax(flow_logits, dim=-1)
        flow_entropy = -flow_entropy.sum(dim=-1)
        flow_patch_entropy = flow_entropy[:, 0]#.mean(dim=1)
        # rgb_patch_entropy = rgb_entropy[:, 0]#.mean(dim=1)
        # Set rgb patch entropy of the unmasked patches to 0
        flow_patch_entropy[unmasked_idxs] = 0.0
        im_size = int(flow_patch_entropy.shape[0] ** 0.5)
        flow_grid_entropy = flow_patch_entropy.view(im_size, im_size)

        return flow_grid_entropy

    def _compute_rgb_grid_varentropy(self, logits: torch.Tensor, unmasked_idxs = []) -> torch.Tensor:
        """
        Compute the varentropy of the RGB grid of the logits.

        Parameters:
            logits: torch.Tensor, shape (B, H * W, C), the logits of the RGB grid
            unmasked_idxs: List[int], the indexes of the unmasked patches
        
        Returns:
            rgb_grid_varentropy: torch.Tensor, shape (H, W), the varentropy of the RGB grid
        """
        # Extract RGB logits based on the range
        rgb_logits = logits[:, :, self.model.config.rgb_range[0]:self.model.config.rgb_range[1]]
        # Compute softmax probabilities
        rgb_probs = F.softmax(rgb_logits, dim=-1)
        # Compute log-probabilities (log-softmax)
        rgb_log_probs = F.log_softmax(rgb_logits, dim=-1)
        # Compute the surprisal (negative log probabilities)
        surprisal = -rgb_log_probs  # shape (B, H*W, C)
        # Compute the first moment (entropy), which is the expected surprisal
        rgb_entropy = (rgb_probs * surprisal).sum(dim=-1)  # shape (B, H*W)
        # Compute the second moment (expected value of the squared surprisal)
        rgb_second_moment = (rgb_probs * surprisal**2).sum(dim=-1)  # shape (B, H*W)
        # Compute the varentropy: second moment - (entropy)^2
        rgb_varentropy = rgb_second_moment - rgb_entropy**2  # shape (B, H*W)
        # Mean across patches to get the per-image varentropy
        rgb_patch_varentropy = rgb_varentropy.mean(dim=1)
        # Set the varentropy of the unmasked patches to 0
        rgb_patch_varentropy[unmasked_idxs] = 0.0
        # Reshape the result into the grid format (H, W)
        im_size = int(rgb_varentropy.shape[0] ** 0.5)
        rgb_grid_varentropy = rgb_patch_varentropy.view(im_size, im_size)
        
        return rgb_grid_varentropy
    
    def _compute_flow_grid_cumulative_probability(
        self,
        logits: torch.Tensor,
        unmasked_idxs,
        target_classes,
        total_range
    ) -> torch.Tensor:
        """
        Compute the cumulative probability of the given target_classes within a 
        specified range of channels (total_range).

        Parameters:
        -----------
        logits : torch.Tensor
            The logits for each patch, shape (B, H*W, C).
        unmasked_idxs : List[int]
            The list of patch indices (0 to H*W-1) for which you want the final 
            probabilities to be set to 0 (e.g., unmasked patches).
        target_classes : List[int]
            The global class indices whose probabilities should be summed.
        total_range : Tuple[int, int]
            The (start, end) slice of the channel dimension to consider when 
            computing probabilities.
        
        Returns:
        --------
        flow_grid_cumulative : torch.Tensor
            A 2D tensor of shape (H, W) containing the summed probability of 
            target_classes at each patch, with unmasked patches set to 0.
        """

        # 1) Slice out the relevant channel range
        #    logits shape: (B, H*W, C) -> (B, H*W, total_range_size)
        start, end = total_range
        subset_logits = logits[:, :, start:end]

        # 2) Compute probabilities with softmax over the sliced channels
        subset_probs = F.softmax(subset_logits, dim=-1)  # shape: (B, H*W, end - start)

        # 3) Adjust target_classes to local indices relative to total_range
        #    (only keep classes that fall within [start, end) )
        local_target_classes = [c - start for c in target_classes if start <= c < end]

        # 4) Sum the probabilities over the target classes
        #    shape: (B, H*W)
        cumulative_prob = subset_probs[:, :, local_target_classes].sum(dim=-1)

        # 5) We want a single patch-level probability grid, so remove batch dim (assuming B=1)
        flow_patch_cumulative = cumulative_prob.squeeze(0)  # shape: (H*W,)

        # 6) Zero out the probability for the unmasked patches
        flow_patch_cumulative[unmasked_idxs] = 1.0

        # 7) Average the probabilities across all tokens in each patch
        flow_patch_cumulative = flow_patch_cumulative[:, 0]

        # 8) Reshape to (H, W)
        im_size = int(flow_patch_cumulative.shape[0] ** 0.5)
        flow_grid_cumulative = flow_patch_cumulative.view(im_size, im_size)

        return flow_grid_cumulative

    def _compute_rgb_grid_ce_error(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the CE error of the RGB grid of the logits.

        Parameters:
            logits: torch.Tensor, shape (B, H * W, C), the logits of the RGB grid
            targets: torch.Tensor, shape (B, H * W), the target image codes
        
        Returns:
            rgb_grid_ce_error: torch.Tensor, shape (H, W), the CE error of the RGB grid
        """
        # Patchify targets
        patch_targets = patchify(targets, patch_size=self.model.config.patch_size)
        # Compute the CE error
        rgb_ce_error = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), patch_targets.reshape(-1), reduce=False)
        # Reshape the result into the grid format (H, W)
        rgb_ce_error = rgb_ce_error.view(1, logits.shape[0], logits.shape[1]).mean(dim=-1)
        # rgb_ce_error_grid = unpatchify(rgb_ce_error)
        im_size = int(logits.shape[0] ** 0.5)
        rgb_grid_ce_error = rgb_ce_error.view(im_size, im_size)

        return rgb_grid_ce_error

    def _compute_rgb_grid_l1_error(self, pred_frame: torch.Tensor, gt_frame: torch.Tensor) -> torch.Tensor:
        """
        Compute the MSE error of the RGB frame.

        Parameters:
            pred_frame: 
        
        Returns:
            rgb_grid_ce_error: torch.Tensor, shape (H, W), the CE error of the RGB grid
        """
        # Compute the L1 error
        rgb_l1_error = F.l1_loss(pred_frame, gt_frame, reduction='none').mean(dim=1)
        # Patchify the L1 error (use config token patch size * 2 to match the patch size of the RGB grid)
        rgb_l1_error = patchify(rgb_l1_error, patch_size=self.model.config.patch_size*2).mean(dim=-1)
        # Reshape the result into the grid format (H, W)
        im_size = int(rgb_l1_error.shape[1] ** 0.5)
        rgb_l1_error = rgb_l1_error.view(im_size, im_size)

        return rgb_l1_error

    def _compute_rgb_grid_mse_error(self, pred_frame, gt_frame) -> torch.Tensor:
        """
        Compute the MSE error of the RGB frame.

        Parameters:
            pred_frame: 
        
        Returns:
            rgb_grid_ce_error: torch.Tensor, shape (H, W), the CE error of the RGB grid
        """
        # Compute the MSE error
        rgb_mse_error = F.mse_loss(pred_frame, gt_frame, reduction='none').mean(dim=1)
        # Patchify the L1 error (use config token patch size * 2 to match the patch size of the RGB grid)
        rgb_mse_error = patchify(rgb_mse_error, patch_size=self.model.config.patch_size*2).mean(dim=-1)
        # Reshape the result into the grid format (H, W)
        im_size = int(rgb_mse_error.shape[1] ** 0.5)
        rgb_mse_error = rgb_mse_error.view(im_size, im_size)

        return rgb_mse_error

    def _pack_image_codes_into_sequence(
            self, frame_codes: torch.Tensor, mask: float = 0.0, shuffle: bool = True, 
            patch_offset: int = 0, seq_offset: int = 0, shuffle_order: List[int] = None
        ):
        
        frame_patches = patchify(frame_codes, patch_size=self.model.config.patch_size)
        frame_with_idxs = add_patch_indexes(frame_patches, patch_offset)
        frame_pos_idxs = get_pos_idxs(frame_with_idxs, seq_offset)
        shuffled_im0_patches, shuffled_img0_pos_idxs = shuffle_and_trim_values_and_positions(
            frame_with_idxs, frame_pos_idxs, mask=mask, shuffle=shuffle, shuffle_order=shuffle_order)
        
        return shuffled_im0_patches, shuffled_img0_pos_idxs
    
    def _pack_camera_pose_codes_into_sequence(
            self, campose_codes: torch.Tensor, campose_offse: int = 0, 
            patch_idx_offset: int = 0, seq_offset: int = 0
    ):
        campose_with_idxs = torch.cat([torch.tensor([patch_idx_offset], dtype=campose_codes.dtype), 
                                       campose_codes + campose_offse])
        campose_pos_idxs = get_pos_idxs(campose_with_idxs, seq_offset)

        return campose_with_idxs, campose_pos_idxs

    def _pack_flow_codes_into_sequence(
            self, flow_codes: torch.Tensor, mask: float = 0.0, shuffle: bool = True,
            patch_offset: int = 0, seq_offset: int = 0
    ):

        flow_patches_x = patchify(flow_codes[:, 0], patch_size=self.model.config.patch_size)
        flow_patches_y = patchify(flow_codes[:, 1], patch_size=self.model.config.patch_size)
        flow_patches = torch.stack([flow_patches_x, flow_patches_y], dim=-1).flatten(-2, -1)
        flow_patches += self.model.config.flow_range[0]

        flow_with_idxs = add_patch_indexes(flow_patches, patch_offset)
        flow_pos_idxs = get_pos_idxs(flow_with_idxs, seq_offset)

        flow_decode = decode_flow_code(flow_codes, input_size=256, num_bins=512)
        flow_decode = F.interpolate(flow_decode, scale_factor=1 / self.model.config.patch_size, mode='nearest')[0]
        assert mask == 0, "assume mask==0 for num_flow_patches=shuffled_patches_flow.shape[1], alpha=0"
        # with num_flow_patches=shuffled_patches_flow.shape[1], alpha=0, it will shuffle the flow patches randomly
        shuffled_patches_flow, shuffled_flow_pos_idx = sample_flow_values_and_positions(
            flow_with_idxs, flow_pos_idxs, flow_decode, num_flow_patches=flow_with_idxs.shape[1], alpha=0)

        return shuffled_patches_flow, shuffled_flow_pos_idx
    
    def _set_seed(self, seed: int):
        """
        Set the seed for reproducibility.

        Parameters:
            seed: int, the seed to set
        """
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    def _bring_patches_to_front(self, seq: torch.Tensor, pos: torch.Tensor, idxs: List[int],
                                patch_idx_offset: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bring the patches of the given indexes to the front of the sequence and its corresponding positions.

        Parameters:
            seq: torch.Tensor, shape (B, H, W), the sequence of patches
            pos: torch.Tensor, shape (B, H, W), the positions of the patches
            idxs: List[int], the indexes of the patches to bring to the front
        
        Returns:
            reordered_seq: torch.Tensor, shape (B, H, W), the reordered sequence
            reordered_pos: torch.Tensor, shape (B, H, W), the reordered positions
        """
        if patch_idx_offset is None:
            patch_idx_offset = self.model.config.flow_patch_idx_range[0]
        # Find locations of specified patches within the sequence based on patch indexes in the sequence
        patch_idxs = seq[:, :, 0].view(-1) - patch_idx_offset # self.model.config.flow_patch_idx_range[0] # self.model.config.rgb_patch_1_idx_range[0]
        # patch_idxs = seq[:, :, 0].view(-1) - self.model.config.rgb_patch_1_idx_range[0]
        bring_to_front_idxs = [i for i in range(len(patch_idxs)) if patch_idxs[i] in idxs]
        # Reorder the sequence and positions
        reordered_seq = torch.cat([seq[:, bring_to_front_idxs, :], 
            seq[:, [i for i in range(seq.shape[1]) if i not in bring_to_front_idxs], :]], dim=1)
        reordered_pos = torch.cat([pos[:, bring_to_front_idxs, :],
            pos[:, [i for i in range(pos.shape[1]) if i not in bring_to_front_idxs], :]], dim=1)
        return reordered_seq, reordered_pos
    
    def _reorder_patches(self, seq: torch.Tensor, pos: torch.Tensor, idxs: List[int]
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Change the patch order based on the idxs list.

        Parameters:
            seq: torch.Tensor, shape (B, H, W), the sequence of patches
            pos: torch.Tensor, shape (B, H, W), the positions of the patches
            idxs: List[int], the indexes of the patches to reorder

        Returns:
            reordered_seq: torch.Tensor, shape (B, H, W), the reordered sequence
            reordered_pos: torch.Tensor, shape (B, H, W), the reordered positions
        """
        # Find locations of specified patches within the sequence based on patch indexes in the sequence
        patch_idxs = seq[:, :, 0].view(-1) - self.model.config.rgb_patch_1_idx_range[0]
        reorder_idxs = []
        for idx in idxs:
            reorder_idxs.extend([i for i in range(len(patch_idxs)) if patch_idxs[i] == idx])
        # Reorder the sequence and positions
        if len(reorder_idxs) == 0:
            return None, None
        reordered_seq = seq[:, reorder_idxs, :]
        reordered_pos = pos[:, reorder_idxs, :]
        return reordered_seq, reordered_pos

    def _bring_flow_patches_to_front(self, seq: torch.Tensor, pos: torch.Tensor, idxs: List[int], discard: bool = False,
                                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bring the patches of the given indexes to the front of the sequence and its corresponding positions.

        Parameters:
            seq: torch.Tensor, shape (B, H, W), the sequence of patches
            pos: torch.Tensor, shape (B, H, W), the positions of the patches
            idxs: List[int], the indexes of the patches to bring to the front

        Returns:
            reordered_seq: torch.Tensor, shape (B, H, W), the reordered sequence
            reordered_pos: torch.Tensor, shape (B, H, W), the reordered positions
        """
        # Find locations of specified patches within the sequence based on patch indexes in the sequence
        patch_idxs = seq[:, :, 0].view(-1) - self.model.config.flow_patch_idx_range[0]
        bring_to_front_idxs = [i for i in range(len(patch_idxs)) if patch_idxs[i] in idxs]
        # Reorder the sequence and positions
        if discard:
            # print('discard remaining')
            reordered_seq = seq[:, bring_to_front_idxs, :]
            reordered_pos = pos[:, bring_to_front_idxs, :]
        else:
            reordered_seq = torch.cat([seq[:, bring_to_front_idxs, :],
                                       seq[:, [i for i in range(seq.shape[1]) if i not in bring_to_front_idxs], :]], dim=1)
            reordered_pos = torch.cat([pos[:, bring_to_front_idxs, :],
                                       pos[:, [i for i in range(pos.shape[1]) if i not in bring_to_front_idxs], :]], dim=1)
        return reordered_seq, reordered_pos

    def resize_flow(self, gt_flow, new_height, new_width):
        import torch.nn.functional as F

        # Assume gt_flow is a PyTorch tensor of shape (H, W, 2)
        # Transpose to shape (2, H, W) for processing
        gt_flow = gt_flow.permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 2, H, W)

        # Resize the flow components using bilinear interpolation
        resized_flow = F.interpolate(gt_flow, size=(new_height, new_width), mode='bilinear', align_corners=True)

        # Calculate scaling factors
        scale_y = gt_flow.shape[2] / new_height
        scale_x = gt_flow.shape[3] / new_width

        # Scale the flow values
        resized_flow[:, 0, :, :] /= scale_x  # Horizontal component
        resized_flow[:, 1, :, :] /= scale_y  # Vertical component

        # Transpose back to original ordering (new_height, new_width, 2)
        resized_flow = resized_flow.squeeze(0).permute(1, 2, 0)
        return resized_flow

    def create_points_list_from_flow(self, flow_pred_np):
        # Round the flow to integers
        flow_pred_np = np.round(flow_pred_np).astype(int)

        # Get the height and width of the image
        H, W, _ = flow_pred_np.shape

        # Dictionary to map end points to a start point (only first occurrence is stored)
        end_points_to_start_points = {}

        def process_pixels(condition):
            """ Process pixels based on the given condition (motion or stationary). """
            for y in range(H):
                for x in range(W):
                    # Compute the end coordinates
                    flow = flow_pred_np[y, x]
                    if condition(flow):
                        end_x = x + int(flow[0])
                        end_y = y + int(flow[1])

                        # Ensure the end coordinates are within bounds
                        if 0 <= end_x < W and 0 <= end_y < H:
                            end_point = (end_x, end_y)
                            start_point = (x, y)

                            # Store only the first encountered start point for an end point
                            if end_point not in end_points_to_start_points:
                                end_points_to_start_points[end_point] = start_point
                                
        # First pass: Process pixels with motion (flow magnitude > 5)
        process_pixels(lambda flow: abs(flow[0]) + abs(flow[1])  >= 2)

        # Second pass: Process stationary pixels (flow magnitude ≤ 5)
        # process_pixels(lambda flow: abs(flow[0]) + abs(flow[1])  == 0)
        
        # Get border points to be fixed
        for y in range(H):
            for x in range(W):
                # if x == 0 or x == W - 1 or y == 0 or y == H - 1:
                # i want the 3 pixels in each corner
                if (x == 0 and y in [0,1,W-1,W-2]) or \
                (x == W - 1 and y in [0,1,W-1,W-2]) or \
                (y == 0 and x in [0,1,H-1,H-2]) or \
                (y == H - 1 and x in [0,1,H-1,H-2]):
                    end_point = (x, y)
                    start_point = (x, y)
                    if end_point not in end_points_to_start_points:
                        end_points_to_start_points[end_point] = start_point

        # Generate the final points list
        points = []
        for end_point, start_point in end_points_to_start_points.items():
            points.append(list(start_point))  # Start point
            points.append(list(end_point))    # End point

        return points