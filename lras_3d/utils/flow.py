"""
Utility function for processing optical flow data
"""

import torch
import numpy as np
import h5py
from typing import Tuple
import os
# from lras_3d.utils.viz import fig_to_img, frames_to_video
import matplotlib.pyplot as plt
from lras_3d.utils.image_processing import patchify
import sys
import tqdm
import torch.nn.functional as F

def compute_quantize_flow_new(
        flow_values: torch.FloatTensor,
        input_size: int = 256,
        num_bins: int = 512,
    ) -> torch.LongTensor:
    """
    Quantizes the continuous optical flow values into discrete bins.

    Parameters:
    flow_values (FloatTensor): The original optical flow values to be quantized, [B, 2, H, W].
    input_size (int): The input size of the flow map (used to set the quantization range).
    num_bins (int): The number of bins for quantization (i.e., how many discrete levels the flow will be quantized into).

    Returns:
    LongTensor: The quantized flow values, where each value is an integer representing the bin index, [B, 2, H, W].
    """
    # Setting the flow range based on input_size, defining min_flow and max_flow.
    max_range = input_size
    min_flow, max_flow = -max_range, max_range

    # Normalize the flow values to the range [0, 1]
    normalized_flow = torch.round(flow_values - min_flow) #/ (max_flow - min_flow)

    # Ensure the normalized flow stays within the valid range [0, 1]
    normalized_flow = torch.clamp(normalized_flow, 0.0, max_flow - min_flow - 1)

    # Scale the normalized values to the number of bins
    # scaled_flow = normalized_flow * (num_bins)

    # Round to the nearest bin index and convert to long tensor
    return normalized_flow.long() #torch.round(scaled_flow).long()

def unquantize_flow(flow_values: torch.LongTensor, input_size: int = 256) -> torch.FloatTensor:
    """
    Unquantizes the discrete flow values back into continuous values.

    Parameters:
    flow_values (LongTensor): The quantized flow values to be unquantized, [B, 2, H, W].
    input_size (int): The input size of the flow map (used to set the quantization range).
    num_bins (int): The number of bins for quantization (i.e., how many discrete levels the flow will be quantized into).

    Returns:
    FloatTensor: The unquantized flow values, [B, 2, H, W].
    """
    # Setting the flow range based on input_size, defining min_flow and max_flow.
    max_range = input_size
    min_flow, max_flow = -max_range, max_range

    # Scale the values back to the original range
    unquantized_flow = flow_values.float() + min_flow #scaled_flow * (max_flow - min_flow) + min_flow

    return unquantized_flow

def compute_quantize_flow(
        flow_values: torch.FloatTensor,
        input_size: int = 256,
        num_bins: int = 512,
    ) -> torch.LongTensor:
    """
    Quantizes the continuous optical flow values into discrete bins.

    Parameters:
    flow_values (FloatTensor): The original optical flow values to be quantized, [B, 2, H, W].
    input_size (int): The input size of the flow map (used to set the quantization range).
    num_bins (int): The number of bins for quantization (i.e., how many discrete levels the flow will be quantized into).

    Returns:
    LongTensor: The quantized flow values, where each value is an integer representing the bin index, [B, 2, H, W].
    """
    # Setting the flow range based on input_size, defining min_flow and max_flow.
    max_range = input_size
    min_flow, max_flow = -max_range, max_range

    # Normalize the flow values to the range [0, 1]
    normalized_flow = (flow_values - min_flow) / (max_flow - min_flow)

    # Ensure the normalized flow stays within the valid range [0, 1]
    normalized_flow = torch.clamp(normalized_flow, 0.0, 1.0)

    # Scale the normalized values to the number of bins
    scaled_flow = normalized_flow * (num_bins - 1)

    # Round to the nearest bin index and convert to long tensor
    return torch.round(scaled_flow).long()


def visualize_quantized_flow(frames_0, frames_1, flows, args, video_file):
    num_bins = [512, 1024, 4096, 8192]
    decode_flows = {bins: decode_flow_code(compute_quantize_flow(flows, args.input_size, bins), args.input_size, bins) for bins in num_bins}

    frames = []
    for frame_idx in range(min(100, frames_0.shape[0])):
        fig, axs = plt.subplots(1, 3 + len(num_bins), figsize=(30, 5))
        axs[0].imshow(frames_0[frame_idx].permute(1, 2, 0).cpu() / 255.)
        axs[1].imshow(frames_1[frame_idx].permute(1, 2, 0).cpu() / 255.)
        axs[0].set_title('Frame 0')
        axs[1].set_title('Frame 1')

        flow_rgb = flow_viz.flow_to_image(flows[frame_idx].numpy().transpose(1, 2, 0))
        axs[2].imshow(flow_rgb)
        axs[2].set_title('Original flow')

        for bidx, bins in enumerate(num_bins):
            decode_flow_rgb = flow_viz.flow_to_image(decode_flows[bins][frame_idx].numpy().transpose(1, 2, 0))
            mse = ((flows[frame_idx] - decode_flows[bins][frame_idx]) ** 2).mean()
            max = flows[frame_idx].max().item()
            max_abs = (flows[frame_idx] - decode_flows[bins][frame_idx]).abs().max()
            axs[3 + bidx].imshow(decode_flow_rgb)
            axs[3 + bidx].set_title(f'n_bins: {bins}\nmse: {mse:.5f}\nmax: {max:.5f}\nmax_abs: {max_abs:.5f}')
        frames.append(fig_to_img(fig))

    os.makedirs(args.output_dir, exist_ok=True)
    video_name = video_file.replace(args.video_folder, '').replace('/', '_').replace('.mp4', '.webm')
    frames_to_video(frames, os.path.join(args.output_dir, video_name), fps=5)


def visualize_flow(frames_0, frames_1, flows, video_file):

    frames = []
    for frame_idx in range(min(100, frames_0.shape[0])):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(frames_0[frame_idx].permute(1, 2, 0).cpu() / 255.)
        axs[0].set_title('Frame 0')

        flow_rgb = flow_viz.flow_to_image(flows[frame_idx].numpy().transpose(1, 2, 0))
        axs[1].imshow(flow_rgb)
        axs[1].set_title('Original flow')

        axs[2].imshow(frames_1[frame_idx].permute(1, 2, 0).cpu() / 255.)
        axs[2].set_title('Frame 1')

        # Find 10 points with significant flow
        flow_magnitude = np.sqrt(
            flows[frame_idx].numpy()[0, :,:]**2 + flows[frame_idx].numpy()[1, :,:]**2)
        threshold = np.percentile(flow_magnitude, 30)  # Use top 30% as significant flow
        y_coords, x_coords = np.where(flow_magnitude > threshold)
        
        # Randomly sample 10 points if we have more than 10
        if len(y_coords) > 10:
            indices = np.random.choice(len(y_coords), min(len(y_coords), 64), replace=False)
            y_coords = y_coords[indices]
            x_coords = x_coords[indices]

        # Plot flow vectors at selected points
        for y, x in zip(y_coords, x_coords):     
            # Plot predicted flow vector
            axs[1].arrow(x, y, 
                flows[frame_idx].numpy()[0,y,x] * 2, 
                flows[frame_idx].numpy()[1,y,x] * 2,  # Scale by 2 for visibility
                color='red', width=2.0)
        
        # set axs[1] range to be the same as axs[0] and axs[2]
        axs[1].set_xlim(0, 256)
        axs[1].set_ylim(256, 0)

        frames.append(fig_to_img(fig))

    frames_to_video(frames, video_file, fps=2)


def visualize_flow_autoencoded(frames_0, frames_1, flows, flows_q, codes, video_file):

    frames = []
    for frame_idx in range(min(100, frames_0.shape[0])):
        fig, axs = plt.subplots(1, 5, figsize=(15, 5))
        axs[0].imshow(frames_0[frame_idx].permute(1, 2, 0).cpu() / 255.)
        axs[0].set_title('Frame 0')

        axs[1].imshow(frames_1[frame_idx].permute(1, 2, 0).cpu() / 255.)
        axs[1].set_title('Frame 1')

        flow_rgb = flow_viz.flow_to_image(flows[frame_idx].numpy().transpose(1, 2, 0))
        axs[2].imshow(flow_rgb)
        axs[2].set_title('Original Flow')

        flow_q_rgb = flow_viz.flow_to_image(flows_q[frame_idx].numpy().transpose(1, 2, 0))
        axs[3].imshow(flow_q_rgb)
        axs[3].set_title('Autoencoded flow')

        axs[4].imshow(codes[frame_idx].cpu().numpy(), cmap='hsv')
        axs[4].set_title('Flow Codes')

        # Find 10 points with significant flow
        flow_magnitude = np.sqrt(
            flows[frame_idx].numpy()[0, :,:]**2 + flows[frame_idx].numpy()[1, :,:]**2)
        threshold = np.percentile(flow_magnitude, 30)  # Use top 30% as significant flow
        y_coords, x_coords = np.where(flow_magnitude > threshold)
        
        # Randomly sample 10 points if we have more than 10
        if len(y_coords) > 10:
            indices = np.random.choice(len(y_coords), min(len(y_coords), 64), replace=False)
            y_coords = y_coords[indices]
            x_coords = x_coords[indices]

        # Plot flow vectors at selected points
        for y, x in zip(y_coords, x_coords):     
            # Plot predicted flow vector
            axs[1].arrow(x, y, 
                flows[frame_idx].numpy()[0,y,x] * 2, 
                flows[frame_idx].numpy()[1,y,x] * 2,  # Scale by 2 for visibility
                color='red', width=2.0)
        
        # set axs[1] range to be the same as axs[0] and axs[2]
        axs[1].set_xlim(0, 256)
        axs[1].set_ylim(256, 0)

        frames.append(fig_to_img(fig))

    # os.makedirs(args.output_dir, exist_ok=True)
    # video_name = video_file.replace(args.video_folder, '').replace('/', '_').replace('.mp4', '.webm')
    # frames_to_video(frames, os.path.join(args.output_dir, video_name), fps=2)
    frames_to_video(frames, video_file, fps=2)


def decode_flow_code(
        quantized_flow: torch.LongTensor,
        input_size: int = 256,
        num_bins: int = 512
    ) -> torch.FloatTensor:
    """
    Decodes the quantized optical flow values back into their original flow range.

    Parameters:
    quantized_flow (LongTensor): The quantized flow values to be decoded, [B, 2, H, W]
    input_size (int): The maximum expected range of the flow values (used to set the decoding range).
    num_bins (int): The number of bins used in quantization (i.e., how many discrete levels the flow was quantized into).

    Returns:
    flow_values (FloatTensor): The decoded flow values, scaled back to the original flow range. [B, 2, H, W]
    """
    # Setting the flow range based on input_size, defining min_flow and max_flow.
    max_range = input_size
    min_flow, max_flow = -max_range, max_range

    # Normalize the quantized values to the range [0, 1]
    normalized_flow = quantized_flow.float() / (num_bins - 1)

    # Scale the normalized values back to the range [min_flow, max_flow]
    flow_values = normalized_flow * (max_flow - min_flow) + min_flow

    return flow_values


def get_flow_frame(f: h5py.File, frame_idx: int, patch_size: int, key: str) -> torch.LongTensor:
    """
    Retrieves the specified flow frame from an HDF5 file and returns it as a tensor of patches.

    Parameters:
        f (h5py.File): An open HDF5 file object containing the flow data.
        frame_idx (int): The index of the frame to retrieve from the HDF5 file.
        patch_size (int): The size of the patches to extract from the flow data.
        key (str): The key in the HDF5 file corresponding to the flow data to extract.

    Returns:
        torch.LongTensor: A tensor containing the flow patches of shape (1, N, P), where:
            - N is the number of patches created from the flow data.
            - P is the patch size, containing interleaved x and y flow values (i.e., x1, y1, x2, y2, ...).
    """
    # Extract the x component of the flow for the specified frame from the HDF5 file.
    flow_x = torch.from_numpy(f[key][frame_idx, 0].astype(np.int64)).unsqueeze(0)  # [1, H, W]

    # Divide the x component flow into patches of the specified patch size.
    patches_x = patchify(flow_x, patch_size=patch_size)  # [1, N, P]

    # Extract the y component of the flow for the specified frame from the HDF5 file.
    flow_y = torch.from_numpy(f[key][frame_idx, 1].astype(np.int64)).unsqueeze(0)  # [1, H, W]

    # Divide the y component flow into patches of the specified patch size.
    patches_y = patchify(flow_y, patch_size=patch_size)  # [1, N, P]

    # Stack the x and y patches along the last dimension and interleave their values before flattening
    patches = torch.stack([patches_x, patches_y], dim=-1).flatten(-2, -1)  # [1, N, 2 * P]

    return patches  # [1, N, 2 * P]


def get_flow_frame_rtheta(f: h5py.File, frame_idx: int, patch_size: int, key: str) -> torch.LongTensor:
    """
    Retrieves the specified flow frame from an HDF5 file and returns it as a tensor of patches.

    Parameters:
        f (h5py.File): An open HDF5 file object containing the flow data.
        frame_idx (int): The index of the frame to retrieve from the HDF5 file.
        patch_size (int): The size of the patches to extract from the flow data.
        key (str): The key in the HDF5 file corresponding to the flow data to extract.

    Returns:
        torch.LongTensor: A tensor containing the flow patches of shape (1, N, P), where:
            - N is the number of patches created from the flow data.
            - P is the patch size, containing interleaved x and y flow values (i.e., x1, y1, x2, y2, ...).
    """
    # Extract the x component of the flow for the specified frame from the HDF5 file.
    flow_x = torch.from_numpy(f[key][frame_idx, 0].astype(np.int64)).unsqueeze(0)  # [1, H, W]

    # Divide the x component flow into patches of the specified patch size.
    patches_x = patchify(flow_x, patch_size=patch_size)  # [1, N, P]

    # Extract the y component of the flow for the specified frame from the HDF5 file.
    flow_y = torch.from_numpy(f[key][frame_idx, 1].astype(np.int64)).unsqueeze(0)  # [1, H, W]

    # Divide the y component flow into patches of the specified patch size.
    patches_y = patchify(flow_y, patch_size=patch_size)  # [1, N, P]

    # Stack the x and y patches along the last dimension and interleave their values before flattening
    patches = torch.stack([patches_x, patches_y], dim=-1).flatten(-2, -1)  # [1, N, 2 * P]

    rtheta_patches = flow_xy_to_rtheta(patches).unsqueeze(-1)

    return rtheta_patches  # [1, N, P]


def flow_xy_to_rtheta(flow_xy: torch.LongTensor) -> torch.LongTensor:

    # Convert flow to float and center it (subtract 255 such that 0 means no flow)
    centered_flow_xy = flow_xy.float() - 255.0  # [B, N, 2]

    # Convert the x and y flow values to polar coordinates (r, theta)
    r = torch.norm(centered_flow_xy, p=2, dim=-1)  # [B, N]
    theta = torch.atan2(centered_flow_xy[..., 1], centered_flow_xy[..., 0])  # [B, N]

    # Quantize the theta to one of 256 discrete values
    theta_quantized = torch.round((theta + np.pi) / (2 * np.pi) * 255).long()  # [B, N]

    # Quantize the r to one of 256 discrete values (set max value to 127) for a max vocab size of 32,768
    r_quantized = torch.round(r).long().clamp(max=127)  # [B, N]

    # Convert the polar coordinates to a single token
    rtheta = r_quantized * 256 + theta_quantized  # [B, N]

    # If r_quantized is 0, set theta_quantized to 0 to ensure we have a sinlge token for no flow
    rtheta[r_quantized == 0] = 0

    return rtheta  # [B, N]


def flow_rtheta_to_xy(flow_rtheta: torch.LongTensor) -> torch.FloatTensor:

    # Extract the r and theta values from the single token
    r = flow_rtheta // 256  # [B, N]
    theta = (flow_rtheta % 256).float() / 255 * 2 * np.pi - np.pi  # [B, N]

    # Convert the polar coordinates back to Cartesian coordinates (x, y)
    x = r.float() * torch.cos(theta)  # [B, N]
    y = r.float() * torch.sin(theta)  # [B, N]

    # Stack the x and y values along the last dimension
    flow_xy = torch.stack([x, y], dim=-1)  # [B, N, 2]

    return flow_xy  # [B, N, 2]


def sample_flow_values_and_positions(
        tokens: torch.LongTensor,
        positions: torch.LongTensor,
        flows: torch.FloatTensor,
        num_flow_patches: int = 0.0,
        alpha: float = 0.75,
        exclude_mask: torch.BoolTensor = None
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    Sample the tokens and positions along the N (1st) axis based on motion flow magnitudes, mask value, and alpha value.
    Only samples from positions where exclude_mask is False.

    Parameters:
        tokens (torch.LongTensor): Tensor containing N patches of size P shape [B, N, P]
        positions (torch.LongTensor): Tensor of positional indexes for the sequence of tokens,
                                      shape [B, N, P], with same dimensions as `tokens`.
        flows (torch.FloatTensor): Tensor containing the optical flow values corresponding to
                                   the tokens, shape [2, H, W].
        num_flow_patches (int): The number of flow patches to be sampled
        alpha (float): Proportion of patches to be selected based on their motion (optical flow).
                       Must be between 0 and 1 (0 means no motion-based patches, 1 means only
                       motion-based patches).
        exclude_mask (torch.BoolTensor): Boolean mask indicating which positions to exclude from sampling.
                                        True values will be excluded. Shape should match flow_magnitude.

    Returns:
        shuffled_tokens (torch.LongTensor): Shuffled tensor of N_m + N_r patches each of size P in
                                            the same order, shape [B, N_m + N_r, P].
        shuffled_positions (torch.LongTensor): Shuffled tensor of positional indexes for the
                                               sequence of tokens, shape [B, N_m + N_r, P].
    """

    # Ensure that alpha is within a valid range [0, 1]
    assert alpha >= 0. and alpha <= 1., "alpha should be between 0 and 1"

    # Compute the magnitude of the flow along the 0th dimension (e.g. for motion-based sorting)
    flow_magnitude = flows.norm(p=2, dim=0)  # [H, W]

    # Apply exclude mask if provided
    if exclude_mask is not None:
        flow_magnitude = flow_magnitude.clone()
        flow_magnitude[exclude_mask] = 0.0

    # Get valid indices where exclude_mask is False
    valid_indices = None if exclude_mask is None else (~exclude_mask).nonzero().squeeze()

    # Calculate the number of tokens to be selected based on motion (flows) and randomly
    num_motion = int(num_flow_patches * alpha)  # Number of motion-based tokens
    num_random = num_flow_patches - num_motion  # Number of randomly selected tokens

    # Sample tokens based on flow magnitude for motion-based selection
    if num_motion == 0:
        if valid_indices is not None:
            shuffle_order = valid_indices[torch.randperm(len(valid_indices))[:num_random]]
        else:
            all_indices = np.arange(tokens.shape[1])
            shuffle_order = np.random.permutation(all_indices)[:num_random]
    else:
        # Sample based on flow magnitude, but only from valid positions
        motion_order = torch.multinomial(flow_magnitude.flatten(), num_motion).numpy()  # [N_m]

        # Get remaining valid indices for random selection
        if valid_indices is not None:
            remaining_indices = np.setdiff1d(valid_indices.cpu().numpy(), motion_order)
        else:
            remaining_indices = np.setdiff1d(np.arange(tokens.shape[1]), motion_order)

        random_order = np.random.permutation(remaining_indices)[:num_random]
        shuffle_order = np.concatenate([motion_order, random_order], axis=0)  # [N_m + N_r]

    # Shuffle the tokens and positions based on the shuffle_order
    shuffled_tokens = tokens[:, shuffle_order]  # shape [B, N_m + N_r, P]
    shuffled_positions = positions[:, shuffle_order]  # shape [B, N_m + N_r, P]

    # Return the shuffled tokens and their corresponding positions
    return shuffled_tokens, shuffled_positions


@torch.no_grad()
def estimate_flow_cond_rgb_loss_3d(model, dataloader, dataloader_name, device, ddp, ctx, eval_iters, debug=False):
    # Set seed for reproducibility
    np.random.seed(2)
    torch.manual_seed(2)

    out = {}
    raw_model = model.module if ddp else model
    model.eval()

    losses = []
    rgb_losses = []
    rgb_first_few_patches_losses = []
    iter_counter = 0
    image_resolution = (256, 256)
    token_resolution = (4, 4)
    num_token_frame = (raw_model.config.patch_size ** 2 + 1) * image_resolution[0] * image_resolution[
        1] // raw_model.config.patch_size ** 2 // (token_resolution[0] * token_resolution[1])
    num_token_flow = (raw_model.config.patch_size ** 2 * 2 + 1) * image_resolution[0] * image_resolution[
        1] // raw_model.config.patch_size ** 2 // (token_resolution[0] * token_resolution[1])
    dataset = dataloader.dataset

    num_token_frame0 = int(num_token_frame * (1 - dataset.img1_mask))
    num_token_flow = 126 # int(num_token_flow * (1 - dataset.flow_mask))
    num_token_frame1 = int(num_token_frame * (1 - dataset.img2_mask))
    print(f"num_token_frame0: {num_token_frame0}, num_token_flow: {num_token_flow}, num_token_frame1: {num_token_frame1}")

    rgb1_token_index = (-num_token_frame1, None)
    print(f"rgb1_token_index: {rgb1_token_index}")

    for seq, pos, tgt, mask in tqdm.tqdm(dataloader):
        if 'cuda' in device:
            seq = seq.pin_memory().to(device, non_blocking=True)
            pos = pos.pin_memory().to(device, non_blocking=True)
            tgt = tgt.pin_memory().to(device, non_blocking=True)
            mask = mask.pin_memory().to(device, non_blocking=True)
            mask = mask.bool()  # this is necessary for running model forward on gpu
            seq_length = seq.size(1)
            print(f"seq_length: {seq_length}, num_token_frame0: {num_token_frame0}, num_token_frame1: {num_token_frame1}, num_token_flow: {num_token_flow}")
            # assert seq_length + 1 == num_token_frame0 + num_token_frame1 + num_token_flow, f"seq_length: {seq_length}, num_token_frame0: {num_token_frame0}, num_token_frame1: {num_token_frame1}, num_token_flow: {num_token_flow}"

        with ctx:
            logits, loss = raw_model(seq, pos, tgt=tgt, mask=mask)

            rgb_tgt = tgt.reshape(-1)[rgb1_token_index[0]:rgb1_token_index[1]]
            rgb_logits = logits.reshape(-1, logits.size(-1))[rgb1_token_index[0]:rgb1_token_index[1]]
            num_first_few_patches = 10
            num_first_few_tokens = 50
            first_few_rgb_logits = rgb_logits[:num_first_few_tokens]
            first_few_rgb_tgt = rgb_tgt[:num_first_few_tokens]
            loss_rgb = F.cross_entropy(rgb_logits, rgb_tgt, ignore_index=-1)
            loss_rgb_first_few = F.cross_entropy(first_few_rgb_logits, first_few_rgb_tgt, ignore_index=-1)
            losses.append(loss.item())
            rgb_losses.append(loss_rgb.item())
            rgb_first_few_patches_losses.append(loss_rgb_first_few.item())

        iter_counter += 1
        if iter_counter > eval_iters or debug:
            break

    # change to float
    out[f"{dataloader_name}/loss"] = np.mean(losses)
    out[f"{dataloader_name}/loss_rgb"] = np.mean(rgb_losses)
    out[f"{dataloader_name}/loss_rgb_first_{num_first_few_patches}_patches"] = np.mean(rgb_first_few_patches_losses)

    return out

@torch.no_grad()
def estimate_flow_loss(model, dataloader, dataloader_name, device, ddp, ctx, eval_iters, debug=False):
    # Set seed for reproducibility
    np.random.seed(2)
    torch.manual_seed(2)

    out = {}
    raw_model = model.module if ddp else model
    model.eval()

    losses = []
    rgb_losses = []
    rgb_first_few_patches_losses = []
    flow_losses = []
    flow_first_few_patches_losses = []
    iter_counter = 0
    image_resolution = (256, 256)
    token_resolution = (4, 4)
    num_patches = 1024
    num_token_frame = (raw_model.config.patch_size ** 2 + 1) * image_resolution[0] * image_resolution[
        1] // raw_model.config.patch_size ** 2 // (token_resolution[0] * token_resolution[1])
    num_token_flow_per_patch = raw_model.config.patch_size ** 2 * 2 + 1
    num_token_rgb_per_patch = raw_model.config.patch_size ** 2 + 1
    dataset = dataloader.dataset

    seq, pos, tgt, mask = next(iter(dataloader))
    num_token_frame0 = int(num_token_frame * (1 - dataset.img1_mask))
    num_token_flow = dataset.shuffled_patches_flow_shape[1] * dataset.shuffled_patches_flow_shape[2]
    if dataset.max_seq_len is None:
        num_token_frame1 = int(num_token_frame * (1 - dataset.img2_mask))
    else:
        num_token_frame1 = dataset.max_seq_len + 1 - num_token_frame0 - num_token_flow

    flow_token_range = (num_token_frame0, num_token_frame0 + num_token_flow)
    rgb1_token_index = (-num_token_frame1, None)

    print(f"num_token_frame0: {num_token_frame0}, num_token_flow: {num_token_flow}, num_token_frame1: {num_token_frame1}")
    print(f"flow_token_range: {flow_token_range}, rgb1_token_index: {rgb1_token_index}")

    for seq, pos, tgt, mask in tqdm.tqdm(dataloader):
        if 'cuda' in device:
            seq = seq.pin_memory().to(device, non_blocking=True)
            pos = pos.pin_memory().to(device, non_blocking=True)
            tgt = tgt.pin_memory().to(device, non_blocking=True)
            mask = mask.pin_memory().to(device, non_blocking=True)
            mask = mask.bool()  # this is necessary for running model forward on gpu
            seq_length = seq.size(1)
            assert num_token_flow == dataset.shuffled_patches_flow_shape[1] * dataset.shuffled_patches_flow_shape[2], (num_token_flow, dataset.shuffled_patches_flow_shape)
            assert seq_length + 1 == num_token_frame0 + num_token_flow + num_token_frame1, f"seq_length: {seq_length}, num_token_frame0: {num_token_frame0}, num_token_frame1: {num_token_frame1}, num_token_flow: {num_token_flow}"

        with ctx:
            logits, loss = raw_model(seq, pos, tgt=tgt, mask=mask)

            # Flow related loss
            flow_tgt = tgt.reshape(-1)[flow_token_range[0]-1:flow_token_range[1]-1]
            assert flow_tgt.view(-1)[0] == -1
            flow_logits = logits.reshape(-1, logits.size(-1))[flow_token_range[0]-1:flow_token_range[1]-1]
            num_first_few_patches = 14
            num_first_few_tokens = num_first_few_patches * num_token_flow_per_patch
            first_few_flow_logits = flow_logits[:num_first_few_tokens]
            first_few_flow_tgt = flow_tgt[:num_first_few_tokens]
            loss_flow = F.cross_entropy(flow_logits, flow_tgt, ignore_index=-1)
            loss_flow_first_few = F.cross_entropy(first_few_flow_logits, first_few_flow_tgt, ignore_index=-1)
            losses.append(loss.item())
            flow_losses.append(loss_flow.item())
            flow_first_few_patches_losses.append(loss_flow_first_few.item())

            # RGB related loss
            if num_token_frame1 > 0:
                rgb_tgt = tgt.reshape(-1)[rgb1_token_index[0]:rgb1_token_index[1]]
                assert rgb_tgt.view(-1)[0] == -1
                rgb_logits = logits.reshape(-1, logits.size(-1))[rgb1_token_index[0]:rgb1_token_index[1]]
                num_first_few_tokens = num_first_few_patches * num_token_rgb_per_patch
                first_few_rgb_logits = rgb_logits[:num_first_few_tokens]
                first_few_rgb_tgt = rgb_tgt[:num_first_few_tokens]
                loss_rgb = F.cross_entropy(rgb_logits, rgb_tgt, ignore_index=-1)
                loss_rgb_first_few = F.cross_entropy(first_few_rgb_logits, first_few_rgb_tgt, ignore_index=-1)
                rgb_losses.append(loss_rgb.item())
                rgb_first_few_patches_losses.append(loss_rgb_first_few.item())

        iter_counter += 1
        if iter_counter > eval_iters or debug:
            break

    # change to float
    out[f"{dataloader_name}/loss"] = np.mean(losses)
    if num_token_frame1 > 0:
        out[f"{dataloader_name}/loss_rgb"] = np.mean(rgb_losses)
        out[f"{dataloader_name}/loss_rgb_first_{num_first_few_patches}_patches"] = np.mean(rgb_first_few_patches_losses)
    out[f"{dataloader_name}/loss_flow"] = np.mean(flow_losses)
    out[f"{dataloader_name}/loss_flow_first_{num_first_few_patches}_patches"] = np.mean(flow_first_few_patches_losses)

    return out


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)

