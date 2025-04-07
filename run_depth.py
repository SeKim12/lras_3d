import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import sys
from typing import List
from torch.utils.data import DataLoader
import random
import glob

from lras_3d.predictor.lras_3d_predictor import LRASPredictor
from lras_3d.utils.image_processing import video_to_frames, load_image_center_crop, patchify, load_image
from lras_3d.utils.model_wrapper import ModelFactory
from lras_3d.utils.camera import pose_list_to_matrix
from lras_3d.utils.flow import (
    decode_flow_code, compute_quantize_flow, flow_to_image, get_flow_data_path
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--flow_model_name',
        type=str,
        default="flow_predictor/flow_predictor.pt",
        help='LRAS Model Name (from gcloud)'
    )
    parser.add_argument(
        '--quantizer_name',
        type=str,
        default='rgb_quantizer/rgb_quantizer.pt',
        help='Quantizer Model Name (from gcloud)',
    )
    parser.add_argument(
        '--flow_quantizer_name',
        type=str,
        default='flow_quantizer/flow_quantizer.pt',
        help='Quantizer Model Name (from gcloud)',
    )
    parser.add_argument(
        '--input_path', 
        type=str,
        default='./example/laptop.png',
        help='List of paths to input images or videos',
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default="./results/",
        help='Path to output directory'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run on'
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top k for sampling"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top p for sampling"
    )
    parser.add_argument(
        "--num_flow_patches_to_predict",
        type=int,
        default=1024,
        help="Number of flow patches to predict"
    )
    parser.add_argument(
        "--campose", 
        type=float,
        nargs='+',
        default=[0.0, 0.0, 0.0, 0.0, -0.3, 0.0],
        help="Either a 1D list of 6 floats of the form (x_rot, y_rot, z_rot, x_trans, y_trans, z_trans)" +\
              "or a 2D list repreesenting a 4x4 transformation matrix",
    )
    parser.add_argument(
        "--num_seq_patches",
        type=int,
        default=1023,
        help="Number of sequence patches to predict"
    )
    parser.add_argument(
        "--allowed_tokens_npy_path",
        type=str,
        default=None,
        help="Path to npy file containing allowed tokens"
    )
    parser.add_argument(
        "--randomize_campose_scale",
        action='store_true',
        help="Randomize the scale of the campose"
    )
    parser.add_argument(
        "--no_change_aspect_ratio",
        action='store_true',
        help="Do not change the aspect ratio of the input image"
    )
    parser.add_argument("--num_iterations", type=int, default=8, help="Number of iterations")

    return parser.parse_args()


def main(args):

    # Load image
    if args.input_path is None:
        raise ValueError(f"Please provide an image path, {args.input_path} not found!")
    else:
        input_path = args.input_path
    
    if args.no_change_aspect_ratio:
        frames = [load_image_center_crop(input_path)]
    else:
        frames = [load_image(input_path)]

    # Load predictor
    flow_predictor = LRASPredictor(args.flow_model_name, args.quantizer_name, 
                                flow_quantizer_name=args.flow_quantizer_name, device=args.device)

    # Load campose
    campose = pose_list_to_matrix(args.campose)

    # Deal with the paths
    input_path = args.input_path
    if not os.path.exists(input_path):
        raise ValueError(f"Input path {input_path} not found!")
    input_path_basename = os.path.basename(input_path)
    output_path = os.path.join(args.out_dir, input_path_basename.split(".")[0]+"_depth.png")

    if args.allowed_tokens_npy_path is not None:
        allowed_tokens = np.load(args.allowed_tokens_npy_path)
        allowed_tokens += flow_predictor.model.config.flow_range[0]
        allowed_tokens = allowed_tokens.tolist()
    else:
        allowed_tokens = None

    disparities = []
    for i in range(args.num_iterations):
        with torch.no_grad():
            predictions = flow_predictor.quantized_flow_prediction(
                frames[0],
                campose=campose,
                num_seq_patches=args.num_seq_patches,
                mode="seq2par", seed=args.seed + i * 42, mask_out=True,
                temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                # allowed_tokens=allowed_tokens,
            )

        # Find 10 points with significant predicted flow
        flow_magnitude = np.sqrt(
            predictions["flow_pred_np"][:,:,0]**2 + \
                predictions["flow_pred_np"][:,:,1]**2)
        threshold = np.percentile(flow_magnitude, 30)  # Use top 30% as significant flow
        y_coords, x_coords = np.where(flow_magnitude > threshold)
        disparity = flow_magnitude
        disparities.append(disparity)

    average_disparity = np.mean(disparities, axis=0)
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(average_disparity, cmap='hot')
    plt.savefig(output_path)
    plt.close()

    print("Done")

if __name__ == "__main__":
    args = get_args()
    main(args)
