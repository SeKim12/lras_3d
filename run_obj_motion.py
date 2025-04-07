import argparse

import cv2
from PIL import Image
from lras_3d.utils.object_motion_eval.utils import load_data, ImageMetricInput, ImageMetricCalculator, save_metrics_in_h5

from lras_3d.task.obj_motion_class import ObjectMotionCounterfactualLRAS
from lras_3d.task.obj_motion_utils import plot_flow_visualizations
import h5py as h5
import os
import torch

import matplotlib.pyplot as plt
import numpy as np

def euler_to_rotation_matrix_deg(yaw_deg, pitch_deg, roll_deg):
    """
    Convert Euler angles (yaw, pitch, roll) in degrees to a rotation matrix.

    Parameters:
        yaw_deg   : rotation about the Z-axis in degrees
        pitch_deg : rotation about the Y-axis in degrees
        roll_deg  : rotation about the X-axis in degrees

    Returns:
        A 3x3 numpy array representing the rotation matrix.
    """
    # Convert degrees to radians
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)

    # Compute individual rotation matrices
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # Combined rotation matrix: R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    return R



def parse_args():
    """
    Parses command-line arguments for the script.
    """
    parser = argparse.ArgumentParser(description="Visualize counterfactual flow and prediction from HDF5 data.")
    parser.add_argument(
        "--input_path",
        type=str,
        default='./example/laptop.png',
        help="Path to the input image."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Path to save the visualization output. Default is 'counterfactual_image.png'."
    )

    # Add a store true argument to the parser
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Store True"
    )

    #num runs
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of runs to perform."
    )

    parser.add_argument("--pts_0", type=float, nargs='+', default=[370, 513, 544, 565, 537, 345, 689, 665],
                        help="Flattened list of pts_0 (e.g. 370 513 544 565 ...).")
    parser.add_argument("--pts_1", type=float, nargs='+', default=[376, 521, 565, 531, 489, 328, 741, 585],
                        help="Flattened list of pts_1.")
    parser.add_argument("--three_points_on_ground", type=float, nargs='+', default=[80, 709, 26, 803, 160, 810],
                        help="Flattened list of three points on ground.")
    parser.add_argument("--K", type=float, nargs='+', default=[915.71210294, 0, 512,
                                                                0, 913.87479936, 512,
                                                                0, 0, 1],
                        help="Flattened 3x3 camera intrinsic matrix.")

    parser.add_argument("--elevation", type=float, default=0,
                        help="Elevation angle in degrees.")
    parser.add_argument("--azimuth", type=float, default=0,
                        help="Azimuth angle in degrees.")
    parser.add_argument("--tilt", type=float, default=0,
                        help="Tilt angle in degrees.")
    parser.add_argument("--T_world", type=float, nargs='+', default=[0.2, 0.0, 0.0],
                        help="Translation vector.")

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    rollout_config = {"temperature": 0.9, "top_k": 1000, "top_p": 0.9, "rollout_mode": "sequential", "seed": 48}

    model_object_motion_cf = ObjectMotionCounterfactualLRAS(rollout_config, args.viz)

    annotations = args.input_path

    output_path = args.output_dir

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    pts_0 = np.array(args.pts_0).reshape(-1, 2)
    pts_1 = np.array(args.pts_1).reshape(-1, 2)
    three_points_on_ground = np.array(args.three_points_on_ground).reshape(-1, 2)
    K = np.array(args.K).reshape(3, 3)

    for runs in range(args.num_runs):

        model_object_motion_cf.seed = rollout_config['seed'] + runs*10
        
        # open "laptop.png"
        image0 = cv2.imread(args.input_path)
        image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)

        # convert to numpy array with values from 0 to 255
        image0 = image0.astype(np.uint8)
        image1 = image0

        # convert to numpy
        image0_downsampled = cv2.resize(image0, (256, 256), interpolation=cv2.INTER_AREA)
        image1_downsampled = image0_downsampled

        with torch.no_grad():

            elevation = args.elevation
            azimuth = args.azimuth
            tilt = args.tilt

            R_world = euler_to_rotation_matrix_deg(azimuth, tilt, elevation)

            T_world = np.array(args.T_world)

            rgb1_pred, flow_map, unmask_indices, indices_flow_in_256, segment_map, cum_log_prob = model_object_motion_cf.run_forward_with_RT(
                image0, pts_0, three_points_on_ground, R_world, T_world, K, condition_rgb=True,
                new_segment_sampling=True,
                condition_from_nvs=True, full_segment_map=False)

            flow_viz = flow_map[0].detach().cpu().numpy().transpose([1, 2, 0])

            indices_flow_in_256_ = indices_flow_in_256.detach().cpu().numpy()

            # make a dir to save result images
            if not os.path.exists(output_path + '/results/'):
                os.makedirs(output_path + '/results/')

            rgb1_pred.save(output_path + '/results/' + annotations.split('/')[-1].split('.')[0] + '_seed_' + str(runs) + '_obj_motion.png')

            print("Done")



