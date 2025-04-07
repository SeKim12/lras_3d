# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import imageio
import torch

from matplotlib import cm
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def read_video_from_path(path):
    try:
        reader = imageio.get_reader(path)
    except Exception as e:
        print("Error opening video file: ", e)
        return None
    frames = []
    for i, im in enumerate(reader):
        frames.append(np.array(im))
    return np.stack(frames)


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


def draw_circle(rgb, coord, radius, color=(255, 0, 0), visible=True, color_alpha=None):
    # Create a draw object
    draw = ImageDraw.Draw(rgb)
    # Calculate the bounding box of the circle
    left_up_point = (coord[0] - radius, coord[1] - radius)
    right_down_point = (coord[0] + radius, coord[1] + radius)
    # Draw the circle
    color = tuple(list(color) + [color_alpha if color_alpha is not None else 255])

    draw.ellipse(
        [left_up_point, right_down_point],
        fill=tuple(color) if visible else None,
        outline=tuple(color),
    )
    return rgb


def draw_line(rgb, coord_y, coord_x, color, linewidth):
    draw = ImageDraw.Draw(rgb)
    draw.line(
        (coord_y[0], coord_y[1], coord_x[0], coord_x[1]),
        fill=tuple(color),
        width=linewidth,
    )
    return rgb


def add_weighted(rgb, alpha, original, beta, gamma):
    return (rgb * alpha + original * beta + gamma).astype("uint8")


class Visualizer:
    def __init__(
        self,
        save_dir: str = "./results",
        grayscale: bool = False,
        pad_value: int = 0,
        fps: int = 10,
        mode: str = "rainbow",  # 'cool', 'optical_flow'
        linewidth: int = 2,
        show_first_frame: int = 10,
        tracks_leave_trace: int = 0,  # -1 for infinite
    ):
        self.mode = mode
        self.save_dir = save_dir
        if mode == "rainbow":
            self.color_map = cm.get_cmap("gist_rainbow")
        elif mode == "cool":
            self.color_map = cm.get_cmap(mode)
        self.show_first_frame = show_first_frame
        self.grayscale = grayscale
        self.tracks_leave_trace = tracks_leave_trace
        self.pad_value = pad_value
        self.linewidth = linewidth
        self.fps = fps

    def visualize(
        self,
        video: torch.Tensor,  # (B,T,C,H,W)
        tracks: torch.Tensor,  # (B,T,N,2)
        visibility: torch.Tensor = None,  # (B, T, N, 1) bool
        gt_tracks: torch.Tensor = None,  # (B,T,N,2)
        segm_mask: torch.Tensor = None,  # (B,1,H,W)
        filename: str = "video",
        writer=None,  # tensorboard Summary Writer, used for visualization during training
        step: int = 0,
        query_frame=0,
        save_video: bool = True,
        compensate_for_camera_motion: bool = False,
        opacity: float = 1.0,
    ):
        if compensate_for_camera_motion:
            assert segm_mask is not None
        if segm_mask is not None:
            coords = tracks[0, query_frame].round().long()
            segm_mask = segm_mask[0, query_frame][coords[:, 1], coords[:, 0]].long()

        video = F.pad(
            video,
            (self.pad_value, self.pad_value, self.pad_value, self.pad_value),
            "constant",
            255,
        )
        color_alpha = int(opacity * 255)
        tracks = tracks + self.pad_value

        if self.grayscale:
            transform = transforms.Grayscale()
            video = transform(video)
            video = video.repeat(1, 1, 3, 1, 1)

        res_video = self.draw_tracks_on_video(
            video=video,
            tracks=tracks,
            visibility=visibility,
            segm_mask=segm_mask,
            gt_tracks=gt_tracks,
            query_frame=query_frame,
            compensate_for_camera_motion=compensate_for_camera_motion,
            color_alpha=color_alpha,
        )
        if save_video:
            self.save_video(res_video, filename=filename, writer=writer, step=step)
        return res_video

    def save_video(self, video, filename, writer=None, step=0):
        if writer is not None:
            writer.add_video(
                filename,
                video.to(torch.uint8),
                global_step=step,
                fps=self.fps,
            )
        else:
            os.makedirs(self.save_dir, exist_ok=True)
            wide_list = list(video.unbind(1))
            wide_list = [wide[0].permute(1, 2, 0).cpu().numpy() for wide in wide_list]

            # Prepare the video file path
            save_path = os.path.join(self.save_dir, f"{filename}.mp4")

            # Create a writer object
            video_writer = imageio.get_writer(save_path, fps=self.fps)

            # Write frames to the video file
            for frame in wide_list[2:-1]:
                video_writer.append_data(frame)

            video_writer.close()

            print(f"Video saved to {save_path}")

    def draw_tracks_on_video(
        self,
        video: torch.Tensor,
        tracks: torch.Tensor,
        visibility: torch.Tensor = None,
        segm_mask: torch.Tensor = None,
        gt_tracks=None,
        query_frame=0,
        compensate_for_camera_motion=False,
        color_alpha: int = 255,
    ):
        B, T, C, H, W = video.shape
        _, _, N, D = tracks.shape

        assert D == 2
        assert C == 3
        video = video[0].permute(0, 2, 3, 1).byte().detach().cpu().numpy()  # S, H, W, C
        tracks = tracks[0].long().detach().cpu().numpy()  # S, N, 2
        if gt_tracks is not None:
            gt_tracks = gt_tracks[0].detach().cpu().numpy()

        res_video = []

        # process input video
        for rgb in video:
            res_video.append(rgb.copy())
        vector_colors = np.zeros((T, N, 3))

        if self.mode == "optical_flow":
            import flow_vis

            vector_colors = flow_vis.flow_to_color(tracks - tracks[query_frame][None])
        elif segm_mask is None:
            if self.mode == "rainbow":
                y_min, y_max = (
                    tracks[query_frame, :, 1].min(),
                    tracks[query_frame, :, 1].max(),
                )
                norm = plt.Normalize(y_min, y_max)
                for n in range(N):
                    if isinstance(query_frame, torch.Tensor):
                        query_frame_ = query_frame[n]
                    else:
                        query_frame_ = query_frame
                    color = self.color_map(norm(tracks[query_frame_, n, 1]))
                    color = np.array(color[:3])[None] * 255
                    vector_colors[:, n] = np.repeat(color, T, axis=0)
            else:
                # color changes with time
                for t in range(T):
                    color = np.array(self.color_map(t / T)[:3])[None] * 255
                    vector_colors[t] = np.repeat(color, N, axis=0)
        else:
            if self.mode == "rainbow":
                vector_colors[:, segm_mask <= 0, :] = 255

                y_min, y_max = (
                    tracks[0, segm_mask > 0, 1].min(),
                    tracks[0, segm_mask > 0, 1].max(),
                )
                norm = plt.Normalize(y_min, y_max)
                for n in range(N):
                    if segm_mask[n] > 0:
                        color = self.color_map(norm(tracks[0, n, 1]))
                        color = np.array(color[:3])[None] * 255
                        vector_colors[:, n] = np.repeat(color, T, axis=0)

            else:
                # color changes with segm class
                segm_mask = segm_mask.cpu()
                color = np.zeros((segm_mask.shape[0], 3), dtype=np.float32)
                color[segm_mask > 0] = np.array(self.color_map(1.0)[:3]) * 255.0
                color[segm_mask <= 0] = np.array(self.color_map(0.0)[:3]) * 255.0
                vector_colors = np.repeat(color[None], T, axis=0)

        #  draw tracks
        if self.tracks_leave_trace != 0:
            for t in range(query_frame + 1, T):
                first_ind = (
                    max(0, t - self.tracks_leave_trace)
                    if self.tracks_leave_trace >= 0
                    else 0
                )
                curr_tracks = tracks[first_ind : t + 1]
                curr_colors = vector_colors[first_ind : t + 1]
                if compensate_for_camera_motion:
                    diff = (
                        tracks[first_ind : t + 1, segm_mask <= 0]
                        - tracks[t : t + 1, segm_mask <= 0]
                    ).mean(1)[:, None]

                    curr_tracks = curr_tracks - diff
                    curr_tracks = curr_tracks[:, segm_mask > 0]
                    curr_colors = curr_colors[:, segm_mask > 0]

                res_video[t] = self._draw_pred_tracks(
                    res_video[t],
                    curr_tracks,
                    curr_colors,
                )
                if gt_tracks is not None:
                    res_video[t] = self._draw_gt_tracks(
                        res_video[t], gt_tracks[first_ind : t + 1]
                    )

        #  draw points
        for t in range(T):
            img = Image.fromarray(np.uint8(res_video[t]))
            for i in range(N):
                coord = (tracks[t, i, 0], tracks[t, i, 1])
                visibile = True
                if visibility is not None:
                    visibile = visibility[0, t, i]
                if coord[0] != 0 and coord[1] != 0:
                    if not compensate_for_camera_motion or (
                        compensate_for_camera_motion and segm_mask[i] > 0
                    ):
                        img = draw_circle(
                            img,
                            coord=coord,
                            radius=int(self.linewidth * 2),
                            color=vector_colors[t, i].astype(int),
                            visible=visibile,
                            color_alpha=color_alpha,
                        )
            res_video[t] = np.array(img)

        #  construct the final rgb sequence
        if self.show_first_frame > 0:
            res_video = [res_video[0]] * self.show_first_frame + res_video[1:]
        return torch.from_numpy(np.stack(res_video)).permute(0, 3, 1, 2)[None].byte()

    def _draw_pred_tracks(
        self,
        rgb: np.ndarray,  # H x W x 3
        tracks: np.ndarray,  # T x 2
        vector_colors: np.ndarray,
        alpha: float = 0.5,
    ):
        T, N, _ = tracks.shape
        rgb = Image.fromarray(np.uint8(rgb))
        for s in range(T - 1):
            vector_color = vector_colors[s]
            original = rgb.copy()
            alpha = (s / T) ** 2
            for i in range(N):
                coord_y = (int(tracks[s, i, 0]), int(tracks[s, i, 1]))
                coord_x = (int(tracks[s + 1, i, 0]), int(tracks[s + 1, i, 1]))
                if coord_y[0] != 0 and coord_y[1] != 0:
                    rgb = draw_line(
                        rgb,
                        coord_y,
                        coord_x,
                        vector_color[i].astype(int),
                        self.linewidth,
                    )
            if self.tracks_leave_trace > 0:
                rgb = Image.fromarray(
                    np.uint8(
                        add_weighted(
                            np.array(rgb), alpha, np.array(original), 1 - alpha, 0
                        )
                    )
                )
        rgb = np.array(rgb)
        return rgb

    def _draw_gt_tracks(
        self,
        rgb: np.ndarray,  # H x W x 3,
        gt_tracks: np.ndarray,  # T x 2
    ):
        T, N, _ = gt_tracks.shape
        color = np.array((211, 0, 0))
        rgb = Image.fromarray(np.uint8(rgb))
        for t in range(T):
            for i in range(N):
                gt_tracks = gt_tracks[t][i]
                #  draw a red cross
                if gt_tracks[0] > 0 and gt_tracks[1] > 0:
                    length = self.linewidth * 3
                    coord_y = (int(gt_tracks[0]) + length, int(gt_tracks[1]) + length)
                    coord_x = (int(gt_tracks[0]) - length, int(gt_tracks[1]) - length)
                    rgb = draw_line(
                        rgb,
                        coord_y,
                        coord_x,
                        color,
                        self.linewidth,
                    )
                    coord_y = (int(gt_tracks[0]) - length, int(gt_tracks[1]) + length)
                    coord_x = (int(gt_tracks[0]) + length, int(gt_tracks[1]) - length)
                    rgb = draw_line(
                        rgb,
                        coord_y,
                        coord_x,
                        color,
                        self.linewidth,
                    )
        rgb = np.array(rgb)
        return rgb


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



# ---------------- Helper Functions ---------------- #

def load_and_crop_video(video_path):
    """Load a video clip and crop it to the largest centered square."""
    clip = VideoFileClip(video_path)
    side = min(clip.w, clip.h)
    x1 = int((clip.w - side) / 2)
    y1 = int((clip.h - side) / 2)
    x2, y2 = x1 + side, y1 + side

    def crop_to_square(frame):
        return frame[y1:y2, x1:x2]

    cropped_clip = clip.fl_image(crop_to_square)
    cropped_frames = list(cropped_clip.iter_frames())

    #convert to rgb
    cropped_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in cropped_frames]

    return cropped_frames


def downsample_frame(frame, size=(512, 512)):
    """Downsample a frame to a given size."""
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def frame_to_tensor(frame):
    """Convert a numpy image frame to a torch tensor on CUDA with proper shape and type."""
    tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def compute_optical_flow(frame0, frame1, flow_model, unnormalize=False):
    """
    Compute optical flow between two frames.

    frame0, frame1: tensors representing frames
    flow_model: function/model that computes flow (e.g., raft_flow_model)
    """
    flow_map = flow_model(frame0, frame1, unnormalize=unnormalize)
    # Move flow to CPU and convert to numpy with shape HxWx2
    flow_map = flow_map.detach().cpu().numpy()[0].transpose(1, 2, 0)
    return flow_map


def get_flow_visualization(flow_map, pad_value=100, lw=3):
    """
    Convert flow to an RGB image using a helper function (flow_to_image),
    then add padding and border lines.
    """
    # Convert flow map to an image visualization (assumes flow_to_image exists)
    flow_viz = flow_to_image(flow_map)
    flow_viz = torch.from_numpy(flow_viz).permute(2, 0, 1)[None, None]

    flow_viz = F.pad(flow_viz, (pad_value, pad_value, pad_value, pad_value), "constant", 255)
    flow_viz = flow_viz[0, 0].permute(1, 2, 0).detach().cpu().numpy()

    # print(flow_map.shape)

    # Draw borders on the padded image
    flow_viz[pad_value:pad_value + lw, pad_value:flow_viz.shape[1] - pad_value] = 0
    flow_viz[flow_viz.shape[1] - pad_value:flow_viz.shape[1] - pad_value + lw,
    pad_value:flow_viz.shape[1] - pad_value] = 0
    flow_viz[pad_value:flow_viz.shape[1] - pad_value, pad_value:pad_value + lw] = 0
    flow_viz[pad_value:flow_viz.shape[1] - pad_value,
    flow_viz.shape[1] - pad_value:flow_viz.shape[1] - pad_value + lw] = 0
    return flow_viz


def plot_results(frame0_img, frame1_img, flow_img, skip_value, ax):
    """Plot a single result: frame0, frame1 and dense flow in a subplot column."""
    # Create a combined plot for this skip value (vertical stack)
    ax[0].imshow(frame0_img)
    ax[0].set_title('Frame 0')
    ax[0].axis('off')

    ax[1].imshow(frame1_img)
    ax[1].set_title(f'Frame {skip_value}')
    ax[1].axis('off')

    ax[2].imshow(flow_img)
    ax[2].set_title('Dense Flow')
    ax[2].axis('off')



def plot_results_ct(frame0_img, frame1_img, skip_value, ax):
    """Plot a single result: frame0, frame1 and dense flow in a subplot column."""
    # Create a combined plot for this skip value (vertical stack)
    ax[0].imshow(frame0_img)
    ax[0].set_title('Frame 0')
    ax[0].axis('off')

    ax[1].imshow(frame1_img)
    ax[1].set_title(f'Frame {skip_value}')
    ax[1].axis('off')



def process_video(video_path, output_dir, flow_model, vis, frame_skips=[30, 60, 90]):
    """
    Process a video:
      - Load and crop frames.
      - For a set of frame skips (with frame0 fixed), compute optical flow and visualize.
      - Create a combined plot and save the result.

    The output file name is constructed from the dataset name (extracted from the folder)
    and the video file name.
    """
    # Load cropped frames
    frames = load_and_crop_video(video_path)
    if len(frames) < max(frame_skips) + 1:
        print(f"Not enough frames in {video_path} for the maximum skip. Skipping.")
        return

    # Downsample first frame
    image0_downsampled = downsample_frame(frames[0])

    # Prepare a figure with one row per frame skip.

    num_plots = len(frame_skips)
    fig, axs = plt.subplots(num_plots, 3, figsize=(15, 5 * num_plots))

    # In case there's only one row, wrap axs in a list.
    if num_plots == 1:
        axs = [axs]

    # breakpoint()

    flow_norms = []

    for idx, skip in enumerate(frame_skips):
        image1_downsampled = downsample_frame(frames[skip])
        # Convert to tensors
        frame0_tensor = frame_to_tensor(image0_downsampled)
        frame1_tensor = frame_to_tensor(image1_downsampled)

        # Compute optical flow (assuming flow_model behaves like raft_flow_model)
        flow_map = compute_optical_flow(frame0_tensor, frame1_tensor, flow_model)
        flow_norm = np.linalg.norm(flow_map, axis=-1)
        flow_norm = np.mean(flow_norm[flow_norm>2])
        if (flow_norm>2).sum() !=0 :
            flow_norms.append(flow_norm.item())
        else:
            flow_norms.append(0)
        # Get flow visualization image
        flow_viz = get_flow_visualization(flow_map)

        video_np = visualize_flow_with_tracks(flow_map, image0_downsampled, image1_downsampled, vis)

        # Plot the three images (frame0, frame1, flow) on a row
        plot_results(video_np[0], video_np[1], flow_viz, skip, axs[idx])

    plt.tight_layout()

    # Create output file name using dataset name and video file name.
    # Assume the dataset name is the folder name immediately preceding the video file.
    dataset_name = os.path.basename(os.path.dirname(video_path))

    # make video name contain path to video with / replaced with @
    video_name = video_path.replace('/', '@')



    output_filename = f"{dataset_name}_{video_name}.png"
    output_filepath = os.path.join(output_dir, output_filename)
    plt.savefig(output_filepath)
    plt.close(fig)
    print(f"Saved plot for {video_path} as {output_filepath}")

    return flow_norms


# ---------------- Main Processing Loop ---------------- #

def process_all_videos(input_base_dir, output_dir, flow_model, vis, frame_skips=[30, 60, 90], num_videos=100000000):
    """
    Recursively search for videos under input_base_dir and process each one.
    The output images are saved in output_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_flow_norms = []

    fps = None
    # Walk through all subdirectories of input_base_dir
    for root, dirs, files in os.walk(input_base_dir):
        # print(f"Processing videos in {root}", files)
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                print("Processing file", file)
                video_path = os.path.join(root, file)
                flow_norms = process_video(video_path, output_dir, flow_model, vis, frame_skips)
                all_flow_norms.append(flow_norms)

                if fps is None:
                    # get video fps
                    video = cv2.VideoCapture(video_path)
                    fps = video.get(cv2.CAP_PROP_FPS)
                    video.release()

                # print(f"--------------- Video fps: ----------- {fps}")

                # Break if we've processed enough videos
                num_videos -= 1
                if num_videos <= 0:
                    break

        if num_videos <= 0:
            break

    flow_norms = np.array(all_flow_norms) / 2
    flow_norms = np.mean(flow_norms, axis=0)

    #plot flow norms with frame skips
    plt.plot(frame_skips, flow_norms)
    plt.xlabel('Frame Skip')
    plt.ylabel('Average Flow Norm')
    plt.title('Average Flow Norm vs Frame Skip with fps: {}'.format(fps))
    # plt.show()
    #save plot
    plt.savefig(os.path.join(output_dir, 'flow_norms.png'))



@torch.no_grad()
def process_video_cotracker(video_path, output_dir, flow_model, vis, frame_skips=[30, 60, 90]):
    """
    Process a video:
      - Load and crop frames.
      - For a set of frame skips (with frame0 fixed), compute optical flow and visualize.
      - Create a combined plot and save the result.

    The output file name is constructed from the dataset name (extracted from the folder)
    and the video file name.
    """
    # Load cropped frames
    frames = load_and_crop_video(video_path)
    if len(frames) < max(frame_skips) + 1:
        print(f"Not enough frames in {video_path} for the maximum skip. Skipping.")
        return

    # Downsample first frame
    # image0_downsampled = downsample_frame(frames[0])

    # Prepare a figure with one row per frame skip.

    num_plots = len(frame_skips)
    fig, axs = plt.subplots(num_plots, 3, figsize=(15, 5 * num_plots))

    breakpoint()

    # breakpoint()

    # In case there's only one row, wrap axs in a list.
    if num_plots == 1:
        axs = [axs]

    for idx, skip in enumerate(frame_skips):
        all_frames = [frame_to_tensor(downsample_frame(x)) for x in frames[:skip+1]]
        # Convert to tensors
        video = torch.cat(all_frames, 0)[None]

        pred_tracks, pred_visibility = flow_model(video, grid_size=11) # B T N 2,  B T N 1

        sz = 256

        all_frames_256 = [frame_to_tensor(downsample_frame(x, (sz, sz))) for x in frames[:skip + 1]]
        video_256 = torch.cat(all_frames_256, 0)[None]
        y_indices, x_indices = np.mgrid[0:sz, 0:sz]
        coords = np.stack([x_indices.flatten(), y_indices.flatten()], axis=1)
        ones_arr = np.zeros((sz*sz, 1))
        coords = np.concatenate([ones_arr, coords], axis=1)[None]
        coords = torch.from_numpy(coords).float().cuda()
        #split coords into 4 parts and run in sequence
        coords = torch.split(coords, sz*sz//4, dim=1)
        dense_tracks = []
        dense_visibility = []
        for i in range(len(coords)):
            dense_track, dense_vis = flow_model(video_256, queries=coords[i])
            dense_tracks.append(dense_track)
            dense_visibility.append(dense_vis)
        dense_tracks = torch.cat(dense_tracks, dim=2) # B T N 2
        dense_visibility = torch.cat(dense_visibility, dim=2) # B T N 1
        dense_tracks = dense_tracks.reshape(1, video.shape[1], -1, 2) # B T N 2
        dense_visibility = dense_visibility.reshape(1, video.shape[1], -1, 1) # B T N 1

        # dense_tracks, dense_visibility = flow_model(video_256, queries=coords) # B T N 2,  B T N 1

        # breakpoint()

        coords_0 = dense_tracks[:, 0].round().long()
        coords_1 = dense_tracks[:, -1].round().long()
        flow = coords_1 - coords_0 #B N 2
        flow_map = flow[0].cpu().numpy().reshape(sz, sz, 2) * (512/sz)

        #reshape
        flow_viz = get_flow_visualization(flow_map, pad_value=50, lw=2)
        flow_viz = downsample_frame(flow_viz, (512, 512))


        # Convert all_coords and visibility to torch tensors
        all_coords = pred_tracks[:, [0, -1], :, :]
        visibility = pred_visibility[:, [0, -1], :]

        # Filter out tracks with negligible movement based on Euclidean distance
        all_coords_0 = all_coords[0, 0].int()
        all_coords_1 = all_coords[0, 1].int()
        dists = torch.sum((all_coords_1 - all_coords_0) ** 2, dim=1)

        if torch.sum(dists > 1) > 0:
            all_coords = all_coords[:, :, dists > 1]
            visibility = visibility[:, :, dists > 1]

        output_video = vis.visualize(
            video=video[:, [0, -1]],
            tracks=all_coords,
            visibility=visibility,
            filename='teaser',
            save_video=False
        )

        video_np = output_video.detach().cpu().numpy()[0].transpose([0, 2, 3, 1])

        # Plot the three images (frame0, frame1, flow) on a row
        plot_results(video_np[0], video_np[1], flow_viz, skip, axs[idx])

    plt.tight_layout()

    # Create output file name using dataset name and video file name.
    # Assume the dataset name is the folder name immediately preceding the video file.
    dataset_name = os.path.basename(os.path.dirname(video_path))
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_filename = f"{dataset_name}_{video_name}.png"
    output_filepath = os.path.join(output_dir, output_filename)
    plt.savefig(output_filepath)
    plt.close(fig)
    print(f"Saved plot for {video_path} as {output_filepath}")


# ---------------- Main Processing Loop ---------------- #

def process_all_videos_cotracker(input_base_dir, output_dir, flow_model, vis, frame_skips=[30, 60, 90]):
    """
    Recursively search for videos under input_base_dir and process each one.
    The output images are saved in output_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Walk through all subdirectories of input_base_dir
    for root, dirs, files in os.walk(input_base_dir):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(root, file)
                process_video_cotracker(video_path, output_dir, flow_model, vis, frame_skips)

def get_points_on_a_grid(
    size: int,
    extent,
    center = None,
    device = torch.device("cpu"),
):
    r"""Get a grid of points covering a rectangular region

    `get_points_on_a_grid(size, extent)` generates a :attr:`size` by
    :attr:`size` grid fo points distributed to cover a rectangular area
    specified by `extent`.

    The `extent` is a pair of integer :math:`(H,W)` specifying the height
    and width of the rectangle.

    Optionally, the :attr:`center` can be specified as a pair :math:`(c_y,c_x)`
    specifying the vertical and horizontal center coordinates. The center
    defaults to the middle of the extent.

    Points are distributed uniformly within the rectangle leaving a margin
    :math:`m=W/64` from the border.

    It returns a :math:`(1, \text{size} \times \text{size}, 2)` tensor of
    points :math:`P_{ij}=(x_i, y_i)` where

    .. math::
        P_{ij} = \left(
             c_x + m -\frac{W}{2} + \frac{W - 2m}{\text{size} - 1}\, j,~
             c_y + m -\frac{H}{2} + \frac{H - 2m}{\text{size} - 1}\, i
        \right)

    Points are returned in row-major order.

    Args:
        size (int): grid size.
        extent (tuple): height and with of the grid extent.
        center (tuple, optional): grid center.
        device (str, optional): Defaults to `"cpu"`.

    Returns:
        Tensor: grid.
    """
    if size == 1:
        return torch.tensor([extent[1] / 2, extent[0] / 2], device=device)[None, None]

    if center is None:
        center = [extent[0] / 2, extent[1] / 2]

    margin = extent[1] / 64
    range_y = (margin - extent[0] / 2 + center[0], extent[0] / 2 + center[0] - margin)
    range_x = (margin - extent[1] / 2 + center[1], extent[1] / 2 + center[1] - margin)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(*range_y, size, device=device),
        torch.linspace(*range_x, size, device=device),
        indexing="ij",
    )
    return torch.stack([grid_y.contiguous().view(-1), grid_x.contiguous().view(-1)], dim=1).cpu().numpy()#.reshape(1, -1, 2)


def visualize_flow_with_tracks(flow_map, image0_downsampled, image1_downsampled, vis):
    """
    Visualizes optical flow using the given Visualizer instance.

    Parameters:
        flow_map (np.ndarray): Optical flow map with shape (H, W, 2).
        image0_downsampled (np.ndarray): First downsampled image (H x W x 3).
        image1_downsampled (np.ndarray): Second downsampled image (H x W x 3).
        vis (Visualizer): An instance of the Visualizer class.

    Returns:
        video (np.ndarray): The output video frames after visualization,
                            with shape (num_frames, H, W, 3).
    """
    # Create grid indices at intervals of 50 pixels
    #y_indices, x_indices = np.mgrid[0:512:50, 0:512:50]
    # coords = np.stack([y_indices.flatten(), x_indices.flatten()], axis=1)

    # y_indices, x_indices = np.mgrid[0:512:50, 0:512:50]
    coords = get_points_on_a_grid(11, (512, 512)).astype(np.int32)

    # breakpoint()

    # Compute the displacement (deltas) from the flow map
    deltas = flow_map[coords[:, 0], coords[:, 1]][:, [1, 0]]
    coords_in_next_frame = coords + deltas

    # Stack coordinates from both frames and add a batch dimension
    all_coords = np.stack([coords, coords_in_next_frame], axis=0)[None]

    # Create visibility mask
    visibility = np.ones([1, 2, all_coords.shape[2]], dtype=np.float32)

    # Prepare video: stack and transpose images to create a 4D tensor (B, C, H, W)
    video = np.stack([image0_downsampled, image1_downsampled], axis=0).transpose([0, 3, 1, 2])[None]
    video = torch.from_numpy(video)

    # Convert all_coords and visibility to torch tensors
    all_coords = torch.from_numpy(all_coords)
    visibility = torch.from_numpy(visibility)

    # Filter out tracks with negligible movement based on Euclidean distance
    all_coords_0 = all_coords[0, 0]
    all_coords_1 = all_coords[0, 1]
    dists = torch.sum((all_coords_1 - all_coords_0) ** 2, dim=1)

    if torch.sum(dists > 1) > 0:
        all_coords = all_coords[:, :, dists > 1]
        visibility = visibility[:, :, dists > 1]

    # Visualize using the provided Visualizer instance
    # Note: swapping coordinates from [y, x] to [x, y] as needed by the visualizer.
    output_video = vis.visualize(
        video=video,
        tracks=all_coords[:, :, :, [1, 0]],
        visibility=visibility,
        filename='teaser',
        save_video=False
    )

    # Convert output video back to numpy format and rearrange dimensions to (frames, H, W, 3)
    video_np = output_video.detach().cpu().numpy()[0].transpose([0, 2, 3, 1])
    return video_np


