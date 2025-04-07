import numpy
import numpy as np
import torch
from matplotlib import pyplot as plt
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRenderer,
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
    CamerasBase,
)

from pytorch3d.transforms import Transform3d

from pytorch3d.renderer.cameras import PerspectiveCameras

def unproject_pixels(pts, depth, intrinsics):
    '''
    pts: [N, 2] pixel coords
    depth: [N, ] depth values
    returns: [N, 3] world coords
    '''

    img_pixs = pts[:, [1, 0]].T
    img_pix_ones = np.concatenate((img_pixs, np.ones((1, img_pixs.shape[1]))))

    img_inv = np.linalg.inv(intrinsics)
    cam_img_mat = np.dot(img_inv, img_pix_ones)
    # print(img_pix_ones)

    points_in_cam_ = np.multiply(cam_img_mat, depth.reshape(-1))

    return points_in_cam_.T


def project_pixels(pts, intrinsics):
    '''
    pts: [N, 2] pixel coords
    depth: [N, ] depth values
    returns: [N, 3] world coords
    '''

    img_pixs = pts.T  #

    img_inv = intrinsics[:3, :3]

    pts_in_cam = np.dot(img_inv, img_pixs)

    pts_in_cam = pts_in_cam // pts_in_cam[-1:, :]

    pts_in_cam = pts_in_cam[[1, 0, 2], :]

    return pts_in_cam.T


def get_gravity_vector(pts_in_pixel_space, depth, K):
    '''
    pts_in_pixel_space: [3, 2] array of pixel locations on the ground plane
    depth: depth at those points
    K: intrinsics matrix
    '''
    points_in_cam = unproject_pixels(pts_in_pixel_space, depth, K)

    p1 = points_in_cam[0]
    p2 = points_in_cam[1]
    p3 = points_in_cam[2]

    v1 = p1 - p2
    v2 = p3 - p2

    gravity_up = np.cross(v2, v1)

    gravity_up = gravity_up / np.linalg.norm(gravity_up)

    return gravity_up


def compute_r_t(points_set1, points_set2):
    # Ensure inputs are numpy arrays
    points_set1 = np.array(points_set1)
    points_set2 = np.array(points_set2)

    # Compute centroids
    centroid1 = np.mean(points_set1, axis=0)
    centroid2 = np.mean(points_set2, axis=0)

    # Center the points
    centered1 = points_set1 - centroid1
    centered2 = points_set2 - centroid2

    # Compute covariance matrix
    H = np.dot(centered1.T, centered2)

    # Perform Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    R = np.dot(Vt.T, U.T)

    # Ensure proper orientation (det(R) should be 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Compute translation
    T = centroid2 - np.dot(R, centroid1)

    return R, T


def get_true_pixel_coords(segment_map):
    # Get the indices of True values
    rows, cols = np.where(segment_map)
    # Combine into (row, col) coordinates
    coords = np.column_stack((rows, cols))
    return coords


def render_point_cloud_with_intrinsics(points, colors, K, image_size=512):
    """
    Render a colored point cloud using custom camera intrinsics.

    Args:
        points (torch.Tensor): Tensor of shape (N, 3) for 3D points.
        colors (torch.Tensor): Tensor of shape (N, 3) for RGB colors for each point.
        K (torch.Tensor): 3x3 intrinsic camera matrix.
        image_size (int): Size of the rendered image.

    Returns:
        torch.Tensor: Rendered image of the point cloud.
    """
    device = points.device
    # Create a point cloud object
    points[:, 0] = -points[:, 0]

    points[:, 1] = -points[:, 1]

    point_cloud = Pointclouds(points=[points], features=[colors])

    # Create Perspective camera with custom K
    cameras = PerspectiveCameras(
        focal_length=torch.tensor((K[0, 0], K[1, 1]))[None].to(device).to(points.dtype),  # fx, fy from K
        principal_point=torch.tensor((K[0, 2], K[1, 2]))[None].to(device).to(points.dtype),  # cx, cy from K
        image_size=np.array(image_size)[None],
        in_ndc=False,
        device=device
    )

    print(device)

    # Set up the renderer
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=0.005,  # Adjust point radius
        points_per_pixel=10  # Control blending
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    compositor = AlphaCompositor(background_color=(0, 0, 0))  # White background
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor)

    # Render the point cloud
    rendered_image = renderer(point_cloud)[0, ..., :3]  # Output RGB image
    return rendered_image


def render_point_cloud(coords_3d, colors, K, image_size):
    '''
    :param coords_3d: [N, 3]
    :param colors: [N, 3]
    :param K: intrinsic matrix K [3, 3]
    :return:
    '''

    points = coords_3d.copy()  # np.random.uniform(-1, 1, size=(num_points, 3))  # Random 3D points
    colors_ = colors.copy()  # (points + 1) / 2  # Normalize to range [0, 1] for RGB

    K_ = torch.from_numpy(K[:])
    # Convert to PyTorch tensors
    kk = np.arange(len(points))
    np.random.shuffle(kk)

    points = torch.tensor(points[kk[:260000]], dtype=torch.float32).cuda()
    colors_ = torch.tensor(colors_[kk[:260000]], dtype=torch.float32).cuda()
    K_ = K_.cuda()

    # Render the point cloud with intrinsic settings
    rendered_image = render_point_cloud_with_intrinsics(points, colors_, K_, image_size=image_size)
    return rendered_image

def convert_segment_map_to_3d_coords(segment_map, depth_map, K):
    '''
    segment_map: [H, W]
    depth_map: [H, W]
    K: intrinsics
    returns: [N, 3] 3D coords
    '''

    # get 2D pixel coordinates from segment_map
    segment_coords_in_pixels_img0 = get_true_pixel_coords(segment_map)

    # get depth at those locations
    depth_coords = depth_map[segment_coords_in_pixels_img0[:, 0], segment_coords_in_pixels_img0[:, 1]]

    # unproject points
    segment_coords_in_3d_img0 = unproject_pixels(segment_coords_in_pixels_img0, depth_coords, K)

    return segment_coords_in_3d_img0, segment_coords_in_pixels_img0




def get_dense_flow_from_segment_depth_RT(segment_map, depth_map, R, T, K):
    '''
    segment_map: [H, W]
    depth_map: [H, W]
    R: [3, 3] rotation matrix
    T: [3, ] translation vector
    returns: dense flow map [H, W]
    '''

    # get 2D pixel coordinates from segment_map
    segment_coords_in_pixels_img0 = get_true_pixel_coords(segment_map)

    # get depth at those locations
    depth_coords = depth_map[segment_coords_in_pixels_img0[:, 0], segment_coords_in_pixels_img0[:, 1]]

    # unproject points
    segment_coords_in_3d_img0 = unproject_pixels(segment_coords_in_pixels_img0, depth_coords, K)

    # transform with RT
    segment_coords_in_3d_img1 = np.dot(R, segment_coords_in_3d_img0.T) + T[:, None]
    segment_coords_in_3d_img1 = segment_coords_in_3d_img1.T

    # project onto image
    segment_coords_in_pixels_img1 = project_pixels(segment_coords_in_3d_img1, K)

    # compute flow vectors
    flow_ = segment_coords_in_pixels_img1[:, :2] - segment_coords_in_pixels_img0

    # make flow map
    flow_map = np.zeros([segment_map.shape[0], segment_map.shape[1], 2])
    flow_map[segment_coords_in_pixels_img0[:, 0], segment_coords_in_pixels_img0[:, 1]] = flow_

    #make fake segment map
    segment_coords_in_pixels_img1 = segment_coords_in_pixels_img1.astype(int)
    #clip to image size
    segment_coords_in_pixels_img1 = np.clip(segment_coords_in_pixels_img1, 0, segment_map.shape[0] - 1)
    segment_map = np.zeros([segment_map.shape[0], segment_map.shape[1]])
    segment_map[segment_coords_in_pixels_img1[:, 0], segment_coords_in_pixels_img1[:, 1]] = 1

    return flow_map,  segment_coords_in_pixels_img0, segment_coords_in_3d_img1, segment_map

import cv2
def combine_dilated_bounding_boxes(seg1, seg2, kernel_size=5, iterations=1):
    """
    Combines two binary segmentation maps by performing the following steps:
    1. Dilates each segmentation map.
    2. Finds the bounding box for the dilated region.
    3. Creates a new binary map for each bounding box.
    4. Adds the two bounding box maps to produce a combined map.

    Parameters:
        seg1 (np.ndarray): First binary segmentation map.
        seg2 (np.ndarray): Second binary segmentation map.
        kernel_size (int): Size of the square kernel used for dilation.
        iterations (int): Number of dilation iterations.

    Returns:
        combined_map (np.ndarray): Combined binary map with bounding boxes.
    """
    # Create the structuring element for dilation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Dilate both segmentation maps
    dilated1 = cv2.dilate(seg1, kernel, iterations=iterations)
    dilated2 = cv2.dilate(seg2, kernel, iterations=iterations)

    # Nested helper function to extract the bounding box from a binary image
    def get_bounding_box(binary_img):
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        x_min, y_min = np.inf, np.inf
        x_max, y_max = -np.inf, -np.inf
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        return int(x_min), int(y_min), int(x_max), int(y_max)

    # Get bounding boxes for each dilated map
    bbox1 = get_bounding_box(dilated1)
    bbox2 = get_bounding_box(dilated2)

    # Initialize bounding box maps with zeros
    bbox_map1 = np.zeros_like(seg1, dtype=np.uint8)
    bbox_map2 = np.zeros_like(seg2, dtype=np.uint8)

    if bbox1 is not None:
        x_min, y_min, x_max, y_max = bbox1
        bbox_map1[y_min:y_max, x_min:x_max] = 1

    if bbox2 is not None:
        x_min, y_min, x_max, y_max = bbox2
        bbox_map2[y_min:y_max, x_min:x_max] = 1

    # Combine the bounding box maps and ensure the result remains binary (0 or 1)
    combined_map = np.clip(bbox_map1 + bbox_map2, 0, 1)
    return combined_map



def get_RT(depth_map0, depth_map1, points0, points1, K):
    '''
    depth_map0: [H, W]
    points0: [4, 2]
    points1: [4, 2]
    K: intrinsics
    '''
    pts_in_3d_img0 = unproject_pixels_with_depth_map(depth_map0, K, points0)
    pts_in_3d_img1 = unproject_pixels_with_depth_map(depth_map1, K, points1)

    R, T = compute_r_t(pts_in_3d_img0, pts_in_3d_img1)

    return R, T


def unproject_pixels_with_depth_map(depth_map, K, points):
    '''
    depth_map: [H, W]
    K: intrinsics
    points: [N, 2] in pixel coordinates
    '''

    depth_pts = depth_map[points[:, 1], points[:, 0]]

    pts_in_3d = unproject_pixels(points[:, [1, 0]], depth_pts, K)

    return pts_in_3d


def to_tensor_flow(flow):
    '''
    flow: [H, W, 2]
    returns torch tensor: [1, 2, H, W]
    '''

    flow = torch.from_numpy(flow).permute(2, 0, 1)[None]

    return flow


def to_tensor_segment(seg):
    '''
    flow: [H, W]
    returns torch tensor: [1, 1, H, W]
    '''

    flow = torch.from_numpy(seg)[None, None]

    return flow


def downsample_flow(flow_tensor_cuda, kernel_size=4, stride=4):
    '''
    :param flow_tensor_cuda: [1, 2, 256, 256]
    :return:
    '''
    magnitude = torch.sqrt(flow_tensor_cuda[0, 0] ** 2 + flow_tensor_cuda[0, 1] ** 2)

    # max pool flow with 4x4 kernel
    magnitude, indices = torch.nn.functional.max_pool2d(magnitude[None, None], kernel_size=kernel_size,
                                                        stride=kernel_size,
                                                        return_indices=True)

    # Use the saved indices to pool the additional_tensor
    flowmap_x = flow_tensor_cuda[0, 0].view(-1).gather(0, indices.view(-1)).view(magnitude.shape)
    flowmap_y = flow_tensor_cuda[0, 1].view(-1).gather(0, indices.view(-1)).view(magnitude.shape)

    flow_tensor_cuda_maxpooled = torch.concatenate([flowmap_x, flowmap_y], dim=1)  # [0]

    return flow_tensor_cuda_maxpooled, indices[0, 0]


def get_flattened_index_from_2d_index(indices, size):
    '''
    indices: [N, 2]
    '''

    return indices[:, 0] * size + indices[:, 1]


def downsample_and_scale_flow(flow_map, to_tensor=True):
    '''
    flow_map: [H, W, 2]
    '''
    if to_tensor:
        flow_map_orig = to_tensor_flow(flow_map)
    else:
        flow_map_orig = flow_map

    flow_map_, indices = downsample_flow(flow_map_orig.cuda(), 4, 4)

    flow_map, indices = downsample_flow(flow_map_.cuda(), 4, 4)

    flow_map = flow_map / (flow_map_orig.shape[-1] / 256)

    #clip flow
    flow_map = torch.clamp(flow_map, -256, 256)

    return flow_map, indices

def downsample_and_scale_flow_1k_256(flow_map, kernel_size=4, to_tensor=True):
    '''
    flow_map: [H, W, 2]
    '''
    if to_tensor:
        flow_map_orig = to_tensor_flow(flow_map)
    else:
        flow_map_orig = flow_map

    flow_map_, indices = downsample_flow(flow_map_orig.cuda(), kernel_size, kernel_size)

    flow_map = flow_map_ / (flow_map_orig.shape[-1] / 256)

    #clip flow
    flow_map = torch.clamp(flow_map, -256, 256)

    return flow_map, indices




def downsample_and_scale_flow_gt(flow_map, to_tensor=True):
    '''
    flow_map: [H, W, 2]
    '''
    if to_tensor:
        flow_map_orig = to_tensor_flow(flow_map)
    else:
        flow_map_orig = flow_map

    flow_map_, indices = downsample_flow(flow_map_orig.cuda(), 2, 2)

    flow_map, indices = downsample_flow(flow_map_.cuda(), 4, 4)

    flow_map = flow_map / (flow_map_orig.shape[-1] / 256)

    #clip flow
    flow_map = torch.clamp(flow_map, -256, 256)

    return flow_map, indices

def downsample_and_scale_flow_pred(flow_map, to_tensor=True):
    '''
    flow_map: [H, W, 2]
    '''
    if to_tensor:
        flow_map_orig = to_tensor_flow(flow_map)
    else:
        flow_map_orig = flow_map

    flow_map, indices = downsample_flow(flow_map_orig.cuda(), 4, 4)

    flow_map = flow_map / (flow_map_orig.shape[-1] / 256)

    #clip flow
    flow_map = torch.clamp(flow_map, -256, 256)

    return flow_map, indices


def get_unmask_indices_from_flow_map(flow_map, num_fg_flows=80, num_bg_flows=20, use_mag=False, new_sampling_method=False):
    '''
    flow_map: [1, 2, H, W]
    num_fg_flows
    num_bg_flows
    '''

    flow_map = flow_map[0].permute(1, 2, 0).detach().cpu().numpy()

    if use_mag:
        flow_map_mag = np.sqrt(flow_map[:, :, 0] ** 2 + flow_map[:, :, 1] ** 2)
        segment_map = flow_map_mag > 1
    else:
        # get segment map from flow
        segment_map = np.abs(flow_map).mean(-1) > 0
    segment_map_tensor = to_tensor_segment(segment_map)

    # downsample segment map to 32 x 32 resolution

    segment_map_tensor_fg = torch.nn.functional.max_pool2d(segment_map_tensor.float(), kernel_size=2, stride=2)[0, 0]
    segment_map_tensor_fg = segment_map_tensor_fg.bool().cpu().numpy()
    segment_map_tensor_bg = ~segment_map_tensor_fg

    if new_sampling_method:
        segment_map_tensor_fg = -torch.nn.functional.max_pool2d(-segment_map_tensor.float(), kernel_size=2, stride=2)[0, 0]
        segment_map_tensor_fg = segment_map_tensor_fg.bool().cpu().numpy()

        segment_map_tensor_bg = torch.nn.functional.max_pool2d(segment_map_tensor.float(), kernel_size=2, stride=2)[0, 0]
        segment_map_tensor_bg = ~segment_map_tensor_bg.bool().cpu().numpy()


    inds_fg = get_flattened_index_from_2d_index(get_true_pixel_coords(segment_map_tensor_fg), segment_map_tensor_fg.shape[0])

    inds_bg = get_flattened_index_from_2d_index(get_true_pixel_coords(segment_map_tensor_bg), segment_map_tensor_fg.shape[0])

    # randomly select part of these inds
    indices_fg = np.arange(len(inds_fg))
    np.random.shuffle(indices_fg)
    indices_fg = indices_fg[:num_fg_flows]

    indices_bg = np.arange(len(inds_bg))
    np.random.shuffle(indices_bg)
    indices_bg = indices_bg[:num_bg_flows]

    inds_fg = inds_fg[indices_fg]
    inds_bg = inds_bg[indices_bg]

    indices = np.concatenate([inds_fg, inds_bg])

    return list(indices)



def get_unmask_indices_from_rgb(rgb, num_indices):
    '''
    flow_map: rgb image [H, W, 3] in [0, 255] 256x256 resolution
    num_fg_flows
    num_bg_flows
    '''

    rgb = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().cuda()

    rgb = (rgb < 0.5).all(1, keepdim=True)

    # downsample segment map to 32 x 32 resolution
    segment_map_tensor = torch.nn.functional.avg_pool2d(rgb.float(), kernel_size=8, stride=8)[
        0, 0].cpu().numpy() * 64

    segment_map_tensor = segment_map_tensor == 0

    inds_fg = get_flattened_index_from_2d_index(get_true_pixel_coords(segment_map_tensor), segment_map_tensor.shape[0])

    print(inds_fg)

    # randomly select part of these inds
    indices_fg = np.arange(len(inds_fg))
    np.random.shuffle(indices_fg)
    indices_fg = indices_fg[:num_indices]

    inds_fg = inds_fg[indices_fg]

    return list(inds_fg), segment_map_tensor

def plot_sparse_flow(image0_downsampled, pts_0_downscaled, pts_1_downscaled, ax, title):
    '''
    image0_downsampled: [H, W, 3]
    pts_0_downscaled: [N, 2]
    pts_1_downscaled: [N, 2]
    '''

    ax.imshow(image0_downsampled)

    for pt0, pt1 in zip(pts_0_downscaled, pts_1_downscaled):
        u, v = pt1 - pt0
        x, y = pt0
        ax.arrow(x, y, u, v, color='red', head_width=4, head_length=4, length_includes_head=True)
        ax.plot(x, y, 'bo')

    ax.set_title(title)

    return

def plot_dense_flow(image0_downsampled, unmask_indices, indices_flow_in_256_, flow_viz, ax, title):
    '''
    image0_downsampled: [H, W, 3]
    unmask_indices: [N, ]
    indices_flow_in_256_: [H, W]
    flow_viz: [H, W, 2]
    '''

    ax.imshow(image0_downsampled)

    # for ct, coord in enumerate(unmask_indices):
    for ct, coord in enumerate(np.concatenate([unmask_indices[:30], unmask_indices[-10:]])):
        x_coord = coord // 32
        y_coord = coord % 32
        x_coord_64 = x_coord * 2
        y_coord_64 = y_coord * 2
        coords_in_64 = np.array([[x_coord_64, y_coord_64], [x_coord_64, y_coord_64 + 1],
                                 [x_coord_64 + 1, y_coord_64], [x_coord_64 + 1, y_coord_64 + 1]])
        coords_in_256 = indices_flow_in_256_[coords_in_64[:, 0], coords_in_64[:, 1]]
        coords_in_256_x = coords_in_256 // 256
        coords_in_256_y = coords_in_256 % 256
        coords_in_256 = np.stack([coords_in_256_x, coords_in_256_y], axis=1)
        flows = flow_viz[coords_in_64[:, 0], coords_in_64[:, 1]]

        for coord_, fl in zip(coords_in_256, flows):
            y, x = coord_
            v, u = fl
            if ct >= 30:
                ax.plot(x, y, 'bo', markersize=3)
            else:
                ax.arrow(x, y, u, v, color='red', head_width=4, head_length=4, length_includes_head=True)

    ax.set_title(title)

    return


def plot_flow_visualizations(image0_downsampled, pts_0, pts_1, unmask_indices, indices_flow_in_256_, flow_viz, rgb1_pred, image1_downsampled, segment_map, cum_log_prob=0, output_path="counterfactual_image.png"):
    """
    Function to handle all plotting and visualization.

    Parameters:
    - image0_downsampled: Downsampled version of the first image.
    - pts_0: Points from image0.
    - pts_1: Corresponding points from image1.
    - unmask_indices: Indices of points involved in dense flow.
    - indices_flow_in_256_: Dense flow mapping in 256x256 resolution.
    - flow_viz: Dense flow visualization data.
    - rgb1_pred: Counterfactual RGB prediction from the model.
    - image1_downsampled: Downsampled version of the second (ground truth) image.
    - output_path: Path to save the output plot.
    """
    fig, ax = plt.subplots(1, 5, figsize=(16, 4))

    pts_0 = pts_0 #/ 4
    pts_1 = pts_1 #/ 4

    # Sparse flow visualization
    plot_sparse_flow(image0_downsampled, pts_0, pts_1, ax[0], title="Image0 + Sparse Flow")

    # Dense flow visualization
    plot_dense_flow(image0_downsampled, unmask_indices, indices_flow_in_256_, flow_viz, ax[1], title="Dense Flow (Estimated)")

    # LRAS prediction visualization
    ax[2].imshow(segment_map)
    ax[2].set_title("Segment Map from SAM")

    # LRAS prediction visualization
    ax[3].imshow(rgb1_pred)
    #format cum_log_prb to 5 decimal places
    cum_log_prob = "{:.5f}".format(cum_log_prob)
    ax[3].set_title("LRAS prediction" + "\n" + "log prob: " + str(cum_log_prob))


    # Ground truth image
    ax[4].imshow(image1_downsampled)
    ax[4].set_title("GT")

    # Save the figure
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)




def plot_flow_visualizations_with_gt(image0_downsampled, pts_0, pts_1, unmask_indices, unmask_indices_gt, indices_flow_in_256_, indices_flow_in_256__gt, flow_viz, flow_viz_gt, rgb1_pred, rgb1_pred_gt_flow, image1_downsampled, segment_map, rgb1_pred_flow_model, flow_map_flow_model, unmask_indices_flow_model, indices_flow_in_256__flow_model, flow_map_256,  output_path="counterfactual_image.png"):
    """
    Function to handle all plotting and visualization.

    Parameters:
    - image0_downsampled: Downsampled version of the first image.
    - pts_0: Points from image0.
    - pts_1: Corresponding points from image1.
    - unmask_indices: Indices of points involved in dense flow.
    - indices_flow_in_256_: Dense flow mapping in 256x256 resolution.
    - flow_viz: Dense flow visualization data.
    - rgb1_pred: Counterfactual RGB prediction from the model.
    - image1_downsampled: Downsampled version of the second (ground truth) image.
    - output_path: Path to save the output plot.
    """
    fig, ax = plt.subplots(3, 5, figsize=(16, 10))


    # Sparse flow visualization
    plot_sparse_flow(image0_downsampled, pts_0, pts_1, ax[0, 0], title="Image0 + Sparse Flow")

    # Dense flow visualization
    plot_dense_flow(image0_downsampled, unmask_indices, indices_flow_in_256_, flow_viz, ax[0, 1], title="Estimated Dense Flow")

    #viz segment
    ax[0, 2].imshow(segment_map)
    ax[0, 2].set_title("Seg Map from SAM")

    # LRAS prediction visualization
    ax[0, 3].imshow(rgb1_pred)
    ax[0, 3].set_title("LRAS Prediction")


    # Ground truth image
    ax[0, 4].imshow(image1_downsampled)
    ax[0, 4].set_title("GT")

    ax[1, 0].remove()

    # Dense flow visualization
    plot_dense_flow(image0_downsampled, unmask_indices_gt, indices_flow_in_256__gt, flow_viz_gt, ax[1, 1], title="GT Flow")

    ax[1, 2].remove()

    # LRAS prediction visualization
    ax[1, 3].imshow(rgb1_pred_gt_flow)
    ax[1, 3].set_title("LRAS Prediction")

    ax[1, 4].remove()

    # Dense flow visualization
    plot_dense_flow(image0_downsampled, unmask_indices_flow_model, indices_flow_in_256__flow_model, flow_map_flow_model, ax[2, 1],
                    title="LRAS Dense Flow")

    flow_map_256_mag = np.sqrt(flow_map_256[:, :, 0] ** 2 + flow_map_256[:, :, 1] ** 2)

    ax[2, 2].imshow(flow_map_256_mag, cmap='gray')
    ax[2, 2].set_title("LRAS Flow Magnitude")

    # LRAS prediction visualization
    ax[2, 3].imshow(rgb1_pred_flow_model)
    ax[2, 3].set_title("LRAS Prediction")

    ax[2, 4].remove()

    ax[2, 0].remove()

    # Save the figure
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def get_decoding_order(flow_map_64):
    flow_map_32, indices = downsample_flow(flow_map_64.cuda(), 2, 2)

    # sample flow
    magnitude = torch.sqrt(flow_map_32[0, 0] ** 2 + flow_map_32[0, 1] ** 2)
    magnitude = torch.nn.functional.max_pool2d(magnitude[None, None], kernel_size=2, stride=2)

    pos_indices = torch.where(magnitude.flatten() > 3)[0]

    x_indices = pos_indices // 32
    y_indices = pos_indices % 32

    pos_indices_f1 = flow_map_32[:, :, x_indices, y_indices]
    dx = pos_indices_f1[0, 1]
    dy = pos_indices_f1[0, 0]
    x_indices_f1 = x_indices + dx
    y_indices_f1 = y_indices + dy

    pos_indices_f1 = x_indices_f1 * 32 + y_indices_f1

    all_pos_indices = torch.cat([pos_indices, pos_indices_f1], dim=0)
    all_pos_indices = torch.unique(all_pos_indices.to(torch.int64))

    decode_order = list(np.array(all_pos_indices.cpu().numpy()))

    # add remaining
    decode_order += list(set(range(1024)) - set(decode_order))

    np.random.shuffle(decode_order)

    decode_order = np.array(decode_order)

    return decode_order