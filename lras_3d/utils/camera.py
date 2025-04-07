"""
Utilities for performing 3D transformations and camera motion.
"""

import torch
import numpy as np
import lras_3d.utils.self_motion.transforms as py3d_t

def parse_camera_orientation_dict(camera_orientation_dict):
    if "pitch" in camera_orientation_dict and "yaw" in camera_orientation_dict and "roll" in camera_orientation_dict:
        pitch = torch.tensor(camera_orientation_dict['pitch'], dtype=torch.float32)
        yaw = torch.tensor(camera_orientation_dict['yaw'], dtype=torch.float32)
        roll = torch.tensor(camera_orientation_dict['roll'], dtype=torch.float32)
        rotation_world_from_camera = py3d_t.euler_angles_to_matrix(torch.tensor([-90+pitch, yaw, roll], dtype=torch.float32) * np.pi / 180, 'XYZ')
        transform_world_from_camera = torch.eye(4, dtype=torch.float32)
        transform_world_from_camera[:3, :3] = rotation_world_from_camera
        transform_world_from_camera = transform_world_from_camera.unsqueeze(0)
        return transform_world_from_camera
    elif "transform_world_from_camera" in camera_orientation_dict:
        transform_world_from_camera = torch.tensor(camera_orientation_dict['transform_world_from_camera'], dtype=torch.float32)
        if transform_world_from_camera.dim() == 2:
            transform_world_from_camera = transform_world_from_camera.unsqueeze(0)
        return transform_world_from_camera
    else:
        raise ValueError("Invalid camera_orientation_dict")
    
def get_transform_world_from_camera0_from_pitch_yaw_roll(pitch, yaw, roll):
    pitch = torch.tensor(pitch, dtype=torch.float32)
    yaw = torch.tensor(yaw, dtype=torch.float32)
    roll = torch.tensor(roll, dtype=torch.float32)
    rotation_world_from_camera = py3d_t.euler_angles_to_matrix(torch.tensor([-90+pitch, yaw, roll], dtype=torch.float32) * np.pi / 180, 'XYZ')
    transform_world_from_camera = torch.eye(4, dtype=torch.float32)
    transform_world_from_camera[:3, :3] = rotation_world_from_camera
    transform_world_from_camera = transform_world_from_camera.unsqueeze(0)
    return transform_world_from_camera

def get_campose_from_transform_world_from_camera0_and_transform_world_from_camera1(transform_world_from_camera0, transform_world_from_camera1):
    assert len(transform_world_from_camera0.shape) == 3
    assert len(transform_world_from_camera1.shape) == 3
    transform_world_from_camera0 = torch.tensor(transform_world_from_camera0, dtype=torch.float32)
    transform_world_from_camera1 = torch.tensor(transform_world_from_camera1, dtype=torch.float32)
    transform_camera0_from_camera1 = torch.einsum('bij,bjk->bik', torch.inverse(transform_world_from_camera0), transform_world_from_camera1)
    return transform_camera0_from_camera1

def get_campose_from_translation_and_axis_angle_rotation(translation, axis_angle_rotation):
    assert isinstance(translation, list) or isinstance(translation, tuple) or (isinstance(translation, np.ndarray) and len(translation.shape) == 1)
    assert isinstance(axis_angle_rotation, list) or isinstance(axis_angle_rotation, tuple) or (isinstance(axis_angle_rotation, np.ndarray) and len(axis_angle_rotation.shape) == 1)
    assert len(translation) == 3
    assert len(axis_angle_rotation) == 3
    translation = torch.tensor(translation, dtype=torch.float32)
    axis_angle_rotation = torch.tensor(axis_angle_rotation, dtype=torch.float32)
    if translation.dim() == 1:
        translation = translation[None, ...]
    if axis_angle_rotation.dim() == 1:
        axis_angle_rotation = axis_angle_rotation[None, ...]
    if translation.size(0) != axis_angle_rotation.size(0):
        raise ValueError("Translation and axis_angle_rotation should have the same batch size")
    rotation = py3d_t.axis_angle_to_rotation_matrix(axis_angle_rotation * np.pi / 180)
    campose = torch.eye(4, dtype=torch.float32)
    campose[:3, :3] = rotation
    campose[:3, 3] = translation
    return campose.unsqueeze(0)

def get_campose_from_translation_and_euler_angle_rotation(translation, euler_angle_rotation):
    assert isinstance(translation, list) or isinstance(translation, tuple) or (isinstance(translation, np.ndarray) and len(translation.shape) == 1)
    assert isinstance(euler_angle_rotation, list) or isinstance(euler_angle_rotation, tuple) or (isinstance(euler_angle_rotation, np.ndarray) and len(euler_angle_rotation.shape) == 1)
    assert len(translation) == 3
    assert len(euler_angle_rotation) == 3
    translation = torch.tensor(translation, dtype=torch.float32)
    euler_angle_rotation = torch.tensor(euler_angle_rotation, dtype=torch.float32)
    if translation.dim() == 1:
        translation = translation[None, ...]
    if euler_angle_rotation.dim() == 1:
        euler_angle_rotation = euler_angle_rotation[None, ...]
    if translation.size(0) != euler_angle_rotation.size(0):
        raise ValueError("Translation and euler_angle_rotation should have the same batch size")
    rotation = py3d_t.euler_angles_to_matrix(euler_angle_rotation * np.pi / 180, 'XYZ')
    campose = torch.eye(4, dtype=torch.float32)
    campose[:3, :3] = rotation
    campose[:3, 3] = translation
    return campose.unsqueeze(0)

def get_campose_from_transform_world_from_camera_and_rotation_z_world(transform_world_from_camera, rotation_z_world):
    assert len(transform_world_from_camera.shape) == 3
    assert type(rotation_z_world) in [int, float]
    transform_world_from_camera = torch.tensor(transform_world_from_camera, dtype=torch.float32)
    camera_position_in_world = transform_world_from_camera[:, :3, 3]
    rotation_matrix_in_world = py3d_t.euler_angles_to_matrix(torch.tensor([0, 0, rotation_z_world], dtype=torch.float32) * np.pi / 180, 'XYZ')
    rotation_matrix_in_world = rotation_matrix_in_world.unsqueeze(0)
    # below this can be reduced to a single line of matrix multiplication, but we are doing it step by step for clarity on what is happening
    rotated_camera_position_in_world = torch.einsum('bij,bjk->bik', rotation_matrix_in_world, camera_position_in_world.unsqueeze(-1)).squeeze(-1)
    camera_x_axis_in_world = transform_world_from_camera[:, :3, 0]
    camera_y_axis_in_world = transform_world_from_camera[:, :3, 1]
    camera_z_axis_in_world = transform_world_from_camera[:, :3, 2]
    rotated_camera_x_axis_in_world = torch.einsum('bij,bjk->bik', rotation_matrix_in_world, camera_x_axis_in_world.unsqueeze(-1)).squeeze(-1)
    rotated_camera_y_axis_in_world = torch.einsum('bij,bjk->bik', rotation_matrix_in_world, camera_y_axis_in_world.unsqueeze(-1)).squeeze(-1)
    rotated_camera_z_axis_in_world = torch.einsum('bij,bjk->bik', rotation_matrix_in_world, camera_z_axis_in_world.unsqueeze(-1)).squeeze(-1)
    transform_world_from_rotated_camera = torch.eye(4, dtype=torch.float32)
    transform_world_from_rotated_camera[:3, 0] = rotated_camera_x_axis_in_world
    transform_world_from_rotated_camera[:3, 1] = rotated_camera_y_axis_in_world
    transform_world_from_rotated_camera[:3, 2] = rotated_camera_z_axis_in_world
    transform_world_from_rotated_camera[:3, 3] = rotated_camera_position_in_world
    transform_world_from_rotated_camera = transform_world_from_rotated_camera.unsqueeze(0)
    campose = torch.einsum('bij,bjk->bik', torch.inverse(transform_world_from_camera), transform_world_from_rotated_camera)
    return campose

def get_campose_from_rotation_xyz_camera(rotation_xyz_camera):
    rotation_x = torch.tensor(rotation_xyz_camera[0], dtype=torch.float32)
    rotation_y = torch.tensor(rotation_xyz_camera[1], dtype=torch.float32)
    rotation_z = torch.tensor(rotation_xyz_camera[2], dtype=torch.float32)
    rotation_matrix = py3d_t.euler_angles_to_matrix(torch.tensor([rotation_x, rotation_y, rotation_z], dtype=torch.float32) * np.pi / 180, 'XYZ')
    campose = torch.eye(4, dtype=torch.float32)
    campose[:3, :3] = rotation_matrix
    return campose.unsqueeze(0)

def get_intrinsics_from_fov_and_image_size(fov, image_size):
    assert len(image_size) == 2
    fov_rad = fov * np.pi / 180
    # make 1, 3, 3 intrinsics matrix
    intrinsics = torch.eye(3, dtype=torch.float32)
    intrinsics[0, 0] = 1 / np.tan(fov_rad / 2) * image_size[1] / 2
    intrinsics[1, 1] = 1 / np.tan(fov_rad / 2) * image_size[1] / 2
    intrinsics[0, 2] = image_size[1] / 2
    intrinsics[1, 2] = image_size[0] / 2
    intrinsics = intrinsics.unsqueeze(0)
    return intrinsics

def parse_campose_dict(campose_dict):
    if "campose" in campose_dict:
        return torch.tensor(campose_dict['campose'], dtype=torch.float32)
    elif "transform_world_from_camera0" in campose_dict and "transform_world_from_camera1" in campose_dict:
        transform_world_from_camera0 = torch.tensor(campose_dict['transform_world_from_camera0'], dtype=torch.float32)
        transform_world_from_camera1 = torch.tensor(campose_dict['transform_world_from_camera1'], dtype=torch.float32)
        transform_camera0_from_camera1 = torch.einsum('bij,bjk->bik', torch.inverse(transform_world_from_camera0), transform_world_from_camera1)
        return transform_camera0_from_camera1
    if "translation" in campose_dict and "rotation" in campose_dict:
        translation = torch.tensor(campose_dict['translation'], dtype=torch.float32)
        rotation = torch.tensor(campose_dict['rotation'], dtype=torch.float32)
        if translation.dim() == 1:
            translation = translation[None, ...]
        if rotation.dim() == 1:
            rotation = rotation[None, ...]
        if translation.size(0) != rotation.size(0):
            raise ValueError("Translation and rotation should have the same batch size")
        # Convert rotation to rotation matrix
        rotation = py3d_t.axis_angle_to_rotation_matrix(rotation * np.pi / 180)
        # Combine translation and rotation into a 4x4 transformation matrix
        campose = torch.eye(4, dtype=torch.float32)
        campose[:3, :3] = rotation
        campose[:3, 3] = translation
        return campose.unsqueeze(0)
    if "translation" in campose_dict and "euler_angle" in campose_dict:
        translation = torch.tensor(campose_dict['translation'], dtype=torch.float32)
        euler_angle = torch.tensor(campose_dict['euler_angle'], dtype=torch.float32)
        if translation.dim() == 1:
            translation = translation[None, ...]
        if euler_angle.dim() == 1:
            euler_angle = euler_angle[None, ...]
        if translation.size(0) != euler_angle.size(0):
            raise ValueError("Translation and euler_angle should have the same batch size")
        # Convert euler_angle to rotation matrix
        rotation = py3d_t.euler_angles_to_matrix(euler_angle * np.pi / 180, 'XYZ')
        # Combine translation and rotation into a 4x4 transformation matrix
        campose = torch.eye(4, dtype=torch.float32)
        campose[:3, :3] = rotation
        campose[:3, 3] = translation
        return campose.unsqueeze(0)
    else:
        raise ValueError("Invalid campose_dict")
    
def parse_intrinsics_dict(intrinsics_dict):
    if "intrinsics" in intrinsics_dict:
        intrinsics = torch.tensor(intrinsics_dict['intrinsics'], dtype=torch.float32)
    elif "fov" in intrinsics_dict and "image_size" in intrinsics_dict:
        fov_deg = intrinsics_dict['fov'] # this is horizontal fov
        fov_rad = fov_deg * np.pi / 180
        image_size = intrinsics_dict['image_size'] # height, width in pixels
        # make 1, 3, 3 intrinsics matrix
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics[0, 0] = 1 / np.tan(fov_rad / 2) * image_size[1] / 2
        intrinsics[1, 1] = 1 / np.tan(fov_rad / 2) * image_size[1] / 2
        intrinsics[0, 2] = image_size[1] / 2
        intrinsics[1, 2] = image_size[0] / 2
        intrinsics = intrinsics.unsqueeze(0)
    else:
        raise ValueError("Invalid intrinsics_dict")
    return intrinsics


def unproject_pixels(pts, depth, intrinsics):
    '''
    pts: [N, 2] pixel coords
    depth: [N, ] depth values
    returns: [N, 3] world coords
    '''

    img_pixs = pts.T
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

def get_camera_orientation_dict_from_threepoints_depth_intrinsics(threepoints, intrinsics, depthmap):
    threepoints_tensor = torch.tensor(threepoints, dtype=torch.float32).unsqueeze(-1)
    threepoints_homogeneous = torch.cat((threepoints_tensor, torch.ones_like(threepoints_tensor[:, [0], :])), dim=1)
    intrinsics_tensor = torch.tensor(intrinsics, dtype=torch.float32)
    threepoints_in_camera_unnormalized = torch.einsum('bij,bjk->bik', torch.inverse(intrinsics_tensor), threepoints_homogeneous)
    # breakpoint()
    # threepoints_in_camera_normalized = threepoints_in_camera_unnormalized / torch.norm(threepoints_in_camera_unnormalized, dim=1, keepdim=True)

    depth_of_threepoints = torch.tensor(depthmap[threepoints[:, 1], threepoints[:, 0]])
    threepoints_in_camera = threepoints_in_camera_unnormalized * depth_of_threepoints.unsqueeze(-1).unsqueeze(-1) # 3, 3, 1

    p1 = threepoints_in_camera[0].squeeze()
    p2 = threepoints_in_camera[1].squeeze()
    p3 = threepoints_in_camera[2].squeeze()

    v1 = p1 - p2 
    v2 = p3 - p2

    gravity_up = np.cross(v2, v1) # we are assuming the points are in counterclockwise order

    gravity_up_in_camera = gravity_up / np.linalg.norm(gravity_up)
    gravity_up_in_camera = torch.tensor(gravity_up_in_camera, dtype=torch.float32)
    camera_front = torch.tensor([0, 0, 1], dtype=torch.float32)
    world_x_in_camera = torch.cross(camera_front, gravity_up_in_camera)
    world_x_in_camera = world_x_in_camera / torch.norm(world_x_in_camera)
    world_y_in_camera = torch.cross(gravity_up_in_camera, world_x_in_camera)
    world_y_in_camera = world_y_in_camera / torch.norm(world_y_in_camera)


    transform_camera_from_world = torch.eye(4, dtype=torch.float32)
    transform_camera_from_world[:3, 2] = gravity_up_in_camera
    transform_camera_from_world[:3, 0] = world_x_in_camera
    transform_camera_from_world[:3, 1] = world_y_in_camera
    transform_world_from_camera = torch.inverse(transform_camera_from_world).unsqueeze(0)

    return {"transform_world_from_camera": transform_world_from_camera, 'transform_camera_from_world': transform_camera_from_world}


def resize_intrinsics(intrinsics, original_size, new_size=(256, 256)):
    """
    Resizes the camera intrinsics based on resizing the image while maintaining the aspect ratio and center cropping.
    
    Parameters:
        intrinsics (np.array): The original camera intrinsics matrix of shape (B, 3, 3).
        original_size (tuple): The original image size as (H, W).
        new_size (tuple): The target image size as (height, width), default is (256, 256).
    
    Returns:
        np.array: The resized intrinsics matrix of shape (B, 3, 3).
    """
    H, W = original_size
    new_H, new_W = new_size
    
    if H < W:
        scale = new_H / H
        cropped_W = scale * W
        crop_offset = (cropped_W - new_W) / 2
        intrinsics[:, 0, 0] *= scale  # fx
        intrinsics[:, 1, 1] *= scale  # fy
        intrinsics[:, 0, 2] = intrinsics[:, 0, 2] * scale - crop_offset  # cx
        intrinsics[:, 1, 2] *= scale
    else:
        scale = new_W / W
        cropped_H = scale * H
        crop_offset = (cropped_H - new_H) / 2
        intrinsics[:, 0, 0] *= scale  # fx
        intrinsics[:, 1, 1] *= scale  # fy
        intrinsics[:, 1, 2] = intrinsics[:, 1, 2] * scale - crop_offset  # cy
        intrinsics[:, 0, 2] *= scale
    
    return intrinsics

def pose_list_to_matrix(pose_list):
    """
    Convert a list of 6 floats [x_rot, y_rot, z_rot, x_trans, y_trans, z_trans]
    into a 4x4 transformation matrix (torch float tensor).
    """
    # Unpack the pose list
    x_rot, y_rot, z_rot, x_trans, y_trans, z_trans = pose_list

    # Convert rotations to torch tensors (assuming rotations are in radians)
    x_rot = torch.tensor(x_rot, dtype=torch.float32)
    y_rot = torch.tensor(y_rot, dtype=torch.float32)
    z_rot = torch.tensor(z_rot, dtype=torch.float32)

    # Compute cosine and sine of rotation angles
    cos_x, sin_x = torch.cos(x_rot), torch.sin(x_rot)
    cos_y, sin_y = torch.cos(y_rot), torch.sin(y_rot)
    cos_z, sin_z = torch.cos(z_rot), torch.sin(z_rot)

    # Rotation matrix around the x-axis
    Rx = torch.tensor([
        [1,     0,      0     ],
        [0,  cos_x, -sin_x],
        [0,  sin_x,  cos_x]
    ], dtype=torch.float32)

    # Rotation matrix around the y-axis
    Ry = torch.tensor([
        [ cos_y, 0, sin_y],
        [     0, 1,     0],
        [-sin_y, 0, cos_y]
    ], dtype=torch.float32)

    # Rotation matrix around the z-axis
    Rz = torch.tensor([
        [cos_z, -sin_z, 0],
        [sin_z,  cos_z, 0],
        [    0,      0, 1]
    ], dtype=torch.float32)

    # Combine the rotation matrices (order matters: R = Rz * Ry * Rx)
    R = Rz @ Ry @ Rx  # Matrix multiplication

    # Translation vector
    t = torch.tensor([x_trans, y_trans, z_trans], dtype=torch.float32)

    # Construct the 4x4 transformation matrix
    T = torch.eye(4, dtype=torch.float32)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def quantize_6dof_campose(
        campose: np.array, 
        translation_quantization: float = 0.001,
        translation_offset: np.array = 1001,
        rotation_quantization: float = 0.1,
        rotation_offset: np.array = 1801,
        translation_scale_quantization: float = 0.001,
        translation_scale_offset: float = 1,
        rotation_indexes: np.array = [0, 1, 2],
        translation_indexes: np.array = [3, 4, 5],
        translation_scale_index: int = 6,
        **kwargs
    ) -> np.array:
    """
    Quantize a 6-DOF camera pose vector

    Parameters:
        - campose: 6-DOF camera pose vector
        - translation_quantization: Quantization value for the translation
        - translation_offset: Offset value for the translation
        - rotation_quantization: Quantization value for the rotation
        - rotation_offset: Offset value for the rotation

    Returns:
        - quantized_campose: Quantized 6-DOF camera pose vector
    """

    # Extract the translation and rotation components of the camera pose
    translation = campose[translation_indexes]
    rotation = campose[rotation_indexes]
    if translation_scale_index is not None:
        translation_scale = campose[translation_scale_index]

    # Quantize the translation and rotation components
    quantized_translation = np.round(translation / translation_quantization)
    quantized_rotation = np.round(rotation / rotation_quantization)
    if translation_scale_index is not None:
        quantized_translation_scale = np.round(translation_scale / translation_scale_quantization)

    # Apply the offset values
    quantized_translation += translation_offset
    quantized_rotation += rotation_offset
    if translation_scale_index is not None:
        quantized_translation_scale += translation_scale_offset

    # Combine the quantized translation and rotation components into a single 6-DOF vector
    # quantized_campose = np.concatenate((quantized_translation, quantized_rotation))
    quantized_campose = np.zeros(6, dtype=np.int16) if translation_scale_index is None else np.zeros(7, dtype=np.int16)
    quantized_campose[translation_indexes] = quantized_translation
    quantized_campose[rotation_indexes] = quantized_rotation
    if translation_scale_index is not None:
        quantized_campose[translation_scale_index] = quantized_translation_scale

    # Convert the quantized camera pose to integers
    quantized_campose = quantized_campose.astype(np.int16)

    return quantized_campose


def transform_matrix_to_six_dof_axis_angle(matrix: np.ndarray, scale: bool = False) -> np.ndarray:
    """
    Convert a 4x4 transformation matrix to a 6-DOF vector (translation + axis-angle)

    Parameters:
        - matrix: 4x4 transformation matrix

    Returns:
        - six_dof_vector: 6-DOF vector (translation + axis-angle)
    """

    # Ensure the matrix is a numpy array
    matrix = np.array(matrix)
    
    # Extract the translation vector (last column of the first three rows)
    translation = matrix[:3, 3]
    
    # Extract the rotation matrix (first three rows and columns)
    rotation_matrix = matrix[:3, :3]

    # Convert to torch tensor with torch.float64
    rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float64)[None, ...]
    
    # Convert rotation matrix to Euler angles (XYZ sequence)
    rotation_so3 = py3d_t.so3_log_map(rotation_matrix)

    # Convert to numpy float32 and degrees, and remove the batch dimension
    rotation_so3 = rotation_so3[0].detach().numpy().astype(np.float32) * 180 / np.pi

    if scale:
        translation_scale = np.linalg.norm(translation, keepdims=True)
        translation_dir = translation / (translation_scale + 1e-6)
        six_dof_vector = np.concatenate((rotation_so3, translation_dir, translation_scale))
    else:
        # Combine translation and Euler angles into a single 6D vector
        six_dof_vector = np.concatenate((rotation_so3, translation))
        
    return six_dof_vector
