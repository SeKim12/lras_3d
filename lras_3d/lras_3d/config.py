from dataclasses import dataclass, field

from lras_3d.utils.model_wrapper import BaseConfig


@dataclass
class LRAS1B_Campose_FlowQuantized_Patch8_RGBConfig(BaseConfig):
    model_class: str = "lras_3d.lras_3d.model.LRAS"
    block_size: int = 22536  # divisible by 256
    vocab_size: int = 111616  # 111377 + 239 = 111616 to make it divisible by 256
    n_layer: int = 36
    n_head: int = 20
    n_embd: int = 1280
    patch_size: int = 2
    dropout: float = 0.0
    bias: bool = False

    # Position token ranges
    frame_0_rgb_pos_range: tuple = field(default=(0, 5120))  # positions of 1st frame patch idx + rgb tokens (1024 * 5 = 5120 values)
    frame_1_rgb_pos_range: tuple = field(default=(5120, 10240))  # positions of 2nd frame patch idx + rgb tokens (1024 * 5 = 5120 values)
    flow_pos_range: tuple = field(default=(10240, 22528))  # positions of flow patch idx + flow tokens (4096 * 3 = 12288 values)
    campose_pos_range: tuple = field(default=(22528, 22536))  # positions of camera pose tokens (8 values)

    # RGB token ranges
    rgb_range: tuple = field(default=(0, 65536))  # range of rgb token values - 16 bits (65,536 values)
    rgb_patch_0_idx_range: tuple = field(default=(65536, 66560))  # range of patch indexes for 1st frame (1024 values)
    rgb_patch_1_idx_range: tuple = field(default=(66560, 67584))  # range of patch indexes for 2nd frame (1024 values)

    # Flow token ranges
    flow_range: tuple = field(default=(67584, 100352))  # range of flow token values (32,768 value)
    flow_patch_idx_range: tuple = field(default=(100352, 101376))  # range of patch indexes for flow (1024 values)

    # Campose token ranges
    campose_range: tuple = field(default=(101376, 111376))  # range of cam pose token values (10,000 values)
    campose_patch_idx_range: tuple = field(default=(111376, 111377))  # range of patch indexes for camera pose (1 value)

@dataclass
class LRAS7B_Patch8_RGB_Flow_Config(BaseConfig):
    model_class: str = "lras_3d.lras_3d.model.LRAS"
    block_size: int = 19456  # divisible by 256
    vocab_size: int = 69120  # 66049 + 255 = 66304 to make it divisible by 256
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096
    patch_size: int = 2
    dropout: float = 0.0
    bias: bool = False

    # position token ranges
    frame_0_rgb_pos_range: tuple = field(default=(0, 5120))  # positions of first frame patch idx + rgb tokens
    frame_1_rgb_pos_range: tuple = field(default=(5120, 10240))  # positions of second frame patch idx + rgb tokens
    flow_pos_range: tuple = field(default=(10240, 19456))  # positions of second frame patch idx + rgb tokens

    # value token ranges
    rgb_range: tuple = field(default=(0, 65536))  # range of rgb token values - 16 bits (65,536 values)
    rgb_patch_0_idx_range: tuple = field(default=(65536, 66560))  # range of patch indexes for first frame (1024 values)
    rgb_patch_1_idx_range: tuple = field(
        default=(66560, 67584))  # range of patch indexes for second frame (1024 values)

    # flow token range
    flow_range: tuple = field(default=(67584, 68096))  # range of flow token values (1 value)
    flow_patch_idx_range: tuple = field(default=(68096, 69120))



