from dataclasses import dataclass, field
from typing import List, Tuple

from lras_3d.utils.model_wrapper import BaseConfig


@dataclass
class LPQConfig(BaseConfig):
    model_class: str = "lras_3d.local_patch_quantizer.model.LPQ"
    patch_size: int = 4
    dropout: float = 0.1
    vocab_size: int = 65536
    num_codebooks: int = 1
    num_in_channels: int = 3
    num_out_channels: int = 3
    entropy_loss_weight = 0.001
    commitment_loss_weight = 0.0025
    diversity_gamma = 1.0
    # encoder made up of either "ResBlock" or "Downsample" blocks
    # each block arguments are of type:
    # (block_type, in_channels, out_channels, hidden_channels, kernel_size, stride, padding) if ResBlock
    # (block_type, in_channels, out_channels) if Downsample
    encoder_blocks: List[Tuple] = field(default_factory=lambda: [
        ("ResBlock", 512, 512, 1024, 1, 1, 0),
    ])
    # decoder made up of either "ResBlock" or "Upsample" blocks
    # each block arguments are of type:
    # (block_type, in_channels, out_channels, hidden_channels, kernel_size, stride, padding) if ResBlock
    # (block_type, in_channels, out_channels) if Upsample
    decoder_blocks: List[Tuple] = field(default_factory=lambda: [

        ("ResBlock", 512, 512, 1024, 3, 1, 1),
        ("ResBlock", 512, 512, 1024, 3, 1, 1),
        ("ResBlock", 512, 512, 1024, 3, 1, 1),
        ("Upsample", 512, 256),
        ("ResBlock", 256, 256, 512, 3, 1, 1),
        ("ResBlock", 256, 256, 512, 3, 1, 1),
        ("ResBlock", 256, 256, 512, 3, 1, 1),
        ("Upsample", 256, 128),
        ("ResBlock", 128, 128, 256, 3, 1, 1),
        ("ResBlock", 128, 128, 256, 3, 1, 1),
        ("ResBlock", 128, 128, 256, 3, 1, 1),
    ])


@dataclass
class LPQFlowConfig(BaseConfig):
    model_class: str = "lras_3d.local_patch_quantizer.model.LPQ"
    patch_size: int = 4
    dropout: float = 0.0
    vocab_size: int = 32768
    num_codebooks: int = 1
    num_in_channels: int = 2
    num_out_channels: int = 2
    entropy_loss_weight = 0.001
    commitment_loss_weight = 0.0025
    diversity_gamma = 1.0
    # encoder made up of either "ResBlock" or "Downsample" blocks
    # each block arguments are of type:
    # (block_type, in_channels, out_channels, hidden_channels, kernel_size, stride, padding) if ResBlock
    # (block_type, in_channels, out_channels) if Downsample
    encoder_blocks: List[Tuple] = field(default_factory=lambda: [
        ("ResBlock", 512, 512, 1024, 1, 1, 0),
    ])
    # decoder made up of either "ResBlock" or "Upsample" blocks
    # each block arguments are of type:
    # (block_type, in_channels, out_channels, hidden_channels, kernel_size, stride, padding) if ResBlock
    # (block_type, in_channels, out_channels) if Upsample
    decoder_blocks: List[Tuple] = field(default_factory=lambda: [

        ("ResBlock", 512, 512, 1024, 3, 1, 1),
        ("ResBlock", 512, 512, 1024, 3, 1, 1),
        ("ResBlock", 512, 512, 1024, 3, 1, 1),
        ("Upsample", 512, 256),
        ("ResBlock", 256, 256, 512, 3, 1, 1),
        ("ResBlock", 256, 256, 512, 3, 1, 1),
        ("ResBlock", 256, 256, 512, 3, 1, 1),
        ("Upsample", 256, 128),
        ("ResBlock", 128, 128, 256, 3, 1, 1),
        ("ResBlock", 128, 128, 256, 3, 1, 1),
        ("ResBlock", 128, 128, 256, 3, 1, 1),
    ])

