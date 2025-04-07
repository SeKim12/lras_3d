"""
Local Patch Quantizer (LPQ) model
An autoencoder which uses a local encoder and a (semi) global decoder.
"""

import inspect
from typing import Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
from vector_quantize_pytorch import LFQ

from lras_3d.utils.modeling import Upsample, Downsample, PatchResidualConvBlock
from lras_3d.utils.model_wrapper import WrappedModel


class LPQ(WrappedModel):

    def __init__(self, config):
        super().__init__(config)
        print("using config:", config)
        self.config = config

        self.encoder = nn.Sequential(
            nn.Conv2d(config.num_in_channels, config.encoder_blocks[0][1], kernel_size=config.patch_size, stride=config.patch_size),
            nn.SiLU(),
            *[
                PatchResidualConvBlock(*block_params[1:]) if block_params[0] == "ResBlock" else Downsample(*block_params[1:]) 
                for block_params in config.encoder_blocks
            ]
        )

        self.quantizer = LFQ(
            codebook_size = config.vocab_size,                      # codebook size, must be a power of 2
            dim = config.encoder_blocks[-1][2],                            # this is the input feature dimension, defaults to log2(codebook_size) if not defined
            num_codebooks = config.num_codebooks,                   # number of codebooks to use, defaults to 1
            entropy_loss_weight = config.entropy_loss_weight,       # how much weight to place on entropy loss
            commitment_loss_weight = config.commitment_loss_weight, # how much weight to place on the commitment loss
            diversity_gamma = config.diversity_gamma,               # within entropy loss, how much weight to give to diversity of codes, taken from https://arxiv.org/abs/1911.05894
            force_quantization_f32=True,                          # force quantization to float32
            spherical=True,                                         # use spherical quantization
        )

        self.decoder = nn.Sequential(
            *[
                PatchResidualConvBlock(*block_params[1:]) if block_params[0] == "ResBlock" else Upsample(*block_params[1:]) 
                for block_params in config.decoder_blocks
            ],
            nn.Conv2d(config.decoder_blocks[-1][2], config.num_out_channels, kernel_size=3, stride=1, padding=1)
        )


    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model

        Parameters:
            c (torch.Tensor): The input tensor. Size b, c, h, w
        
        Returns:
            torch.Tensor: The predicted output tensor. Size b, c, h, w
            torch.Tensor: The loss of the model. Size 1
        """
        # grab device to perform operations on
        device = x.device
        # grab dimensions
        b, c, h, w = x.size()

        # encode the input
        z = self.encoder(x)

        quantized, indices, entropy_aux_loss = self.quantizer(z)
        # quantized = z
        # entropy_aux_loss = torch.tensor(0.0).to(device)

        # decode the quantized input
        y = self.decoder(quantized)

        # calculate the loss
        loss = F.mse_loss(y, x)

        return y, loss, entropy_aux_loss
    
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize the input tensor

        Parameters:
            x (torch.Tensor): The input tensor. Size b, c, h, w

        Returns:
            torch.Tensor: The indices tensor. Size b, h, w
        """
        # encode the input
        z = self.encoder(x)

        # quantize the input
        quantized, indices, _ = self.quantizer(z)

        return indices
    
    @torch.no_grad()
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Parameters: 
            indices: torch.Tensor of shape (b, t, n_freq_bins)
        Returns: 
            emb: torch.Tensor of shape (b, t, n_embd)
        """
        b, h, w = indices.size()  # batch, seq_len
        # Transform the indices back to quantized embeddings
        emb = self.quantizer.indices_to_codes(indices)
        # decode the quantized embeddings to cochleagram
        pred_coch = self.decoder(emb)

        return pred_coch
