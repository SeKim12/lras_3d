"""
Modeling utils and layers for the KPT models
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np
import math
from einops import rearrange
import os
from collections import namedtuple
from torchvision import models


class Rotary4D(nn.Module):
    def __init__(self, dim, num_dims=4, base=10000):
        super().__init__()
        assert dim % (num_dims * 2) == 0, "Embedding dim must be divisible by num_dims*2"

        self.dim = dim
        self.num_dims = num_dims
        self.dim_per_axis = dim // num_dims

        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim_per_axis, 2).float() / self.dim_per_axis))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, pos):
        """
        x: [batch, seq_len, dim]
        pos: [batch, seq_len, 4] integer positions along each dimension
        """

        batch_size, seq_len, _ = x.shape
        assert pos.shape == (batch_size, seq_len, self.num_dims), "pos shape mismatch"

        freqs = []
        for axis in range(self.num_dims):
            pos_axis = pos[:, :, axis].float()  # [batch, seq_len]
            freqs_axis = torch.einsum('bs,d->bsd', pos_axis, self.inv_freq)  # [batch, seq_len, dim_per_axis//2]
            freqs.append(freqs_axis)

        cos_emb = torch.cat([f.cos() for f in freqs], dim=-1)  # [batch, seq_len, dim//2]
        sin_emb = torch.cat([f.sin() for f in freqs], dim=-1)

        x1, x2 = x[..., :self.dim // 2], x[..., self.dim // 2:]
        x_rotated = torch.cat([
            x1 * cos_emb - x2 * sin_emb,
            x1 * sin_emb + x2 * cos_emb
        ], dim=-1)

        return x_rotated



# class LayerNorm(nn.Module):
#     """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

#     def __init__(self, ndim, bias):
#         super().__init__()
#         self.weight = nn.Parameter(torch.zeros(ndim))
#         self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

#     def forward(self, input):
#         return F.layer_norm(input, self.weight.shape, self.weight + 1.0, self.bias, 1e-5)


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


# class LayerNorm(torch.nn.Module):
#     def __init__(self, dim: int, bias: bool = False, eps: float = 1e-6):
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(dim))

#     def _norm(self, x):
#         return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

#     def forward(self, x):
#         output = self._norm(x.float()).type_as(x)
#         return output * self.weight


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.use_kvc = True

        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
        #                             .view(1, 1, config.block_size, config.block_size))
        # self.bias = torch.tril(torch.ones(config.block_size, config.block_size)
        #                                 ).view(1, 1, config.block_size, config.block_size)

    def forward(self, x, return_kv=False, spmd_mesh=None, mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        if spmd_mesh is not None:
            import torch_xla.distributed.spmd.xla_sharding as xs
            xs.mark_sharding(q, spmd_mesh,  (('dcn', 'data'), None, 'model'))
            xs.mark_sharding(k, spmd_mesh,  (('dcn', 'data'), None, 'model'))
            xs.mark_sharding(v, spmd_mesh,  (('dcn', 'data'), None, 'model'))

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # If mask is not provided, create a fully causal mask
        if mask is None:
            mask = torch.triu(torch.ones(T, T), 1).to(dtype=torch.bool).to(x.device)
            mask = mask.view(1, 1, T, T)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash and spmd_mesh is None and not return_kv:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, # attn_mask=mask.logical_not(), 
                dropout_p=self.dropout if self.training else 0, is_causal=True) #is_causal=mask==None)
        elif (spmd_mesh is not None or self.use_kvc) and not return_kv:
            # Integrated with PyTorch/XLA Pallas Flash Attention:
            from torch_xla.experimental.custom_kernel import flash_attention
            q_norm = q / math.sqrt(k.size(-1))
            y = flash_attention(q_norm, k, v, causal=True, partition_spec=('data', 'model', None, None))
        else:
            # manual implementation of attention
            att = torch.einsum('bnsh,bnkh->bnsk', q, k) * (1.0 / math.sqrt(k.size(-1)))
            
            if spmd_mesh is not None:
                xs.mark_sharding(att, spmd_mesh, (('dcn', 'data'), 'model', None, None))
                xs.mark_sharding(mask, spmd_mesh, (('dcn', 'data'), 'model', None, None))

            att = att.masked_fill(mask, float('-inf'))
            # subtracting the max value for numerical stability
            # att = att - torch.max(att, dim=-1, keepdim=True).values
            
            # upcast to float32 for numerical stability, as per llama implementation
            att = F.softmax(att, dim=-1, dtype=torch.float32).to(q.dtype)
            att = self.attn_dropout(att)

            y = torch.einsum('bnsk,bnkh->bnsh', att, v)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        if spmd_mesh is not None:
            xs.mark_sharding(y, spmd_mesh,  (('dcn', 'data'), None, 'model'))

        # return key and value caches if requested
        if return_kv:
            return y, k, v

        return y

    def kv_cache_forward(self, x, k_cache=None, v_cache=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # append cached keys and values with new keys and values
        if k_cache is not None:
            k = torch.cat((k_cache, k), dim=2)
        if v_cache is not None:
            v = torch.cat((v_cache, v), dim=2)

        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        return y, k, v
    
    # def to(self, *args, **kwargs):
    #     """
    #     Overriding the to method to ensure that the bias tensor is moved to the correct device
    #     """
    #     # Call the parent class's .to() method to move all parameters to the desired device
    #     module = super(CausalSelfAttention, self).to(*args, **kwargs)
    #     # Move the bias tensor as well
    #     module.bias = self.bias.to(*args, **kwargs)
    #     return module

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, spmd_mesh=None):
        
        x = self.c_fc(x)
        x = self.gelu(x)

        if spmd_mesh is not None:
            import torch_xla.distributed.spmd.xla_sharding as xs
            xs.mark_sharding(x, spmd_mesh,  (('dcn', 'data'), None, 'model'))

        x = self.c_proj(x)
        x = self.dropout(x)

        if spmd_mesh is not None:
            xs.mark_sharding(x, spmd_mesh,  (('dcn', 'data'), None, 'model'))

        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, k_cache=None, v_cache=None, return_kv=False, spmd_mesh=None, mask=None):
        # If we are given a key and value cache, we will use the pre-computed values to minimize
        # the computation cost
        if k_cache is not None and v_cache is not None:
            # Pass the key and value cache to the attention layer, obtain new key and value caches
            x_attn, k, v = self.attn.kv_cache_forward(self.ln_1(x), k_cache, v_cache)
            x = x + x_attn
            x = x + self.mlp(self.ln_2(x))
            return x, k, v
        # We might want to encode the caches of a whole block of keys and values at once using the
        # fast flash attention impelmentation while still returning the key and value caches
        elif return_kv:
            # Pass the key and value cache to the attention layer, obtain new key and value caches
            x_attn, k, v = self.attn(self.ln_1(x), return_kv=True, mask=mask)
            x = x + x_attn
            x = x + self.mlp(self.ln_2(x))
            return x, k, v
        # Else we proceed with the regular forward pass
        x = x + self.attn(self.ln_1(x), spmd_mesh=spmd_mesh, mask=mask)
        x = x + self.mlp(self.ln_2(x), spmd_mesh=spmd_mesh)
        return x


class PatchResidualConvBlock(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim, kernel_size, stride, padding, dorpout=0.1) -> None:
        super().__init__()
        self.nonlinearity = nn.SiLU()
        self.ln1 = LayerNorm(in_dim, bias=True)
        self.dropout = nn.Dropout(dorpout)
        self.conv1 = nn.Conv2d(in_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(hidden_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        
    def forward(self, x):
        b, c, h, w = x.shape
        z = self.ln1(x.permute(0, 2, 3, 1).reshape(b * h * w, c)).reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        z = self.nonlinearity(self.conv1(z))
        z = self.dropout(z)
        z = self.nonlinearity(self.conv2(z))
        return z + x


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = torch.nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=2,
                                    padding=0)

    def forward(self, x):
        pad = (0,1,0,1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


# utility function for unpatchifying the reconstruced rgb images
def unpatchify(labels):
    # Define the input tensor
    B = labels.shape[0]  # batch size
    N_patches = int(np.sqrt(labels.shape[1]))  # number of patches along each dimension
    patch_size = int(np.sqrt(labels.shape[2] / 3))  # patch size along each dimension
    channels = 3  # number of channels

    rec_imgs = rearrange(labels, 'b n (p c) -> b n p c', c=3)
    # Notice: To visualize the reconstruction video, we add the predict and the original mean and var of each patch.
    rec_imgs = rearrange(rec_imgs,
                         'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)',
                         p0=1,
                         p1=patch_size,
                         p2=patch_size,
                         h=N_patches,
                         w=N_patches)

    return rec_imgs


class CheckpointWrapper(nn.Module):
    def __init__(self, module, use_checkpointing=True):
        """
        Wrapper around a PyTorch module to perform gradient checkpointing.

        Args:
            module (nn.Module): The module to be wrapped.
            use_checkpointing (bool): Whether to use gradient checkpointing or not.
        """
        super(CheckpointWrapper, self).__init__()
        self.module = module
        self.use_checkpointing = use_checkpointing

    def forward(self, *inputs, **kwargs):
        if self.use_checkpointing:
            # Define a lambda to accept kwargs and pass them into the module
            def function_with_kwargs(*inputs):
                return self.module(*inputs, **kwargs)

            # Use checkpoint with the function that handles both inputs and kwargs
            return checkpoint(function_with_kwargs, *inputs)
        else:
            # Regular forward pass without checkpointing
            return self.module(*inputs, **kwargs)


class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        try:
            data = torch.load("vgg.pth", map_location=torch.device("cpu"))
        except:
            print("Failed to load vgg.pth, downloading...")
            os.system(
                "wget --trust-server-names 'https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1'"
            )
            data = torch.load("vgg.pth", map_location=torch.device("cpu"))

        self.load_state_dict(
            data,
            strict=False,
        )

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(
                outs1[kk]
            )
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [
            spatial_average(lins[kk].model(diffs[kk]), keepdim=True)
            for kk in range(len(self.chns))
        ]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer(
            "shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None]
        )
        self.register_buffer(
            "scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None]
        )

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = (
            [
                nn.Dropout(),
            ]
            if (use_dropout)
            else []
        )
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),
        ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
        )
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        self.scaling_layer = ScalingLayer()

        _vgg = models.vgg16(pretrained=True)

        self.slice1 = nn.Sequential(_vgg.features[:4])
        self.slice2 = nn.Sequential(_vgg.features[4:9])
        self.slice3 = nn.Sequential(_vgg.features[9:16])
        self.slice4 = nn.Sequential(_vgg.features[16:23])
        self.slice5 = nn.Sequential(_vgg.features[23:30])

        self.binary_classifier1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=4, stride=4, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=4, stride=4, padding=0, bias=True),
        )
        nn.init.zeros_(self.binary_classifier1[-1].weight)

        self.binary_classifier2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=4, stride=4, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=2, stride=2, padding=0, bias=True),
        )
        nn.init.zeros_(self.binary_classifier2[-1].weight)

        self.binary_classifier3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=2, stride=2, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=2, stride=2, padding=0, bias=True),
        )
        nn.init.zeros_(self.binary_classifier3[-1].weight)

        self.binary_classifier4 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=2, stride=2, padding=0, bias=True),
        )
        nn.init.zeros_(self.binary_classifier4[-1].weight)

        self.binary_classifier5 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=True),
        )
        nn.init.zeros_(self.binary_classifier5[-1].weight)

    def forward(self, x):
        x = self.scaling_layer(x)
        features1 = self.slice1(x)
        features2 = self.slice2(features1)
        features3 = self.slice3(features2)
        features4 = self.slice4(features3)
        features5 = self.slice5(features4)

        # torch.Size([1, 64, 256, 256]) torch.Size([1, 128, 128, 128]) torch.Size([1, 256, 64, 64]) torch.Size([1, 512, 32, 32]) torch.Size([1, 512, 16, 16])

        bc1 = self.binary_classifier1(features1).flatten(1)
        bc2 = self.binary_classifier2(features2).flatten(1)
        bc3 = self.binary_classifier3(features3).flatten(1)
        bc4 = self.binary_classifier4(features4).flatten(1)
        bc5 = self.binary_classifier5(features5).flatten(1)

        return bc1 + bc2 + bc3 + bc4 + bc5


dec_lo, dec_hi = (
    torch.Tensor([-0.1768, 0.3536, 1.0607, 0.3536, -0.1768, 0.0000]),
    torch.Tensor([0.0000, -0.0000, 0.3536, -0.7071, 0.3536, -0.0000]),
)

filters = torch.stack(
    [
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1),
    ],
    dim=0,
)

filters_expanded = filters.unsqueeze(1)


def prepare_filter(device):
    global filters_expanded
    filters_expanded = filters_expanded.to(device)


def wavelet_transform_multi_channel(x, levels=4):
    B, C, H, W = x.shape
    padded = torch.nn.functional.pad(x, (2, 2, 2, 2))

    # use predefined filters
    global filters_expanded

    ress = []
    for ch in range(C):
        res = torch.nn.functional.conv2d(
            padded[:, ch : ch + 1], filters_expanded, stride=2
        )
        ress.append(res)

    res = torch.cat(ress, dim=1)
    H_out, W_out = res.shape[2], res.shape[3]
    res = res.view(B, C, 4, H_out, W_out)
    res = res.view(B, 4 * C, H_out, W_out)
    return res



class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.SiLU()  # or nn.ReLU, etc.

    def forward(self, x):
        residual = x
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        return self.activation(x + residual)


class UNetEncoder(nn.Module):
    """
    Goes from (B, in_channels, 512, 512) -> (B, out_channels, 64, 64)
    Collects skip features at intermediate resolutions.
    """
    def __init__(self, in_channels, base_channels, out_channels, num_res_blocks=2):
        super().__init__()
        self.initial_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.num_res_blocks = num_res_blocks
        
        # We'll downsample 3 times: base -> 2x -> 4x
        self.down_channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
        ]
        self.down_stages = nn.ModuleList()
        in_ch = base_channels
        for out_ch in self.down_channels:
            stage_blocks = []
            # Some ResBlocks at the current resolution
            for _ in range(num_res_blocks):
                stage_blocks.append(ResBlock(in_ch))
            # Downsample at the end of the stage
            stage_blocks.append(Downsample(in_ch, out_ch))
            self.down_stages.append(nn.Sequential(*stage_blocks))
            in_ch = out_ch

        # final conv to get exactly out_channels at the 64x64 resolution
        self.final_conv = nn.Conv2d(in_ch, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Returns:
          bottleneck: (B, out_channels, 64, 64)
          skips: list of skip-connection features
        """
        x = self.initial_conv(x)
        skips = []

        out = x
        for stage in self.down_stages:
            # separate out the resblocks vs. downsample
            *res_blocks, down_block = stage
            for rb in res_blocks:
                out = rb(out)
            # store skip
            skips.append(out)
            # downsample
            out = down_block(out)

        out = self.final_conv(out)  # => Nx64x64
        return out, skips


class UNetDecoder(nn.Module):
    """
    Goes from (B, in_channels, 64, 64) -> (B, out_channels, 512, 512)
    Each upsampling stage merges skip from the *corresponding* encoder stage.
    """
    def __init__(self, in_channels, base_channels, out_channels, num_res_blocks=2):
        super().__init__()
        self.num_res_blocks = num_res_blocks

        # This is the reverse of the above 3 downsamplings
        self.up_channels = [
            base_channels * 4,
            base_channels * 2,
            base_channels,
        ]
        
        self.up_stages = nn.ModuleList()
        self.post_cat_stages = nn.ModuleList()

        prev_ch = in_channels
        for out_ch in self.up_channels:
            # Upsample from prev_ch -> out_ch
            up = Upsample(prev_ch, out_ch)
            self.up_stages.append(up)

            # After upsample, we cat the skip => channels = out_ch + skip_out_ch
            # Typically skip_out_ch == out_ch (if symmetrical).
            # Then ResBlocks -> 1x1 conv to reduce back to out_ch.
            blocks = []
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(2 * out_ch))
            blocks.append(nn.Conv2d(2 * out_ch, out_ch, kernel_size=1))
            self.post_cat_stages.append(nn.Sequential(*blocks))

            prev_ch = out_ch

        self.final_conv = nn.Conv2d(self.up_channels[-1], out_channels, kernel_size=3, padding=1)

    def forward(self, bottleneck, skips):
        # reverse skip order so the last skip is used first
        skips = skips[::-1]

        out = bottleneck
        for i, (up, post_cat) in enumerate(zip(self.up_stages, self.post_cat_stages)):
            out = up(out)
            skip_feat = skips[i]
            out = torch.cat([out, skip_feat], dim=1)  # cat skip
            out = post_cat(out)

        out = self.final_conv(out)
        return out
