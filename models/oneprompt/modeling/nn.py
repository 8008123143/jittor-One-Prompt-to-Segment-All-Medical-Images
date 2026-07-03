"""
Neural network utilities (Jittor version).
"""

import math
import jittor as jt
from jittor import nn


class SiLU(nn.Module):
    """SiLU activation (same as Swish)."""
    def execute(self, x):
        return x * jt.nn.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def execute(self, x):
        return super().execute(x.float())


def conv_nd(dims, *args, **kwargs):
    """Create 1D, 2D, or 3D convolution."""
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def layer_norm(shape, *args, **kwargs):
    return nn.LayerNorm(shape, *args, **kwargs)


def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """Create 1D, 2D, or 3D average pooling."""
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def zero_module(module):
    """Zero out all parameters in a module."""
    for p in module.parameters():
        p.assign(jt.zeros_like(p))
    return module


def scale_module(module, scale):
    """Scale all parameters in a module."""
    for p in module.parameters():
        p.assign(p * scale)
    return module


def mean_flat(tensor):
    """Take mean over all non-batch dimensions."""
    dims = list(range(1, len(tensor.shape)))
    if dims:
        return tensor.mean(dim=dims)
    return tensor


def normalization(channels):
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = jt.exp(
        -math.log(max_period) * jt.arange(start=0, end=half, dtype=jt.float32) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = jt.concat([jt.cos(args), jt.sin(args)], dim=-1)
    if dim % 2:
        embedding = jt.concat([embedding, jt.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
