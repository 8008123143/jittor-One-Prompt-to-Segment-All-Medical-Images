"""
Image Encoder for One-Prompt (Jittor version).
Supports ViT-based and U-Net-based encoders.
"""

import math
import numpy as np
import jittor as jt
from jittor import nn
from typing import Optional, Tuple, Type, List, Union
from collections import OrderedDict
from copy import deepcopy

from .common import LayerNorm2d, MLPBlock, Adapter
from .utils import softmax_helper, sigmoid_helper, InitWeights_He, no_op, to_cuda, maybe_to_torch
from .nn import (
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)


# ═══════════════════════════════════════════════════════════════════════
# ViT Components
# ═══════════════════════════════════════════════════════════════════════

class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=kernel_size, stride=stride, padding=padding
        )

    def execute(self, x: jt.Var) -> jt.Var:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


def window_partition(x: jt.Var, window_size: int) -> Tuple[jt.Var, Tuple[int, int]]:
    """Partition into non-overlapping windows with padding if needed."""
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = jt.nn.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.reshape(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: jt.Var, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> jt.Var:
    """Window unpartition into original sequences and remove padding."""
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.reshape(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: jt.Var) -> jt.Var:
    """Get relative positional embeddings from a lookup table."""
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = jt.nn.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    q_coords = jt.arange(q_size).unsqueeze(1).float() * max(k_size / q_size, 1.0)
    k_coords = jt.arange(k_size).unsqueeze(0).float() * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.int().long()]


def add_decomposed_rel_pos(
    attn: jt.Var,
    q: jt.Var,
    rel_pos_h: jt.Var,
    rel_pos_w: jt.Var,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> jt.Var:
    """Add decomposed relative position embeddings (from MViTv2)."""
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)

    # rel_h: einsum("bhwc,hkc->bhwk", r_q, Rh)
    # r_q: (B, q_h, q_w, C), Rh: (q_h, K, C) → result: (B, q_h, q_w, K)
    Rh_expanded = Rh.unsqueeze(0).unsqueeze(2)  # (1, q_h, 1, K, C)
    rel_h = (r_q.unsqueeze(3) * Rh_expanded).sum(dim=-1)  # (B, q_h, q_w, K)

    # rel_w: einsum("bhwc,wkc->bhwk", r_q, Rw)
    Rw_expanded = Rw.unsqueeze(0).unsqueeze(1)  # (1, 1, q_w, K, C)
    rel_w = (r_q.unsqueeze(3) * Rw_expanded).sum(dim=-1)  # (B, q_h, q_w, K)

    attn = (
        attn.reshape(B, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, None]
        + rel_w[:, :, :, None, :]
    ).reshape(B, q_h * q_w, k_h * k_w)

    return attn


class AttentionViT(nn.Module):
    """Multi-head Attention for ViT with optional relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None
            self.rel_pos_h = jt.zeros(2 * input_size[0] - 1, head_dim)
            self.rel_pos_w = jt.zeros(2 * input_size[1] - 1, head_dim)

    def execute(self, x: jt.Var) -> jt.Var:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = jt.nn.softmax(attn, dim=-1)
        x = (attn @ v).reshape(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)
        return x


class Block(nn.Module):
    """Transformer block with window attention support."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionViT(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        self.window_size = window_size

    def execute(self, x: jt.Var) -> jt.Var:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


# ═══════════════════════════════════════════════════════════════════════
# ViT Encoder
# ═══════════════════════════════════════════════════════════════════════

class ImageEncoderViT(nn.Module):
    """Original SAM ViT image encoder."""

    def __init__(
        self,
        args=None,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.args = args

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[jt.Var] = None
        if use_abs_pos:
            self.pos_embed = jt.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

    def execute(self, x: jt.Var) -> jt.Var:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return x


class OnePromptEncoderViT(nn.Module):
    """ViT encoder for One-Prompt, returns multi-scale skip features."""

    def __init__(
        self,
        args=None,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.args = args

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[jt.Var] = None
        if use_abs_pos:
            self.pos_embed = jt.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

    def execute(self, x: jt.Var):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        skips = []
        for blk in self.blocks:
            x = blk(x)
            skips.append(x)

        return x, skips


# ═══════════════════════════════════════════════════════════════════════
# U-Net Encoder
# ═══════════════════════════════════════════════════════════════════════

class ConvDropoutNormNonlin(nn.Module):
    """Conv → Dropout → Norm → Nonlin block."""

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super().__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.conv = conv_op(input_channels, output_channels, **conv_kwargs)
        self.dropout = dropout_op(**dropout_op_kwargs) if (dropout_op is not None and dropout_op_kwargs.get('p', 0) > 0) else None
        self.instnorm = norm_op(output_channels, **norm_op_kwargs)
        self.lrelu = nonlin(**nonlin_kwargs)

    def execute(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


class StackedConvLayers(nn.Module):
    """Stack of ConvDropoutNormNonlin layers."""

    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None,
                 first_stride=None, basic_block=ConvDropoutNormNonlin):
        super().__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.conv_kwargs = conv_kwargs
        conv_kwargs_first = deepcopy(conv_kwargs)
        if first_stride is not None:
            conv_kwargs_first['stride'] = first_stride

        layers = [basic_block(input_feature_channels, output_feature_channels, conv_op,
                              conv_kwargs_first, norm_op, norm_op_kwargs,
                              dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)]
        for _ in range(num_convs - 1):
            layers.append(basic_block(output_feature_channels, output_feature_channels, conv_op,
                                      conv_kwargs, norm_op, norm_op_kwargs,
                                      dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs))
        self.blocks = nn.Sequential()
        for layer in layers:
            self.blocks.append(layer)

    def execute(self, x):
        return self.blocks(x)


class OnePromptEncoderUnet(nn.Module):
    """U-Net based encoder for One-Prompt."""

    def __init__(self, input_channels=3, base_num_features=128, final_num_features=256,
                 fea_size=64, num_pool=4, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None,
                 weightInitializer=InitWeights_He(1e-2),
                 pool_op_kernel_sizes=None, conv_kernel_sizes=None,
                 max_num_features=None):
        super().__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        if pool_op_kernel_sizes is None:
            pool_op_kernel_sizes = [(2, 2)] * num_pool
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [(3, 3)] * (num_pool + 1)

        self.conv_pad_sizes = []
        for krnl in conv_kernel_sizes:
            self.conv_pad_sizes.append(tuple(1 if i == 3 else 0 for i in krnl))

        if max_num_features is None:
            self.max_num_features = 480
        else:
            self.max_num_features = max_num_features

        self.conv_blocks_context = []
        self.td = []
        self.al = []

        output_features = base_num_features
        input_features = input_channels

        for d in range(num_pool):
            first_stride = None

            self.conv_kwargs['kernel_size'] = conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            self.conv_blocks_context.append(
                StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                  conv_op, self.conv_kwargs, norm_op,
                                  norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                  nonlin, nonlin_kwargs, first_stride)
            )
            self.al.append(nn.Linear(output_features, final_num_features))

            self.td.append(nn.MaxPool2d(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))
            output_features = min(output_features, self.max_num_features)

        # Bottleneck
        self.conv_kwargs['kernel_size'] = conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1,
                              conv_op, self.conv_kwargs, norm_op, norm_op_kwargs,
                              dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs),
            StackedConvLayers(output_features, final_num_features, 1,
                              conv_op, self.conv_kwargs, norm_op, norm_op_kwargs,
                              dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
        ))

        self.up = []
        for u in range(num_pool):
            self.up.append(nn.Resize(size=(fea_size, fea_size), mode='bilinear'))

        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.up = nn.ModuleList(self.up)
        self.al = nn.ModuleList(self.al)

        if weightInitializer is not None:
            self.apply(weightInitializer)

    def execute(self, raw):
        skips_raw = []
        for d in range(len(self.conv_blocks_context) - 1):
            raw = self.conv_blocks_context[d](raw)
            raw_arch = self.up[d](raw)
            raw_arch = raw_arch.permute(0, 2, 3, 1)
            raw_arch = self.al[d](raw_arch)
            skips_raw.append(raw_arch)
            raw = self.td[d](raw)

        raw = self.conv_blocks_context[-1](raw)
        raw = raw.permute(0, 2, 3, 1)

        return raw, skips_raw
