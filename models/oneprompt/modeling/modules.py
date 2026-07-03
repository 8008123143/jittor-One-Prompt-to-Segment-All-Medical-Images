"""
Core attention and transformer modules, migrated from PyTorch to Jittor.
Contains: Attention, GaussianConv2d, PromptMixer, PromptParser, OnePromptFormer,
TwoWayTransformer, TwoWayAttentionBlock, CrossAttentionBlock.
"""

import math
import jittor as jt
from jittor import nn
from typing import Tuple, Type
from .common import MLPBlock


# ── Helper ────────────────────────────────────────────────────────────
def gaussian_kernel_2d(size, mean, std):
    """Generate a 2D Gaussian kernel (replaces torch.distributions.Normal)."""
    d = jt.array([-(x - mean) ** 2 / (2 * std ** 2) for x in range(size)])
    d = d.unsqueeze(1) + d.unsqueeze(0)
    grid = jt.exp(d)
    grid = grid / grid.sum()
    return grid


class GaussianConv2d(nn.Module):
    """2D convolution with a learnable Gaussian kernel."""
    def __init__(self, in_channels=1, out_channels=1, kernel_size=3,
                 stride=1, padding=1, mean=0.0, std=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.mean = jt.array(float(mean))
        self.std = jt.array(float(std))
        init_kernel = gaussian_kernel_2d(kernel_size, float(mean), float(std))
        self.weights = init_kernel.reshape(1, 1, kernel_size, kernel_size)
        self.weights = self.weights.repeat(out_channels, in_channels, 1, 1)
        self.bias = jt.zeros(out_channels)

    def execute(self, x):
        return jt.nn.conv2d(x, self.weights, stride=self.stride, padding=self.padding)


# ── PromptMLP / PromptMixer ───────────────────────────────────────────
class PromptMLP(nn.Module):
    """MLP that reduces to 1 channel (used inside PromptMixer)."""
    def __init__(self, dim=3, expansion_factor=4, dropout=0.):
        super().__init__()
        inner_dim = int(dim * expansion_factor)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, 1),
            nn.Dropout(dropout),
        )

    def execute(self, x):
        return self.net(x)


class PromptMixer(nn.Module):
    """Mixes query/key/value prompts with batch broadcasting."""
    def __init__(self, dim=3, depth=1, expansion_factor=4, dropout=0.):
        super().__init__()
        self.depth = depth
        self.dim = dim
        self.mlp = nn.Sequential()
        for _ in range(depth):
            self.mlp.append(PromptMLP(dim, expansion_factor, dropout))

    def execute(self, q, k, v):
        # q, k, v each: (b, n, d) — broadcast to match batch dims
        max_b = max(q.shape[0], k.shape[0], v.shape[0])
        if q.shape[0] != max_b:
            q = q.repeat(max_b, 1, 1)
        if k.shape[0] != max_b:
            k = k.repeat(max_b, 1, 1)
        if v.shape[0] != max_b:
            v = v.repeat(max_b, 1, 1)
        qk = jt.stack([q, k, v], dim=0)
        qk = qk.transpose(1, 2, 3, 0)
        res = self.mlp(qk)
        return res.squeeze(-1)


# ── Attention ──────────────────────────────────────────────────────────
class Attention(nn.Module):
    """Multi-head attention with optional downsampling rate."""

    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: jt.Var, num_heads: int) -> jt.Var:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: jt.Var) -> jt.Var:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def execute(self, q: jt.Var, k: jt.Var, v: jt.Var) -> jt.Var:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)
        _, _, _, c_per_head = q.shape
        attn = q @ k.transpose(2, 3)
        attn = attn / math.sqrt(c_per_head)
        attn = jt.nn.softmax(attn, dim=-1)
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


# ── PromptParser ───────────────────────────────────────────────────────
class PromptParser(nn.Module):
    """Cross-attention parser between template and image embeddings."""

    def __init__(self, embedding_dim: int, token_num: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pt_mix = PromptMixer()
        self.gauss = GaussianConv2d(in_channels=token_num)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def execute(self, image_embedding: jt.Var, tmp_embedding: jt.Var,
                prompt_embedding1: jt.Var, prompt_embedding2: jt.Var):
        # Broadcast batch dims
        max_b = max(image_embedding.shape[0], tmp_embedding.shape[0],
                    prompt_embedding1.shape[0], prompt_embedding2.shape[0])
        if image_embedding.shape[0] != max_b:
            image_embedding = image_embedding.repeat(max_b, 1, 1)
        if tmp_embedding.shape[0] != max_b:
            tmp_embedding = tmp_embedding.repeat(max_b, 1, 1)
        if prompt_embedding1.shape[0] != max_b:
            prompt_embedding1 = prompt_embedding1.repeat(max_b, 1, 1)
        if prompt_embedding2.shape[0] != max_b:
            prompt_embedding2 = prompt_embedding2.repeat(max_b, 1, 1)

        pt_pe = prompt_embedding1 + prompt_embedding2
        etpp = self.pt_mix(tmp_embedding, prompt_embedding1, prompt_embedding2)
        att_m = jt.matmul(image_embedding.unsqueeze(-1), etpp.unsqueeze(-2))
        att_m = self.gauss(att_m)
        etq = jt.matmul(image_embedding.unsqueeze(-1),
                        (tmp_embedding + pt_pe).unsqueeze(-2))
        eg = jt.maximum(att_m * etq, etq)
        res = (eg * (tmp_embedding + pt_pe).unsqueeze(2)).sum(dim=-1)
        return image_embedding, res


# ── OnePromptFormer ────────────────────────────────────────────────────
class OnePromptFormer(nn.Module):
    """Fuses image embeddings, template embeddings, and prompt embeddings."""

    def __init__(self, embedding_dim: int, prompt_embed_dim: int,
                 token_num: int, num_heads: int, mlp_dim: int,
                 activation: Type[nn.Module] = nn.ReLU):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.nn_proj = nn.Linear(embedding_dim, prompt_embed_dim)
        self.attns1 = Attention(prompt_embed_dim, num_heads)
        self.attns2 = Attention(prompt_embed_dim, num_heads)
        self.mlps1 = MLPBlock(prompt_embed_dim, mlp_dim, activation)
        self.norms1 = nn.LayerNorm(prompt_embed_dim)
        self.norms2 = nn.LayerNorm(prompt_embed_dim)
        self.parser = PromptParser(embedding_dim=prompt_embed_dim,
                                   token_num=token_num)
        self.attnt1 = Attention(prompt_embed_dim, num_heads)
        self.mlpt1 = MLPBlock(prompt_embed_dim, mlp_dim, activation)
        self.normt1 = nn.LayerNorm(prompt_embed_dim)
        self.normt2 = nn.LayerNorm(prompt_embed_dim)
        self.attnm1 = Attention(prompt_embed_dim, num_heads)
        self.attnm2 = Attention(prompt_embed_dim, num_heads)
        self.final = nn.Sequential(
            MLPBlock(prompt_embed_dim, mlp_dim, activation),
            nn.LayerNorm(prompt_embed_dim),
        )

    def execute(self, emb: jt.Var, image_embedding: jt.Var,
                tmp_embedding: jt.Var, prompt_embedding1: jt.Var,
                prompt_embedding2: jt.Var):
        image_embedding, et = self.parser(
            image_embedding, tmp_embedding,
            prompt_embedding1, prompt_embedding2)
        es = self.attns1(q=image_embedding, k=emb, v=emb)
        es_bk = es
        es = self.attns2(q=et, k=es, v=es)
        es = self.norms1(es + et)
        es = self.norms2(self.mlps1(es) + es)
        et = self.attnt1(q=es_bk, k=et, v=et)
        et = self.normt1(es_bk + et)
        et = self.normt2(self.mlps1(et) + et)
        e = self.attnm1(q=et, k=es, v=es)
        e = self.attnm2(q=e, k=e, v=e)
        e = self.final(e)
        return e


# ── Transformer Blocks ─────────────────────────────────────────────────
class TwoWayAttentionBlock(nn.Module):
    """Standard two-way attention block (used in SAM)."""

    def __init__(self, embedding_dim: int, num_heads: int,
                 mlp_dim: int = 2048,
                 activation: Type[nn.Module] = nn.ReLU,
                 attention_downsample_rate: int = 2,
                 skip_first_layer_pe: bool = False):
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads,
            downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads,
            downsample_rate=attention_downsample_rate)
        self.skip_first_layer_pe = skip_first_layer_pe

    def execute(self, queries: jt.Var, keys: jt.Var,
                query_pe: jt.Var, key_pe: jt.Var):
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)
        return queries, keys


class TwoWayTransformer(nn.Module):
    """Multi-layer two-way transformer."""

    def __init__(self, depth: int, embedding_dim: int, num_heads: int,
                 mlp_dim: int, activation: Type[nn.Module] = nn.ReLU,
                 attention_downsample_rate: int = 2):
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(TwoWayAttentionBlock(
                embedding_dim=embedding_dim, num_heads=num_heads,
                mlp_dim=mlp_dim, activation=activation,
                attention_downsample_rate=attention_downsample_rate,
                skip_first_layer_pe=(i == 0)))
        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads,
            downsample_rate=attention_downsample_rate)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def execute(self, image_embedding: jt.Var, image_pe: jt.Var,
                point_embedding: jt.Var):
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.reshape(bs, c, h * w).transpose(1, 2)
        image_pe = image_pe.reshape(bs, c, h * w).transpose(1, 2)
        queries = point_embedding
        keys = image_embedding
        for layer in self.layers:
            queries, keys = layer(queries=queries, keys=keys,
                                  query_pe=point_embedding, key_pe=image_pe)
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        return queries, keys


class CrossAttentionBlock(nn.Module):
    """Cross-attention block (similar to TwoWayAttentionBlock)."""

    def __init__(self, embedding_dim: int, num_heads: int,
                 mlp_dim: int = 2048,
                 activation: Type[nn.Module] = nn.ReLU,
                 attention_downsample_rate: int = 2,
                 skip_first_layer_pe: bool = False):
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads,
            downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads,
            downsample_rate=attention_downsample_rate)
        self.skip_first_layer_pe = skip_first_layer_pe

    def execute(self, queries: jt.Var, keys: jt.Var,
                query_pe: jt.Var, key_pe: jt.Var):
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)
        return queries, keys
