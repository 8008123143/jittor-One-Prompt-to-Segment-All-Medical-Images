"""
One-Prompt Mask Decoder (Jittor version).
Core innovation: fuses template and query image features via OnePromptFormer.
"""

import math
import jittor as jt
from jittor import nn
from typing import List, Tuple, Type

from .common import LayerNorm2d
from .modules import (
    CrossAttentionBlock,
    OnePromptFormer,
    TwoWayTransformer,
)
from .image_encoder import PatchEmbed


# ═══════════════════════════════════════════════════════════════════════
# MLP (from SAM)
# ═══════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """Simple MLP with ReLU activations."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList([
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        ])
        self.sigmoid_output = sigmoid_output

    def execute(self, x):
        for i, layer in enumerate(self.layers):
            x = jt.nn.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = jt.nn.sigmoid(x)
        return x


# ═══════════════════════════════════════════════════════════════════════
# MaskDecoder (SAM-style, produces final mask from fused features)
# ═══════════════════════════════════════════════════════════════════════

class MaskDecoder(nn.Module):
    """Standard SAM mask decoder that produces masks from image embeddings."""

    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
             for _ in range(self.num_mask_tokens)]
        )
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def execute(
        self,
        image_embeddings: jt.Var,
        image_pe: jt.Var,
        mix_embeddings: jt.Var,
        multimask_output: bool,
    ) -> Tuple[jt.Var, jt.Var]:
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            mix_embeddings=mix_embeddings,
        )
        # Select mask slice
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: jt.Var,
        image_pe: jt.Var,
        mix_embeddings: jt.Var,
    ) -> Tuple[jt.Var, jt.Var]:
        """Predict masks from image and mixed embeddings."""
        output_tokens = jt.concat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(image_embeddings.size(0), -1, -1)
        tokens = jt.concat((output_tokens, mix_embeddings), dim=1)

        if image_embeddings.shape[0] != tokens.shape[0]:
            src = jt.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings

        # Resize image_pe to match src spatial resolution (needed for U-Net)
        h_s, w_s = src.shape[2], src.shape[3]
        if image_pe.shape[2] != h_s or image_pe.shape[3] != w_s:
            image_pe = jt.nn.interpolate(image_pe, size=(h_s, w_s),
                                          mode='bilinear', align_corners=False)
        pos_src = jt.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]

        # Upscale & predict masks
        src = src.transpose(1, 2).reshape(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = jt.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.reshape(b, c, h * w)).reshape(b, -1, h, w)

        # IoU predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred


# ═══════════════════════════════════════════════════════════════════════
# Decode_Align: Aligns template & query features
# ═══════════════════════════════════════════════════════════════════════

class Decode_Align(nn.Module):
    """Aligns decoder features with prompt-based positional embeddings."""

    def __init__(self, *, embed_dim: int, transformer_dim: int, stages: int = 4096):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.num_mask_tokens = stages
        self.p1_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.p2_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.layer = nn.Linear(embed_dim, transformer_dim)

    def execute(
        self,
        x: jt.Var,
        src_embeddings: jt.Var,
        image_embeddings: jt.Var,
        image_pe: jt.Var,
        pt1: jt.Var,
        pt2: jt.Var,
        dense_prompt_embeddings: jt.Var,
    ):
        image_embeddings = self.layer(image_embeddings)
        src_embeddings = self.layer(src_embeddings)

        p1 = self.p1_tokens.weight.unsqueeze(0).expand(pt1.size(0), -1, -1)
        p2 = self.p2_tokens.weight.unsqueeze(0).expand(pt1.size(0), -1, -1)

        p1_tokens = jt.concat((p1, pt1), dim=1)
        p2_tokens = jt.concat((p2, pt2), dim=1)

        if image_embeddings.shape[0] != p1_tokens.shape[0]:
            src = jt.repeat_interleave(image_embeddings, p1_tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        src = src.permute(0, 3, 1, 2)
        img = src_embeddings.permute(0, 3, 1, 2)
        x = x.permute(0, 3, 1, 2)
        src = src + dense_prompt_embeddings
        # pos_src not used further but computed for consistency
        # pos_src = jt.repeat_interleave(image_pe, p1_tokens.shape[0], dim=0)

        return x, img, src, image_pe, p1_tokens, p2_tokens


# ═══════════════════════════════════════════════════════════════════════
# OnePromptDecoder (main decoder)
# ═══════════════════════════════════════════════════════════════════════

class OnePromptDecoder(nn.Module):
    """
    The core One-Prompt decoder.
    Fuses template (skips_tmp/tmp_emb) with query (skips_raw/raw_emb) features
    through multiple OnePromptFormer layers, then generates mask predictions.
    """

    def __init__(
        self,
        *,
        depth: int = 4,
        prompt_embed_dim: int = 256,
        embed_dim: int = 768,
        out_chans: int = 256,
        token_num: int,
        patch_size: int,
        mlp_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.of = nn.ModuleList()
        self.deals = nn.ModuleList()

        # Final mask decoder (SAM-style)
        self.updecode = MaskDecoder(
            transformer_dim=prompt_embed_dim,
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=256,
                num_heads=2,
            )
        )

        # Neck: reduce feature dims
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )

        for i in range(depth):
            self.of.append(
                OnePromptFormer(
                    embedding_dim=prompt_embed_dim,
                    prompt_embed_dim=prompt_embed_dim,
                    token_num=token_num,
                    num_heads=2,
                    mlp_dim=mlp_dim,
                )
            )
            self.deals.append(
                Decode_Align(
                    embed_dim=embed_dim,
                    transformer_dim=prompt_embed_dim,
                    stages=token_num - 1,
                )
            )

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=prompt_embed_dim,
            embed_dim=out_chans,
        )

    def execute(
        self,
        skips_raw: list,
        skips_tmp: list,
        raw_emb: jt.Var,
        tmp_emb: jt.Var,
        pt1: jt.Var,
        pt2: jt.Var,
        image_pe: jt.Var,
        sparse_prompt_embeddings: jt.Var,
        dense_prompt_embeddings: jt.Var,
        multimask_output: bool,
    ) -> Tuple[jt.Var, jt.Var]:
        # Fuse raw and template embeddings
        x = raw_emb + tmp_emb
        x = self.neck(x.permute(0, 3, 1, 2))
        x = x.permute(0, 2, 3, 1)

        raw_emb_proj = self.neck(raw_emb.permute(0, 3, 1, 2))

        for u in range(self.depth):
            if u == 0:
                x, img_embed, tmp_embed, temp_pos, p1, p2 = self.deals[u](
                    x, skips_raw[-(u + 1)], skips_tmp[-(u + 1)],
                    image_pe, pt1, pt2, dense_prompt_embeddings
                )
                p1 = p1 + temp_pos.reshape(temp_pos.shape[0], temp_pos.shape[1], -1).permute(0, 2, 1)
                p2 = p2 + temp_pos.reshape(temp_pos.shape[0], temp_pos.shape[1], -1).permute(0, 2, 1)
                img_embed = img_embed.reshape(img_embed.shape[0], img_embed.shape[1], -1).permute(0, 2, 1)
                tmp_embed = tmp_embed.reshape(tmp_embed.shape[0], tmp_embed.shape[1], -1).permute(0, 2, 1)
                x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

            x = self.of[u](x, img_embed, tmp_embed, p1, p2)

        # Reshape back to spatial
        sqrt_n = int(math.sqrt(x.shape[1]))
        x = x.reshape(x.shape[0], sqrt_n, sqrt_n, -1).permute(0, 3, 1, 2)
        x = self.patch_embed(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])

        # Generate masks
        low_res_masks, iou_predictions = self.updecode(
            image_embeddings=raw_emb_proj,
            image_pe=image_pe,
            mix_embeddings=x,
            multimask_output=multimask_output,
        )

        return low_res_masks, iou_predictions
