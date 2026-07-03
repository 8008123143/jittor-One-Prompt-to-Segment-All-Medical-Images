"""
One-Prompt main model (Jittor version).
"""

import jittor as jt
from jittor import nn
from typing import Any, Dict, List, Tuple


class OnePrompt(nn.Module):
    """One-Prompt: One-Prompt to Segment All Medical Images."""

    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        args,
        image_encoder: nn.Module,
        prompt_encoder: nn.Module,
        mask_decoder: nn.Module,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        super().__init__()
        self.args = args
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

        # Jittor: use jt.Var directly instead of register_buffer
        self.pixel_mean = jt.array(pixel_mean).reshape(-1, 1, 1)
        self.pixel_std = jt.array(pixel_std).reshape(-1, 1, 1)

    @property
    def device(self):
        return "cuda"  # Jittor auto-manages device

    def execute(
        self,
        batched_input: List[Dict[str, Any]],
        template_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, jt.Var]]:
        """Full forward pass for inference (with gradient disabled)."""
        # Disable gradient for inference
        prev_grad = jt.flag.grad
        jt.flag.grad = False

        input_images = jt.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        template_images = jt.stack([self.preprocess(x["image"]) for x in template_input], dim=0)
        r_emb, r_list = self.image_encoder(input_images)
        t_emb, t_list = self.image_encoder(template_images)

        outputs = []
        for image_record, r_l, t_l, r_e, t_e in zip(batched_input, r_list, t_list, r_emb, t_emb):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            p1, p2, sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                skips_raw=r_l,
                skips_tmp=t_l,
                raw_emb=r_e,
                tmp_emb=t_e,
                pt1=p1,
                pt2=p2,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append({
                "masks": masks,
                "iou_predictions": iou_predictions,
                "low_res_logits": low_res_masks,
            })

        # Restore gradient flag
        jt.flag.grad = prev_grad
        return outputs

    def postprocess_masks(
        self,
        masks: jt.Var,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> jt.Var:
        """Remove padding and upscale masks to original image size."""
        masks = jt.nn.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = jt.nn.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: jt.Var) -> jt.Var:
        """Normalize pixel values and pad to a square input."""
        x = (x - self.pixel_mean) / self.pixel_std

        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = jt.nn.pad(x, (0, padw, 0, padh))
        return x
