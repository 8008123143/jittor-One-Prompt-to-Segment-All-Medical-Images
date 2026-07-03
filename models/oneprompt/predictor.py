"""
One-Prompt Predictor (Jittor version).
Wraps the model with image pre-processing and efficient mask prediction.
"""

import numpy as np
import jittor as jt
from typing import Optional, Tuple
from .utils.transforms import ResizeLongestSide


class OnePredictor:
    """Predictor for One-Prompt model."""

    def __init__(self, one_model) -> None:
        super().__init__()
        self.model = one_model
        self.transform = ResizeLongestSide(one_model.image_encoder.img_size)
        self.reset_image()

    def set_image(self, image: np.ndarray, image_format: str = "RGB") -> None:
        """Calculate image embeddings for the provided image."""
        assert image_format in ["RGB", "BGR"], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        input_image = self.transform.apply_image(image)
        input_image_jt = jt.array(input_image).permute(2, 0, 1)[None, :, :, :]
        self.set_torch_image(input_image_jt, image.shape[:2])

    def set_torch_image(self, transformed_image: jt.Var, original_image_size: Tuple[int, ...]) -> None:
        """Set pre-transformed image tensor."""
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(transformed_image.shape[2:]) == self.model.image_encoder.img_size
        )
        self.reset_image()
        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        input_image = self.model.preprocess(transformed_image)
        self.features = self.model.image_encoder(input_image)
        self.is_image_set = True

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict masks for given prompts."""
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        coords_jt, labels_jt, box_jt, mask_input_jt = None, None, None, None
        if point_coords is not None:
            assert point_labels is not None
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_jt = jt.array(point_coords).float()
            labels_jt = jt.array(point_labels).int()
            coords_jt, labels_jt = coords_jt[None, :, :], labels_jt[None, :]
        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_jt = jt.array(box).float()[None, :]
        if mask_input is not None:
            mask_input_jt = jt.array(mask_input).float()[None, :, :, :]

        masks, iou_predictions, low_res_masks = self.predict_torch(
            coords_jt, labels_jt, box_jt, mask_input_jt,
            multimask_output, return_logits=return_logits,
        )

        masks_np = masks[0].numpy()
        iou_predictions_np = iou_predictions[0].numpy()
        low_res_masks_np = low_res_masks[0].numpy()
        return masks_np, iou_predictions_np, low_res_masks_np

    def predict_torch(
        self,
        point_coords: Optional[jt.Var],
        point_labels: Optional[jt.Var],
        boxes: Optional[jt.Var] = None,
        mask_input: Optional[jt.Var] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[jt.Var, jt.Var, jt.Var]:
        """Predict masks from pre-transformed tensors."""
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points, boxes=boxes, masks=mask_input,
        )

        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)
        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks

    def get_image_embedding(self):
        """Return cached image embeddings."""
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...).")
        return self.features

    def reset_image(self) -> None:
        """Reset cached image state."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None
