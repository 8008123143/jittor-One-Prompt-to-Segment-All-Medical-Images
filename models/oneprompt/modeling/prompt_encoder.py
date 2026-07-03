"""
Prompt Encoder for One-Prompt (Jittor version).
Encodes points, boxes, doodles and masks into embeddings.
"""

import numpy as np
import jittor as jt
from jittor import nn
from typing import Any, Optional, Tuple, Type

from .common import LayerNorm2d


class PositionEmbeddingRandom(nn.Module):
    """Positional encoding using random spatial frequencies."""

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.num_pos_feats = num_pos_feats
        # Jittor: register_buffer equivalent — assign jt.Var directly
        self.positional_encoding_gaussian_matrix = scale * jt.randn(2, num_pos_feats)

    def _pe_encoding(self, coords: jt.Var) -> jt.Var:
        """Positionally encode points normalized to [0,1]."""
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return jt.concat([jt.sin(coords), jt.cos(coords)], dim=-1)

    def execute(self, size: Tuple[int, int]) -> jt.Var:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        grid = jt.ones((h, w))
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(jt.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: jt.Var, image_size: Tuple[int, int]
    ) -> jt.Var:
        """Positionally encode unnormalized coordinates."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.float())  # B x N x C


class PromptEncoder(nn.Module):
    """Encodes prompts (points, boxes, doodles, masks) for One-Prompt."""

    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 6  # pos/neg point/doodle + 2 box corners
        point_embeddings = [nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)

        self.not_a_point_embed = nn.Embedding(1, embed_dim)
        self.not_a_doodle_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> jt.Var:
        """Returns dense positional encoding for the image embedding grid."""
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self, points: jt.Var, labels: jt.Var, pad: bool
    ):
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = jt.zeros((points.shape[0], 1, 2))
            padding_label = -jt.ones((labels.shape[0], 1))
            points = jt.concat([points, padding_point], dim=1)
            labels = jt.concat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding = point_embedding * (labels != -1).unsqueeze(-1).float()
        point_embedding = point_embedding + self.not_a_point_embed.weight * (labels == -1).unsqueeze(-1).float()
        point_embedding = point_embedding + self.point_embeddings[0].weight * (labels == 0).unsqueeze(-1).float()
        point_embedding = point_embedding + self.point_embeddings[1].weight * (labels == 1).unsqueeze(-1).float()
        return point_embedding[:, 0, :], point_embedding[:, 1, :]

    def _embed_boxes(self, boxes: jt.Var):
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] = corner_embedding[:, 0, :] + self.point_embeddings[2].weight
        corner_embedding[:, 1, :] = corner_embedding[:, 1, :] + self.point_embeddings[3].weight
        return corner_embedding[:, 0, :], corner_embedding[:, 1, :]

    def _embed_doodles(self, doodles: jt.Var, labels: jt.Var):
        """Embeds doodle prompts."""
        doodles = doodles + 0.5  # Shift to center of pixel
        doodle_embedding = self.pe_layer.forward_with_coords(doodles, self.input_image_size)
        doodle_embedding = doodle_embedding * (labels != -1).unsqueeze(-1).float()
        doodle_embedding = doodle_embedding + self.not_a_doodle_embed.weight * (labels == -1).unsqueeze(-1).float()
        doodle_embedding = doodle_embedding + self.point_embeddings[4].weight * (labels == 0).unsqueeze(-1).float()
        doodle_embedding = doodle_embedding + self.point_embeddings[5].weight * (labels == 1).unsqueeze(-1).float()
        return doodle_embedding[:, 0, :], doodle_embedding[:, 1, :]

    def _embed_masks(self, masks: jt.Var):
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding, mask_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[jt.Var, jt.Var]],
        boxes: Optional[jt.Var],
        masks: Optional[jt.Var],
    ) -> int:
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def execute(
        self,
        points: Optional[Tuple[jt.Var, jt.Var]],
        boxes: Optional[jt.Var],
        doodles: Optional[Tuple[jt.Var, jt.Var]],
        masks: Optional[jt.Var],
    ) -> Tuple[jt.Var, jt.Var, jt.Var, jt.Var]:
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = jt.empty((bs, 0, self.embed_dim))

        p1, p2 = None, None

        if points is not None:
            coords, labels = points
            p1, p2 = self._embed_points(coords, labels, pad=(boxes is None))
            p1 = jt.concat([sparse_embeddings, p1.unsqueeze(0)], dim=1)
            p2 = jt.concat([sparse_embeddings, p2.unsqueeze(0)], dim=1)
            sparse_embeddings = jt.concat([sparse_embeddings, p1, p2], dim=1)

        if boxes is not None:
            p1, p2 = self._embed_boxes(boxes)
            p1 = jt.concat([sparse_embeddings, p1.unsqueeze(0)], dim=1)
            p2 = jt.concat([sparse_embeddings, p2.unsqueeze(0)], dim=1)
            sparse_embeddings = jt.concat([sparse_embeddings, p1, p2], dim=1)

        if doodles is not None:
            coords, labels = doodles
            p1, p2 = self._embed_doodles(coords, labels)
            p1 = jt.concat([sparse_embeddings, p1.unsqueeze(0)], dim=1)
            p2 = jt.concat([sparse_embeddings, p2.unsqueeze(0)], dim=1)
            sparse_embeddings = jt.concat([sparse_embeddings, p1, p2], dim=1)

        if masks is not None:
            p1, p2 = self._embed_masks(masks)
            dense_embeddings = p1
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return p1, p2, sparse_embeddings, dense_embeddings
