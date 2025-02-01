# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from sam2.modeling.sam2_utils import MLP, LayerNorm2d


class BoxDecoder(nn.Module):
    """
    Predict bounding boxes given an image and prompt embeddings,
    using a transformer architecture similar to MaskDecoder,
    but now specialized for bounding box regression.
    """

    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        activation: nn.Module = nn.GELU,
        num_box_outputs: int = 1,
        hidden_dim_for_box: int = 256,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_box_outputs = num_box_outputs  # typically 1 box per object

        # Instead of iou_token + mask_tokens, we define a box_token
        self.box_token = nn.Embedding(self.num_box_outputs, transformer_dim)

        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)

        # Upscaling layers: same idea as mask decoder, but we keep minimal usage
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

        # Box regression head: [x1, y1, x2, y2] plus an optional confidence
        # Using an MLP of depth 3 as a generic example
        self.box_head = MLP(
            input_dim=transformer_dim,
            hidden_dim=hidden_dim_for_box,
            output_dim=4,  # bounding box coords: x1, y1, x2, y2
            num_layers=3,
        )

        if self.pred_obj_scores:
            # Object score head to classify whether object is present
            if pred_obj_scores_mlp:
                self.obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)
            else:
                self.obj_score_head = nn.Linear(transformer_dim, 1)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool = False,
        high_res_features=None
    ):
        """
        Return bounding boxes instead of masks.
        """
        b = sparse_prompt_embeddings.size(0)

        # Concatenate box tokens
        if self.pred_obj_scores:
            # shape: [B, 1 + num_box_outputs, C]
            output_tokens = torch.cat(
                [self.obj_score_token.weight, self.box_token.weight], dim=0
            ).unsqueeze(0).expand(b, -1, -1)
            s = 1  # index offset for box tokens
        else:
            output_tokens = self.box_token.weight.unsqueeze(0).expand(b, -1, -1)
            s = 0

        # Combine with the prompt embeddings
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch dimension if needed
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, b, dim=0)
            pos_src = torch.repeat_interleave(image_pe, b, dim=0)
        else:
            src = image_embeddings
            pos_src = image_pe

        # Add dense prompt (like mask_prompt) into the image embeddings
        src = src + dense_prompt_embeddings

        # Run the transformer
        hs, src_out = self.transformer(src, pos_src, tokens)

        # box_tokens_out = hs[:, s : s + self.num_box_outputs, :]
        box_tokens_out = hs[:, s : s + self.num_box_outputs, :]

        # Predict bounding boxes
        box_coords = self.box_head(box_tokens_out)  # [B, num_box_outputs, 4]
        # Optionally, predict object scores
        if self.pred_obj_scores:
            object_score_logits = self.obj_score_head(hs[:, 0, :])
        else:
            object_score_logits = torch.zeros((b, 1), device=box_tokens_out.device)

        return box_coords, object_score_logits, box_tokens_out
