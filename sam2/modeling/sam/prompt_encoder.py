# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Type

import torch
from torch import nn

from sam2.modeling.position_encoding import PositionEmbeddingRandom

from sam2.modeling.sam2_utils import LayerNorm2d


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [
            nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)
        ]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
        )
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

        # Object identity embeddings to differentiate masks
        self.object_id_embeddings = nn.Embedding(3, embed_dim)
        
        # Attention mechanism to fuse multiple mask embeddings
        self.mask_attention = nn.MultiheadAttention(
            embed_dim, 
            num_heads=8, 
            batch_first=True
        )
        
        # Final projection to maintain embedding dimensionality
        self.mask_fusion = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            LayerNorm2d(embed_dim),
            activation(),
        )

    def _embed_multi_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Embeds multiple mask inputs and combines them into a single dense embedding.
        
        Args:
            masks: Tensor of shape (B, N, H, W) where:
                B is batch size, N is number of masks per sample,
                H and W are mask height and width
                
        Returns:
            Tensor of shape (B, embed_dim, embed_H, embed_W)
        """
        B, N, H, W = masks.shape
        
        # Process each mask individually
        mask_embeddings = []
        for i in range(N):
            # Process each mask through the downscaling network
            single_mask_embedding = self.mask_downscaling(masks[:, i:i+1, :, :])  # (B, embed_dim, embed_H, embed_W)
            
            # Add object identity embedding
            object_id_embed = self.object_id_embeddings(torch.tensor([i], device=masks.device))  # (1, embed_dim)
            object_id_embed = object_id_embed.view(1, self.embed_dim, 1, 1).expand_as(single_mask_embedding)
            
            # Combine spatial features with object identity
            enhanced_embedding = single_mask_embedding + object_id_embed
            mask_embeddings.append(enhanced_embedding)
        
        if not mask_embeddings:
            # No masks provided, use the no_mask embedding
            return self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                B, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )
        
        # Stack all embeddings
        stacked_embeddings = torch.stack(mask_embeddings, dim=1)  # (B, N, embed_dim, h, w)
        B, N, C, H, W = stacked_embeddings.shape
        
        # Method 1: Spatial-aware attention (applying attention across masks at each spatial location)
        # Reshape to (B*H*W, N, C) - treating each spatial location separately
        reshaped_emb = stacked_embeddings.permute(0, 3, 4, 1, 2).reshape(B*H*W, N, C)
        
        # Apply attention across the mask dimension for each spatial location
        attn_output, _ = self.mask_attention(reshaped_emb, reshaped_emb, reshaped_emb)
        
        # Reshape back: (B*H*W, N, C) -> (B, H, W, N, C) -> (B, N, C, H, W)
        attn_output = attn_output.reshape(B, H, W, N, C).permute(0, 3, 4, 1, 2)
        
        # Combine masks (weighted sum across mask dimension)
        fused_embedding = torch.sum(attn_output, dim=1)  # (B, C, H, W)
        
        # Final projection to ensure proper embedding
        final_embedding = self.mask_fusion(fused_embedding)
        
        return final_embedding

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size
        )

        point_embedding = torch.where(
            (labels == -1).unsqueeze(-1),
            torch.zeros_like(point_embedding) + self.not_a_point_embed.weight,
            point_embedding,
        )
        point_embedding = torch.where(
            (labels == 0).unsqueeze(-1),
            point_embedding + self.point_embeddings[0].weight,
            point_embedding,
        )
        point_embedding = torch.where(
            (labels == 1).unsqueeze(-1),
            point_embedding + self.point_embeddings[1].weight,
            point_embedding,
        )
        point_embedding = torch.where(
            (labels == 2).unsqueeze(-1),
            point_embedding + self.point_embeddings[2].weight,
            point_embedding,
        )
        point_embedding = torch.where(
            (labels == 3).unsqueeze(-1),
            point_embedding + self.point_embeddings[3].weight,
            point_embedding,
        )
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(
            coords, self.input_image_size
        )
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            print("masks in prompt enc", masks.shape)
            dense_embeddings = self._embed_multi_masks(masks)
            print("dense_embeddings", dense_embeddings.shape)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings
