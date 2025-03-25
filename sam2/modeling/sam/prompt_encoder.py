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

        # Add an instance ID embedding to differentiate between instance masks
        # Support up to 500 different instances
        self.instance_id_embed = nn.Embedding(500, embed_dim)
        
        # Add cross-instance attention mechanism
        self.instance_attention = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
        
        # Add a mask embedding fusion mechanism
        self.mask_fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

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

    def _embed_mask(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs for single mask."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _embed_masks(self, masks: torch.Tensor, instance_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds mask inputs with optional instance IDs.
        
        Args:
            masks: Tensor of shape (batch_size, num_instances, H, W) or (batch_size, 1, H, W)
            instance_ids: Optional tensor of shape (batch_size, num_instances) with IDs for each mask
            
        Returns:
            Tensor of embeddings for each mask and instance
        """
        b, n, h, w = masks.shape
        
        # Process each mask individually
        mask_embeddings = []
        for i in range(n):
            mask_slice = masks[:, i:i+1, :, :]  # (batch_size, 1, H, W)
            mask_embedding = self.mask_downscaling(mask_slice)  # (batch_size, embed_dim, h', w')
            
            # Add instance ID embeddings if provided
            if instance_ids is not None:
                instance_embedding = self.instance_id_embed(instance_ids[:, i])  # (batch_size, embed_dim)
                # Reshape for broadcasting addition
                instance_embedding = instance_embedding.unsqueeze(-1).unsqueeze(-1)  # (batch_size, embed_dim, 1, 1)
                # Add instance embedding to mask embedding
                mask_embedding = mask_embedding + instance_embedding
            
            mask_embeddings.append(mask_embedding)
        
        return mask_embeddings

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

    def _fuse_mask_embeddings(self, mask_embeddings: list[torch.Tensor]) -> torch.Tensor:
        """
        Fuses multiple mask embeddings into a single embedding.
        Uses cross-attention to allow masks to interact with each other.
        
        Args:
            mask_embeddings: List of mask embeddings, each of shape (batch_size, embed_dim, h, w)
            
        Returns:
            Fused mask embedding of shape (batch_size, embed_dim, h, w)
        """
        if len(mask_embeddings) == 0:
            return None
        
        if len(mask_embeddings) == 1:
            return mask_embeddings[0]
        
        # Average the embeddings as initial fusion
        #fused_embedding = sum(mask_embeddings) / len(mask_embeddings)
        
        # Reshape for attention: (batch_size, h*w, embed_dim)
        b, c, h, w = mask_embeddings[0].shape
        reshaped_embeddings = [emb.reshape(b, c, -1).permute(0, 2, 1) for emb in mask_embeddings]
        
        # Stack along a new dimension: (batch_size, num_masks, h*w, embed_dim)
        stacked_embeddings = torch.stack(reshaped_embeddings, dim=1)
        
        # Flatten for attention: (batch_size*num_masks, h*w, embed_dim)
        num_masks = len(mask_embeddings)
        flattened_embeddings = stacked_embeddings.reshape(b * num_masks, h * w, c)
        
        # Self-attention to allow interaction between spatial locations
        attended_embeddings, _ = self.instance_attention(
            flattened_embeddings, flattened_embeddings, flattened_embeddings
        )
        
        # Reshape back to original dimensions
        attended_embeddings = attended_embeddings.reshape(b, num_masks, h * w, c)
        
        # Process each attended mask embedding
        processed_embeddings = []
        for i in range(num_masks):
            mask_emb = attended_embeddings[:, i]  # (batch_size, h*w, embed_dim)
            processed_emb = self.mask_fusion(mask_emb)  # Apply fusion MLP
            processed_emb = processed_emb.permute(0, 2, 1).reshape(b, c, h, w)  # (batch_size, embed_dim, h, w)
            processed_embeddings.append(processed_emb)
        
        # Sum all processed embeddings
        final_embedding = sum(processed_embeddings)
        
        return final_embedding

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

        mask_embeddings = None
        if masks is not None:
            use_original_mask_input = False
            if use_original_mask_input:
                dense_embeddings = self._embed_mask(masks)
            else:
                # Handle multiple masks with instance IDs
                if masks.dim() == 3:
                    masks = masks.unsqueeze(1)  # Add instance dimension if not present
                
                instance_ids = None
                if instance_ids is None and masks.shape[1] > 1:
                    # If instance IDs not provided but multiple masks present, create sequential IDs
                    #instance_ids = torch.arange(masks.shape[1], device=masks.device).unsqueeze(0).expand(masks.shape[0], -1)
                    pass

                # Embed each mask
                mask_embeddings = self._embed_masks(masks, instance_ids)
                
                # Fuse mask embeddings
                dense_embeddings = self._fuse_mask_embeddings(mask_embeddings)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings #, mask_embeddings  # Return individual mask embeddings too
