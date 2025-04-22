# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.modeling.sam2_utils import DropPath, get_clones, LayerNorm2d

from training.utils.visualize import visualize_frame, quick_visualize_mask, visualize_4d_tensor


class MaskDownSampler(nn.Module):
    """
    Progressively downsample a mask by total_stride, each time by stride.
    Note that LayerNorm is applied per *token*, like in ViT.

    With each downsample (by a factor stride**2), channel capacity increases by the same factor.
    In the end, we linearly project to embed_dim channels.
    """

    def __init__(
        self,
        embed_dim=256,
        kernel_size=4,
        stride=4,
        padding=0,
        total_stride=16,
        activation=nn.GELU,
    ):
        super().__init__()
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        assert stride**num_layers == total_stride
        self.encoder = nn.Sequential()
        mask_in_chans, mask_out_chans = 3, 1
        for _ in range(num_layers):
            mask_out_chans = mask_in_chans * (stride**2)
            self.encoder.append(
                nn.Conv2d(
                    mask_in_chans,
                    mask_out_chans,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            self.encoder.append(LayerNorm2d(mask_out_chans))
            self.encoder.append(activation())
            mask_in_chans = mask_out_chans

        self.encoder.append(nn.Conv2d(mask_out_chans, mask_out_chans // 3, kernel_size=1))  # 768 -> 256
        self.encoder.append(LayerNorm2d(mask_out_chans // 3))
        self.encoder.append(activation())
        
        self.encoder.append(nn.Conv2d(mask_out_chans // 3, embed_dim, kernel_size=1))

    def forward(self, x):
        return self.encoder(x)


# Lightly adapted from ConvNext (https://github.com/facebookresearch/ConvNeXt)
class CXBlock(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        dim,
        kernel_size=7,
        padding=3,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        use_dwconv=True,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim if use_dwconv else 1,
        )  # depthwise conv
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Fuser(nn.Module):
    def __init__(self, layer, num_layers, dim=None, input_projection=False):
        super().__init__()
        self.proj = nn.Identity()
        self.layers = get_clones(layer, num_layers)

        if input_projection:
            assert dim is not None
            self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        # normally x: (N, C, H, W)
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x

class MemoryEncoder(nn.Module):
    def __init__(
        self,
        out_dim,
        mask_downsampler,
        fuser,
        position_encoding,
        in_dim=256,  # in_dim of pix_feats
    ):
        super().__init__()

        self.mask_downsampler = mask_downsampler

        self.pix_feat_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.fuser = fuser
        self.position_encoding = position_encoding
        self.out_proj = nn.Identity()
        if out_dim != in_dim:
            self.out_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

        
        # new stuff for multi mask emb fusing
        self.embed_dim = in_dim

        # Object identity embeddings to differentiate masks
        self.object_id_embeddings = nn.Embedding(3, self.embed_dim * 2)  # Double dimension
        self.object_proj = nn.Linear(self.embed_dim * 2, self.embed_dim)  # Project back down

        # Attention mechanism to fuse multiple mask embeddings
        self.mask_attention = nn.MultiheadAttention(
            self.embed_dim, 
            num_heads=8, 
            batch_first=True
        )
        
        # Final projection to maintain embedding dimensionality
        self.mask_fusion = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1),
            LayerNorm2d(self.embed_dim),
            nn.GELU(),
        )

        self.no_mask_embed = nn.Embedding(1, self.embed_dim)

    def _embed_multi_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Embeds multiple mask inputs and combines them into a single dense embedding
        per batch element.
        
        Args:
            masks: Tensor of shape (B, N, H, W) where:
                B is batch size
                N is number of masks per sample (typically 1)
                H and W are mask height and width
                    
        Returns:
            Tensor of shape (B, embed_dim, embed_H, embed_W) - one embedding per batch
        """
        B, N, H, W = masks.shape
        
        # Process each mask in the batch independently
        batch_embeddings = []
        for b in range(B):
            # Process the N masks for this batch element
            mask_embeddings = []
            for n in range(N):
                # Process through the downscaling network - shape (1, 1, H, W)
                single_mask = masks[b:b+1, n:n+1]
                single_mask_embedding = self.mask_downsampler(single_mask)  # (1, embed_dim, embed_H, embed_W)
                
                # Add object identity embedding
                object_id_embed = self.object_id_embeddings(torch.tensor([min(n, 2)], device=masks.device))
                object_id_embed = self.object_proj(object_id_embed)  # Project down to embed dim shape again
                object_id_embed = object_id_embed.view(1, self.embed_dim, 1, 1).expand_as(single_mask_embedding)
                
                # Combine spatial features with object identity
                enhanced_embedding = single_mask_embedding + object_id_embed
                mask_embeddings.append(enhanced_embedding)
            
            if not mask_embeddings:
                # No masks for this batch, use the no_mask embedding
                batch_embedding = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                    1, -1, self.image_embedding_size[0], self.image_embedding_size[1]
                )
            elif len(mask_embeddings) == 1:
                # Only one mask for this batch, no need for attention mechanism
                batch_embedding = mask_embeddings[0]
            else:
                # Multiple masks for this batch, use attention to fuse them
                # Stack along a new dimension - each element is (1, embed_dim, h, w)
                stacked_embeddings = torch.stack(mask_embeddings, dim=1)  # (1, N, embed_dim, h, w)
                
                C = self.embed_dim
                H_emb, W_emb = stacked_embeddings.shape[-2], stacked_embeddings.shape[-1]
                
                # Reshape for attention: (1, N, C, H, W) -> (1*H*W, N, C)
                reshaped_emb = stacked_embeddings.permute(0, 3, 4, 1, 2).reshape(H_emb*W_emb, N, C)
                
                # Apply attention across the mask dimension for each spatial location
                attn_output, _ = self.mask_attention(reshaped_emb, reshaped_emb, reshaped_emb)
                
                # Reshape back and sum across mask dimension
                attn_output = attn_output.reshape(1, H_emb, W_emb, N, C).permute(0, 4, 3, 1, 2)  # (1, C, N, H, W)
                fused_embedding = torch.sum(attn_output, dim=2)  # (1, C, H, W)
                
                # Final projection to ensure proper embedding
                batch_embedding = self.mask_fusion(fused_embedding)
            
            batch_embeddings.append(batch_embedding)
        
        # Concatenate the batch embeddings
        return torch.cat(batch_embeddings, dim=0)

    def forward(
        self,
        pix_feat: torch.Tensor,
        masks: torch.Tensor,
        skip_mask_sigmoid: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # sigmoid, so that less domain shift from gt masks which are bool
        if not skip_mask_sigmoid:
            masks = F.sigmoid(masks)
        
        #visualize_4d_tensor(masks.float(), f"masks_memenc/masks memory encoder.png")
        masks = self.mask_downsampler(masks)
        
        # Use the prompt encoder's multi-mask embedding functionality
        # This will handle the attention fusion between masks
        #mask_features = self._embed_multi_masks(masks)
        #visualize_4d_tensor(masks.float(), "MASKS.png")

        ## Fuse pix_feats and downsampled masks
        # in case the visual features are on CPU, cast them to CUDA
        pix_feat = pix_feat.to(masks.device)

        x = self.pix_feat_proj(pix_feat)
        x = x + masks #mask_features
        x = self.fuser(x)
        x = self.out_proj(x)

        #visualize_4d_tensor(x.float(), f"masks_memenc/vision features memory encoder.png")

        pos = self.position_encoding(x).to(x.dtype)

        return {"vision_features": x, "vision_pos_enc": [pos]}
