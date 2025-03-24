# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type

import torch
from torch.nn import functional as F
from torch import nn
from sam2.modeling.sam.transformer import Attention

from sam2.modeling.sam2_utils import LayerNorm2d, MLP

from enum import IntEnum, auto

class InstanceIdMode(IntEnum):
    DIRECT = 0
    EMBEDDING_HEAD_CNN = 1
    EMBEDDING_HEAD_MLP = 2
    EMBEDDING_SHARED_MLP = 3

class SpatialEmbeddingAggregator(nn.Module):
    """Aggregates embeddings using cross-attention that preserves spatial information."""
    
    def __init__(self, embedding_dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Query projection for each spatial position
        self.query_proj = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1)
        
        # Cross-attention for combining embeddings
        self.cross_attention = Attention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, embedding_list):
        """
        Args:
            embedding_list: List of tensors, each of shape [b, c, h, w]
            
        Returns:
            Tensor of shape [b, c, h, w]
        """
        b, c, h, w = embedding_list[0].shape
        num_embeddings = len(embedding_list)
        
        # Generate queries from the first embedding
        # This could also be the average, or a learned combination
        queries = self.query_proj(embedding_list[0])  # [b, c, h, w]
        
        # Reshape queries for attention: [b, h*w, c]
        queries = queries.view(b, c, h*w).permute(0, 2, 1)
        
        # Prepare keys and values by stacking embeddings along sequence dimension
        # First reshape each embedding to [b, h*w, c]
        keys_values_list = []
        for embed in embedding_list:
            # [b, c, h, w] -> [b, h*w, c]
            embed_flat = embed.view(b, c, h*w).permute(0, 2, 1)
            keys_values_list.append(embed_flat)
        
        # Stack along a new sequence dimension: [b, num_embeddings*h*w, c]
        keys_values = torch.cat(keys_values_list, dim=1)
        
        # Apply cross attention
        attn_out = self.cross_attention(q=queries, k=keys_values, v=keys_values)
        output = queries + attn_out
        output = self.norm(output)
        
        # Reshape back to [b, c, h, w]
        output = output.permute(0, 2, 1).view(b, c, h, w)
        
        return output

class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid=False,
        dynamic_multimask_via_stability=False,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1 = nn.Conv2d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
            )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )
        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

        # When outputting a single mask, optionally we can dynamically fall back to the best
        # multimask output token if the single mask output token gives low stability scores.
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

        # *** New: Add an instance ID head for dual-channel output ***
        self.instance_id_mode = InstanceIdMode.EMBEDDING_HEAD_MLP
        self.embedding_dim = 64
        self.num_embedding_tokens = 64

        # We need to match the dimension of upscaled features
        self.output_feature_dim = transformer_dim // 8
        print("transforemr dim", transformer_dim)
        
        # We assume that output_upscaling produces feature maps of channel dimension transformer_dim//8?
        match self.instance_id_mode:
            case InstanceIdMode.DIRECT:
                self.instance_id_head = nn.Conv2d(self.output_feature_dim, 1, kernel_size=1)
            case InstanceIdMode.EMBEDDING_HEAD_CNN:
                # Just a more advanced head to predict instance embeddings, using CNNs
                self.instance_embedding_head = nn.Sequential(
                    nn.Conv2d(self.output_feature_dim, transformer_dim // 4, kernel_size=3, padding=1),
                    nn.GroupNorm(8, transformer_dim // 4),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(transformer_dim // 4, self.embedding_dim, kernel_size=1)
                )
            case InstanceIdMode.EMBEDDING_HEAD_MLP:
                # Extended transformer MLP head to predict instance embeddings
                
                # Add embedding tokens alongside mask tokens
                self.embedding_tokens = nn.Embedding(self.num_embedding_tokens, transformer_dim)
                
                # Add embedding prediction hypernetworks (parallel to mask hypernetworks)
                self.embedding_hypernetworks_mlps = nn.ModuleList(
                    [
                        MLP(transformer_dim, transformer_dim, self.output_feature_dim, 3)
                    ]
                )
            
            case InstanceIdMode.EMBEDDING_SHARED_MLP:
                # Shared MLP head to predict instance embeddings
                # Use the same hypernetworks but with different output dimension
                self.instance_id_head = nn.ModuleList(
                    [
                        nn.Linear(self.output_feature_dim, self.embedding_dim)
                        for i in range(self.num_mask_tokens)
                    ]
                )
                
        # *** New: For the EMBEDDING_HEAD_MLP mode, add heads for bandwidth (sigma) and offsets ***
        if self.instance_id_mode == InstanceIdMode.EMBEDDING_HEAD_MLP:
            self.pred_bandwidth_head = nn.Conv2d(self.output_feature_dim, 1, kernel_size=1)
            self.pred_offset_head = nn.Conv2d(self.output_feature_dim, 2, kernel_size=1)
        else:
            self.pred_bandwidth_head = None
            self.pred_offset_head = None
            
        # Compute timer events
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
    
    def process_embeddings(self, embedding_list):
        """Process a list of instance embeddings to create a single embedding volume.
        
        Args:
            embedding_list: List of tensors, each with shape [b, c, h, w]
            
        Returns:
            Tensor with shape [b, c, h, w] representing the combined embedding
        """
        #print("len embedding_list", len(embedding_list), embedding_list[0].shape)
        # If we only have one embedding, just return it
        if len(embedding_list) == 1:
            print("Returning just one embed")
            return embedding_list[0]
            
        # Create the aggregator if it doesn't exist
        if not hasattr(self, 'embedding_aggregator'):
            b, c, h, w = embedding_list[0].shape
            self.embedding_aggregator = SpatialEmbeddingAggregator(
                embedding_dim=c,
                num_heads=8,
                dropout=0.1
            ).to(embedding_list[0].device)
        
        # Use the aggregator to combine embeddings
        return self.embedding_aggregator(embedding_list)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
          torch.Tensor: batched SAM token for mask output
        """
        
        self.start.record()
        masks, iou_pred, mask_tokens_out, object_score_logits, instance_embeddings = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]  # [b, 3, c] shape
        else:
            # Take the mask output token. Here we *always* use the token for single mask output.
            # At test time, even if we track after 1-click (and using multimask_output=True),
            # we still take the single mask token here. The rationale is that we always track
            # after multiple clicks during training, so the past tokens seen during training
            # are always the single mask token (and we'll let it be the object-memory token).
            sam_tokens_out = mask_tokens_out[:, 0:1]  # [b, 1, c] shape


        self.end.record()
        torch.cuda.synchronize()
        encoder_time = self.start.elapsed_time(self.end)  # milliseconds
        #print("mask decoder forward run time in ms: ", encoder_time)

        # Check gradients flowing to embeddings
        #print(f"Dual output requires grad: {instance_embeddings.requires_grad}")

        # Prepare output
        return masks, iou_pred, sam_tokens_out, object_score_logits, instance_embeddings

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight,
                    self.iou_token.weight,
                    self.mask_tokens.weight,
                    self.embedding_tokens.weight,
                ],
                dim=0,
            )
            s = 1
        else:
            output_tokens = torch.cat(
                [self.iou_token.weight, self.mask_tokens.weight, self.embedding_tokens.weight], dim=0
            )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        src = src + dense_prompt_embeddings
        assert (
            image_pe.size(0) == 1
        ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]
        embedding_tokens_out = hs[:,  : (s + 1 + self.num_mask_tokens + 1), :]
        
        start_idx = (s + 1 + self.num_mask_tokens)
        end_idx = start_idx + self.num_embedding_tokens
        embedding_tokens_out = hs[:, start_idx : end_idx, :]
        #print("shape of embed tokens out: ", embedding_tokens_out.shape, "hs shape", hs.shape)

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        # *** New: Generate an instance ID map using the new head ***
        pred_bandwidth = None
        pred_offsets = None
        match self.instance_id_mode:
            case InstanceIdMode.DIRECT:
                instance_id_map = self.instance_id_head(upscaled_embedding)  # (B, 1, h, w)
            case InstanceIdMode.EMBEDDING_HEAD_CNN:
                # Use instance embeddings to predict instance ID map
                instance_id_map = self.instance_embedding_head(upscaled_embedding)  # (B, embedding_dim, h, w)
            case InstanceIdMode.EMBEDDING_HEAD_MLP:
                embedding_hyper_in_list: List[torch.Tensor] = []
                # Process each embedding token through the hypernetwork
                for j in range(self.num_embedding_tokens):
                    embedding_hyper_in_list.append(
                        self.embedding_hypernetworks_mlps[0](embedding_tokens_out[:, j, :])
                    )
                embedding_hyper_in = torch.stack(embedding_hyper_in_list, dim=1)
                
                # Create instance embeddings using hypernetworks
                instance_embedding_list = []
                b, c, h, w = upscaled_embedding.shape
                
                for i in range(self.num_embedding_tokens):
                    # Get hypernetwork weights and ensure compatible dtype
                    hyper_weights = embedding_hyper_in[:, i]  # [b, embedding_dim]
                    
                    # Create a projection that outputs a c×c transformation matrix
                    if not hasattr(self, f'hyper_proj_{i}'):
                        self.register_module(
                            f'hyper_proj_{i}',
                            nn.Linear(hyper_weights.shape[1], c*c).to(device=upscaled_embedding.device, dtype=upscaled_embedding.dtype)
                        )
                    
                    # Project to get a full transformation matrix
                    proj = getattr(self, f'hyper_proj_{i}')
                    projected_weights = proj(hyper_weights.to(dtype=upscaled_embedding.dtype))
                    
                    # Reshape to create a transformation matrix for each batch item
                    transformation_matrix = projected_weights.view(b, c, c)  # [b, c, c]
                    
                    # Apply the transformation to all pixels while preserving the channel dimension
                    reshaped_features = upscaled_embedding.reshape(b, c, -1)  # [b, c, h*w]
                    transformed_features = torch.bmm(transformation_matrix, reshaped_features)  # [b, c, h*w]
                    
                    # Reshape back to spatial dimensions, preserving all c channels
                    base_embedding = transformed_features.view(b, c, h, w)  # [b, c, h, w]
                    
                    instance_embedding_list.append(base_embedding)
                    #print("base emb shape", base_embedding.shape)

                # Use the first token's embedding
                instance_embeddings = self.process_embeddings(instance_embedding_list)  # [b, embed_dim, h, w]
                
                # Use the first token's embeddings (consistent with taking first mask)
                #instance_embeddings = embeddings[:, 0, :, :] # [b, embed_dim, h, w]
                
                # Combine information from all token embeddings using attention or weighted sum
                #instance_embeddings = self.token_fusion(embeddings)  # [b, embed_dim, h, w]
                
                # Reshape and normalize embeddings (L2 normalization scales all vector to same size so only direction matters)
                instance_embeddings = F.normalize(instance_embeddings, p=2, dim=1)
                #print("instance_embeddings requries grad:", instance_embeddings.requires_grad, "instance_embeddings shape", instance_embeddings.shape)
                # *** New: Predict bandwidth and offsets ***
                #pred_bandwidth = self.pred_bandwidth_head(upscaled_embedding)  # (B, 1, H, W)
                #pred_offsets = self.pred_offset_head(upscaled_embedding)        # (B, 2, H, W)

            case InstanceIdMode.EMBEDDING_SHARED_MLP:
                instance_embeddings = []
                for i in range(self.num_mask_tokens):
                    # Apply hypernetwork weights to get features, then project to embedding space
                    features_i = (hyper_in[:, i] @ upscaled_embedding.view(b, c, h * w)).view(b, 1, h, w)
                    embeddings_i = self.instance_id_head[i](features_i.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                    instance_embeddings.append(embeddings_i)
                instance_embeddings = torch.cat(instance_embeddings, dim=1)
                
        # Standard mask prediction using hypernetworks
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            # Obj scores logits - default to 10.0, i.e. assuming the object is present, sigmoid(10)=1
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        # *** New: For final output, select the primary mask token as binary mask
        # Here we take the first channel as the binary mask output.
        binary_mask = masks[:, 0:1, :, :]

        match self.instance_id_mode:
            case InstanceIdMode.DIRECT:
                # Combine binary mask and instance ID map to form a dual-channel output.
                instance_embeddings = instance_id_map  # shape: (B, 1, h, w)
            case InstanceIdMode.EMBEDDING_HEAD_CNN| InstanceIdMode.EMBEDDING_HEAD_MLP:
                pass
            case _:
                raise NotImplementedError(f"InstanceIdMode {self.instance_id_mode} not implemented")
                
        return masks, iou_pred, mask_tokens_out, object_score_logits, instance_embeddings#, pred_bandwidth, pred_offsets

    def _get_stability_scores(self, mask_logits):
        """
        Compute stability scores of the mask logits based on the IoU between upper and
        lower thresholds.
        """
        mask_logits = mask_logits.flatten(-2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
        stability_scores = torch.where(area_u > 0, area_i / area_u, 1.0)
        return stability_scores

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        """
        When outputting a single mask, if the stability score from the current single-mask
        output (based on output token 0) falls below a threshold, we instead select from
        multi-mask outputs (based on output token 1~3) the mask with the highest predicted
        IoU score. This is intended to ensure a valid mask for both clicking and tracking.
        """
        # The best mask from multimask output tokens (1~3)
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(
            multimask_iou_scores.size(0), device=all_iou_scores.device
        )
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_logits = best_multimask_logits.unsqueeze(1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)

        # The mask from singlemask output token 0 and its stability score
        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        # Dynamically fall back to best multimask output upon low stability scores.
        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits),
            singlemask_logits,
            best_multimask_logits,
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out
