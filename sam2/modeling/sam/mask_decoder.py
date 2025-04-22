# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type

import torch
from torch import nn

from sam2.modeling.sam2_utils import LayerNorm2d, MLP

from training.utils.visualize import visualize_frame, quick_visualize_mask, visualize_4d_tensor

class MaskTokenCommunication(nn.Module):
    """
    Module for inter-object communication between mask tokens.
    Allows mask tokens to attend to each other and develop awareness
    of other potential objects in the scene.
    """
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        # Add an MLP after attention for more complex relationships
        self.mlp = MLP(dim, dim * 2, dim, 2)
        
    def forward(self, x):
        # x shape: [batch_size, num_tokens, dim]
        normed_x = self.norm(x)
        attended_x, _ = self.attention(normed_x, normed_x, normed_x)
        return x + self.mlp(attended_x)

class MaskCompetition(nn.Module):
    """
    Module that creates competition between masks to reduce overlap.
    Makes mask outputs compete for spatial regions through a softmax-like mechanism.
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, masks):
        # masks shape: [batch_size, num_masks, height, width]
        # Apply softmax across the mask dimension to create competition
        # Higher temperature makes the competition softer
        masks_exp = torch.exp(masks / self.temperature)
        masks_sum = masks_exp.sum(dim=1, keepdim=True)
        normalized_masks = masks_exp / (masks_sum + 1e-6)
        return normalized_masks

class PromptMaskCrossAttention(nn.Module):
    """
    Module for cross-attention between mask tokens and individual mask prompts.
    This helps each mask token focus on the correct object from the input prompts.
    """
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.norm_tokens = nn.LayerNorm(embed_dim)
        self.norm_prompts = nn.LayerNorm(embed_dim)
        
        # Cross-attention from mask tokens to prompts
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        
        # MLP for processing the attended output
        self.mlp = MLP(embed_dim, embed_dim * 2, embed_dim, 2)
        
    def forward(self, mask_tokens, prompt_embeddings):
        """
        Arguments:
            mask_tokens: [batch_size, num_masks, embed_dim]
            prompt_embeddings: list of [batch_size, embed_dim, h, w] tensors
        
        Returns:
            Updated mask tokens with prompt information
        """
        B, N, C = mask_tokens.shape
        
        # Normalize the mask tokens - this doesn't modify the original
        normed_tokens = self.norm_tokens(mask_tokens)
        
        # Track accumulated updates for all tokens
        accumulated_updates = torch.zeros_like(mask_tokens)
        
        # Process each prompt embedding (one per object)
        for i, prompt_emb in enumerate(prompt_embeddings):
            if i >= N:  # Skip if we have more prompts than mask tokens
                break
                
            # Reshape spatial prompt embedding to sequence format for attention
            # Convert from [B, C, h, w] to [B, h*w, C]
            flattened_prompt = prompt_emb.flatten(2).permute(0, 2, 1)
            normed_prompt = self.norm_prompts(flattened_prompt)
            
            # Get the corresponding mask token for this prompt
            # We only need the i-th token since each token corresponds to one prompt
            token_query = normed_tokens[:, i:i+1]  # Shape: [B, 1, C]
            
            # Apply cross-attention: token attends to all spatial locations in the prompt
            attended_output, _ = self.cross_attention(
                token_query,     # Query: single token [B, 1, C]
                normed_prompt,   # Key: all spatial locations [B, h*w, C]
                normed_prompt    # Value: all spatial locations [B, h*w, C]
            )
            
            # Store the update for this specific token
            accumulated_updates[:, i:i+1] = attended_output
        
        # Apply the accumulated updates
        updated_tokens = mask_tokens + accumulated_updates
        
        # Apply the MLP to further process the tokens
        final_tokens = updated_tokens + self.mlp(updated_tokens)
        
        return final_tokens

class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int,
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
        use_multimask_token_for_obj_ptr: bool = True, #False,
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

        # shared_output_hypernetwork = MLP(
        #     transformer_dim, 
        #     transformer_dim, 
        #     transformer_dim // 8 * self.num_mask_tokens, 
        #     3
        # )

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
            # for original:
            # self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            # if pred_obj_scores_mlp:
            #     self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

            # for multi obj
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, self.num_mask_tokens, 3)
            else:
                self.pred_obj_score_head = nn.Linear(transformer_dim, self.num_mask_tokens)

        # When outputting a single mask, optionally we can dynamically fall back to the best
        # multimask output token if the single mask output token gives low stability scores.
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh


        # Add cross-attention between mask tokens and mask prompts
        self.prompt_mask_attention = PromptMaskCrossAttention(transformer_dim)

        # Add inter-object communication module
        self.mask_token_communication = MaskTokenCommunication(transformer_dim)
        
        # Add mask competition module
        self.mask_competition = MaskCompetition(temperature=0.5)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
        individual_mask_embeddings: Optional[List[torch.Tensor]] = None,  # New parameter for explicit prompt attention
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
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
            individual_mask_embeddings=individual_mask_embeddings,
        )

        # Select the correct mask or masks for output
        # if multimask_output:
        #     masks = masks[:, 1:, :, :]
        #     iou_pred = iou_pred[:, 1:]
        # elif self.dynamic_multimask_via_stability and not self.training:
        #     masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        # else:
        #     masks = masks[:, 0:1, :, :]
        #     iou_pred = iou_pred[:, 0:1]

        # if multimask_output and self.use_multimask_token_for_obj_ptr:
        #     sam_tokens_out = mask_tokens_out[:, 1:]  # [b, 3, c] shape
        # else:
        #     # Take the mask output token. Here we *always* use the token for single mask output.
        #     # At test time, even if we track after 1-click (and using multimask_output=True),
        #     # we still take the single mask token here. The rationale is that we always track
        #     # after multiple clicks during training, so the past tokens seen during training
        #     # are always the single mask token (and we'll let it be the object-memory token).
        #     sam_tokens_out = mask_tokens_out[:, 0:1]  # [b, 1, c] shape
        sam_tokens_out = mask_tokens_out

        #visualize_4d_tensor(masks.float(), f"loss_viz/mask_decoder_masks_iter_{0}.png")
        
        # Prepare output
        return masks, iou_pred, sam_tokens_out, object_score_logits

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
        individual_mask_embeddings: Optional[List[torch.Tensor]] = None,
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
                ],
                dim=0,
            )
            s = 1
        else:
            output_tokens = torch.cat(
                [self.iou_token.weight, self.mask_tokens.weight], dim=0
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
        src = src #+ dense_prompt_embeddings
        #visualize_4d_tensor(src.float(), "masks_memenc/mask decoder src.png")
        assert (
            image_pe.size(0) == 1
        ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]

        # Apply cross-attention between mask tokens and individual mask prompts
        # This helps each mask token focus on its corresponding prompt
        if individual_mask_embeddings is not None and len(individual_mask_embeddings) > 0:
            mask_tokens_out = self.prompt_mask_attention(
                mask_tokens_out, individual_mask_embeddings
            )

        # Apply inter-object communication to the mask tokens
        # This allows each mask token to be aware of other mask tokens
        # and helps ensure better separation between objects
        mask_tokens_out = self.mask_token_communication(mask_tokens_out)

        #print("hs shape, s", hs.shape, s, mask_tokens_out.shape, torch.equal(mask_tokens_out[0, 0], mask_tokens_out[0, 1]))

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Apply mask competition to reduce overlap between masks
        if not self.training:
            masks = self.mask_competition(masks)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            # Obj scores logits - default to 10.0, i.e. assuming the object is present, sigmoid(10)=1
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)
        
        return masks, iou_pred, mask_tokens_out, object_score_logits

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
