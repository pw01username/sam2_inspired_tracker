# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Optional, Tuple, Type

import torch
from torch import nn

from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.modeling.sam2_utils import LayerNorm2d, MLP


class MaskRefiner(nn.Module):
    """
    Takes raw mask logits for M objects plus a feature map, and predicts
    a per-mask residual to sharpen boundaries under occlusion.
    """
    def __init__(self, mask_channels: int, feat_channels: int, hidden_dim: int = 128):
        super().__init__()
        in_ch = mask_channels + feat_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        # predict a residual for each mask channel
        self.conv3 = nn.Conv2d(hidden_dim, mask_channels, kernel_size=1)

    def forward(self, masks: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        # masks: raw logits [B, M, H, W]
        # feats: feature map [B, C, H, W]
        m_prob = masks.sigmoid()
        x = torch.cat([m_prob, feats], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        delta = self.conv3(x)
        # apply as a logit-space residual
        return masks + delta

class TokenCompetition(nn.Module):
    """
    Alternative implementation with learned attention-based suppression
    """
    
    def __init__(self, transformer_dim: int, num_mask_tokens: int = 4, num_heads: int = 4):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.num_mask_tokens = num_mask_tokens
        
        # Cross-attention for competition
        self.competition_attention = nn.ModuleList([
            nn.MultiheadAttention(
                transformer_dim, 
                num_heads, 
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_mask_tokens)
        ])
        
        # Gating network to blend original vs. competed tokens
        self.gate_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(transformer_dim * 2, transformer_dim),
                nn.ReLU(),
                nn.Linear(transformer_dim, 1),
                nn.Sigmoid()
            ) for _ in range(num_mask_tokens)
        ])
        
        self.norm = nn.LayerNorm(transformer_dim)
        
    def forward(
        self,
        mask_tokens_out: torch.Tensor,  # [B, T, D]
        img_ids: torch.Tensor           # [B]
    ) -> torch.Tensor:
        """
        Competition through attention mechanism
        """
        B, T, D = mask_tokens_out.shape
        updated_tokens = mask_tokens_out.clone()
        
        for slot_idx in range(T):
            for img_id in img_ids.unique():
                idx = (img_ids == img_id).nonzero(as_tuple=True)[0]
                G = idx.numel()
                if G <= 1:
                    continue
                
                # Get tokens for this slot in this image
                group_tokens = mask_tokens_out[idx, slot_idx, :]  # [G, D]
                group_tokens_expanded = group_tokens.unsqueeze(0)  # [1, G, D]
                
                # Self-attention with competition
                # Keys and values are all tokens, but we modify attention pattern
                competed_tokens, attention_weights = self.competition_attention[slot_idx](
                    query=group_tokens_expanded,
                    key=group_tokens_expanded,
                    value=group_tokens_expanded
                )
                competed_tokens = competed_tokens.squeeze(0)  # [G, D]
                
                # Compute gate values
                for i in range(G):
                    # How different is the competed token from original?
                    gate_input = torch.cat([
                        group_tokens[i], 
                        competed_tokens[i]
                    ], dim=0)
                    gate = self.gate_nets[slot_idx](gate_input)
                    
                    # Blend original and competed tokens
                    updated_tokens[idx[i], slot_idx] = (
                        gate * group_tokens[i] + 
                        (1 - gate) * competed_tokens[i]
                    )
        
        # Final normalization
        return self.norm(updated_tokens + mask_tokens_out)

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


        # 1) instantiate the SAME sinePE you use for image_pe,
        #   but now to encode an (object × token) grid:
        #   we want final PE dim == transformer_dim,
        #   and PositionEmbeddingSine takes num_pos_feats*2 == transformer_dim
        self.obj_token_pe = PositionEmbeddingSine(
            num_pos_feats=transformer_dim,
            normalize=True,
            scale=2 * math.pi,
        )

        # 2) a full TransformerEncoder layer for inter-object communication
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=1,                    # you can tune this
            dim_feedforward=transformer_dim * 4,
            activation="gelu",
            dropout=0.1,
            batch_first=True            # allow (B*, Seq, Dim)
        )
        self.inter_object_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4,
            norm=nn.LayerNorm(transformer_dim)
        )
        self.norm_inter_obj = nn.LayerNorm(transformer_dim)



        self.token_competition = TokenCompetition(
            transformer_dim=transformer_dim,
            num_mask_tokens=self.num_mask_tokens
        )


    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
        img_ids = None,
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
            img_ids=img_ids,
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
        img_ids = None,
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

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        
        
        # --- interobject PE + attention ---
        
        # mask_tokens_out: [B, T, C]
        # B, T, C = mask_tokens_out.shape
        # device = mask_tokens_out.device

        # # a) build a dummy tensor of shape [B, 1, B, T] so that
        # #    obj_token_pe.forward() will return a [B, C, B, T] map:
        # dummy = torch.zeros((B, 1, B, T), device=device)
        # pe_map = self.obj_token_pe(dummy)           # [B, C, B, T]

        # # b) we only want the diagonal slice along the objectaxis:
        # #    for each batchitem b, pick the PE row at index b.
        # #    First permute to [B, B, T, C]:
        # pe_map = pe_map.permute(0, 2, 3, 1)          # [B, H=B, W=T, C]
        # #    then index the H axis = objectindex:
        # idx = torch.arange(B, device=device)
        # pe_ot = pe_map[idx, idx, :, :]              # [B, T, C]

        # # c) add onto your mask tokens
        # x = mask_tokens_out + pe_ot                  # [B, T, C]

        # # d) merge into a single sequence of length B*T
        # x = x.view(1, B * T, C)                      # [1, B*T, C]

        # # e) run your interobject TransformerEncoder
        # x = self.inter_object_encoder(x)             # [1, B*T, C]

        # # f) unmerge back to [B, T, C] and residual+norm
        # x = x.view(B, T, C)
        # mask_tokens_out = self.norm_inter_obj(mask_tokens_out + x)
        
        # --- end interobject block ---

        if img_ids is not None:
            # mask_tokens_out: [N, T, C]; img_ids: [N] ↦ group tokens by image
            mask_tokens_out = self._cross_object_attention(mask_tokens_out, img_ids)

            mask_tokens_out = self.token_competition(mask_tokens_out, img_ids)
        else:
            print("WARNING: img_ids missing—skipping cross-object attention")


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

        return masks, iou_pred, mask_tokens_out, object_score_logits

    def _cross_object_attention(
        self,
        mask_tokens_out: torch.Tensor,  # [B, T, D]
        img_ids: torch.Tensor           # [B]
    ) -> torch.Tensor:

        # 1) Split off the “main” slot:
        main_tokens = mask_tokens_out[:, 0, :]  # [B, D]
        updated_main = main_tokens.clone()

        # 2) For each image group, mix only those primary tokens:
        for img_id in img_ids.unique():
            idx = (img_ids == img_id).nonzero(as_tuple=True)[0]
            G = idx.numel()
            if G <= 1:
                continue

            group = main_tokens[idx]         # [G, D]
            x = group.unsqueeze(0)           # → [1, G, D]

            # (optional) you can add a small learned obj-index embedding here,
            # but since these are unordered “objects” you can also skip PE altogether:
            #   pe = self.obj_index_embed(torch.arange(G, device=device))[None]
            #   x = x + pe

            # 3) Run your shared TransformerEncoder:
            x = self.inter_object_encoder(x)  # [1, G, D]
            x = x.squeeze(0)                  # [G, D]

            # 4) Write back with a block-level residual + norm:
            updated_main[idx] = self.norm_inter_obj(group + x)

        # 5) Put the updated main slot back into mask_tokens_out:
        mask_tokens_out[:, 0, :] = updated_main

        return mask_tokens_out


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
