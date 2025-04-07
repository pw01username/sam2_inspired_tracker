# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List

import os
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from training.trainer import CORE_LOSS_KEY

from training.utils.distributed import get_world_size, is_dist_avail_and_initialized

from training.utils.visualize import visualize_frame, quick_visualize_mask, visualize_4d_tensor

def dice_loss(inputs, targets, num_objects, loss_on_multimask=False):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        Dice loss tensor
    """
    inputs = inputs.sigmoid()
    if loss_on_multimask:
        # inputs and targets are [N, M, H, W] where M corresponds to multiple predicted masks
        assert inputs.dim() == 4 and targets.dim() == 4
        # flatten spatial dimension while keeping multimask channel dimension
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        numerator = 2 * (inputs * targets).sum(-1)
    else:
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects

def diversity_loss(predictions, targets, threshold=0.5, eps=1e-5):
    """
    Penalize similarity between different mask channels only for predicted objects
    within target regions.
    
    Args:
        predictions: tensor of shape [B, N, H, W] - predicted logits or probabilities
        targets: tensor of shape [B, N, H, W] - ground truth masks
        threshold: probability threshold to consider a pixel as foreground
        eps: small value to avoid division by zero
    
    Returns:
        loss tensor of shape [B]
    """
    # Convert logits to probabilities if needed
    if predictions.max() > 1.0 or predictions.min() < 0.0:
        probs = torch.sigmoid(predictions)
    else:
        probs = predictions
        
    B, N, H, W = probs.shape
    
    # Flatten spatial dimensions
    probs_flat = probs.view(B, N, -1)  # [B, N, H*W]
    targets_flat = targets.view(B, N, -1)  # [B, N, H*W]
    
    # Create target regions mask - pixels where any target channel has an object
    target_regions = (targets_flat > 0.5).any(dim=1, keepdim=True)  # [B, 1, H*W]
    
    # Calculate overlap only for pixels where at least one channel predicts an object
    # AND where there's at least one target object
    cosine_sim_matrix = torch.zeros(B, N, N, device=probs.device)
    
    for b in range(B):
        for i in range(N):
            for j in range(i+1, N):  # Only calculate upper triangle (symmetric matrix)
                # Only consider pixels where either channel predicts an object
                # AND are within target regions
                pred_pixels = ((probs_flat[b, i] > threshold) | (probs_flat[b, j] > threshold))
                active_pixels = pred_pixels & target_regions[b, 0]
                
                # Skip if no active pixels
                if not active_pixels.any():
                    continue
                
                # Calculate intersection over regions where either predicts an object
                intersection = (probs_flat[b, i] * probs_flat[b, j] * active_pixels).sum()
                union = ((probs_flat[b, i] + probs_flat[b, j]) * active_pixels).sum() - intersection + eps
                
                # IoU-like metric between these two channels
                overlap = intersection / union
                
                cosine_sim_matrix[b, i, j] = overlap
                cosine_sim_matrix[b, j, i] = overlap  # Symmetric
    
    # Sum the upper triangle of the matrix (excluding diagonal)
    mask = torch.triu(torch.ones(N, N, device=probs.device), diagonal=1)
    channel_overlaps = (cosine_sim_matrix * mask.unsqueeze(0)).sum(dim=(1, 2))
    
    # Normalize by number of channel pairs
    num_pairs = (N * (N - 1)) / 2
    return channel_overlaps / (num_pairs + eps)

def sigmoid_focal_loss(
    inputs,
    targets,
    num_objects,
    alpha: float = 0.25,
    gamma: float = 2,
    loss_on_multimask=False,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        focal loss tensor
    """
    prob = inputs.sigmoid()
    visualize_4d_tensor(prob.float(), "prob_sigmoid.png")

    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if loss_on_multimask:
        # loss is [N, M, H, W] where M corresponds to multiple predicted masks
        #assert loss.dim() == 4
        #return loss.flatten(2).mean(-1) / num_objects  # average over spatial dims

        # For multi-object segmentation, we compute loss per object (channel)
        # and return loss for each object separately
        # Average over spatial dimensions but keep object dimension
        # loss is [B, N, H, W] -> [B, N]
        div_loss = diversity_loss(inputs, targets)
        diversity_weight = 0.001
        print("----", loss.flatten(2).shape, loss.flatten(2).mean(-1).shape)
        visualize_4d_tensor(loss.float(), "loss.png")
        visualize_4d_tensor(ce_loss.float(), "loss_ce.png")
        visualize_4d_tensor((1-p_t).float(), "p_t.png")

        # Average over spatial dimensions but keep batch dimension
        focal_loss_term = loss.flatten(2).mean(-1) / num_objects
        
        # Add diversity penalty (already at batch level)
        return focal_loss_term + diversity_weight * div_loss

    return loss.mean(1).sum() / num_objects

def debug_gradients(tensor, name):
    """Helper function to debug gradient flow"""
    if isinstance(tensor, torch.Tensor):
        print(f"{name}: requires_grad={tensor.requires_grad}, has_grad_fn={tensor.grad_fn is not None}")
        if tensor.grad_fn is not None:
            print(f"  grad_fn type: {type(tensor.grad_fn).__name__}")
    else:
        print(f"{name} is not a tensor: {type(tensor)}")

def iou_loss(
    inputs, targets, pred_ious, num_objects, loss_on_multimask=False, use_l1_loss=False
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        pred_ious: A float tensor containing the predicted IoUs scores per mask
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
        use_l1_loss: Whether to use L1 loss is used instead of MSE loss
    Returns:
        IoU loss tensor
    """
    assert inputs.dim() == 4 and targets.dim() == 4
    pred_mask = inputs.flatten(2) > 0
    gt_mask = targets.flatten(2) > 0
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)

    #print(f"DEBUG - pred_ious shape: {pred_ious.shape}, actual_ious shape: {actual_ious.shape}")
    
    # Expand pred_ious to match actual_ious if needed
    if pred_ious.shape[1] != actual_ious.shape[1]:
        print(f"Expanding pred_ious from {pred_ious.shape} to match actual_ious {actual_ious.shape}")
        pred_ious = pred_ious.expand(-1, actual_ious.shape[1])
        #print(f"New pred_ious shape: {pred_ious.shape}")

    if use_l1_loss:
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
    else:
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


class MultiStepMultiMasksAndIous(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
    ):
        """
        This class computes the multi-step multi-mask and IoU losses.
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            focal_alpha: alpha for sigmoid focal loss
            focal_gamma: gamma for sigmoid focal loss
            supervise_all_iou: if True, back-prop iou losses for all predicted masks
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
            pred_obj_scores: if True, compute loss for object scores
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
        """

        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores

        # Viz
        self.iteration = 0

    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor):
        assert len(outs_batch) == len(targets_batch)
        num_objects = torch.tensor(
            (targets_batch.shape[1]), device=targets_batch.device, dtype=torch.float
        )  # Number of objects is fixed within a batch
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objects)
        num_objects = torch.clamp(num_objects / get_world_size(), min=1).item()

        losses = defaultdict(int)
        for outs, targets in zip(outs_batch, targets_batch):
            cur_losses = self._forward(outs, targets, num_objects)
            for k, v in cur_losses.items():
                losses[k] += v

        # Visualize masks
        if False:
            # Save visualization only on the main process in distributed training
            if not is_dist_avail_and_initialized() or torch.distributed.get_rank() == 0:
                # Create a base directory for this iteration
                base_save_dir = os.path.join('mask_drawings', f"iter_{self.iteration}")
                os.makedirs(base_save_dir, exist_ok=True)
                
                # Process each batch item (frame)
                for batch_idx in range(len(outs_batch)):
                    #print("batch length: ", len(targets_batch))

                    if batch_idx < len(outs_batch):
                        # Create a directory for this specific frame
                        frame_save_dir = os.path.join(base_save_dir, f"frame_{batch_idx}")
                        os.makedirs(frame_save_dir, exist_ok=True)
                        
                        # Get the data for this frame
                        outs = outs_batch[batch_idx]
                        targets = targets_batch[batch_idx]

                        #visualize_4d_tensor(outs, f"outs_{self.iteration}.png")
                        #visualize_4d_tensor(targets, f"targets_{self.iteration}.png")

                        #print(f"Batch shape, targets:", len(outs["multistep_pred_multimasks_high_res"]), outs["multistep_pred_multimasks_high_res"][0].shape, ", preds: ", targets.shape)
                        
                        # Visualize this frame
                        # visualize_frame(
                        #     outs,
                        #     targets,
                        #     self.iteration,
                        #     frame_save_dir,
                        #     batch_idx
                        # )
                
        self.iteration += 1
        return losses

    def _forward(self, outputs: Dict, targets: torch.Tensor, num_objects):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        and also the MAE or MSE loss between predicted IoUs and actual IoUs.

        Here "multistep_pred_multimasks_high_res" is a list of multimasks (tensors
        of shape [N, M, H, W], where M could be 1 or larger, corresponding to
        one or multiple predicted masks from a click.

        We back-propagate focal, dice losses only on the prediction channel
        with the lowest focal+dice loss between predicted mask and ground-truth.
        If `supervise_all_iou` is True, we backpropagate ious losses for all predicted masks.
        """
        # original sam2 does this, but we need to unsqueeze on 0 dim instead
        #target_masks = targets.unsqueeze(1).float()
        #assert target_masks.dim() == 4  # [N, 1, H, W]
        
        if targets.dim() == 3:
            # Add batch dimension if not present
            target_masks = targets.unsqueeze(0).float()
        else:
            # If targets already have batch dimension [B, N, H, W]
            target_masks = targets.float()
        assert target_masks.dim() == 4  # [B, N, H, W]
        
        src_masks_list = outputs["multistep_pred_multimasks_high_res"]
        ious_list = outputs["multistep_pred_ious"]
        object_score_logits_list = outputs["multistep_object_score_logits"]
        #print("len src and iou masks", len(src_masks_list), len(ious_list), src_masks_list[0].shape, ious_list[0].shape)
        assert len(src_masks_list) == len(ious_list)
        assert len(object_score_logits_list) == len(ious_list)

        # accumulate the loss over prediction steps
        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0}
        for src_masks, ious, object_score_logits in zip(
            src_masks_list, ious_list, object_score_logits_list
        ):
            self._update_losses(
                losses, src_masks, target_masks, ious, num_objects, object_score_logits
            )
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        
        return losses

    def _update_losses(
        self, losses, src_masks, target_masks, ious, num_objects, object_score_logits
    ):
        debug_gradients(src_masks, "src_masks")
        debug_gradients(target_masks, "target_masks")
        debug_gradients(ious, "ious")
        
        visualize_4d_tensor(src_masks, f"loss_viz/src_predicted_mask_{0}.png")
        visualize_4d_tensor(target_masks, f"loss_viz/target_masks_{0}.png")

        B, N, H, W = src_masks.shape  # Batch size, number of predicted masks, height, width
        M = target_masks.shape[1]     # Number of ground truth masks
        #print("batch: ", B, target_masks.shape)

        # Convert logits to probabilities for IoU calculation
        src_probs = torch.sigmoid(src_masks)
        
        # Compute IoU between all pred and target masks for this batch item
        iou_matrix = torch.zeros((N, M), device=src_masks.device)
        
        # Calculate IoU matrix
        for n in range(N):
            for m in range(M):
                pred_mask = src_probs[:, n] > 0  # Binarize prediction
                gt_mask = target_masks[:, m] > 0  # Binarize ground truth
                
                # Calculate intersection and union
                intersection = (pred_mask & gt_mask).sum()
                union = (pred_mask | gt_mask).sum()
                
                if union > 0:
                    iou_matrix[n, m] = intersection.float() / union.float()
        
        # Get optimal matching using greedy approach
        # This rearranges target_masks to match order of src_masks
        matched_indices = []
        matched_gt_indices = list(range(M))
        
        for n in range(N):
            if not matched_gt_indices:
                break
                
            # Find best match among remaining ground truth masks
            best_iou = -1
            best_m = -1
            best_m_idx = -1
            
            for idx, m in enumerate(matched_gt_indices):
                if iou_matrix[n, m] > best_iou:
                    best_iou = iou_matrix[n, m]
                    best_m = m
                    best_m_idx = idx
            
            # Only match if IoU is above threshold
            if best_iou > 0:
                matched_indices.append((n, best_m))
                # Remove this ground truth mask from available matches
                matched_gt_indices.pop(best_m_idx)
        
        # Create new rearranged target masks based on matching
        rearranged_targets = torch.zeros_like(target_masks)
        
        # For each matched pair, put the target mask in the corresponding prediction slot
        for pred_idx, gt_idx in matched_indices:
            rearranged_targets[0, pred_idx] = target_masks[0, gt_idx]

        target_masks = rearranged_targets
        #visualize_4d_tensor(target_masks2, f"loss_viz/target_masks_retarget{0}.png")
        #print("matched_indices", matched_indices)
        # -------- 

        # get focal, dice and iou loss on all output masks in a prediction step
        loss_multimask = sigmoid_focal_loss(
            src_masks,
            target_masks,
            num_objects,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            loss_on_multimask=True,
        )

        #print("sigmoid_focal_loss: ", loss_multimask)

        obj_presence = torch.any(target_masks > 0, dim=(2,3)).float()
        
        loss_multidice = dice_loss(
            src_masks, target_masks, num_objects, loss_on_multimask=True
        )

        if not self.pred_obj_scores or 1 == 1:
            loss_class = torch.tensor(
                0.0, dtype=loss_multimask.dtype, device=loss_multimask.device
            )
            target_obj = torch.ones(
                loss_multimask.shape[0],
                1,
                dtype=loss_multimask.dtype,
                device=loss_multimask.device,
            )
        else:
            loss_class = sigmoid_focal_loss(
                object_score_logits,
                obj_presence,
                num_objects,
                alpha=self.focal_alpha_obj_score,
                gamma=self.focal_gamma_obj_score,
            )
            
            # target_obj = torch.any((target_masks[:, :3] > 0).flatten(1), dim=-1)[
            #     ..., None
            # ].float()
            # print("tar", target_obj.shape, object_score_logits.shape)
            # loss_class = sigmoid_focal_loss(
            #     object_score_logits[:, 0:1],
            #     target_obj,
            #     num_objects,
            #     alpha=self.focal_alpha_obj_score,
            #     gamma=self.focal_gamma_obj_score,
            # )

        loss_multiiou = iou_loss(
            src_masks,
            target_masks,
            ious,
            num_objects,
            loss_on_multimask=True,
            use_l1_loss=self.iou_use_l1_loss,
        )

        assert loss_multimask.dim() == 2
        assert loss_multidice.dim() == 2
        assert loss_multiiou.dim() == 2
        # Original code selected best of multimasks, we predict different objects in each multimask channel now.
        # if loss_multimask.size(1) > 1:
        #     # take the mask indices with the smallest focal + dice loss for back propagation
        #     loss_combo = (
        #         loss_multimask * self.weight_dict["loss_mask"]
        #         + loss_multidice * self.weight_dict["loss_dice"]
        #     )
        #     best_loss_inds = torch.argmin(loss_combo, dim=-1)
        #     batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
        #     loss_mask = loss_multimask[batch_inds, best_loss_inds].unsqueeze(1)
        #     loss_dice = loss_multidice[batch_inds, best_loss_inds].unsqueeze(1)
        #     # calculate the iou prediction and slot losses only in the index
        #     # with the minimum loss for each mask (to be consistent w/ SAM)
        #     if self.supervise_all_iou:
        #         loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)
        #     else:
        #         loss_iou = loss_multiiou[batch_inds, best_loss_inds].unsqueeze(1)
        # else:
        #     loss_mask = loss_multimask
        #     loss_dice = loss_multidice
        #     loss_iou = loss_multiiou
        
        # Therefore select the multimask loss as loss
        loss_mask = loss_multimask
        loss_dice = loss_multidice
        loss_iou = loss_multiiou

        # backprop focal, dice and iou loss only if obj present
        # loss_mask = loss_mask * target_obj
        # loss_dice = loss_dice * target_obj
        # loss_iou = loss_iou * target_obj

        #loss_mask = loss_mask * target_obj.unsqueeze(-1) if target_obj.dim() < loss_mask.dim() else loss_mask * target_obj
        #loss_dice = loss_dice * target_obj.unsqueeze(-1) if target_obj.dim() < loss_dice.dim() else loss_dice * target_obj
        #loss_iou = loss_iou * target_obj.unsqueeze(-1) if target_obj.dim() < loss_iou.dim() else loss_iou * target_obj

        # sum over batch dimension (note that the losses are already divided by num_objects)
        losses["loss_mask"] += loss_mask.sum()
        losses["loss_dice"] += loss_dice.sum()
        losses["loss_iou"] += loss_iou.sum()
        losses["loss_class"] += loss_class

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight

        return reduced_loss
