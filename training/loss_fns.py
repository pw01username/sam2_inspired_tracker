# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import os
from typing import Dict, List
import math

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from training.trainer import CORE_LOSS_KEY

from training.utils.distributed import get_world_size, is_dist_avail_and_initialized
from training.utils.visualize_instance_segmentation import visualize_frame, visualize_instance_predictions


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
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if loss_on_multimask:
        # loss is [N, M, H, W] where M corresponds to multiple predicted masks
        assert loss.dim() == 4
        return loss.flatten(2).mean(-1) / num_objects  # average over spatial dims
    return loss.mean(1).sum() / num_objects


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
        if "loss_instance" not in self.weight_dict:
            self.weight_dict["loss_instance"] = 20.0  # Default weight for instance loss
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores
            
        # Add visualization parameters
        self.enable_visualization = False
        self.vis_save_dir = "./vis"
        self.vis_freq = 100
        self.iteration = 0
        
        if self.enable_visualization:
            os.makedirs(self.vis_save_dir, exist_ok=True)

    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor, instance_ids_batch: torch.Tensor):
        
        assert len(outs_batch) == len(targets_batch)
        assert len(outs_batch) == len(instance_ids_batch)
        num_objects = torch.tensor(
            (targets_batch.shape[1]), device=targets_batch.device, dtype=torch.float
        )  # Number of objects is fixed within a batch
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objects)
        num_objects = torch.clamp(num_objects / get_world_size(), min=1).item()

        losses = defaultdict(int)
        for outs, targets, instance_ids in zip(outs_batch, targets_batch, instance_ids_batch):
            cur_losses = self._forward(outs, targets, instance_ids, num_objects)
            for k, v in cur_losses.items():
                losses[k] += v
                
        
        # Add visualization after loss calculation
        if self.enable_visualization and self.iteration % self.vis_freq == 0:
            # Save visualization only on the main process in distributed training
            if not is_dist_avail_and_initialized() or torch.distributed.get_rank() == 0:
                try:
                    # Print batch shape information
                    #print(f"Batch shape: {targets_batch.shape}")
                    
                    # Create a base directory for this iteration
                    base_save_dir = os.path.join(self.vis_save_dir, f"iter_{self.iteration}")
                    os.makedirs(base_save_dir, exist_ok=True)
                    
                    # Process each batch item (frame)
                    for batch_idx in range(len(outs_batch)):
                        if batch_idx < len(outs_batch):
                            # Create a directory for this specific frame
                            frame_save_dir = os.path.join(base_save_dir, f"frame_{batch_idx}")
                            os.makedirs(frame_save_dir, exist_ok=True)
                            
                            # Get the data for this frame
                            outs = outs_batch[batch_idx]
                            targets = targets_batch[batch_idx]
                            instance_ids = instance_ids_batch[batch_idx]
                            
                            # Visualize this frame
                            visualize_frame(
                                outs,
                                targets,
                                instance_ids,
                                self.iteration,
                                frame_save_dir,
                                batch_idx
                            )
                            
                            #print(f"Visualization saved for frame {batch_idx} at iteration {self.iteration}")
                    
                    #print(f"All frame visualizations saved at iteration {self.iteration}")
                except Exception as e:
                    print(f"Visualization error at iteration {self.iteration}: {e}")
                    import traceback
                    traceback.print_exc()
    
            
        # Save visualization only on the main process in distributed training
        if not is_dist_avail_and_initialized() or torch.distributed.get_rank() == 0 and (self.enable_visualization and self.iteration % self.vis_freq == 0):
            try:
                # Create a base directory for this iteration
                base_save_dir = os.path.join(self.vis_save_dir, f"iter_{self.iteration}", "instance_visualizations")
                os.makedirs(base_save_dir, exist_ok=True)
                
                # Process each batch item (frame)
                for batch_idx in range(len(outs_batch)):
                    if batch_idx < len(outs_batch):
                        # Get the data for this frame
                        outs = outs_batch[batch_idx]
                        targets = targets_batch[batch_idx]
                        instance_ids = instance_ids_batch[batch_idx]
                        
                        # Create a directory for this specific frame
                        frame_save_dir = os.path.join(base_save_dir, f"frame_{batch_idx}")
                        os.makedirs(frame_save_dir, exist_ok=True)
                        
                        # Get the predicted masks
                        if "multistep_pred_multimasks_high_res" in outs and len(outs["multistep_pred_multimasks_high_res"]) > 0:
                            pred_masks = outs["multistep_pred_multimasks_high_res"][-1]  # Latest prediction
                            
                            # Get the predicted instance IDs
                            if "multistep_pred_instance_ids" in outs and len(outs["multistep_pred_instance_ids"]) > 0:
                                pred_instance_ids = outs["multistep_pred_instance_ids"][-1]  # Latest prediction
                                
                                # Call our new visualization function
                                visualize_instance_predictions(
                                    pred_masks,
                                    pred_instance_ids, 
                                    targets.unsqueeze(1).float(),  # [N, H, W] -> [N, 1, H, W]
                                    instance_ids,
                                    self.iteration,
                                    frame_save_dir,
                                    batch_idx
                                )
                        
                print(f"Instance ID visualizations saved at iteration {self.iteration}")
            except Exception as e:
                print(f"Instance visualization error at iteration {self.iteration}: {e}")
                import traceback
                traceback.print_exc()
    
        # Update iteration counter
        self.iteration += 1

        return losses

    def _forward(self, outputs: Dict, targets: torch.Tensor, instance_ids: torch.Tensor, num_objects):
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

        target_masks = targets.unsqueeze(1).float()
        assert target_masks.dim() == 4  # [N, 1, H, W]
        src_masks_list = outputs["multistep_pred_multimasks_high_res"]
        ious_list = outputs["multistep_pred_ious"]
        object_score_logits_list = outputs["multistep_object_score_logits"]

        pred_instance_ids_list = outputs["multistep_pred_instance_ids"]

        assert len(src_masks_list) == len(ious_list)
        assert len(object_score_logits_list) == len(ious_list)
        assert len(pred_instance_ids_list) == len(ious_list)

        # accumulate the loss over prediction steps
        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0, "loss_instance": 0}
        for src_masks, ious, object_score_logits, pred_instance_ids  in zip(
            src_masks_list, ious_list, object_score_logits_list, pred_instance_ids_list 
        ):
            self._update_losses(
                losses, src_masks, target_masks, ious, num_objects, object_score_logits, pred_instance_ids, instance_ids
            )
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        return losses

    def _update_losses(
        self, losses, src_masks, target_masks, ious, num_objects, object_score_logits, pred_instance_ids, target_instance_ids
    ):
        #print("target_instance_ids: ", target_instance_ids.shape)
        #print("pred_instance_ids: ", pred_instance_ids.shape)
        
        # Get the binary masks for foreground/background segmentation
        # Ensure target_masks has the right shape for comparisons
        if target_masks.shape[0] != src_masks.shape[0]:
            # Expand target_masks to match batch size of src_masks
            target_masks = target_masks.expand(src_masks.shape[0], -1, -1, -1)
        
        # Expand target_masks along mask dimension if needed
        if target_masks.shape[1] != src_masks.shape[1]:
            target_masks = target_masks.expand(-1, src_masks.shape[1], -1, -1)
        
        # Convert target_instance_ids to match pred_instance_ids shape if needed
        if target_instance_ids.dim() == 3:  # [B, H, W]
            target_instance_ids = target_instance_ids.unsqueeze(1)  # [B, 1, H, W]
        
        # Handle batch size mismatch for instance IDs
        if target_instance_ids.shape[0] != pred_instance_ids.shape[0]:
            target_instance_ids = target_instance_ids.expand(pred_instance_ids.shape[0], -1, -1, -1)
        
        # target_masks = target_masks.expand_as(src_masks)
        # get focal, dice and iou loss on all output masks in a prediction step
        loss_multimask = sigmoid_focal_loss(
            src_masks,
            target_masks,
            num_objects,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            loss_on_multimask=True,
        )
        loss_multidice = dice_loss(
            src_masks, target_masks, num_objects, loss_on_multimask=True
        )
        if not self.pred_obj_scores:
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
            target_obj = torch.any((target_masks[:, 0] > 0).flatten(1), dim=-1)[
                ..., None
            ].float()
            loss_class = sigmoid_focal_loss(
                object_score_logits,
                target_obj,
                num_objects,
                alpha=self.focal_alpha_obj_score,
                gamma=self.focal_gamma_obj_score,
            )

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
        if loss_multimask.size(1) > 1:
            # take the mask indices with the smallest focal + dice loss for back propagation
            loss_combo = (
                loss_multimask * self.weight_dict["loss_mask"]
                + loss_multidice * self.weight_dict["loss_dice"]
            )
            best_loss_inds = torch.argmin(loss_combo, dim=-1)
            batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
            loss_mask = loss_multimask[batch_inds, best_loss_inds].unsqueeze(1)
            loss_dice = loss_multidice[batch_inds, best_loss_inds].unsqueeze(1)
            # calculate the iou prediction and slot losses only in the index
            # with the minimum loss for each mask (to be consistent w/ SAM)
            if self.supervise_all_iou:
                loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)
            else:
                loss_iou = loss_multiiou[batch_inds, best_loss_inds].unsqueeze(1)
        else:
            loss_mask = loss_multimask
            loss_dice = loss_multidice
            loss_iou = loss_multiiou

        # backprop focal, dice and iou loss only if obj present
        loss_mask = loss_mask * target_obj
        loss_dice = loss_dice * target_obj
        loss_iou = loss_iou * target_obj

        # new: instance ID loss
        
        # Instance id embeddings, discriminative emb loss. Assuming pred_instance_ids is embeddings.
        loss_instance = self.instance_id_loss(pred_instance_ids, target_instance_ids, num_objects)
        if torch.isnan(loss_instance).any():
            print("NaN value as loss!")
            loss_instance += 0.01
        
        # L1
        #loss_instance = self.instance_id_loss_L1(pred_instance_ids, target_instance_ids, num_objects)
        #print("target instance ids: ", target_instance_ids)
        
        #self.debug_instance_map(pred_instance_ids.cpu().detach())
        #self.debug_instance_map(target_instance_ids.cpu())
        
        # sum over batch dimension (note that the losses are already divided by num_objects)
        losses["loss_mask"] += loss_mask.sum()
        losses["loss_dice"] += loss_dice.sum()
        losses["loss_iou"] += loss_iou.sum()
        losses["loss_class"] += loss_class
        losses["loss_instance"] += loss_instance.mean()
        #print("losses instance: ", losses["loss_instance"])

    def debug_instance_map(self, instance_map_tensor):
        # Move to CPU and convert to numpy
        instance_map = instance_map_tensor[0].numpy()  # Take first batch item
        
        # Print shape and statistics
        print(f"Instance map shape: {instance_map.shape}")
        print(f"Min value: {np.min(instance_map)}, Max value: {np.max(instance_map)}")
        
        # Print unique values and their counts
        unique_ids, counts = np.unique(instance_map, return_counts=True)
        for id_val, count in zip(unique_ids, counts):
            if id_val > 0:  # Only show non-background IDs
                print(f"Instance ID {id_val}: {count} pixels")
        
        # Print a small sample of the map (center region)
        b, h, w = instance_map.shape
        center_y, center_x = h // 2, w // 2
        sample_size = 10
        sample = instance_map[
            center_y - sample_size // 2:center_y + sample_size // 2,
            center_x - sample_size // 2:center_x + sample_size // 2
        ]
        print(f"Center {sample_size}x{sample_size} sample of instance map:")
        print(sample)

    def instance_id_loss(self, pred_embeddings, target_instance_ids, num_objects, delta_v=0.1, delta_d=2.0, reg_weight=0.01):
        """
        Direct-access implementation of discriminative loss with guaranteed non-zero values.
        
        Args:
            pred_embeddings: Embedding vectors of shape (B, E, H, W) where E is embedding dimension
            target_instance_ids: Ground truth instance ID map of shape (B, 1, H, W) or (B, H, W)
            num_objects: Number of objects (for normalization)
            delta_v: Variance margin for pull loss
            delta_d: Distance margin for push loss
            reg_weight: Regularization weight (increased to 0.01)
        """
        batch_size = pred_embeddings.size(0)
        #print("Batch size", pred_embeddings.shape)
        embedding_dim = pred_embeddings.size(1)
        device = pred_embeddings.device
        
        # Ensure target has the right shape
        if target_instance_ids.dim() == 4:
            target_instance_ids = target_instance_ids.squeeze(1)
        
        # Return value for empty batch
        if batch_size == 0 or num_objects == 0:
            return torch.ones(1, device=device, requires_grad=True) * 0.1

        # Initialize loss components directly within computational graph
        pull_losses = []
        push_losses = []
        reg_losses = []
        
        total_instances = 0
        
        # Process each sample in batch
        for b in range(batch_size):
            # Extract this batch item
            emb = pred_embeddings[b]  # (E, H, W)
            ids = target_instance_ids[b]  # (H, W)
            
            # Get unique instance IDs (excluding background)
            instance_ids = torch.unique(ids)
            instance_ids = instance_ids[instance_ids > 0]
            
            if len(instance_ids) == 0:
                # Create dummy loss for samples with no instances
                dummy_loss = torch.sum(emb) * 0.0 + 0.1
                pull_losses.append(dummy_loss)
                continue
            
            # Collect centers and features
            centers = []
            pixel_embeddings = []
            
            for inst_id in instance_ids:
                # Get mask for this instance
                mask = (ids == inst_id)
                
                # Skip tiny instances
                if mask.sum() < 10:
                    continue
                    
                # Extract coordinates where mask is True
                y_indices, x_indices = torch.where(mask)
                
                # Limit to 1000 random pixels for very large instances
                if len(y_indices) > 1000:
                    idx = torch.randperm(len(y_indices))[:1000]
                    y_indices = y_indices[idx]
                    x_indices = x_indices[idx]
                
                # Directly extract embeddings for these coordinates
                # Important: we need to keep gradients flowing here
                inst_pixel_embeddings = []
                for i in range(len(y_indices)):
                    y, x = y_indices[i], x_indices[i]
                    pixel_emb = emb[:, y, x]  # Extract embedding vector for this pixel
                    inst_pixel_embeddings.append(pixel_emb)
                
                # Skip if no embeddings extracted
                if not inst_pixel_embeddings:
                    continue
                    
                # Stack into a tensor
                inst_pixel_embeddings = torch.stack(inst_pixel_embeddings)  # (N_pixels, E)
                
                # Calculate center embedding (mean)
                center = torch.mean(inst_pixel_embeddings, dim=0)  # (E)
                
                centers.append(center)
                pixel_embeddings.append(inst_pixel_embeddings)
            
            # Skip if no valid centers found
            if not centers:
                # Create dummy loss
                dummy_loss = torch.sum(emb) * 0.0 + 0.1
                pull_losses.append(dummy_loss)
                continue
                
            total_instances += len(centers)
            centers_tensor = torch.stack(centers)  # (N_instances, E)
            
            # 1. Pull loss calculation: Force pixels to cluster around their centers
            batch_pull_loss = 0
            for i, (center, pixels) in enumerate(zip(centers, pixel_embeddings)):
                # For each instance:
                # Calculate distance from each pixel to center
                dists = torch.sqrt(torch.sum((pixels - center.unsqueeze(0))**2, dim=1) + 1e-6)
                
                # Apply minimum pull (even for pixels within delta_v)
                min_pull = torch.mean(dists**2) * 0.1
                
                # Apply hinge for pixels outside delta_v
                hinge_dists = torch.clamp(dists - delta_v, min=0.0)**2
                hinge_pull = torch.mean(hinge_dists)
                
                # Combine both
                instance_pull = min_pull + hinge_pull
                
                # Add directly to computational graph
                batch_pull_loss = batch_pull_loss + instance_pull
            
            # Normalize pull loss
            if len(centers) > 0:
                batch_pull_loss = batch_pull_loss / len(centers)
                # Add small constant to ensure non-zero
                batch_pull_loss = batch_pull_loss + 1e-6
                pull_losses.append(batch_pull_loss)
            
            # 2. Push loss calculation: Force centers apart from each other
            if len(centers) > 1:
                batch_push_loss = 0
                pair_count = 0
                
                for i in range(len(centers)):
                    for j in range(i+1, len(centers)):
                        # Calculate distance between centers
                        dist = torch.sqrt(torch.sum((centers[i] - centers[j])**2) + 1e-6)
                        
                        # Apply hinge
                        margin = 2.0 * delta_d
                        hinge_dist = torch.clamp(margin - dist, min=0.0)**2
                        
                        # Add directly to computational graph
                        batch_push_loss = batch_push_loss + hinge_dist
                        pair_count += 1
                
                # Normalize by pair count
                if pair_count > 0:
                    batch_push_loss = batch_push_loss / pair_count
                    push_losses.append(batch_push_loss)
            
            # 3. Regularization term: Keep centers at a reasonable magnitude
            # Calculate L2 norm of the centers
            center_norms = torch.norm(centers_tensor, p=2, dim=1)
            
            # Target norm of 1.0
            target_norm = torch.ones_like(center_norms)
            reg_loss = torch.mean((center_norms - target_norm)**2)
            
            # Add small constant to ensure non-zero
            reg_loss = reg_loss + 1e-4
            reg_losses.append(reg_loss)
        
        # Combine losses with appropriate weights
        # If we have pull losses, combine them
        if pull_losses:
            pull_loss = torch.mean(torch.stack(pull_losses))
        else:
            # Create dummy pull loss (detached from graph)
            pull_loss = torch.tensor(0.001, device=device)
        
        # If we have push losses, combine them
        if push_losses:
            push_loss = torch.mean(torch.stack(push_losses))
        else:
            # Create dummy push loss (detached from graph)
            push_loss = torch.tensor(0.0, device=device)
        
        # If we have reg losses, combine them
        if reg_losses:
            reg_loss = torch.mean(torch.stack(reg_losses))
        else:
            # Create dummy reg loss (detached from graph)
            reg_loss = torch.tensor(0.001, device=device)
        
        # Calculate weights
        pull_weight = 1.0
        push_weight = 0.05 / math.sqrt(embedding_dim)
        
        # This is key: we need to directly combine these tensors
        # to ensure gradient flow through the combined loss
        loss = pull_weight * pull_loss + push_weight * push_loss + reg_weight * reg_loss
        
        if torch.isnan(emb).any():
            print("NaN detected in embeddings!")
        if torch.isnan(centers_tensor).any():
            print("NaN detected in centers tensor!")
        if torch.isnan(dists).any():
            print("NaN detected in distance calculation!")

        # Debug output
        if self.training and is_dist_avail_and_initialized() and torch.distributed.get_rank() == 0:
            print(f"Embed dim: {embedding_dim}, "
                f"Instances: {total_instances}, "
                f"Pull: {pull_loss.item():.6f} (w={pull_weight:.1f}), "
                f"Push: {push_loss.item():.6f} (w={push_weight:.6f}), "
                f"Reg: {reg_loss.item():.6f} (w={reg_weight:.2f}), "
                f"Total: {loss.item():.6f}")
        
        # We need to ensure the loss has the proper gradient.
        # Instead of using torch.full, we'll return the scalar loss
        # expanded to batch size, directly preserving gradients.
        return loss.expand(batch_size) / num_objects
    
    def instance_id_loss_first(self, pred_embeddings, target_instance_ids, num_objects, delta_v=0.2, delta_d=1.5, reg_weight=0.001):
        """
        Discriminative loss for instance segmentation with embeddings.
        
        Args:
            pred_embeddings: Embedding vectors of shape (B, E, H, W) where E is embedding dimension
            target_instance_ids: Ground truth instance ID map of shape (B, 1, H, W) or (B, H, W)
            num_objects: Number of objects (for normalization)
            delta_v: Variance margin - pull loss margin
            delta_d: Distance margin - push loss margin
        
        Returns:
            Instance embedding loss (discriminative loss)
        """
        batch_size = pred_embeddings.size(0)
        #print("Input pred_embeddings shape:", pred_embeddings.shape)
        
        avg_instances = sum(len(torch.unique(target_instance_ids[i])[torch.unique(target_instance_ids[i]) > 0]) 
                    for i in range(batch_size)) / batch_size
        
        # Ensure target has the right shape - we need (B, H, W)
        if target_instance_ids.dim() == 4:  # (B, 1, H, W)
            target_instance_ids = target_instance_ids.squeeze(1)
        
        # Move embeddings to shape expected by discriminative loss (B, H, W, E)
        embeddings = pred_embeddings.permute(0, 2, 3, 1)  # (B, E, H, W) -> (B, H, W, E)
        
        # Get embedding dimension
        embedding_dim = embeddings.size(3)
        #print("Embeddings shape: ", embeddings.shape)
        
        # Initialize loss components
        pull_losses = []
        push_losses = []
        reg_losses = []
        
        # Process each sample in the batch
        for b in range(batch_size):
            sample_embeddings = embeddings[b]  # (H, W, E)
            sample_targets = target_instance_ids[b]  # (H, W)
            
            # Get unique instance IDs (excluding background)
            instance_ids = torch.unique(sample_targets)
            instance_ids = instance_ids[instance_ids > 0]
            
            if len(instance_ids) == 0:
                # No instances in this sample, skip
                print(f"WARNING: No instances found in batch {b}")
                continue
            
            # Print debug info about instances
            #print(f"Batch {b}: Found {len(instance_ids)} instances")
            
            # Calculate mean embeddings for each instance
            means = []
            instance_masks = []
            
            for instance_id in instance_ids:
                mask = (sample_targets == instance_id)
                if not torch.any(mask):
                    continue
                    
                # Get embeddings for this instance
                instance_embeddings = sample_embeddings[mask.nonzero(as_tuple=True)]  # (N, E) where N is number of pixels
                
                # Skip if too few pixels (avoid numerical issues)
                if instance_embeddings.size(0) < 10:
                    continue
                
                # Calculate mean embedding (centroid)
                mean_embedding = torch.mean(instance_embeddings, dim=0)  # (E)
                
                # L2 normalize the mean embedding to constrain it to unit hypersphere
                mean_embedding = F.normalize(mean_embedding, p=2, dim=0)
                
                means.append(mean_embedding)
                instance_masks.append(mask.sum())
            
            if len(means) == 0:
                continue
                
            # Convert to tensor
            means = torch.stack(means, dim=0)  # (I, E) where I is number of instances
            
            # 1. Variance (pull) term - pull embeddings towards their cluster center
            pull_loss = 0
            for i, instance_id in enumerate(instance_ids[:len(means)]):
                mask = instance_masks[i]
                if not torch.any(mask):
                    continue
                    
                instance_embeddings = sample_embeddings[mask.nonzero(as_tuple=True)]  # (N, E)
            
                # L2 normalize instance embeddings
                instance_embeddings = F.normalize(instance_embeddings, p=2, dim=1)
                
                # Calculate distances from pixels to cluster center
                mean = means[i].unsqueeze(0)  # (1, E)
                distances = torch.sqrt(torch.sum((instance_embeddings - mean) ** 2, dim=1))  # (N)
                
                
                # Apply hinge - only penalize distances beyond delta_v
                hinge_distances = torch.clamp(distances - delta_v, min=0.0)
                pull_loss += torch.mean(hinge_distances ** 2)
            
             # Average pull loss over all instances
            if len(means) > 0:
                pull_loss = pull_loss / len(means)
                pull_losses.append(pull_loss)
            
            # 2. Distance (push) term - push cluster centers apart
            if len(means) > 1:
                push_loss = 0
                n_pairs = 0
                
                # Calculate pairwise distances between cluster centers
                for i in range(len(means)):
                    for j in range(i+1, len(means)):
                        # Calculate cosine similarity between centers
                        similarity = torch.sum(means[i] * means[j])
                        
                        # Convert to distance (1 - similarity) and scale
                        distance = 1.0 - similarity
                        
                        # Apply hinge with 2*delta_d margin
                        margin = 2.0 * delta_d
                        hinge_distance = torch.clamp(margin - distance, min=0.0)
                        
                        # Square the hinge distance and add to loss
                        push_loss += hinge_distance ** 2
                        n_pairs += 1
                
                # Normalize push loss by number of pairs and scale down
                if n_pairs > 0:
                    # Apply a scaling factor that decreases with more instances
                    # This prevents push loss from growing too large with many instances
                    scale_factor = 1.0 / (1.0 + math.log(max(1, len(means) - 1)))
                    push_loss = push_loss * scale_factor / n_pairs
                    push_losses.append(push_loss)
            
            # 3. Regularization term - keep embeddings bounded
            if len(means) > 0:
                # L2 regularization on centers (should be small since we normalize)
                reg_loss = torch.mean(torch.norm(means, p=2, dim=1))
                reg_losses.append(reg_loss)
        
        # Return early with zero loss if no valid losses found
        if not pull_losses and not push_losses:
            return torch.zeros(batch_size, device=pred_embeddings.device, requires_grad=True)
        
        # Combine batch losses with appropriate weights
        batch_losses = []
        
        # Calculate appropriate weight for push loss based on embedding dimension
        # Higher embedding dimensions need smaller push weights
        embed_scale = 1.0 / math.sqrt(embedding_dim)
        push_weight = 0.1 * embed_scale  # Reduce push weight significantly
        
        for b in range(batch_size):
            # Get losses for this batch item (or zero if none)
            batch_pull = pull_losses[b] if b < len(pull_losses) else torch.tensor(0.0, device=pred_embeddings.device)
            batch_push = push_losses[b] if b < len(push_losses) else torch.tensor(0.0, device=pred_embeddings.device)
            batch_reg = reg_losses[b] if b < len(reg_losses) else torch.tensor(0.0, device=pred_embeddings.device)
            
            # Combine losses with appropriate weights
            batch_loss = batch_pull + push_weight * batch_push + reg_weight * batch_reg
            batch_losses.append(batch_loss)
        
        # Stack batch losses and normalize by num_objects
        final_loss = torch.stack(batch_losses) if batch_losses else torch.zeros(batch_size, device=pred_embeddings.device)
        
        if torch.isnan(emb).any():
            print("NaN detected in embeddings!")
        if torch.isnan(centers_tensor).any():
            print("NaN detected in centers tensor!")
        if torch.isnan(dists).any():
            print("NaN detected in distance calculation!")


        # Print debug info
        if pull_losses and push_losses and self.training and is_dist_avail_and_initialized() and torch.distributed.get_rank() == 0:
            pull_loss_mean = torch.mean(torch.stack(pull_losses))
            push_loss_mean = torch.mean(torch.stack(push_losses))
            reg_loss_mean = torch.mean(torch.stack(reg_losses)) if reg_losses else torch.tensor(0.0)
            
            print(f"Embed dim: {embedding_dim}, "
                f"Avg instances: {avg_instances:.1f}, "
                f"Pull loss: {pull_loss_mean.item():.4f}, "
                f"Push loss: {push_loss_mean.item():.4f} (weight: {push_weight:.6f}), "
                f"Reg loss: {reg_loss_mean.item():.4f}")
        
        return final_loss / num_objects

    def instance_id_loss_stem_seg(self, pred_embeddings, target_instance_ids, num_objects,
                         pred_bandwidth=None, pred_offsets=None,
                         delta_v=0.5, delta_d=2, reg_weight=0.1,
                         offset_weight=1.0, sigma_reg_weight=0.001):
        """
        STEm-Seg inspired instance embedding loss.
        Args:
            pred_embeddings: (B, E, H, W) predicted instance embeddings.
            target_instance_ids: (B, 1, H, W) or (B, H, W) ground-truth instance ID map.
            pred_bandwidth: (B, 1, H, W) predicted per-pixel bandwidth (σ); if None, fixed delta_v is used.
            pred_offsets: (B, 2, H, W) predicted per-pixel offset vectors.
            delta_v: Base margin for pull loss if no σ is provided.
            delta_d: Base margin for push loss.
            reg_weight: Regularization weight on instance centers.
            offset_weight: Weight for offset regression loss.
            sigma_reg_weight: Weight for bandwidth (σ) regression loss.
        Returns:
            Instance loss scalar.
        """
        B, E, H, W = pred_embeddings.shape
        device = pred_embeddings.device
        # Ensure target shape is (B, H, W)
        if target_instance_ids.dim() == 4:
            target_instance_ids = target_instance_ids.squeeze(1)
        # Create coordinate grid for offset regression (shape: H x W x 2)
        yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        coord_grid = torch.stack([yy, xx], dim=-1).float()  # (H, W, 2)

        batch_loss_list = []
        embeddings = pred_embeddings.permute(0, 2, 3, 1)  # (B, H, W, E)
        for b in range(B):
            sample_embeds = embeddings[b]  # (H, W, E)
            sample_targets = target_instance_ids[b]  # (H, W)
            # Get unique instance IDs (exclude background 0)
            instance_ids = torch.unique(sample_targets)
            instance_ids = instance_ids[instance_ids > 0]
            if len(instance_ids) == 0:
                continue

            inst_pull_losses = []
            inst_sigma_losses = []
            inst_offset_losses = []
            inst_centers = []
            inst_sigmas = []

            for inst in instance_ids:
                mask = (sample_targets == inst)  # (H, W) boolean
                if mask.sum() == 0:
                    continue
                # Instance embeddings: (N, E)
                inst_embeds = sample_embeds[mask]
                # Center in embedding space (mean vector)
                center_emb = inst_embeds.mean(dim=0)
                inst_centers.append(center_emb)
                # Compute distances for pull loss
                distances = torch.norm(inst_embeds - center_emb, p=2, dim=1)
                # Determine predicted σ over the instance region; if not provided, use delta_v.
                if pred_bandwidth is not None:
                    bandwidth_map = pred_bandwidth[b, 0, :, :]  # (H, W)
                    pred_sigma = bandwidth_map[mask].mean()
                else:
                    pred_sigma = delta_v
                inst_sigmas.append(pred_sigma)
                # Pull loss: penalize distances exceeding pred_sigma
                pull_loss = torch.mean(torch.clamp(distances - pred_sigma, min=0.0) ** 2)
                inst_pull_losses.append(pull_loss)
                # Sigma regression loss: encourage pred_sigma to match average spread.
                sigma_loss = torch.abs(pred_sigma - distances.mean())
                inst_sigma_losses.append(sigma_loss)
                # Offset regression loss:
                if pred_offsets is not None:
                    offsets_map = pred_offsets[b]  # (2, H, W)
                    # Compute target center in pixel space using coordinate grid:
                    target_coords = coord_grid[mask]  # (N, 2)
                    center_pixel = target_coords.mean(dim=0)  # (2)
                    # Predicted offsets for instance pixels: (N, 2)
                    pred_off = offsets_map[:, mask].permute(1, 0)
                    # Ground-truth offsets: difference between pixel coordinates and center.
                    target_off = target_coords - center_pixel
                    off_loss = F.l1_loss(pred_off, target_off, reduction='mean')
                    inst_offset_losses.append(off_loss)
                else:
                    inst_offset_losses.append(torch.tensor(0.0, device=device))
            
            if len(inst_pull_losses) == 0:
                continue
            pull_loss_inst = torch.stack(inst_pull_losses).mean()
            sigma_loss_inst = torch.stack(inst_sigma_losses).mean()
            offset_loss_inst = torch.stack(inst_offset_losses).mean()

            # Push loss: for all pairs of instance centers
            push_losses = []
            centers_tensor = torch.stack(inst_centers)  # (num_inst, E)
            sigmas_tensor = torch.stack(inst_sigmas)      # (num_inst)
            num_inst = centers_tensor.shape[0]
            if num_inst > 1:
                for i in range(num_inst):
                    for j in range(i + 1, num_inst):
                        center_dist = torch.norm(centers_tensor[i] - centers_tensor[j], p=2)
                        margin = sigmas_tensor[i] + sigmas_tensor[j]
                        push_losses.append(torch.clamp(margin - center_dist, min=0.0) ** 2)
                push_loss_inst = torch.stack(push_losses).mean() if push_losses else torch.tensor(0.0, device=device)
            else:
                push_loss_inst = torch.tensor(0.0, device=device)

            # Optional regularization on centers (L2 norm)
            reg_loss_inst = centers_tensor.norm(p=2, dim=1).mean() if num_inst > 0 else torch.tensor(0.0, device=device)
            
            total_inst_loss = (pull_loss_inst +
                               push_loss_inst +
                               reg_weight * reg_loss_inst +
                               sigma_reg_weight * sigma_loss_inst +
                               offset_weight * offset_loss_inst)
            batch_loss_list.append(total_inst_loss)
        if len(batch_loss_list) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        final_loss = torch.stack(batch_loss_list).mean() / num_objects
        return final_loss

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight

        return reduced_loss

