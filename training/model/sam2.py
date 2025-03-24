# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch.nn.functional as F
import torch
import torch.distributed
from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.sam2_utils import (
    get_1d_sine_pe,
    get_next_point,
    sample_box_points,
    select_closest_cond_frames,
)

from sam2.utils.misc import concat_points

from training.utils.data_utils import BatchedVideoDatapoint


class SAM2Train(SAM2Base):
    def __init__(
        self,
        image_encoder,
        memory_attention=None,
        memory_encoder=None,
        prob_to_use_pt_input_for_train=0.0,
        prob_to_use_pt_input_for_eval=0.0,
        prob_to_use_box_input_for_train=0.0,
        prob_to_use_box_input_for_eval=0.0,
        # if it is greater than 1, we interactive point sampling in the 1st frame and other randomly selected frames
        num_frames_to_correct_for_train=1,  # default: only iteratively sample on first frame
        num_frames_to_correct_for_eval=1,  # default: only iteratively sample on first frame
        rand_frames_to_correct_for_train=False,
        rand_frames_to_correct_for_eval=False,
        # how many frames to use as initial conditioning frames (for both point input and mask input; the first frame is always used as an initial conditioning frame)
        # - if `rand_init_cond_frames` below is True, we randomly sample 1~num_init_cond_frames initial conditioning frames
        # - otherwise we sample a fixed number of num_init_cond_frames initial conditioning frames
        # note: for point input, we sample correction points on all such initial conditioning frames, and we require that `num_frames_to_correct` >= `num_init_cond_frames`;
        # these are initial conditioning frames because as we track the video, more conditioning frames might be added
        # when a frame receives correction clicks under point input if `add_all_frames_to_correct_as_cond=True`
        num_init_cond_frames_for_train=1,  # default: only use the first frame as initial conditioning frame
        num_init_cond_frames_for_eval=1,  # default: only use the first frame as initial conditioning frame
        rand_init_cond_frames_for_train=True,  # default: random 1~num_init_cond_frames_for_train cond frames (to be constent w/ previous TA data loader)
        rand_init_cond_frames_for_eval=False,
        # if `add_all_frames_to_correct_as_cond` is True, we also append to the conditioning frame list any frame that receives a later correction click
        # if `add_all_frames_to_correct_as_cond` is False, we conditioning frame list to only use those initial conditioning frames
        add_all_frames_to_correct_as_cond=False,
        # how many additional correction points to sample (on each frame selected to be corrected)
        # note that the first frame receives an initial input click (in addition to any correction clicks)
        num_correction_pt_per_frame=7,
        # method for point sampling during evaluation
        # "uniform" (sample uniformly from error region) or "center" (use the point with the largest distance to error region boundary)
        # default to "center" to be consistent with evaluation in the SAM paper
        pt_sampling_for_eval="center",
        # During training, we optionally allow sampling the correction points from GT regions
        # instead of the prediction error regions with a small probability. This might allow the
        # model to overfit less to the error regions in training datasets
        prob_to_sample_from_gt_for_train=0.0,
        use_act_ckpt_iterative_pt_sampling=False,
        # whether to forward image features per frame (as it's being tracked) during evaluation, instead of forwarding image features
        # of all frames at once. This avoids backbone OOM errors on very long videos in evaluation, but could be slightly slower.
        forward_backbone_per_frame_for_eval=False,
        freeze_image_encoder=False,
        **kwargs,
    ):
        super().__init__(image_encoder, memory_attention, memory_encoder, **kwargs)
        self.use_act_ckpt_iterative_pt_sampling = use_act_ckpt_iterative_pt_sampling
        self.forward_backbone_per_frame_for_eval = forward_backbone_per_frame_for_eval

        # Point sampler and conditioning frames
        self.prob_to_use_pt_input_for_train = prob_to_use_pt_input_for_train
        self.prob_to_use_box_input_for_train = prob_to_use_box_input_for_train
        self.prob_to_use_pt_input_for_eval = prob_to_use_pt_input_for_eval
        self.prob_to_use_box_input_for_eval = prob_to_use_box_input_for_eval
        if prob_to_use_pt_input_for_train > 0 or prob_to_use_pt_input_for_eval > 0:
            logging.info(
                f"Training with points (sampled from masks) as inputs with p={prob_to_use_pt_input_for_train}"
            )
            assert num_frames_to_correct_for_train >= num_init_cond_frames_for_train
            assert num_frames_to_correct_for_eval >= num_init_cond_frames_for_eval

        self.num_frames_to_correct_for_train = num_frames_to_correct_for_train
        self.num_frames_to_correct_for_eval = num_frames_to_correct_for_eval
        self.rand_frames_to_correct_for_train = rand_frames_to_correct_for_train
        self.rand_frames_to_correct_for_eval = rand_frames_to_correct_for_eval
        # Initial multi-conditioning frames
        self.num_init_cond_frames_for_train = num_init_cond_frames_for_train
        self.num_init_cond_frames_for_eval = num_init_cond_frames_for_eval
        self.rand_init_cond_frames_for_train = rand_init_cond_frames_for_train
        self.rand_init_cond_frames_for_eval = rand_init_cond_frames_for_eval
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond
        self.num_correction_pt_per_frame = num_correction_pt_per_frame
        self.pt_sampling_for_eval = pt_sampling_for_eval
        self.prob_to_sample_from_gt_for_train = prob_to_sample_from_gt_for_train
        # A random number generator with a fixed initial seed across GPUs
        self.rng = np.random.default_rng(seed=42)

        if freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

    def forward(self, input: BatchedVideoDatapoint):
        # Store the input image for visualization
        self.current_input_image = input.flat_img_batch
        if self.training or not self.forward_backbone_per_frame_for_eval:
            backbone_out = self.forward_image(input.flat_img_batch)
        else:
            # defer image feature computation on a frame until it's being tracked
            backbone_out = {"backbone_fpn": None, "vision_pos_enc": None}
            
        backbone_out = self.prepare_prompt_inputs(backbone_out, input)
        previous_stages_out = self.forward_tracking(backbone_out, input)

        return previous_stages_out

    def _prepare_backbone_features_per_frame(self, img_batch, img_ids):
        """Compute the image backbone features on the fly for the given img_ids."""
        # Only forward backbone on unique image ids to avoid repetitive computation
        # (if `img_ids` has only one element, it's already unique so we skip this step).
        if img_ids.numel() > 1:
            unique_img_ids, inv_ids = torch.unique(img_ids, return_inverse=True)
        else:
            unique_img_ids, inv_ids = img_ids, None

        # Compute the image features on those unique image ids
        image = img_batch[unique_img_ids]
        backbone_out = self.forward_image(image)
        (
            _,
            vision_feats,
            vision_pos_embeds,
            feat_sizes,
        ) = self._prepare_backbone_features(backbone_out)
        
        # Inverse-map image features for `unique_img_ids` to the final image features
        # for the original input `img_ids`.
        if inv_ids is not None:
            image = image[inv_ids]
            vision_feats = [x[:, inv_ids] for x in vision_feats]
            vision_pos_embeds = [x[:, inv_ids] for x in vision_pos_embeds]
        return image, vision_feats, vision_pos_embeds, feat_sizes

    def prepare_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
        """
        Prepare input mask, point or box prompts. Optionally, we allow tracking from
        a custom `start_frame_idx` to the end of the video (for evaluation purposes).
        """
        
        border_emphasis_mask_GT_per_obj = False  # Control for per-object masks
        border_emphasis_combined_with_instance_borders = True  # Control for adding instance borders to combined masks
        border_width = 1  # Width of the border in pixels
        
        
        # Get the individual object masks (instead of combined masks)
        # The masks tensor is of shape [T, O, H, W] where O is the number of objects across all frames
        
        # Get the number of objects and create a mapping from frame to objects in that frame
        num_frames = input.num_frames
        frame_to_objects = {}
        
        # We need to know which objects are in which frames
        # obj_to_frame_idx is of shape [T, O, 2] containing [frame_idx, video_idx]
        # We'll use this to map from frame_idx to object indices
        for t in range(num_frames):
            # Get all objects that appear in frame t
            # Find all indices where obj_to_frame_idx[t, :, 0] == t
            # This is already the case by construction, so we just need all objects indices for frame t
            frame_to_objects[t] = list(range(input.obj_to_frame_idx[t].shape[0]))
        
        # Store individual object masks per frame
        gt_masks_per_frame_per_obj = {}
        for t, obj_indices in frame_to_objects.items():
            gt_masks_per_frame_per_obj[t] = {}
            for obj_idx in obj_indices:
                # Get the mask for this object in this frame
                obj_mask = input.masks[t, obj_idx].unsqueeze(0).unsqueeze(0)  # Add batch and channel dims [1, 1, H, W]
                if border_emphasis_mask_GT_per_obj:
                    obj_mask_border = self.create_border_emphasized_masks(obj_mask, border_width=border_width)
                    gt_masks_per_frame_per_obj[t][obj_idx] = obj_mask_border
                else:
                    gt_masks_per_frame_per_obj[t][obj_idx] = obj_mask
        
        # Load the ground-truth masks on all frames (so that we can later
        # sample correction points from them)
        # gt_masks_per_frame = {
        #     stage_id: targets.segments.unsqueeze(1)  # [B, 1, H_im, W_im]
        #     for stage_id, targets in enumerate(input.find_targets)
        # }
        # Original:
        # gt_masks_per_frame = {
        #     stage_id: masks.unsqueeze(1)  # [B, 1, H_im, W_im]
        #     for stage_id, masks in enumerate(input.masks)
        # }
        
        # Get the ground-truth masks on all frames (so that we can later
        # sample correction points from them)
        gt_masks_per_frame = {}
    
        for t in range(num_frames):
            # Replace the instance border creation code with this approach
            if border_emphasis_combined_with_instance_borders:
                # Start with an empty mask
                obj_indices = frame_to_objects[t]
                if not obj_indices:
                    gt_masks_per_frame[t] = torch.zeros_like(input.combined_masks[t]).unsqueeze(1)
                    continue
                    
                # Get shape from the first object mask
                first_obj_mask = input.masks[t, obj_indices[0]]
                h, w = first_obj_mask.shape
                combined_mask = torch.zeros((1, 1, h, w), device=first_obj_mask.device)
                
                # First create a mask where interior pixels have values equal to instance IDs
                # This ensures non-overlapping values for each instance
                instance_id_mask = torch.zeros((h, w), device=first_obj_mask.device)
                for i, obj_idx in enumerate(obj_indices):
                    # Use i+1 as the instance ID (keeping 0 as background)
                    instance_id = i + 1
                    # Fill the instance mask with the instance ID
                    instance_mask = input.masks[t, obj_idx]
                    instance_id_mask = torch.where(instance_mask > 0, 
                                                torch.tensor(instance_id, device=instance_mask.device),
                                                instance_id_mask)
                
                # Now detect borders by checking if any adjacent pixel has a different non-zero value
                border_mask = torch.zeros((h, w), device=first_obj_mask.device)
                
                # Check each pixel's neighborhood for different instance IDs
                for i in range(1, h-1):
                    for j in range(1, w-1):
                        if instance_id_mask[i, j] > 0:  # If this is part of an instance
                            # Check 8-connected neighborhood
                            center_id = instance_id_mask[i, j]
                            
                            # Check if any neighbor has a different non-zero ID or is background
                            is_border = False
                            
                            # Check if any neighbor is background (0)
                            if (instance_id_mask[i-1:i+2, j-1:j+2] == 0).any():
                                is_border = True
                            # Check if any neighbor has a different instance ID
                            else:
                                neighbors = instance_id_mask[i-1:i+2, j-1:j+2]
                                if ((neighbors != center_id) & (neighbors > 0)).any():
                                    is_border = True
                                    
                            if is_border:
                                border_mask[i, j] = 1.0
                
                # Create combined mask with:
                # - Borders = 1.0
                # - Interior = 0.5
                # - Background = 0.0
                combined_mask[0, 0] = torch.where(border_mask > 0,
                                                torch.tensor(1.0, device=border_mask.device),
                                                torch.where(instance_id_mask > 0,
                                                        torch.tensor(0.5, device=instance_id_mask.device),
                                                        torch.tensor(0.0, device=instance_id_mask.device)))
                
                gt_masks_per_frame[t] = combined_mask
                
                # Optionally add this line to visualize the result
                self.visualize_masks(combined_mask, combined_mask)
            else:
                # Just use the regular combined mask without instance borders
                gt_masks_per_frame[t] = input.combined_masks[t].unsqueeze(1)
        
        # gt_masks_per_frame = input.masks.unsqueeze(2) # [T,B,1,H_im,W_im] keep everything in tensor form
        
        # this will store GT mask for every frame, making model know GT and predict perfect every time due to overfitting.
        #backbone_out["gt_masks_per_frame"] = gt_masks_per_frame
        #backbone_out["gt_masks_per_frame_per_obj"] = gt_masks_per_frame_per_obj
        
        backbone_out["frame_to_objects"] = frame_to_objects
        backbone_out["num_frames"] = num_frames

        # Randomly decide whether to use point inputs or mask inputs
        if self.training:
            prob_to_use_pt_input = 0# self.prob_to_use_pt_input_for_train
            prob_to_use_box_input = 0 #self.prob_to_use_box_input_for_train
            num_frames_to_correct = self.num_frames_to_correct_for_train
            rand_frames_to_correct = self.rand_frames_to_correct_for_train
            num_init_cond_frames = self.num_init_cond_frames_for_train
            rand_init_cond_frames = self.rand_init_cond_frames_for_train
        else:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_eval
            prob_to_use_box_input = self.prob_to_use_box_input_for_eval
            num_frames_to_correct = self.num_frames_to_correct_for_eval
            rand_frames_to_correct = self.rand_frames_to_correct_for_eval
            num_init_cond_frames = self.num_init_cond_frames_for_eval
            rand_init_cond_frames = self.rand_init_cond_frames_for_eval
        if num_frames == 1:
            # here we handle a special case for mixing video + SAM on image training,
            # where we force using point input for the SAM task on static images
            prob_to_use_pt_input = 0 #1.0
            num_frames_to_correct = 1
            num_init_cond_frames = 1
        assert num_init_cond_frames >= 1
        # (here `self.rng.random()` returns value in range 0.0 <= X < 1.0)
        use_pt_input = self.rng.random() < prob_to_use_pt_input
        if rand_init_cond_frames and num_init_cond_frames > 1:
            # randomly select 1 to `num_init_cond_frames` frames as initial conditioning frames
            num_init_cond_frames = self.rng.integers(
                1, num_init_cond_frames, endpoint=True
            )
        if (
            use_pt_input
            and rand_frames_to_correct
            and num_frames_to_correct > num_init_cond_frames
        ):
            # randomly select `num_init_cond_frames` to `num_frames_to_correct` frames to sample
            # correction clicks (only for the case of point input)
            num_frames_to_correct = self.rng.integers(
                num_init_cond_frames, num_frames_to_correct, endpoint=True
            )
        backbone_out["use_pt_input"] = use_pt_input

        # Sample initial conditioning frames
        if num_init_cond_frames == 1:
            init_cond_frames = [start_frame_idx]  # starting frame
        else:
            # starting frame + randomly selected remaining frames (without replacement)
            init_cond_frames = [start_frame_idx] + self.rng.choice(
                range(start_frame_idx + 1, num_frames),
                num_init_cond_frames - 1,
                replace=False,
            ).tolist()
        backbone_out["init_cond_frames"] = init_cond_frames
        backbone_out["frames_not_in_init_cond"] = [
            t for t in range(start_frame_idx, num_frames) if t not in init_cond_frames
        ]
        
        backbone_out["gt_masks_per_frame"] = {t: gt_masks_per_frame[t] for t in init_cond_frames}
        backbone_out["gt_masks_per_frame_per_obj"] = {t: gt_masks_per_frame_per_obj[t] for t in init_cond_frames}
        
        # Prepare mask or point inputs on initial conditioning frames
        #backbone_out["mask_inputs_per_frame"] = {}  # {frame_idx: <input_masks>}
        #backbone_out["point_inputs_per_frame"] = {}  # {frame_idx: <input_points>}
        
        # Prepare mask or point inputs for each object in initial conditioning frames
        backbone_out["mask_inputs_per_frame_per_obj"] = {}  # {frame_idx: {obj_idx: mask}}
        backbone_out["point_inputs_per_frame_per_obj"] = {}  # {frame_idx: {obj_idx: points}}
        
        for t in init_cond_frames:
            backbone_out["mask_inputs_per_frame_per_obj"][t] = {}
            backbone_out["point_inputs_per_frame_per_obj"][t] = {}
        
            # Process each object in this frame
            for obj_idx in frame_to_objects[t]:
                gt_mask = gt_masks_per_frame_per_obj[t][obj_idx]
                
                if not use_pt_input:
                    backbone_out["mask_inputs_per_frame_per_obj"][t][obj_idx] = gt_mask
                else:
                    # During training # P(box) = prob_to_use_pt_input * prob_to_use_box_input
                    use_box_input = self.rng.random() < prob_to_use_box_input
                    if use_box_input:
                        points, labels = sample_box_points(
                            gt_masks_per_frame[t],
                        )
                    else:
                        # (here we only sample **one initial point** on initial conditioning frames from the
                        # ground-truth mask; we may sample more correction points on the fly)
                        points, labels = get_next_point(
                            gt_masks=gt_masks_per_frame[t],
                            pred_masks=None,
                            method=(
                                "uniform" if self.training else self.pt_sampling_for_eval
                            ),
                        )

                    point_inputs = {"point_coords": points, "point_labels": labels}
                    backbone_out["point_inputs_per_frame_per_obj"][t][obj_idx] = point_inputs

        # Sample frames where we will add correction clicks on the fly
        # based on the error between prediction and ground-truth masks
        if not use_pt_input:
            # no correction points will be sampled when using mask inputs
            frames_to_add_correction_pt = []
        elif num_frames_to_correct == num_init_cond_frames:
            frames_to_add_correction_pt = init_cond_frames
        else:
            assert num_frames_to_correct > num_init_cond_frames
            # initial cond frame + randomly selected remaining frames (without replacement)
            extra_num = num_frames_to_correct - num_init_cond_frames
            frames_to_add_correction_pt = (
                init_cond_frames
                + self.rng.choice(
                    backbone_out["frames_not_in_init_cond"], extra_num, replace=False
                ).tolist()
            )
        # This will only be used for training interactive segmentation model.
        #backbone_out["frames_to_add_correction_pt"] = frames_to_add_correction_pt
        backbone_out["frames_to_add_correction_pt"] = []
    
         
        return backbone_out

    def visualize_masks(self, masks, border_enhanced_masks, path='./mask_input.png'):
        """
        Helper function to visualize original masks side by side with border enhanced masks.
        Returns a visualization array that can be displayed with matplotlib.
        
        Args:
            masks: Original binary masks
            border_enhanced_masks: Masks with emphasized borders
            
        Returns:
            Visualization array
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        
        
        # Convert to numpy for visualization
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        if isinstance(border_enhanced_masks, torch.Tensor):
            border_enhanced_masks = border_enhanced_masks.cpu().numpy()
        
        # Skip visualization if either input is 1D or doesn't have the right shape
        if len(masks.shape) < 2 or len(border_enhanced_masks.shape) < 2:
            print(f"Skipping visualization due to invalid dimensions. Masks shape: {masks.shape}, Border masks shape: {border_enhanced_masks.shape}")
            return
        
        # Create a custom colormap for the border enhanced masks
        colors = [(0, 0, 0, 0), (0, 0, 1, 0.5), (1, 0, 0, 1)]  # transparent -> blue -> red
        positions = [0, 0.5, 1]
        border_cmap = LinearSegmentedColormap.from_list("border_cmap", list(zip(positions, colors)))
        
        # Prepare visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Display original masks
        axes[0].imshow(masks[0, 0], cmap='gray')
        axes[0].set_title("Original Mask")
        axes[0].axis('off')
        
        # Display border enhanced masks
        axes[1].imshow(border_enhanced_masks[0, 0], cmap=border_cmap)
        axes[1].set_title("Border Enhanced Mask")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)

    def create_border_emphasized_masks(self, masks, border_width=2, border_value=1.0, interior_value=0.5):
        """
        Transforms binary masks to emphasize borders.
        
        Args:
            masks: Binary masks tensor [T, O, H, W] or any shape ending with H, W
            border_width: Width of the border in pixels
            border_value: Value to assign to border pixels (default: 1.0)
            interior_value: Value to assign to interior (non-border) pixels (default: 0.5)
        
        Returns:
            Enhanced masks with borders emphasized
        """
        # Make sure we're working with float tensors for proper value assignment
        if masks.dtype != torch.float32:
            masks = masks.float()
        
        # Create a copy to avoid modifying the original
        enhanced_masks = masks.clone()
        
        # Identify borders using erosion: border pixels are those that are in the original
        # mask but not in the eroded mask
        if border_width > 0:
            # Create a square kernel for erosion
            kernel_size = 2 * border_width + 1
            weight = torch.ones(1, 1, kernel_size, kernel_size, device=masks.device)
            
            # Reshape masks for 2D convolution if needed, preserving batch dimensions
            original_shape = masks.shape
            if len(original_shape) > 4:
                masks_reshaped = masks.reshape(-1, 1, original_shape[-2], original_shape[-1])
            else:
                # Assuming shape is [T, O, H, W]
                masks_reshaped = masks.reshape(-1, 1, original_shape[-2], original_shape[-1])
            
            # Perform erosion using convolution with thresholding
            # Padding to maintain shape
            padding = border_width
            # Normalize by kernel size to get average
            eroded = F.conv2d(masks_reshaped, weight, padding=padding) / (kernel_size * kernel_size)
            # Threshold to get eroded binary mask: only pixels where all kernel elements are 1 will remain 1
            eroded = (eroded > 0.999).float()  # Using 0.999 instead of 1.0 for numerical stability
            
            # Reshape back to original shape
            eroded = eroded.reshape(original_shape)
            
            # Create border mask: original mask minus eroded mask
            borders = masks - eroded
            
            # Set values: border_value for borders, interior_value for interior (eroded) parts
            enhanced_masks = torch.where(borders > 0, 
                                        torch.tensor(border_value, device=masks.device),
                                        torch.where(masks > 0, 
                                                    torch.tensor(interior_value, device=masks.device),
                                                    torch.tensor(0.0, device=masks.device)))
        
        return enhanced_masks

    def forward_tracking(
        self, backbone_out, input: BatchedVideoDatapoint, return_dict=False
    ):
        """Forward video tracking on each frame and each object (and sample correction clicks if doing segmentation)."""
        
        img_feats_already_computed = backbone_out["backbone_fpn"] is not None
        if img_feats_already_computed:
            # Prepare the backbone features
            # - vision_feats and vision_pos_embeds are in (HW)BC format
            (
                _,
                vision_feats,
                vision_pos_embeds,
                feat_sizes,
            ) = self._prepare_backbone_features(backbone_out)
        
        # Starting the stage loop
        num_frames = backbone_out["num_frames"]
        init_cond_frames = backbone_out["init_cond_frames"]
        frames_to_add_correction_pt = backbone_out["frames_to_add_correction_pt"]
        frame_to_objects = backbone_out["frame_to_objects"]
        
        # First process all the initial conditioning frames to encode them as memory,
        # and then conditioning on them to track the remaining frames
        processing_order = init_cond_frames + backbone_out["frames_not_in_init_cond"]
        output_dict = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: {obj_idx: <out>}}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: {obj_idx: <out>}}
            "frame_to_objects": frame_to_objects,  # Pass this through for reference
        }
        
        for stage_id in processing_order:
            # Get the image features for the current frames
            # img_ids = input.find_inputs[stage_id].img_ids
            #img_ids = input.flat_obj_to_img_idx[stage_id] # this causes error with duplicate
            img_ids = torch.unique(input.flat_obj_to_img_idx[stage_id])
            if img_feats_already_computed:
                # Retrieve image features according to img_ids (if they are already computed).
                current_vision_feats = [x[:, img_ids] for x in vision_feats]
                current_vision_pos_embeds = [x[:, img_ids] for x in vision_pos_embeds]
            else:
                # Otherwise, compute the image features on the fly for the given img_ids
                # (this might be used for evaluation on long videos to avoid backbone OOM).
                (
                    _,
                    current_vision_feats,
                    current_vision_pos_embeds,
                    feat_sizes,
                ) = self._prepare_backbone_features_per_frame(
                    input.flat_img_batch, img_ids
                )
            
            # Process each object in this frame by generating an individual prompt for it
            objects_in_frame = frame_to_objects.get(stage_id, [])
            instance_id_map = None
            
            # Get instance ID map for this frame if available
            instance_id_map = None
            if hasattr(input, 'instance_maps'):
                instance_id_map = input.instance_maps[stage_id]
            
            # Combine prompts from all objects in this frame
            combined_point_inputs = self._combine_object_point_prompts(
                backbone_out.get("point_inputs_per_frame_per_obj", {}).get(stage_id, {}),
                objects_in_frame
            )
            
            mask_inputs = None
            if not backbone_out.get("use_pt_input", False):
                # Option 1: Use the combined mask (all objects together)
                #mask_inputs = backbone_out.get("mask_inputs_per_frame_per_obj", {}).get(stage_id, {})
                mask_inputs = backbone_out.get("gt_masks_per_frame", {}).get(stage_id, {})[0]
                # Option 2: Create a combined mask from individual object masks
                # mask_inputs = self._combine_object_mask_prompts(
                #         backbone_out.get("mask_inputs_per_frame_per_obj", {}).get(stage_id, {}),
                #         objects_in_frame
                #     )
            
            # Get combined ground truth masks, only for conditioning frames
            gt_masks = None
            if stage_id in init_cond_frames:
                gt_masks = backbone_out["gt_masks_per_frame"].get(stage_id, None)
                # Gt masks have borders internally so just quickfix make global binary mask for GT mask:
                gt_masks = self._combine_object_mask_prompts(
                            backbone_out.get("mask_inputs_per_frame_per_obj", {}).get(stage_id, {}),
                            objects_in_frame
                        )

                self.visualize_masks(gt_masks, mask_inputs, path='./gtmask_vs_inputmask.png')

            # Get output masks based on this frame's prompts and previous memory
            current_out = self.track_step(
                frame_idx=stage_id,
                is_init_cond_frame=stage_id in init_cond_frames,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=combined_point_inputs,
                mask_inputs=mask_inputs,
                gt_masks=gt_masks,
                frames_to_add_correction_pt=frames_to_add_correction_pt,
                output_dict=output_dict,
                num_frames=num_frames,
                gt_instance_ids=instance_id_map,  # NEW: pass instance ids if available
            )
        
            # Append the output, depending on whether it's a conditioning frame
            add_output_as_cond_frame = stage_id in init_cond_frames or (
                self.add_all_frames_to_correct_as_cond
                and stage_id in frames_to_add_correction_pt
            )
            
            if add_output_as_cond_frame:
                output_dict["cond_frame_outputs"][stage_id] = current_out
            else:
                output_dict["non_cond_frame_outputs"][stage_id] = current_out

        if return_dict:
            return output_dict
        
        # turn `output_dict` into a list for loss function
        # We need to aggregate results per frame across all objects
        all_frame_outputs = {}
        all_frame_outputs.update(output_dict["cond_frame_outputs"])
        all_frame_outputs.update(output_dict["non_cond_frame_outputs"])
        all_frame_outputs = [all_frame_outputs[t] for t in range(num_frames)]
        
        # Make DDP happy with activation checkpointing by removing unused keys
        all_frame_outputs = [
            {k: v for k, v in d.items() if k != "obj_ptr"} for d in all_frame_outputs
        ]

        return all_frame_outputs

    def _combine_object_point_prompts(self, point_inputs_dict, objects_in_frame):
        """
        Combine point prompts from all objects into a single point prompt.
        
        Args:
            point_inputs_dict: Dictionary of {obj_idx: point_inputs} where each point_inputs 
                            contains 'point_coords' [B, P, 2] and 'point_labels' [B, P]
            objects_in_frame: List of object indices in this frame
            
        Returns:
            Combined point inputs with all object points concatenated
        """
        if not point_inputs_dict or not objects_in_frame:
            return None
            
        # Collect point coordinates and labels from all objects
        all_coords = []
        all_labels = []
        
        for obj_idx in objects_in_frame:
            if obj_idx not in point_inputs_dict:
                continue
                
            obj_point_inputs = point_inputs_dict[obj_idx]
            if obj_point_inputs is None:
                continue
                
            all_coords.append(obj_point_inputs["point_coords"])
            all_labels.append(obj_point_inputs["point_labels"])
        
        if not all_coords:
            return None
            
        # Concatenate along the point dimension (dim=1)
        combined_coords = torch.cat(all_coords, dim=1)
        combined_labels = torch.cat(all_labels, dim=1)
        
        return {"point_coords": combined_coords, "point_labels": combined_labels}

    def _combine_object_mask_prompts(self, mask_inputs_dict, objects_in_frame):
        """
        Combine mask prompts from all objects into a single mask prompt.
        
        Args:
            mask_inputs_dict: Dictionary of {obj_idx: mask_inputs} where each mask_inputs
                            is a tensor of shape [B, 1, H, W]
            objects_in_frame: List of object indices in this frame
            
        Returns:
            Combined mask inputs with all object masks combined (using logical OR)
        """
        if not mask_inputs_dict or not objects_in_frame:
            return None
            
        # Collect masks from all objects
        all_masks = []
        
        for obj_idx in objects_in_frame:
            if obj_idx not in mask_inputs_dict:
                continue
                
            obj_mask_inputs = mask_inputs_dict[obj_idx]
            if obj_mask_inputs is None:
                continue
                
            all_masks.append(obj_mask_inputs)
        
        if not all_masks:
            return None
            
        # Start with the first mask
        combined_mask = all_masks[0].clone()
        
        # Combine with remaining masks using logical OR
        for mask in all_masks[1:]:
            # Use maximum operation for logical OR (works for both binary and float masks)
            combined_mask = torch.max(combined_mask, mask)
        
        return combined_mask

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        run_mem_encoder=True,  # Whether to run the memory encoder on the predicted masks.
        prev_sam_mask_logits=None,  # The previously predicted SAM mask logits.
        frames_to_add_correction_pt=None,
        gt_masks=None,
        gt_instance_ids=None,
    ):
        if frames_to_add_correction_pt is None:
            frames_to_add_correction_pt = []
            
        # Get objects in this frame
        frame_to_objects = output_dict.get("frame_to_objects", {})
        objects_in_frame = frame_to_objects.get(frame_idx, [])
        
        # Create a combined output dictionary that will contain aggregated results
        combined_output = {
            "frame_idx": frame_idx,
            "object_outputs": {},  # Store per-object outputs here
            "num_objects": len(objects_in_frame)
        }
        
        # Process each object in the frame separately
        #print("objects_in_frame", objects_in_frame)
        
        current_out, sam_outputs, high_res_features, pix_feat = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )

        (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
            high_res_instance_ids,
        ) = sam_outputs

        current_out["multistep_pred_masks"] = low_res_masks
        current_out["multistep_pred_masks_high_res"] = high_res_masks
        current_out["multistep_pred_multimasks"] = [low_res_multimasks]
        current_out["multistep_pred_multimasks_high_res"] = [high_res_multimasks]
        current_out["multistep_pred_ious"] = [ious]
        current_out["multistep_point_inputs"] = [point_inputs]
        current_out["multistep_object_score_logits"] = [object_score_logits]
        current_out["multistep_pred_instance_ids"] = [high_res_instance_ids]

        # Optionally, sample correction points iteratively to correct the mask
        # if frame_idx in frames_to_add_correction_pt:
        #     point_inputs, final_sam_outputs = self._iter_correct_pt_sampling(
        #         is_init_cond_frame,
        #         point_inputs,
        #         gt_masks,
        #         high_res_features,
        #         pix_feat,
        #         low_res_multimasks,
        #         high_res_multimasks,
        #         ious,
        #         low_res_masks,
        #         high_res_masks,
        #         object_score_logits,
        #         current_out,
        #     )
        #     (
        #         _,
        #         _,
        #         _,
        #         low_res_masks,
        #         high_res_masks,
        #         obj_ptr,
        #         object_score_logits,
        #     ) = final_sam_outputs

        # Use the final prediction (after all correction steps for output and eval)
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        current_out["pred_instance_ids"] = high_res_instance_ids 

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
            gt_instance_ids=gt_instance_ids  # NEW: pass instance ids to memory encoder
        )
        
        return current_out

    def _iter_correct_pt_sampling(
        self,
        is_init_cond_frame,
        point_inputs,
        gt_masks,
        high_res_features,
        pix_feat_with_mem,
        low_res_multimasks,
        high_res_multimasks,
        ious,
        low_res_masks,
        high_res_masks,
        object_score_logits,
        current_out,
    ):

        assert gt_masks is not None
        all_pred_masks = [low_res_masks]
        all_pred_high_res_masks = [high_res_masks]
        all_pred_multimasks = [low_res_multimasks]
        all_pred_high_res_multimasks = [high_res_multimasks]
        all_pred_ious = [ious]
        all_point_inputs = [point_inputs]
        all_object_score_logits = [object_score_logits]
        for _ in range(self.num_correction_pt_per_frame):
            # sample a new point from the error between prediction and ground-truth
            # (with a small probability, directly sample from GT masks instead of errors)
            if self.training and self.prob_to_sample_from_gt_for_train > 0:
                sample_from_gt = (
                    self.rng.random() < self.prob_to_sample_from_gt_for_train
                )
            else:
                sample_from_gt = False
            # if `pred_for_new_pt` is None, only GT masks will be used for point sampling
            pred_for_new_pt = None if sample_from_gt else (high_res_masks > 0)
            new_points, new_labels = get_next_point(
                gt_masks=gt_masks,
                pred_masks=pred_for_new_pt,
                method="uniform" if self.training else self.pt_sampling_for_eval,
            )
            point_inputs = concat_points(point_inputs, new_points, new_labels)
            # Feed the mask logits of the previous SAM outputs in the next SAM decoder step.
            # For tracking, this means that when the user adds a correction click, we also feed
            # the tracking output mask logits along with the click as input to the SAM decoder.
            mask_inputs = low_res_masks
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            if self.use_act_ckpt_iterative_pt_sampling and not multimask_output:
                sam_outputs = torch.utils.checkpoint.checkpoint(
                    self._forward_sam_heads,
                    backbone_features=pix_feat_with_mem,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    high_res_features=high_res_features,
                    multimask_output=multimask_output,
                    use_reentrant=False,
                )
            else:
                sam_outputs = self._forward_sam_heads(
                    backbone_features=pix_feat_with_mem,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    high_res_features=high_res_features,
                    multimask_output=multimask_output,
                )
            (
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                _,
                object_score_logits,
            ) = sam_outputs
            all_pred_masks.append(low_res_masks)
            all_pred_high_res_masks.append(high_res_masks)
            all_pred_multimasks.append(low_res_multimasks)
            all_pred_high_res_multimasks.append(high_res_multimasks)
            all_pred_ious.append(ious)
            all_point_inputs.append(point_inputs)
            all_object_score_logits.append(object_score_logits)

        # Concatenate the masks along channel (to compute losses on all of them,
        # using `MultiStepIteractiveMasks`)
        current_out["multistep_pred_masks"] = torch.cat(all_pred_masks, dim=1)
        current_out["multistep_pred_masks_high_res"] = torch.cat(
            all_pred_high_res_masks, dim=1
        )
        current_out["multistep_pred_multimasks"] = all_pred_multimasks
        current_out["multistep_pred_multimasks_high_res"] = all_pred_high_res_multimasks
        current_out["multistep_pred_ious"] = all_pred_ious
        current_out["multistep_point_inputs"] = all_point_inputs
        current_out["multistep_object_score_logits"] = all_object_score_logits

        return point_inputs, sam_outputs
