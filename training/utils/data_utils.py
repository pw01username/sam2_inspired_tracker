# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from PIL import Image as PILImage
from tensordict import tensorclass


@tensorclass
class BatchedVideoMetaData:
    """
    This class represents metadata about a batch of videos.
    Attributes:
        unique_objects_identifier: A tensor of shape Bx3 containing unique identifiers for each object in the batch. Index consists of (video_id, obj_id, frame_id)
        frame_orig_size: A tensor of shape Bx2 containing the original size of each frame in the batch.
    """

    unique_objects_identifier: torch.LongTensor
    frame_orig_size: torch.LongTensor


@tensorclass
class BatchedVideoDatapoint:
    """
    This class represents a batch of videos with associated annotations and metadata.
    Attributes:
        img_batch: A [TxBxCxHxW] tensor containing the image data for each frame in the batch, where T is the number of frames per video, and B is the number of videos in the batch.
        obj_to_frame_idx: A [TxOx2] tensor containing the image_batch index which the object belongs to. O is the number of objects in the batch.
        masks: A [TxOxHxW] tensor containing binary masks for each object in the batch.
        metadata: An instance of BatchedVideoMetaData containing metadata about the batch.
        dict_key: A string key used to identify the batch.
    """

    img_batch: torch.FloatTensor
    obj_to_frame_idx: torch.IntTensor
    masks: torch.BoolTensor  # Individual object masks
    combined_masks: torch.BoolTensor  # Combined mask of all objects
    instance_maps: torch.Tensor  # Instance ID maps
    metadata: BatchedVideoMetaData

    dict_key: str
    
    def to(self, device, non_blocking=False):
        self.img_batch = self.img_batch.to(device, non_blocking=non_blocking)
        self.masks = self.masks.to(device, non_blocking=non_blocking)
        self.combined_masks = self.combined_masks.to(device, non_blocking=non_blocking)
        self.instance_maps = self.instance_maps.to(device, non_blocking=non_blocking)
        return self

    def pin_memory(self, device=None):
        return self.apply(torch.Tensor.pin_memory, device=device)

    @property
    def num_frames(self) -> int:
        """
        Returns the number of frames per video.
        """
        return self.batch_size[0]

    @property
    def num_videos(self) -> int:
        """
        Returns the number of videos in the batch.
        """
        return self.img_batch.shape[1]

    @property
    def flat_obj_to_img_idx(self) -> torch.IntTensor:
        """
        Returns a flattened tensor containing the object to img index.
        The flat index can be used to access a flattened img_batch of shape [(T*B)xCxHxW]
        """
        frame_idx, video_idx = self.obj_to_frame_idx.unbind(dim=-1)
        flat_idx = video_idx * self.num_frames + frame_idx
        return flat_idx

    @property
    def flat_img_batch(self) -> torch.FloatTensor:
        """
        Returns a flattened img_batch_tensor of shape [(B*T)xCxHxW]
        """

        return self.img_batch.transpose(0, 1).flatten(0, 1)


@dataclass
class Object:
    # Id of the object in the media
    object_id: int
    # Index of the frame in the media (0 if single image)
    frame_index: int
    segment: Union[torch.Tensor, dict]  # RLE dict or binary mask


@dataclass
class Frame:
    data: Union[torch.Tensor, PILImage.Image]
    objects: List[Object]
    instance_id_map: Optional[torch.Tensor] = None  # Instance ID map

@dataclass
class VideoDatapoint:
    """Refers to an image/video and all its annotations"""

    frames: List[Frame]
    video_id: int
    size: Tuple[int, int]
    instance_id_maps: Optional[List[torch.Tensor]] = None  # List of instance ID maps


def collate_fn(
    batch: List[VideoDatapoint],
    dict_key,
) -> BatchedVideoDatapoint:
    """
    Args:
        batch: A list of VideoDatapoint instances.
        dict_key (str): A string key used to identify the batch.
    """
    img_batch = []
    for video in batch:
        img_batch += [torch.stack([frame.data for frame in video.frames], dim=0)]

    img_batch = torch.stack(img_batch, dim=0).permute((1, 0, 2, 3, 4))
    T = img_batch.shape[0]
    
    # Prepare data structures for sequential processing. Per-frame processing but batched across videos.
    step_t_objects_identifier = [[] for _ in range(T)]
    step_t_frame_orig_size = [[] for _ in range(T)]
    step_t_masks = [[] for _ in range(T)]
    step_t_obj_to_frame_idx = [
        [] for _ in range(T)
    ]  # List to store frame indices for each time step

    # For instance maps, we'll create one per video per frame
    instance_maps = []
    combined_masks = []  # For binary masks that combine all objects

    for video_idx, video in enumerate(batch):
        orig_video_id = video.video_id
        orig_frame_size = video.size
        h, w = orig_frame_size
        
        # Process each frame in the video
        video_instance_maps = []
        video_combined_masks = []
        
        for t, frame in enumerate(video.frames):
            # Create instance ID map for this frame
            res = 256#1024
            frame_instance_map = torch.zeros((res, res), dtype=torch.float)
            # Create combined binary mask for this frame (all objects vs background)
            frame_combined_mask = torch.zeros((res, res), dtype=torch.bool)
            
            objects = frame.objects
            for obj in objects:
                orig_obj_id = obj.object_id
                orig_frame_idx = obj.frame_index
                obj_mask = obj.segment.to(torch.bool)
                
                step_t_obj_to_frame_idx[t].append(
                    torch.tensor([t, video_idx], dtype=torch.int)
                )
                step_t_masks[t].append(obj_mask)
                step_t_objects_identifier[t].append(
                    torch.tensor([orig_video_id, orig_obj_id, orig_frame_idx])
                )
                step_t_frame_orig_size[t].append(torch.tensor(orig_frame_size))

                # Update the instance ID map - assign object ID to all pixels of this object
                frame_instance_map[obj_mask] = float(orig_obj_id)
                
                # Update the combined binary mask - mark all pixels that belong to any object
                frame_combined_mask = frame_combined_mask | obj_mask
            
            video_instance_maps.append(frame_instance_map)
            video_combined_masks.append(frame_combined_mask)
            
        # Stack maps and masks for this video
        instance_maps.append(torch.stack(video_instance_maps, dim=0))
        combined_masks.append(torch.stack(video_combined_masks, dim=0))
        
    obj_to_frame_idx = torch.stack(
        [
            torch.stack(obj_to_frame_idx, dim=0)
            for obj_to_frame_idx in step_t_obj_to_frame_idx
        ],
        dim=0,
    )
    # Stack individual object masks [T, O, H, W]
    masks = torch.stack([torch.stack(masks, dim=0) for masks in step_t_masks], dim=0)
    
    objects_identifier = torch.stack(
        [torch.stack(id, dim=0) for id in step_t_objects_identifier], dim=0
    )
    frame_orig_size = torch.stack(
        [torch.stack(id, dim=0) for id in step_t_frame_orig_size], dim=0
    )
    
    # Stack instance maps and combined masks across videos and transpose to [T, B, H, W]
    instance_maps = torch.stack(instance_maps, dim=0).permute((1, 0, 2, 3))
    combined_masks = torch.stack(combined_masks, dim=0).permute((1, 0, 2, 3))
    
    return BatchedVideoDatapoint(
        img_batch=img_batch,
        obj_to_frame_idx=obj_to_frame_idx,
        masks=masks,  # Individual object masks
        combined_masks=combined_masks,  # Combined masks of all objects
        instance_maps=instance_maps,  # Instance ID maps
        metadata=BatchedVideoMetaData(
            unique_objects_identifier=objects_identifier,
            frame_orig_size=frame_orig_size,
        ),
        dict_key=dict_key,
        batch_size=[T],
    )
