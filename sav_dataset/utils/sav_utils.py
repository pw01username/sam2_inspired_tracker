# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the sav_dataset directory of this source tree.
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_util


def decode_video(video_path: str) -> List[np.ndarray]:
    """
    Decode the video and return the RGB frames
    """
    video = cv2.VideoCapture(video_path)
    video_frames = []
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frames.append(frame)
        else:
            break
    return video_frames


def show_anns(masks, colors: List, borders=True) -> None:
    """
    show the annotations
    """
    # return if no masks
    if len(masks) == 0:
        return

    # sort masks by size
    sorted_annot_and_color = sorted(
        zip(masks, colors), key=(lambda x: x[0].sum()), reverse=True
    )
    H, W = sorted_annot_and_color[0][0].shape[0], sorted_annot_and_color[0][0].shape[1]

    canvas = np.ones((H, W, 4))
    canvas[:, :, 3] = 0  # set the alpha channel
    contour_thickness = max(1, int(min(5, 0.01 * min(H, W))))
    for mask, color in sorted_annot_and_color:
        canvas[mask] = np.concatenate([color, [0.55]])
        if borders:
            contours, _ = cv2.findContours(
                np.array(mask, dtype=np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
            )
            cv2.drawContours(
                canvas, contours, -1, (0.05, 0.05, 0.05, 1), thickness=contour_thickness
            )

    ax = plt.gca()
    ax.imshow(canvas)

import json
import os
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import pycocotools.mask as mask_util
from PIL import Image
import matplotlib.pyplot as plt
import re

class DAVISDataset:
    """
    DAVISDataset is a class to load the DAVIS dataset and visualize the annotations.
    Adapts the DAVIS format to be compatible with SAM training pipeline.
    """

    def __init__(self, davis_dir, annot_sample_rate=1):
        """
        Args:
            davis_dir: the directory of the DAVIS dataset
            annot_sample_rate: the sampling rate of the annotations.
                DAVIS annotations are typically available for each frame.
        """
        self.davis_dir = davis_dir
        self.annot_sample_rate = annot_sample_rate
        self.mask_colors = np.random.random((256, 3))
        
        # Cache for finding sequence directories case-insensitive
        self._sequence_dirs = {}
        self._scan_sequences()
        
    def _scan_sequences(self):
        """Scan available sequences and create a case-insensitive lookup"""
        frames_base_dir = os.path.join(self.davis_dir, 'JPEGImages')
        annotations_base_dir = os.path.join(self.davis_dir, 'Annotations')
        
        # Check if the base directories exist
        if not os.path.exists(frames_base_dir):
            print(f"Warning: Frames directory {frames_base_dir} does not exist")
            
        if not os.path.exists(annotations_base_dir):
            print(f"Warning: Annotations directory {annotations_base_dir} does not exist")
        
        # Get all sequence directories (case-insensitive lookup)
        if os.path.exists(frames_base_dir):
            for sequence_dir in os.listdir(frames_base_dir):
                full_path = os.path.join(frames_base_dir, sequence_dir)
                if os.path.isdir(full_path):
                    self._sequence_dirs[sequence_dir.lower()] = sequence_dir
            
            print(f"Found {len(self._sequence_dirs)} sequences in {frames_base_dir}")
        else:
            print("No sequences found as frames directory doesn't exist")

    def _get_sequence_dir(self, sequence_name):
        """Get the actual sequence directory name (case-insensitive lookup)"""
        if sequence_name in self._sequence_dirs:
            return self._sequence_dirs[sequence_name]
        elif sequence_name.lower() in self._sequence_dirs:
            return self._sequence_dirs[sequence_name.lower()]
        else:
            return None

    def _get_frame_number(self, filename):
        """Extract frame number from filename (e.g., 00042.jpg -> 42)"""
        # Look for a number in the filename
        match = re.search(r'(\d+)', os.path.splitext(filename)[0])
        if match:
            return int(match.group(1))
        return 0  # Default to 0 if no number found

    def read_frames(self, sequence_name):
        """
        Read frames from a DAVIS sequence, maintaining precise frame order by number
        """
        # Get the actual sequence directory name (case-insensitive)
        actual_sequence = self._get_sequence_dir(sequence_name)
        if actual_sequence is None:
            print(f"Sequence '{sequence_name}' not found. Available sequences: {list(self._sequence_dirs.values())[:5]} and {len(self._sequence_dirs)-5} more")
            return None, None
            
        frames_dir = os.path.join(self.davis_dir, 'JPEGImages', actual_sequence)
        
        # Debugging information
        print(f"Looking for frames in: {frames_dir}")
        print(f"Directory exists: {os.path.exists(frames_dir)}")
        
        if not os.path.exists(frames_dir):
            print(f"Frames directory does not exist: {frames_dir}")
            return None, None
            
        # Get all frame files with full details for precise matching
        frame_files = [f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not frame_files:
            print(f"No frames found for {sequence_name} in {frames_dir}")
            return None, None
            
        # Sort files by frame number (not just alphabetically)
        frame_files.sort(key=self._get_frame_number)
        
        # Store frame numbers for accurate annotation matching
        frame_numbers = [self._get_frame_number(f) for f in frame_files]
        
        # Sample frames based on annot_sample_rate
        sampled_indices = list(range(0, len(frame_files), self.annot_sample_rate))
        sampled_frame_files = [frame_files[i] for i in sampled_indices]
        sampled_frame_numbers = [frame_numbers[i] for i in sampled_indices]
        
        # Read the sampled frames
        frames = []
        for frame_file in sampled_frame_files:
            img_path = os.path.join(frames_dir, frame_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
            
        print(f"Loaded {len(frames)} frames for sequence {sequence_name}")
        return frames, sampled_frame_numbers

    def _load_davis_annotations(self, sequence_name, frame_numbers):
        """
        Load DAVIS annotations and convert to a format compatible with SAV,
        ensuring precise matching between frames and annotations
        
        Args:
            sequence_name: Name of the sequence
            frame_numbers: List of frame numbers corresponding to the loaded frames
        """
        # Get the actual sequence directory name (case-insensitive)
        actual_sequence = self._get_sequence_dir(sequence_name)
        if actual_sequence is None:
            print(f"Sequence '{sequence_name}' not found for annotations")
            return None
            
        annotations_dir = os.path.join(self.davis_dir, 'Annotations', actual_sequence)
        
        # Debugging information
        print(f"Looking for annotations in: {annotations_dir}")
        print(f"Directory exists: {os.path.exists(annotations_dir)}")
        
        if not os.path.exists(annotations_dir):
            print(f"Annotations for {sequence_name} not found at {annotations_dir}")
            return None
            
        # Get all annotation files
        all_annotation_files = [f for f in os.listdir(annotations_dir) if f.lower().endswith('.png')]
        if not all_annotation_files:
            print(f"No annotation files found in {annotations_dir}")
            return None
            
        # Create a mapping of frame numbers to annotation files for precise matching
        annotation_map = {}
        for ann_file in all_annotation_files:
            frame_num = self._get_frame_number(ann_file)
            annotation_map[frame_num] = ann_file
            
        # Create SAV-compatible annotation structure
        sav_compatible_annot = {"masklet": []}
        
        # For each frame we've loaded, find its corresponding annotation
        for frame_num in frame_numbers:
            if frame_num in annotation_map:
                annotation_file = annotation_map[frame_num]
                mask_path = os.path.join(annotations_dir, annotation_file)
                
                try:
                    mask = np.array(Image.open(mask_path))
                except Exception as e:
                    print(f"Error loading mask {mask_path}: {e}")
                    # If we can't load this annotation, add an empty list as placeholder
                    sav_compatible_annot["masklet"].append([])
                    continue
                
                # Process this frame's annotation
                frame_rles = []
                unique_ids = np.unique(mask)
                # Skip 0 which is background
                for obj_id in unique_ids[unique_ids > 0]:
                    binary_mask = (mask == obj_id).astype(np.uint8)
                    rle = mask_util.encode(np.asfortranarray(binary_mask))
                    # Convert to dictionary to make it JSON serializable
                    rle_dict = {'counts': rle['counts'].decode('utf-8'), 'size': rle['size']}
                    frame_rles.append(rle_dict)
                
                sav_compatible_annot["masklet"].append(frame_rles)
            else:
                print(f"Warning: No annotation found for frame {frame_num}")
                # No annotation for this frame, add an empty list as placeholder
                sav_compatible_annot["masklet"].append([])
        
        print(f"Loaded {len(sav_compatible_annot['masklet'])} annotation frames for {sequence_name}")
        
        # Verify frame and annotation alignment
        if len(frame_numbers) != len(sav_compatible_annot["masklet"]):
            print(f"WARNING: Mismatch between number of frames ({len(frame_numbers)}) and annotations ({len(sav_compatible_annot['masklet'])})")
            
        return sav_compatible_annot

    def get_frames_and_annotations(self, sequence_name):
        """
        Get the frames and annotations for a DAVIS sequence, ensuring proper alignment
        """
        frames, frame_numbers = self.read_frames(sequence_name)
        if frames is None:
            return None, None, None
            
        # Load manual annotations with precise frame number matching
        manual_annot = self._load_davis_annotations(sequence_name, frame_numbers)
        
        # For DAVIS, we don't have auto annotations
        auto_annot = None
        
        return frames, manual_annot, auto_annot
        
    def visualize_annotation(
        self,
        frames: List[np.ndarray],
        auto_annot: Optional[Dict],
        manual_annot: Optional[Dict],
        annotated_frame_id: int,
        show_auto=True,
        show_manual=True,
    ):
        """
        Visualize the annotations on the annotated_frame_id.
        """
        if frames is None or len(frames) == 0:
            print("No frames to visualize")
            return
            
        if annotated_frame_id >= len(frames):
            print(f"Invalid annotated_frame_id {annotated_frame_id}, max is {len(frames)-1}")
            return

        rles = []
        colors = []
        if show_manual and manual_annot is not None:
            if annotated_frame_id < len(manual_annot["masklet"]):
                rles.extend(manual_annot["masklet"][annotated_frame_id])
                colors.extend(
                    self.mask_colors[
                        : len(manual_annot["masklet"][annotated_frame_id])
                    ]
                )
                print(f"Frame {annotated_frame_id}: Found {len(manual_annot['masklet'][annotated_frame_id])} manual annotation objects")
            else:
                print(f"No manual annotation for frame {annotated_frame_id}")
                
        if show_auto and auto_annot is not None:
            if auto_annot.get("masklet") and annotated_frame_id < len(auto_annot["masklet"]):
                rles.extend(auto_annot["masklet"][annotated_frame_id])
                colors.extend(
                    self.mask_colors[: len(auto_annot["masklet"][annotated_frame_id])]
                )
                print(f"Frame {annotated_frame_id}: Found {len(auto_annot['masklet'][annotated_frame_id])} auto annotation objects")
            else:
                print(f"No auto annotation for frame {annotated_frame_id}")

        plt.figure(figsize=(10, 8))
        plt.imshow(frames[annotated_frame_id])
        plt.title(f"Frame {annotated_frame_id}")

        if len(rles) > 0:
            # Convert dictionary RLEs back to pycocotools format if needed
            decoded_rles = []
            for rle in rles:
                if isinstance(rle, dict) and 'counts' in rle and 'size' in rle:
                    if isinstance(rle['counts'], str):
                        rle_obj = {'counts': rle['counts'].encode('utf-8'), 'size': rle['size']}
                    else:
                        rle_obj = rle
                    decoded_rles.append(rle_obj)
                else:
                    decoded_rles.append(rle)
                    
            masks = [mask_util.decode(rle) > 0 for rle in decoded_rles]
            self.show_anns(masks, colors)
        else:
            print("No annotation will be shown")

        plt.axis("off")
        plt.show()
    
    def show_anns(self, masks, colors: List, borders=True) -> None:
        """
        Show the annotations
        """
        # return if no masks
        if len(masks) == 0:
            return

        # sort masks by size
        sorted_annot_and_color = sorted(
            zip(masks, colors), key=(lambda x: x[0].sum()), reverse=True
        )
        H, W = sorted_annot_and_color[0][0].shape[0], sorted_annot_and_color[0][0].shape[1]

        canvas = np.ones((H, W, 4))
        canvas[:, :, 3] = 0  # set the alpha channel
        contour_thickness = max(1, int(min(5, 0.01 * min(H, W))))
        for mask, color in sorted_annot_and_color:
            canvas[mask] = np.concatenate([color, [0.55]])
            if borders:
                contours, _ = cv2.findContours(
                    np.array(mask, dtype=np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
                )
                cv2.drawContours(
                    canvas, contours, -1, (0.05, 0.05, 0.05, 1), thickness=contour_thickness
                )

        ax = plt.gca()
        ax.imshow(canvas)
        
    def list_available_sequences(self):
        """List all available sequences in the DAVIS dataset"""
        return list(self._sequence_dirs.values())
        
    def visualize_sequence(self, sequence_name, frame_indices=None, max_frames=5):
        """
        Visualize multiple frames from a sequence to verify annotation alignment
        
        Args:
            sequence_name: Name of the sequence
            frame_indices: Specific frame indices to visualize, if None, evenly samples frames
            max_frames: Maximum number of frames to visualize if frame_indices is None
        """
        frames, manual_annot, auto_annot = self.get_frames_and_annotations(sequence_name)
        
        if frames is None or len(frames) == 0:
            print(f"No frames available for sequence {sequence_name}")
            return
            
        if frame_indices is None:
            # Sample evenly spaced frames
            if len(frames) <= max_frames:
                frame_indices = range(len(frames))
            else:
                step = len(frames) // max_frames
                frame_indices = range(0, len(frames), step)[:max_frames]
        
        # Visualize each selected frame
        for frame_id in frame_indices:
            if frame_id < len(frames):
                print(f"\nVisualizing frame {frame_id} of sequence {sequence_name}")
                self.visualize_annotation(frames, auto_annot, manual_annot, frame_id)
            else:
                print(f"Frame index {frame_id} out of range (max: {len(frames)-1})")

class SAVDataset:
    """
    SAVDataset is a class to load the SAV dataset and visualize the annotations.
    """

    def __init__(self, sav_dir, annot_sample_rate=4):
        """
        Args:
            sav_dir: the directory of the SAV dataset
            annot_sample_rate: the sampling rate of the annotations.
                The annotations are aligned with the videos at 6 fps.
        """
        self.sav_dir = sav_dir
        self.annot_sample_rate = annot_sample_rate
        self.manual_mask_colors = np.random.random((256, 3))
        self.auto_mask_colors = np.random.random((256, 3))

    def read_frames(self, mp4_path: str) -> None:
        """
        Read the frames and downsample them to align with the annotations.
        """
        if not os.path.exists(mp4_path):
            print(f"{mp4_path} doesn't exist.")
            return None
        else:
            # decode the video
            frames = decode_video(mp4_path)
            print(f"There are {len(frames)} frames decoded from {mp4_path} (24fps).")

            # downsample the frames to align with the annotations
            frames = frames[:: self.annot_sample_rate]
            print(
                f"Videos are annotated every {self.annot_sample_rate} frames. "
                "To align with the annotations, "
                f"downsample the video to {len(frames)} frames."
            )
            return frames

    def get_frames_and_annotations(
        self, video_id: str
    ) -> Tuple[List | None, Dict | None, Dict | None]:
        """
        Get the frames and annotations for video.
        """
        # load the video
        mp4_path = os.path.join(self.sav_dir, video_id + ".mp4")
        frames = self.read_frames(mp4_path)
        if frames is None:
            return None, None, None

        # load the manual annotations
        manual_annot_path = os.path.join(self.sav_dir, video_id + "_manual.json")
        if not os.path.exists(manual_annot_path):
            print(f"{manual_annot_path} doesn't exist. Something might be wrong.")
            manual_annot = None
        else:
            manual_annot = json.load(open(manual_annot_path))

        # load the manual annotations
        auto_annot_path = os.path.join(self.sav_dir, video_id + "_auto.json")
        if not os.path.exists(auto_annot_path):
            print(f"{auto_annot_path} doesn't exist.")
            auto_annot = None
        else:
            auto_annot = json.load(open(auto_annot_path))

        return frames, manual_annot, auto_annot

    def visualize_annotation(
        self,
        frames: List[np.ndarray],
        auto_annot: Optional[Dict],
        manual_annot: Optional[Dict],
        annotated_frame_id: int,
        show_auto=True,
        show_manual=True,
    ) -> None:
        """
        Visualize the annotations on the annotated_frame_id.
        If show_manual is True, show the manual annotations.
        If show_auto is True, show the auto annotations.
        By default, show both auto and manual annotations.
        """

        if annotated_frame_id >= len(frames):
            print("invalid annotated_frame_id")
            return

        rles = []
        colors = []
        if show_manual and manual_annot is not None:
            rles.extend(manual_annot["masklet"][annotated_frame_id])
            colors.extend(
                self.manual_mask_colors[
                    : len(manual_annot["masklet"][annotated_frame_id])
                ]
            )
        if show_auto and auto_annot is not None:
            rles.extend(auto_annot["masklet"][annotated_frame_id])
            colors.extend(
                self.auto_mask_colors[: len(auto_annot["masklet"][annotated_frame_id])]
            )

        plt.imshow(frames[annotated_frame_id])

        if len(rles) > 0:
            masks = [mask_util.decode(rle) > 0 for rle in rles]
            show_anns(masks, colors)
        else:
            print("No annotation will be shown")

        plt.axis("off")
        plt.show()
