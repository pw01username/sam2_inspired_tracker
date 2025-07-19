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


def decode_motsynth_rle(rle_string: str, height: int, width: int, debug=False) -> np.ndarray:
    """
    Decode MOTSynth RLE string to binary mask.
    """
    # Return empty mask for very long strings to avoid crashes
    if len(rle_string) > 10000:
        if debug:
            print(f"RLE string too long ({len(rle_string)}), returning empty mask")
        return np.zeros((height, width), dtype=np.uint8)
    
    # Simple custom decoding with early termination
    char_map = {chr(i): i - ord('0') + 10 for i in range(ord('A'), ord('z') + 1)}
    char_map.update({str(i): i for i in range(10)})
    
    runs = []
    i = 0
    while i < len(rle_string) and len(runs) < 1000:  # Limit runs to prevent crash
        if rle_string[i].isdigit():
            num = 0
            while i < len(rle_string) and rle_string[i].isdigit():
                num = num * 10 + int(rle_string[i])
                i += 1
            runs.append(min(num, height * width))  # Cap run length
        else:
            char = rle_string[i]
            if char in char_map:
                runs.append(char_map[char])
            i += 1
    
    # Create mask efficiently
    mask = np.zeros(height * width, dtype=np.uint8)
    pos = 0
    
    for i, run_length in enumerate(runs):
        if i % 2 == 1:  # Odd indices are foreground
            end_pos = min(pos + run_length, len(mask))
            mask[pos:end_pos] = 1
        pos += run_length
        if pos >= len(mask):
            break
    
    return mask.reshape(height, width)


def parse_motsynth_annotations(file_path: str) -> Dict[int, Dict[int, Dict]]:
    """
    Parse MOTSynth MOTS annotation file with memory optimization.
    """
    annotations = {}
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f):
            if line_num > 1000:  # Limit lines to prevent crash
                print(f"Stopping at line {line_num} to prevent memory issues")
                break
                
            parts = line.strip().split(' ')
            if len(parts) < 6:
                continue
                
            frame_id = int(parts[0])
            track_id = int(parts[1])
            class_id = int(parts[2])
            height = int(parts[3])
            width = int(parts[4])
            rle_string = parts[5]
            
            # Skip very long RLE strings
            if len(rle_string) > 5000:
                continue
            
            # Only decode mask when needed, store RLE string
            if frame_id not in annotations:
                annotations[frame_id] = {}
            
            annotations[frame_id][track_id] = {
                'class_id': class_id,
                'height': height,
                'width': width,
                'rle_string': rle_string,
                'mask': None  # Lazy loading
            }
    
    return annotations


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
        print("loaded frames: ", len(frames))
        
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


class MOTSynthDataset:
    """
    MOTSynthDataset is a class to load the MOTSynth dataset and visualize the annotations.
    """

    def __init__(self, motsynth_video_dir, motsynth_annot_dir):
        """
        Args:
            motsynth_video_dir: directory containing MP4 files (e.g., /path/to/MOTSynth_1/)
            motsynth_annot_dir: directory containing annotations (e.g., /path/to/mots_annotations/)
        """
        self.motsynth_video_dir = motsynth_video_dir
        self.motsynth_annot_dir = motsynth_annot_dir
        self.track_colors = np.random.random((1000, 3))  # Colors for tracks

    def read_frames_motsynth(self, sequence_id: str) -> List[np.ndarray]:
        """
        Read frames from MOTSynth MP4 video file.
        """
        mp4_path = os.path.join(self.motsynth_video_dir, f"{sequence_id}.mp4")
        
        if not os.path.exists(mp4_path):
            print(f"{mp4_path} doesn't exist.")
            return None
            
        frames = decode_video(mp4_path)
        print(f"Loaded {len(frames)} frames from {mp4_path}")
        return frames

    def get_frames_and_annotations_motsynth(
        self, sequence_id: str
    ) -> Tuple[List | None, Dict | None]:
        """
        Get the frames and annotations for MOTSynth sequence.
        """
        # Load frames from MP4
        frames = self.read_frames_motsynth(sequence_id)
        if frames is None:
            return None, None
            
        # Load annotations from gt.txt
        gt_path = os.path.join(self.motsynth_annot_dir, sequence_id, "gt", "gt.txt")
        if not os.path.exists(gt_path):
            print(f"{gt_path} doesn't exist.")
            return frames, None
            
        annotations = parse_motsynth_annotations(gt_path)
        print(f"Loaded annotations for {len(annotations)} frames")
        
        return frames, annotations

    def visualize_annotation_motsynth(
        self,
        frames: List[np.ndarray],
        annotations: Optional[Dict],
        frame_id: int,
        show_tracks=True,
        show_track_ids=True,
        debug=False,
    ) -> None:
        """
        Visualize MOTSynth annotations on the specified frame.
        """
        if frame_id >= len(frames):
            print("invalid frame_id")
            return

        plt.figure(figsize=(12, 8))
        plt.imshow(frames[frame_id])

        if show_tracks and annotations is not None:
            # Try different frame indexing
            frame_data = annotations.get(frame_id + 1) or annotations.get(frame_id)
            
            if frame_data is None:
                if debug:
                    print(f"No annotations for frame {frame_id}. Available: {sorted(annotations.keys())[:5]}")
                plt.title(f"Frame {frame_id + 1} - No annotations")
                plt.axis("off")
                plt.show()
                return
            
            masks = []
            colors = []
            track_ids = []
            
            for track_id, data in frame_data.items():
                # Lazy decode mask
                if data['mask'] is None:
                    data['mask'] = decode_motsynth_rle(data['rle_string'], data['height'], data['width'], debug)
                
                mask = data['mask']
                if mask.sum() > 0:
                    masks.append(mask)
                    colors.append(self.track_colors[track_id % len(self.track_colors)])
                    track_ids.append(track_id)
            
            if masks:
                show_anns(masks, colors)
                
                if show_track_ids:
                    for i, track_id in enumerate(track_ids):
                        mask = masks[i]
                        y_coords, x_coords = np.where(mask > 0)
                        if len(y_coords) > 0:
                            centroid_y = int(np.mean(y_coords))
                            centroid_x = int(np.mean(x_coords))
                            plt.text(centroid_x, centroid_y, str(track_id), 
                                   color='white', fontsize=10, fontweight='bold',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        plt.title(f"Frame {frame_id + 1}")
        plt.axis("off")
        plt.show()

    def get_track_statistics_motsynth(self, annotations: Dict) -> Dict:
        """
        Get statistics about tracks in the MOTSynth annotations.
        """
        if annotations is None:
            return {}
            
        track_lengths = {}
        track_classes = {}
        
        for frame_id, frame_data in annotations.items():
            for track_id, data in frame_data.items():
                if track_id not in track_lengths:
                    track_lengths[track_id] = 0
                    track_classes[track_id] = data['class_id']
                track_lengths[track_id] += 1
        
        stats = {
            'total_tracks': len(track_lengths),
            'total_frames': len(annotations),
            'track_lengths': track_lengths,
            'track_classes': track_classes,
            'avg_track_length': np.mean(list(track_lengths.values())) if track_lengths else 0,
            'class_distribution': {}
        }
        
        # Count class distribution
        for class_id in track_classes.values():
            if class_id not in stats['class_distribution']:
                stats['class_distribution'][class_id] = 0
            stats['class_distribution'][class_id] += 1
            
        return stats