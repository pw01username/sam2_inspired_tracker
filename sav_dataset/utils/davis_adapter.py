# davis_adapter.py
import json
import os
import re
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import pycocotools.mask as mask_util
from PIL import Image
import matplotlib.pyplot as plt
import shutil

class DAVIStoSAVAdapter:
    """
    Adapter class that maintains the SAVDataset API but works with DAVIS dataset,
    with improved frame-to-annotation alignment and proper manual annotation handling.
    """
    def __init__(self, davis_dir, sav_dir=None, temp_dir=None, annot_sample_rate=4):
        """
        Initialize the adapter
        
        Args:
            davis_dir: Path to the DAVIS dataset
            sav_dir: Path to save/load the converted SAV format (if None, uses temp_dir)
            temp_dir: Temporary directory for conversion (if None, creates one)
            annot_sample_rate: Annotation sample rate (same as SAVDataset)
        """
        self.davis_dir = davis_dir
        self.annot_sample_rate = annot_sample_rate
        self.mask_colors = np.random.random((256, 3))
        
        # Setup conversion directory
        self.temp_created = False
        if sav_dir is None:
            if temp_dir is None:
                import tempfile
                self.sav_dir = tempfile.mkdtemp()
                self.temp_created = True
            else:
                self.sav_dir = temp_dir
                os.makedirs(self.sav_dir, exist_ok=True)
        else:
            self.sav_dir = sav_dir
            os.makedirs(self.sav_dir, exist_ok=True)
        
        # Scan sequences and create lookup maps
        self._sequence_dirs = {}
        self._scan_sequences()
        
        # Check if conversion has been done already
        self.mapping_file = os.path.join(self.sav_dir, "davis_to_sav_mapping.json")
        if os.path.exists(self.mapping_file):
            with open(self.mapping_file, 'r') as f:
                self.davis_to_sav_mapping = json.load(f)
            self.sav_to_davis_mapping = {v: k for k, v in self.davis_to_sav_mapping.items()}
            print(f"Loaded existing conversion from {self.mapping_file}")
            
            # Check if we need to regenerate the conversion
            need_reconversion = False
            for sav_id in self.sav_to_davis_mapping.keys():
                manual_annot_path = os.path.join(self.sav_dir, f"{sav_id}_manual.json")
                if not os.path.exists(manual_annot_path):
                    need_reconversion = True
                    break
            
            if need_reconversion:
                print("Missing annotation files. Forcing full reconversion...")
                self.davis_to_sav_mapping = self.convert_davis_to_sav(self.davis_dir, self.sav_dir, force=True)
                self.sav_to_davis_mapping = {v: k for k, v in self.davis_to_sav_mapping.items()}
        else:
            # Need to do conversion
            print(f"Converting DAVIS dataset to SAV format in {self.sav_dir}")
            self.davis_to_sav_mapping = self.convert_davis_to_sav(self.davis_dir, self.sav_dir)
            self.sav_to_davis_mapping = {v: k for k, v in self.davis_to_sav_mapping.items()}
    
    def _scan_sequences(self):
        """Scan available sequences and create a case-insensitive lookup"""
        frames_base_dir = os.path.join(self.davis_dir, 'JPEGImages')
        
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
    
    def __del__(self):
        """Cleanup temp directory if created"""
        if hasattr(self, 'temp_created') and self.temp_created and os.path.exists(self.sav_dir):
            try:
                shutil.rmtree(self.sav_dir)
            except:
                pass
    
    def convert_davis_to_sav(self, davis_dir, output_dir, fps=6, force=False):
        """
        Convert DAVIS dataset to SAV format for SAM training, with improved frame-annotation alignment
        
        Args:
            davis_dir: Path to the DAVIS dataset (should contain JPEGImages and Annotations folders)
            output_dir: Output directory to save the converted dataset
            fps: Frames per second for the output videos (default: 6, matching SAV dataset)
            force: Force reconversion even if files exist
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all sequences from DAVIS
        sequences_dir = os.path.join(davis_dir, 'JPEGImages')
        sequences = [d for d in os.listdir(sequences_dir) if os.path.isdir(os.path.join(sequences_dir, d))]
        
        sequence_mapping = {}  # Maps DAVIS sequence names to SAV-style IDs
        
        for i, sequence in enumerate(sequences):
            sav_id = f"sav_{i+1:06d}"  # Format: sav_000001, sav_000002, etc.
            sequence_mapping[sequence] = sav_id
            
            video_path = os.path.join(output_dir, f"{sav_id}.mp4")
            manual_annot_path = os.path.join(output_dir, f"{sav_id}_manual.json")
            
            # Skip if files already exist and not forcing reconversion
            if not force and os.path.exists(video_path) and os.path.exists(manual_annot_path):
                print(f"Files for {sequence} already exist, skipping. Set force=True to reconvert.")
                continue
                
            # Create video from frames
            frames_dir = os.path.join(davis_dir, 'JPEGImages', sequence)
            if not os.path.exists(frames_dir):
                print(f"Frames directory for {sequence} not found at {frames_dir}, skipping")
                continue
                
            frame_files = [f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not frame_files:
                print(f"No frames found for {sequence}, skipping")
                continue
            
            # Sort by frame number, not just alphabetically
            frame_files.sort(key=self._get_frame_number)
                
            # Read first frame to get dimensions
            first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
            if first_frame is None:
                print(f"Could not read first frame for {sequence}, skipping")
                continue
                
            height, width = first_frame.shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            # Get frame numbers for all frames
            frame_numbers = []
            
            for frame_file in frame_files:
                img_path = os.path.join(frames_dir, frame_file)
                frame = cv2.imread(img_path)
                if frame is not None:
                    video_writer.write(frame)
                    frame_numbers.append(self._get_frame_number(frame_file))
                else:
                    print(f"Could not read frame {img_path}")
            
            video_writer.release()
            print(f"Created video for {sequence} at {video_path} with {len(frame_numbers)} frames")
            
            # Process annotations
            annotations_dir = os.path.join(davis_dir, 'Annotations', sequence)
            if not os.path.exists(annotations_dir):
                print(f"Annotations directory for {sequence} not found at {annotations_dir}")
                continue
                
            all_annotation_files = [f for f in os.listdir(annotations_dir) if f.lower().endswith('.png')]
            
            if not all_annotation_files:
                print(f"No annotation files found for {sequence}")
                continue
                
            # Create mapping of frame number to annotation file
            annotation_map = {}
            for ann_file in all_annotation_files:
                frame_num = self._get_frame_number(ann_file)
                annotation_map[frame_num] = ann_file
            
            # Create SAV-compatible annotation format
            manual_annot = {"masklet": []}
            
            # Sample frame numbers at the correct rate (this must match how the frames are sampled in read_frames)
            sampled_indices = list(range(0, len(frame_numbers), self.annot_sample_rate))
            sampled_frame_numbers = [frame_numbers[i] for i in sampled_indices]
            
            # Process annotations
            for frame_num in sampled_frame_numbers:
                if frame_num in annotation_map:
                    annotation_file = annotation_map[frame_num]
                    mask_path = os.path.join(annotations_dir, annotation_file)
                    
                    try:
                        mask = np.array(Image.open(mask_path))
                    except Exception as e:
                        print(f"Error loading mask {mask_path}: {e}")
                        # If we can't load this annotation, add an empty list as placeholder
                        manual_annot["masklet"].append([])
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
                    
                    manual_annot["masklet"].append(frame_rles)
                else:
                    # No annotation for this frame, add an empty list as placeholder
                    manual_annot["masklet"].append([])
            
            # Save manual annotations 
            with open(manual_annot_path, 'w') as f:
                json.dump(manual_annot, f)
                
            print(f"Created manual annotations for {sequence} with {len(manual_annot['masklet'])} frames")
            print(f"Sampled frame numbers: {sampled_frame_numbers[:5]}...")
        
        # Save sequence mapping for reference
        mapping_path = os.path.join(output_dir, "davis_to_sav_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump(sequence_mapping, f)
            
        print(f"Conversion complete. Saved {len(sequence_mapping)} sequences to {output_dir}")
        return sequence_mapping
    
    def read_frames(self, mp4_path):
        """Read frames from MP4 file with proper downsampling"""
        if not os.path.exists(mp4_path):
            print(f"{mp4_path} doesn't exist.")
            return None
        else:
            # Decode the video
            video = cv2.VideoCapture(mp4_path)
            frames = []
            while video.isOpened():
                ret, frame = video.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    break
            video.release()
            
            total_frames = len(frames)
            print(f"There are {total_frames} frames decoded from {mp4_path}")

            # Downsample the frames to align with the annotations
            # This must match how frames are sampled during conversion
            sampled_frames = frames[::self.annot_sample_rate]
            
            print(
                f"Videos are annotated every {self.annot_sample_rate} frames. "
                "To align with the annotations, "
                f"downsample the video to {len(sampled_frames)} frames."
            )
            return sampled_frames
    
    def get_frames_and_annotations(self, video_id):
        """
        Get frames and annotations using the SAV ID format or DAVIS sequence name
        
        Args:
            video_id: SAV-style ID (e.g., "sav_000001") or DAVIS sequence name
        
        Returns:
            Tuple of (frames, manual_annot, auto_annot)
        """
        # Check if this is a DAVIS sequence name rather than a SAV ID
        if video_id in self.davis_to_sav_mapping:
            sav_id = self.davis_to_sav_mapping[video_id]
            print(f"Converted DAVIS sequence name '{video_id}' to SAV ID: {sav_id}")
            video_id = sav_id
        elif video_id.lower() in {k.lower(): v for k, v in self.davis_to_sav_mapping.items()}:
            # Case-insensitive lookup in davis names
            lower_mapping = {k.lower(): v for k, v in self.davis_to_sav_mapping.items()}
            sav_id = lower_mapping[video_id.lower()]
            print(f"Converted DAVIS sequence name '{video_id}' to SAV ID: {sav_id}")
            video_id = sav_id
        elif not video_id.startswith("sav_"):
            davis_sequences = list(self.davis_to_sav_mapping.keys())
            print(f"Invalid video_id: {video_id}. Available DAVIS sequences: {davis_sequences[:5]} (and {len(davis_sequences)-5} more)")
            return None, None, None
        
        # Load the video
        mp4_path = os.path.join(self.sav_dir, video_id + ".mp4")
        frames = self.read_frames(mp4_path)
        if frames is None:
            return None, None, None

        # Load the manual annotations
        manual_annot_path = os.path.join(self.sav_dir, video_id + "_manual.json")
        if not os.path.exists(manual_annot_path):
            print(f"{manual_annot_path} doesn't exist.")
            manual_annot = None
        else:
            with open(manual_annot_path, 'r') as f:
                manual_annot = json.load(f)
                
            # Verify we have the right number of annotations
            if manual_annot and "masklet" in manual_annot:
                print(f"Loaded {len(manual_annot['masklet'])} manual annotation frames")
                if len(manual_annot["masklet"]) != len(frames):
                    print(f"WARNING: Mismatch between frames ({len(frames)}) and annotations ({len(manual_annot['masklet'])})")
                    
                    # Fix the mismatch by adjusting annotations to match frame count
                    if len(manual_annot["masklet"]) > len(frames):
                        print(f"Trimming annotation array to match frame count")
                        manual_annot["masklet"] = manual_annot["masklet"][:len(frames)]
                    elif len(manual_annot["masklet"]) < len(frames):
                        print(f"Padding annotation array with empty annotations")
                        # Add empty annotations to match frame count
                        while len(manual_annot["masklet"]) < len(frames):
                            manual_annot["masklet"].append([])

        # For DAVIS dataset converted to SAV format, there are no auto annotations
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
        
        # For DAVIS, we should primarily use manual annotations
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
                
        # For DAVIS, auto_annot should be None, but include this for compatibility
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
        Same as the original SAVDataset.show_anns
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
        
    def get_available_sequences(self):
        """
        Get a list of available sequences in SAV ID format
        
        Returns:
            List of SAV IDs
        """
        return list(self.sav_to_davis_mapping.keys())
        
    def get_davis_sequence_name(self, sav_id):
        """
        Get the original DAVIS sequence name from a SAV ID
        
        Args:
            sav_id: SAV-style ID (e.g., "sav_000001")
        
        Returns:
            Original DAVIS sequence name
        """
        return self.sav_to_davis_mapping.get(sav_id)
        
    def get_sav_id(self, davis_sequence):
        """
        Get the SAV ID from a DAVIS sequence name
        
        Args:
            davis_sequence: DAVIS sequence name (e.g., "bear")
        
        Returns:
            SAV-style ID
        """
        return self.davis_to_sav_mapping.get(davis_sequence)
        
    def visualize_sequence(self, sequence_name, frame_indices=None, max_frames=5):
        """
        Visualize multiple frames from a sequence to verify annotation alignment
        
        Args:
            sequence_name: SAV ID or DAVIS sequence name
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
                # For DAVIS, we want to show manual annotations, not auto
                self.visualize_annotation(
                    frames, auto_annot, manual_annot, frame_id, 
                    show_auto=False, show_manual=True
                )
            else:
                print(f"Frame index {frame_id} out of range (max: {len(frames)-1})")
                
    def force_reconversion(self):
        """Force a complete reconversion of the dataset"""
        print("Forcing complete reconversion of DAVIS dataset...")
        self.davis_to_sav_mapping = self.convert_davis_to_sav(self.davis_dir, self.sav_dir, force=True)
        self.sav_to_davis_mapping = {v: k for k, v in self.davis_to_sav_mapping.items()}
        print("Reconversion complete")