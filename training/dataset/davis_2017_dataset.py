import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class DAVISDataset(Dataset):
    """
    DAVIS2017 dataset loader.
    
    Expected DAVIS2017 folder structure:
      DAVIS2017/
         JPEGImages/
             video1/
                 00000.jpg, 00001.jpg, ...
             video2/
                 00000.jpg, 00001.jpg, ...
         Annotations/480p/
             video1/
                 00000.png, 00001.png, ...
             video2/
                 00000.png, 00001.png, ...
    """
    def __init__(self, root, split="train", transform=None, normalize_instance=True):
        """
        Args:
            root (str): Path to DAVIS2017 root folder.
            split (str): Dataset split to use. DAVIS typically has "train" and "val".
            transform (callable, optional): A function/transform that takes in a sample and returns a transformed version.
            normalize_instance (bool): Whether to normalize instance IDs by dividing by 125.
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.normalize_instance = normalize_instance
        
        # Define paths for images and annotations.
        self.image_dir = os.path.join(root, "JPEGImages")
        self.ann_dir = os.path.join(root, "Annotations", "480p")
        
        # Get list of video folders.
        self.videos = sorted(os.listdir(self.image_dir))
        self.samples = []  # list of tuples: (video, frame_name)
        for video in self.videos:
            video_img_dir = os.path.join(self.image_dir, video)
            frame_files = sorted([f for f in os.listdir(video_img_dir) if f.endswith(".jpg")])
            for frame_file in frame_files:
                self.samples.append((video, frame_file))
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video, frame_file = self.samples[idx]
        img_path = os.path.join(self.image_dir, video, frame_file)
        ann_path = os.path.join(self.ann_dir, video, frame_file.replace(".jpg", ".png"))
        
        # Load image and annotation.
        image = Image.open(img_path).convert("RGB")
        ann = Image.open(ann_path)
        # Convert annotation to numpy array; this array contains instance IDs.
        ann_np = np.array(ann)
        # Convert to tensor.
        ann_tensor = torch.from_numpy(ann_np).long()
        
        # Create instance ID map.
        # Here, we assume ann_tensor already encodes instance IDs (0 for background, 1, 2, ... for objects).
        # If desired, normalize the instance IDs (e.g., by dividing by 125).
        if self.normalize_instance:
            instance_ids = ann_tensor.float() / 125.0
        else:
            instance_ids = ann_tensor.float()
        
        # Construct sample dictionary.
        sample = {
            "image": image,
            "mask": ann_tensor,        # Raw mask with instance labels.
            "instance_ids": instance_ids,  # Normalized instance ID map.
            "video": video,
            "frame": frame_file,
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
