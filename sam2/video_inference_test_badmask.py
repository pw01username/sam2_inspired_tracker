import os
import torch
import cv2
from sam2.build_sam import build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import time

# Initialize model
checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

# Initialize automatic mask generator
mask_generator = SAM2AutomaticMaskGenerator(
    predictor, points_per_side=32, stability_score_thresh=0.95, output_mode="binary_mask"
)

# Output directory
output_dir = "./output_masks_gpufast"
os.makedirs(output_dir, exist_ok=True)

# Input video path
video_path = '../demo/data/gallery/03_blocks.mp4'
cap = cv2.VideoCapture(video_path)

# Retrieve video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Collect frames and prepare output storage
frames = []
masks_list = []
print("Starting GPU tasks, inference.")
# GPU-only inference and mask generation
total_inference_time = 0.0
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    start_total_time = time.time()  # Start GPU inference timer
    
    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to the format expected by the model (HWC, uint8)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)  # Store original frame for later CPU processing

        # Start timer for this frame
        start_frame_time = time.time()

        # Generate masks using the automatic mask generator
        masks = mask_generator.generate(frame_rgb)
        masks_list.append(masks)  # Store masks for this frame

        # End timer for this frame
        total_inference_time += time.time() - start_frame_time

    end_total_time = time.time()  # End GPU inference timer

cap.release()  # Release video reader as we no longer need it

# CPU-bound tasks: Visualization and saving results
output_video_path = os.path.join(output_dir, "output_video.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

for frame_idx, (frame, masks) in enumerate(zip(frames, masks_list)):
    # Overlay masks on the frame for visualization
    for mask_info in masks:
        mask = mask_info['segmentation']
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()  # Convert to numpy if it's a tensor
        mask_overlay = (mask * 255).astype('uint8')
        mask_overlay_colored = cv2.applyColorMap(mask_overlay, cv2.COLORMAP_JET)
        frame = cv2.addWeighted(frame, 0.7, mask_overlay_colored, 0.3, 0)

    # Save processed frame to video
    out_video.write(frame)

    # Save masks for the frame
    mask_dir = os.path.join(output_dir, f"frame_{frame_idx:04d}")
    os.makedirs(mask_dir, exist_ok=True)
    for i, mask_info in enumerate(masks):
        mask_path = os.path.join(mask_dir, f"mask_{i:02d}.png")
        cv2.imwrite(mask_path, (mask_info['segmentation'] * 255).astype('uint8'))

out_video.release()  # Release video writer

# Benchmarking information
avg_inference_time = total_inference_time / frame_count if frame_count > 0 else 0
print(f"Total GPU-only inference time: {total_inference_time:.2f} seconds")
print(f"Average inference time per frame: {avg_inference_time:.2f} seconds")
print(f"Total processing time (GPU + CPU): {end_total_time - start_total_time:.2f} seconds")

print(f"Processing completed. Results saved in {output_dir}.")
