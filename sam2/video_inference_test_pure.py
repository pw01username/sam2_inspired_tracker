import cv2
import torch
import time
from sam2.build_sam import build_sam2_video_predictor
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


import logging
#logging.basicConfig(level=logging.DEBUG)

# Initialize model
checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# Load a frame from the video
video_path = "../demo/data/gallery/03_blocks.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError(f"Unable to open video: {video_path}")

# Read the first frame as an example
ret, frame = cap.read()
cap.release()

if not ret:
    raise ValueError("Unable to read a frame from the video.")

# Convert frame to RGB as the model expects RGB input
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Prepare dummy input prompts (for simplicity, these can be adjusted as needed)


# Benchmark inference time
n_iterations = 100  # Number of iterations for averaging
times = []

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    input_prompts = {
        "point_coords": torch.tensor([[100, 100], [200, 200]]),  # Example points
        "point_labels": torch.tensor([1, 0]),  # Foreground and background labels
    }
    
    # Warm up
    predictor.set_image(frame_rgb)
    predictor.predict(**input_prompts)

    # Timed iterations
    for _ in range(n_iterations):
        start_time = time.time()
        predictor.set_image(frame_rgb)
        
        masks, _, _ = predictor.predict(**input_prompts)
        end_time = time.time()

        times.append(end_time - start_time)

# Calculate average inference time
avg_time = sum(times) / len(times)

# Print results
print(f"Average inference time over {n_iterations} iterations: {avg_time:.4f} seconds")




# import os
# import torch
# import cv2
# from sam2.build_sam import build_sam2_video_predictor
# import time

# # Initialize model
# checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
# predictor = build_sam2_video_predictor(model_cfg, checkpoint)

# # Output directory
# output_dir = "./output_masks_s2"
# os.makedirs(output_dir, exist_ok=True)

# # Input video path
# video_path = '../demo/data/gallery/03_blocks.mp4'
# cap = cv2.VideoCapture(video_path)

# # Retrieve video properties
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# # Output video writer for visualization
# output_video_path = os.path.join(output_dir, "output_video.mp4")
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# # Initialize timers
# total_inference_time = 0.0
# frames = []
# masks_per_frame = []

# # GPU inference using SAM2VideoPredictor
# with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
#     # Start total inference timer
#     start_total_time = time.time()

#     # Initialize state for the video
#     state = predictor.init_state(video_path)

#     # Add initial prompts (adjust as needed for your video)
#     frame_idx = 0  # Start with the first frame
#     obj_id = 1  # Assign a unique object ID
#     box_prompt = [100, 100, 300, 300]  # Example box prompt, adjust coordinates
#     frame_idx, object_ids, masks = predictor.add_new_points_or_box(
#         state, frame_idx=frame_idx, obj_id=obj_id, box=box_prompt
#     )

#     # Process all frames
#     for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
#         total_inference_time += time.time() - start_total_time

#         # Save masks to memory for later processing
#         masks_per_frame.append(masks)

#         # Store frames for later visualization
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#         ret, frame = cap.read()
#         if ret:
#             frames.append(frame)

#     # End total inference timer
#     end_total_time = time.time()

# cap.release()  # Release video reader

# # CPU-bound tasks: Visualization and saving results
# for idx, (frame, masks) in enumerate(zip(frames, masks_per_frame)):
#     # Overlay masks on the frame for visualization
#     for mask_idx, mask in enumerate(masks):
#         mask = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
#         # Ensure mask is 2D
#         if mask.ndim > 2:
#             mask = mask.squeeze()  # Remove extra dimensions if present

#         mask_overlay = (mask * 255).astype('uint8')  # Convert to 8-bit grayscale
#         mask_overlay_colored = cv2.applyColorMap(mask_overlay, cv2.COLORMAP_JET)

#         frame = cv2.addWeighted(frame, 0.7, mask_overlay_colored, 0.3, 0)

#     # Save processed frame to video
#     out_video.write(frame)

#     # Save individual masks
#     mask_dir = os.path.join(output_dir, f"frame_{idx:04d}")
#     os.makedirs(mask_dir, exist_ok=True)
#     for i, mask in enumerate(masks):
#         mask_path = os.path.join(mask_dir, f"mask_{i:02d}.png")
#         if isinstance(mask, torch.Tensor):
#             mask = mask.cpu().numpy()  # Convert PyTorch tensor to NumPy array
#         cv2.imwrite(mask_path, (mask * 255).astype('uint8'))


# out_video.release()  # Release video writer

# # Benchmarking information
# avg_inference_time = total_inference_time / frame_count if frame_count > 0 else 0
# print(f"Total GPU-only inference time: {total_inference_time:.2f} seconds")
# print(f"Average inference time per frame: {avg_inference_time:.2f} seconds")
# print(f"Total video processing time (GPU + CPU): {end_total_time - start_total_time:.2f} seconds")

# print(f"Processing completed. Results saved in {output_dir}.")