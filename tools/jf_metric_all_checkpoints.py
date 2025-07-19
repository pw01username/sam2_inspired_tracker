

#!/usr/bin/env python3
import os
import glob
import subprocess
import concurrent.futures
import time

def process_checkpoint(checkpoint, base_video_dir, input_mask_dir, video_list_file, output_base_dir, sam2_cfg, parent_folder, force_recompute=False):
    """Process a single checkpoint with inference and evaluation"""
    ckpt_name = os.path.splitext(os.path.basename(checkpoint))[0]
    # Name output directory as <parent_folder>_<checkpoint_name>
    output_dir = os.path.join(output_base_dir, f"{parent_folder}_{ckpt_name}")
    
    # Check if output directory already exists and has content
    if os.path.exists(output_dir) and not force_recompute:
        # Look for files in the directory (assuming any files indicate completed processing)
        if len(os.listdir(output_dir)) > 0:
            print(f"Skipping checkpoint {ckpt_name}: Output directory already exists with content")
            return f"{ckpt_name} (skipped)"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Print which checkpoint is being processed
    print(f"Starting processing of checkpoint: {ckpt_name}")

    # 1) Run inference
    infer_cmd = [
        "python", "./tools/vos_inference.py",
        "--sam2_cfg", sam2_cfg,
        "--sam2_checkpoint", checkpoint,
        "--base_video_dir", base_video_dir,
        "--input_mask_dir", input_mask_dir,
        "--video_list_file", video_list_file,
        "--output_mask_dir", output_dir
    ]
    print(f"Running inference for {ckpt_name}: {' '.join(infer_cmd)}")
    subprocess.run(infer_cmd, check=True)

    # 2) Run evaluation on the produced masks
    eval_cmd = [
        "python", "sav_dataset/sav_evaluator.py",
        "--gt_root", input_mask_dir,
        "--pred_root", output_dir
    ]
    print(f"Running evaluator for {ckpt_name}: {' '.join(eval_cmd)}")
    subprocess.run(eval_cmd, check=True)
    
    print(f"Completed processing of checkpoint: {ckpt_name}")
    return f"{ckpt_name} (completed)"

def main():
    # Configuration
    checkpoint_dir = os.path.expanduser(
        "./sam2_logs/configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml/checkpoints"
        #"./sam2_logs/configs/sam2.1_training/sam2.1_hiera_large_MOSE_finetune.yaml/checkpoints"
    )
    base_video_dir = os.path.expanduser("./DAVIS/JPEGImages/")
    input_mask_dir = os.path.expanduser("./DAVIS/Annotations/")
    video_list_file = os.path.expanduser("./DAVIS/ImageSets/2017/val.txt")
    output_base_dir = os.path.expanduser("./outputs")
    sam2_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    #sam2_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    # Number of parallel processes to run
    max_parallel = 4
    
    # Whether to force recomputation even if output directory exists
    force_recompute = False  # Set to True to recompute all checkpoints

    # Derive parent folder name for output naming
    parent_folder = os.path.basename(os.path.normpath(checkpoint_dir))

    # Find all checkpoint files
    checkpoint_pattern = os.path.join(checkpoint_dir, "*.pt")
    checkpoints = sorted(glob.glob(checkpoint_pattern))

    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    
    print(f"Found {len(checkpoints)} checkpoints, processing up to {max_parallel} in parallel")
    print(f"Force recompute mode: {'ON' if force_recompute else 'OFF'}")
    
    # Count skipped checkpoints
    skipped_count = 0
    for checkpoint in checkpoints:
        ckpt_name = os.path.splitext(os.path.basename(checkpoint))[0]
        output_dir = os.path.join(output_base_dir, f"{parent_folder}_{ckpt_name}")
        if os.path.exists(output_dir) and not force_recompute and len(os.listdir(output_dir)) > 0:
            skipped_count += 1
    
    if skipped_count > 0:
        print(f"Will skip {skipped_count} checkpoints that already have output directories")
    
    # Use ProcessPoolExecutor to run up to max_parallel processes simultaneously
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_parallel) as executor:
        # Create a list to store future objects
        futures = []
        
        # Submit all checkpoint processing tasks to the executor
        for checkpoint in checkpoints:
            future = executor.submit(
                process_checkpoint, 
                checkpoint, 
                base_video_dir, 
                input_mask_dir, 
                video_list_file, 
                output_base_dir, 
                sam2_cfg, 
                parent_folder,
                force_recompute
            )
            futures.append(future)
        
        # Process results as they complete
        completed_count = 0
        skipped_count = 0
        error_count = 0
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if "(skipped)" in result:
                    skipped_count += 1
                    print(f"⏩ Skipped already processed: {result}")
                else:
                    completed_count += 1
                    print(f"✅ Successfully completed processing: {result}")
            except Exception as e:
                error_count += 1
                print(f"❌ Error processing checkpoint: {e}")
        
        print(f"\nSummary: {completed_count} completed, {skipped_count} skipped, {error_count} failed")

    print("All checkpoint processing completed!")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")