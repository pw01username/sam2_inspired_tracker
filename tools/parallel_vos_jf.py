#!/usr/bin/env python3
"""
Parallel VOS Inference Runner
Batches videos into buckets and processes them in parallel for better GPU utilization.
"""

import argparse
import subprocess
import sys
import os
import glob
import tempfile
import concurrent.futures
import time
from pathlib import Path


def get_video_list(base_video_dir):
    """Get list of all video directories in the base video directory."""
    video_dirs = []
    if os.path.exists(base_video_dir):
        for item in os.listdir(base_video_dir):
            item_path = os.path.join(base_video_dir, item)
            if os.path.isdir(item_path):
                video_dirs.append(item)
    return sorted(video_dirs)


def create_video_list_file(video_names, temp_dir):
    """Create a temporary video list file."""
    temp_file = tempfile.NamedTemporaryFile(
        mode='w', 
        suffix='.txt', 
        dir=temp_dir, 
        delete=False
    )
    for video_name in video_names:
        temp_file.write(f"{video_name}\n")
    temp_file.close()
    return temp_file.name


def process_video_batch(batch_id, video_batch, base_video_dir, input_mask_dir, 
                       sam2_cfg, checkpoint, output_dir, temp_dir):
    """Process a batch of videos."""
    try:
        print(f"🔄 Batch {batch_id}: Processing {len(video_batch)} videos")
        
        # Create temporary video list file for this batch
        video_list_file = create_video_list_file(video_batch, temp_dir)
        
        # Define the inference command
        inference_cmd = [
            'python', 'tools/vos_inference.py',
            '--sam2_cfg', sam2_cfg,
            '--sam2_checkpoint', checkpoint,
            '--base_video_dir', base_video_dir,
            '--input_mask_dir', input_mask_dir,
            '--video_list_file', video_list_file,
            '--output_mask_dir', output_dir
        ]
        
        print(f"🚀 Batch {batch_id}: Running inference on videos: {', '.join(video_batch)}")
        
        # Run the inference command
        result = subprocess.run(inference_cmd, check=True, capture_output=True, text=True)
        
        # Clean up the temporary video list file
        os.unlink(video_list_file)
        
        print(f"✅ Batch {batch_id}: Completed successfully")
        return f"Batch {batch_id} (completed)"
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Batch {batch_id}: Failed with error code {e.returncode}")
        print(f"Error output: {e.stderr}")
        # Clean up the temporary video list file if it exists
        if 'video_list_file' in locals() and os.path.exists(video_list_file):
            os.unlink(video_list_file)
        raise e
    except Exception as e:
        print(f"❌ Batch {batch_id}: Failed with error: {e}")
        # Clean up the temporary video list file if it exists
        if 'video_list_file' in locals() and os.path.exists(video_list_file):
            os.unlink(video_list_file)
        raise e


def split_into_batches(video_list, num_batches):
    """Split video list into approximately equal batches."""
    if not video_list:
        return []
    
    batch_size = len(video_list) // num_batches
    remainder = len(video_list) % num_batches
    
    batches = []
    start_idx = 0
    
    for i in range(num_batches):
        # Add one extra video to first 'remainder' batches
        current_batch_size = batch_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_batch_size
        
        if start_idx < len(video_list):
            batch = video_list[start_idx:end_idx]
            if batch:  # Only add non-empty batches
                batches.append(batch)
        
        start_idx = end_idx
    
    return batches


def run_evaluation(input_mask_dir, output_dir):
    """Run evaluation after all inference is complete."""
    eval_cmd = [
        'python', 'sav_dataset/sav_evaluator.py',
        '--gt_root', input_mask_dir,
        '--pred_root', output_dir
    ]
    
    print(f"\n{'='*60}")
    print(f"Running: Evaluation")
    print(f"Command: {' '.join(eval_cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(eval_cmd, check=True)
        print(f"✓ Evaluation completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Evaluation failed with error code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run VOS inference with parallel video batch processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python tools/parallel_vos_jf.py --checkpoint checkpoints/sam2.1_hiera_base_plus.pt --output test1 --num-processes 4
    
Or with short options:
    python tools/parallel_vos_jf.py -c checkpoints/sam2.1_hiera_base_plus.pt -o test1 -n 4
        """
    )
    
    parser.add_argument(
        '-c', '--checkpoint',
        required=True,
        help='Path to SAM2 checkpoint file'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output directory for masks'
    )
    
    parser.add_argument(
        '-n', '--num-processes',
        type=int,
        default=4,
        help='Number of parallel processes to run (default: 4)'
    )
    
    parser.add_argument(
        '--sam2-cfg',
        default='configs/sam2.1/sam2.1_hiera_b+.yaml',
        help='SAM2 config file path'
    )
    
    parser.add_argument(
        '--base-video-dir',
        default='/cluster/home/patricwu/Downloads/train/train_mose_davis/JPEGImages',
        help='Base directory containing video frames'
    )
    
    parser.add_argument(
        '--input-mask-dir',
        default='/cluster/home/patricwu/Downloads/train/train_mose_davis/Annotations',
        help='Directory containing input masks'
    )
    
    parser.add_argument(
        '--skip-eval',
        action='store_true',
        help='Skip evaluation step after inference'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without executing them'
    )
    
    args = parser.parse_args()
    
    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_path = Path(f'outputs/{args.output}')
    output_path.mkdir(parents=True, exist_ok=True)
    output_dir = str(output_path)
    
    # Get list of all videos
    print(f"📁 Scanning for videos in: {args.base_video_dir}")
    video_list = get_video_list(args.base_video_dir)
    
    if not video_list:
        print(f"❌ No videos found in {args.base_video_dir}")
        sys.exit(1)
    
    print(f"📊 Found {len(video_list)} videos")
    print(f"🔧 Using {args.num_processes} parallel processes")
    
    # Split videos into batches
    video_batches = split_into_batches(video_list, args.num_processes)
    print(f"📦 Split into {len(video_batches)} batches")
    
    for i, batch in enumerate(video_batches):
        print(f"   Batch {i+1}: {len(batch)} videos - {', '.join(batch[:3])}{'...' if len(batch) > 3 else ''}")
    
    if args.dry_run:
        print("\n🧪 DRY RUN MODE - Commands that would be executed:")
        temp_dir = tempfile.mkdtemp()
        try:
            for i, batch in enumerate(video_batches):
                video_list_file = create_video_list_file(batch, temp_dir)
                inference_cmd = [
                    'python', 'tools/vos_inference.py',
                    '--sam2_cfg', args.sam2_cfg,
                    '--sam2_checkpoint', args.checkpoint,
                    '--base_video_dir', args.base_video_dir,
                    '--input_mask_dir', args.input_mask_dir,
                    '--video_list_file', video_list_file,
                    '--output_mask_dir', output_dir
                ]
                print(f"\nBatch {i+1} command:")
                print(' '.join(inference_cmd))
                os.unlink(video_list_file)
        finally:
            os.rmdir(temp_dir)
        
        if not args.skip_eval:
            print(f"\nEvaluation command:")
            eval_cmd = [
                'python', 'sav_dataset/sav_evaluator.py',
                '--gt_root', args.input_mask_dir,
                '--pred_root', output_dir
            ]
            print(' '.join(eval_cmd))
        return
    
    # Create temporary directory for video list files
    temp_dir = tempfile.mkdtemp()
    
    try:
        start_time = time.time()
        
        # Run inference in parallel
        print(f"\n🚀 Starting parallel inference with {args.num_processes} processes...")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_processes) as executor:
            # Submit all batch processing tasks
            futures = []
            for i, batch in enumerate(video_batches):
                future = executor.submit(
                    process_video_batch,
                    i + 1,  # batch_id (1-indexed)
                    batch,
                    args.base_video_dir,
                    args.input_mask_dir,
                    args.sam2_cfg,
                    args.checkpoint,
                    output_dir,
                    temp_dir
                )
                futures.append(future)
            
            # Process results as they complete
            completed_count = 0
            error_count = 0
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    completed_count += 1
                    print(f"✅ {result}")
                except Exception as e:
                    error_count += 1
                    print(f"❌ Error processing batch: {e}")
            
            print(f"\n📊 Inference Summary: {completed_count} completed, {error_count} failed")
        
        inference_time = time.time() - start_time
        print(f"⏱️  Inference time: {inference_time:.2f} seconds")
        
        # Run evaluation unless skipped
        if not args.skip_eval and error_count == 0:
            print(f"\n🔍 Running evaluation...")
            eval_start_time = time.time()
            
            if run_evaluation(args.input_mask_dir, output_dir):
                eval_time = time.time() - eval_start_time
                print(f"⏱️  Evaluation time: {eval_time:.2f} seconds")
            else:
                print("❌ Evaluation failed.")
                sys.exit(1)
        elif error_count > 0:
            print("⚠️  Skipping evaluation due to inference errors.")
        else:
            print("ℹ️  Skipping evaluation step.")
        
        total_time = time.time() - start_time
        print(f"\n✅ All tasks completed successfully!")
        print(f"⏱️  Total execution time: {total_time:.2f} seconds")
        print(f"📁 Results saved to: {output_dir}")
        
    finally:
        # Clean up temporary directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()