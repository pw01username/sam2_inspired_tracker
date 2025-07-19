#!/usr/bin/env python3
"""
Simple VOS Inference Runner
Simplifies calling vos_inference.py by requiring only checkpoint and output path.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"✗ Command not found: {cmd[0]}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run VOS inference with simplified parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python tools/vos_jf.py --checkpoint 1.7_trainloss_1024res_memca/checkpoint.pt --output test1
    
Or with short options:
    python tools/vos_jf.py -c 1.7_trainloss_1024res_memca/checkpoint.pt -o test1
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
    #if not os.path.exists(f'/cluster/home/patricwu/ondemand/w/sam2.1/sam2/sam2_logs/configs/sam2.1_training/sam2.1_hiera_large_MOSE_finetune.yaml/{args.checkpoint}'):
    # if not os.path.exists(f'/cluster/home/patricwu/ondemand/w/sam2.1/sam2/sam2_logs/configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml/{args.checkpoint}'):
    #     print(f"Error: Checkpoint file not found: {args.checkpoint}")
    #     sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_path = Path(f'outputs/{args.output}')
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define the inference command with hardcoded paths
    inference_cmd = [
        'python', 'tools/vos_inference.py',
        
        #'--sam2_cfg', 'configs/sam2.1/sam2.1_hiera_b+.yaml',
        '--sam2_cfg', 'configs/sam2.1/sam2.1_hiera_l.yaml',
        
        #'--sam2_checkpoint', '/cluster/home/patricwu/ondemand/w/sam2.1/sam2/checkpoints/sam2.1_hiera_base_plus.pt', #
        '--sam2_checkpoint', '/cluster/home/patricwu/ondemand/w/sam2.1/sam2/checkpoints/sam2.1_hiera_large.pt', #        
        #'--sam2_checkpoint', f'/cluster/home/patricwu/ondemand/w/sam2.1/sam2/sam2_logs/configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml/{args.checkpoint}',
        #'--sam2_checkpoint', f'/cluster/home/patricwu/ondemand/w/sam2.1/sam2/sam2_logs/configs/sam2.1_training/sam2.1_hiera_large_MOSE_finetune.yaml/{args.checkpoint}',
        
        # mose val
        #'--base_video_dir', '/cluster/home/patricwu/Downloads/mose_valid/JPEGImages',
        #'--input_mask_dir', '/cluster/home/patricwu/Downloads/mose_valid/Annotations',
        
        # davis val
        #'--base_video_dir', 'DAVIS/JPEGImages/',
        #'--input_mask_dir', 'DAVIS/Annotations/',
        #'--video_list_file', 'DAVIS/ImageSets/2017/val.txt',

        # davis test
        #'--base_video_dir', '/cluster/home/patricwu/Downloads/davis_test/JPEGImages/480p/',
        #'--input_mask_dir', '/cluster/home/patricwu/Downloads/davis_test/Annotations/480p/',

        '--base_video_dir', '/cluster/home/patricwu/Downloads/DAVIS/JPEGImages/Full-Resolution/',
        '--input_mask_dir', '/cluster/home/patricwu/Downloads/DAVIS/Annotations/Full-Resolution/',

        

        # yt vos train, filtered
        #'--base_video_dir', '/cluster/home/patricwu/Downloads/trainytvos2/JPEGImages/',
        #'--input_mask_dir', '/cluster/home/patricwu/Downloads/trainytvos2/Annotations/',

        
        '--output_mask_dir', f'outputs/{args.output}'
    ]
    
    # Define the evaluation command
    eval_cmd = [
        'python', 'sav_dataset/sav_evaluator.py',
        
        #'--gt_root', '/cluster/home/patricwu/Downloads/trainytvos2/Annotations',
        
        # DAVIS VAL
        '--gt_root', 'DAVIS/Annotations',
        
        '--pred_root', f'outputs/{args.output}'
    ]
    
    if args.dry_run:
        print("DRY RUN MODE - Commands that would be executed:")
        print("\n1. Inference command:")
        print(' '.join(inference_cmd))
        if not args.skip_eval:
            print("\n2. Evaluation command:")
            print(' '.join(eval_cmd))
        return
    
    # Run inference
    if not run_command(inference_cmd, "VOS Inference"):
        print("\nInference failed. Exiting.")
        sys.exit(1)
    
    # Run evaluation unless skipped
    if not args.skip_eval:
        if not run_command(eval_cmd, "Evaluation"):
            print("\nEvaluation failed.")
            sys.exit(1)
    else:
        print("\nSkipping evaluation step.")
    
    print(f"\n✓ All tasks completed successfully!")
    print(f"Results saved to: outputs/{args.output}")


if __name__ == '__main__':
    main()