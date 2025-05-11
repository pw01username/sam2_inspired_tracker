#!/usr/bin/env python3
import os
import glob
import subprocess

def main():
    # Configuration
    checkpoint_dir = os.path.expanduser(
        "./sam2_logs/configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml/checkpoints_23epochmemca1.1_latest_1024res"
    )
    base_video_dir = os.path.expanduser("./DAVIS/JPEGImages/")
    input_mask_dir = os.path.expanduser("./DAVIS/Annotations/")
    video_list_file = os.path.expanduser("./DAVIS/ImageSets/2017/val.txt")
    output_base_dir = os.path.expanduser("./outputs")
    sam2_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

    # Derive parent folder name for output naming
    parent_folder = os.path.basename(os.path.normpath(checkpoint_dir))

    # Find all checkpoint files
    checkpoint_pattern = os.path.join(checkpoint_dir, "*.pt")
    checkpoints = sorted(glob.glob(checkpoint_pattern))

    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return

    # Loop over each checkpoint
    for checkpoint in checkpoints:
        ckpt_name = os.path.splitext(os.path.basename(checkpoint))[0]
        # Name output directory as <parent_folder>_<checkpoint_name>
        output_dir = os.path.join(output_base_dir, f"{parent_folder}_{ckpt_name}")
        os.makedirs(output_dir, exist_ok=True)

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
        print(f"Running inference: {' '.join(infer_cmd)}")
        subprocess.run(infer_cmd, check=True)

        # 2) Run evaluation on the produced masks
        eval_cmd = [
            "python", "sav_dataset/sav_evaluator.py",
            "--gt_root", input_mask_dir,
            "--pred_root", output_dir
        ]
        print(f"Running evaluator: {' '.join(eval_cmd)}")
        subprocess.run(eval_cmd, check=True)

if __name__ == "__main__":
    main()
