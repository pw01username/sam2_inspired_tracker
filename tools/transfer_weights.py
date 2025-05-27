#!/usr/bin/env python3
"""
Script to copy inter-object encoder weights from a 1-layer checkpoint to a 4-layer checkpoint.
This handles the case where you trained with num_layers=1 and want to expand to num_layers=4
by duplicating the single layer's weights across all 4 layers.
"""

import torch
import argparse
from pathlib import Path
import sys

def copy_layer_weights(source_checkpoint_path, target_checkpoint_path, output_path):
    """
    Copy inter-object encoder weights from 1-layer to 4-layer configuration.
    
    Args:
        source_checkpoint_path: Path to the checkpoint with 1-layer inter_object_encoder
        target_checkpoint_path: Path to the checkpoint with 4-layer inter_object_encoder 
        output_path: Path where the modified checkpoint will be saved
    """
    
    print(f"Loading source checkpoint: {source_checkpoint_path}")
    source_ckpt = torch.load(source_checkpoint_path, map_location='cpu')
    
    print(f"Loading target checkpoint: {target_checkpoint_path}")
    target_ckpt = torch.load(target_checkpoint_path, map_location='cpu')
    
    # Extract state dicts (handle different checkpoint formats)
    if 'model' in source_ckpt:
        source_state = source_ckpt['model']
    elif 'state_dict' in source_ckpt:
        source_state = source_ckpt['state_dict']
    else:
        source_state = source_ckpt
        
    if 'model' in target_ckpt:
        target_state = target_ckpt['model']
    elif 'state_dict' in target_ckpt:
        target_state = target_ckpt['state_dict']
    else:
        target_state = target_ckpt
    
    # Find inter_object_encoder keys in source (should be for 1 layer)
    source_inter_obj_keys = [k for k in source_state.keys() if 'inter_object_encoder' in k]
    target_inter_obj_keys = [k for k in target_state.keys() if 'inter_object_encoder' in k]
    
    print(f"Found {len(source_inter_obj_keys)} inter_object_encoder keys in source")
    print(f"Found {len(target_inter_obj_keys)} inter_object_encoder keys in target")
    
    if not source_inter_obj_keys:
        print("ERROR: No inter_object_encoder keys found in source checkpoint!")
        return False
        
    if not target_inter_obj_keys:
        print("ERROR: No inter_object_encoder keys found in target checkpoint!")
        return False
    
    # Print some example keys for debugging
    print("\nSource inter_object_encoder keys (first 5):")
    for key in source_inter_obj_keys[:5]:
        print(f"  {key}: {source_state[key].shape}")
    
    print("\nTarget inter_object_encoder keys (first 10):")
    for key in target_inter_obj_keys[:10]:
        print(f"  {key}: {target_state[key].shape}")
    
    # Copy all non-inter_object_encoder weights from source to target
    print("\nCopying non-inter_object_encoder weights from source to target...")
    copied_keys = 0
    for key, value in source_state.items():
        if 'inter_object_encoder' not in key:
            if key in target_state:
                if target_state[key].shape == value.shape:
                    target_state[key] = value.clone()
                    copied_keys += 1
                else:
                    print(f"WARNING: Shape mismatch for {key}: source {value.shape} vs target {target_state[key].shape}")
            else:
                print(f"WARNING: Key {key} not found in target checkpoint")
    
    print(f"Copied {copied_keys} non-inter_object_encoder weights")
    
    # Now handle the inter_object_encoder layers
    # Source should have layers.0.* pattern, target should have layers.0.*, layers.1.*, layers.2.*, layers.3.*
    
    # Find the pattern for layer 0 in source
    layer0_keys = [k for k in source_inter_obj_keys if 'layers.0.' in k]
    print(f"\nFound {len(layer0_keys)} layer 0 keys in source")
    
    if not layer0_keys:
        print("ERROR: No layers.0.* keys found in source inter_object_encoder!")
        return False
    
    # Copy layer 0 weights to all 4 layers in target
    print("Copying layer 0 weights to all target layers...")
    for layer_idx in range(4):  # 0, 1, 2, 3
        for source_key in layer0_keys:
            # Convert source key to target key for this layer
            target_key = source_key.replace('layers.0.', f'layers.{layer_idx}.')
            
            if target_key in target_state:
                if target_state[target_key].shape == source_state[source_key].shape:
                    target_state[target_key] = source_state[source_key].clone()
                    if layer_idx == 0:  # Only print for first layer to avoid spam
                        print(f"  Copied {source_key} -> {target_key}")
                else:
                    print(f"ERROR: Shape mismatch for {target_key}")
                    return False
            else:
                print(f"ERROR: Target key {target_key} not found!")
                return False
    
    # Copy the norm layer weights (these should be the same)
    norm_keys = [k for k in source_inter_obj_keys if 'norm.' in k or 'norm_inter_obj' in k]
    print(f"\nCopying {len(norm_keys)} norm layer weights...")
    for key in norm_keys:
        if key in target_state:
            if target_state[key].shape == source_state[key].shape:
                target_state[key] = source_state[key].clone()
                print(f"  Copied {key}")
            else:
                print(f"ERROR: Shape mismatch for norm key {key}")
                return False
    
    # Save the modified checkpoint
    print(f"\nSaving modified checkpoint to: {output_path}")
    torch.save(target_ckpt, output_path)
    
    print("Successfully copied weights!")
    return True

def main():
    parser = argparse.ArgumentParser(description='Copy inter-object encoder weights from 1-layer to 4-layer checkpoint')
    parser.add_argument('--source', '-s', required=True, help='Path to source checkpoint (1-layer)')
    parser.add_argument('--target', '-t', required=True, help='Path to target checkpoint (4-layer)')
    parser.add_argument('--output', '-o', required=True, help='Path for output checkpoint')
    
    args = parser.parse_args()
    
    # Validate paths
    source_path = Path(args.source)
    target_path = Path(args.target)
    output_path = Path(args.output)
    
    if not source_path.exists():
        print(f"ERROR: Source checkpoint not found: {source_path}")
        sys.exit(1)
        
    if not target_path.exists():
        print(f"ERROR: Target checkpoint not found: {target_path}")
        sys.exit(1)
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Perform the copy
    success = copy_layer_weights(source_path, target_path, output_path)
    
    if success:
        print("\n✅ Checkpoint weights copied successfully!")
        sys.exit(0)
    else:
        print("\n❌ Failed to copy checkpoint weights!")
        sys.exit(1)

if __name__ == "__main__":
    main()