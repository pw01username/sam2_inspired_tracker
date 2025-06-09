#!/usr/bin/env python3
"""
Script to copy memory attention weights (cross_obj_attn, obj_embeddings, obj_emb_scale) 
from one checkpoint to another.
"""

import torch
import argparse
from pathlib import Path
import sys

def copy_memory_attention_weights(source_checkpoint_path, target_checkpoint_path, output_path):
    """
    Copy memory attention weights from source to target checkpoint.
    
    Args:
        source_checkpoint_path: Path to the source checkpoint
        target_checkpoint_path: Path to the target checkpoint 
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
    
    # Find memory attention keys we're interested in
    memory_patterns = ['cross_obj_attn', 'obj_embeddings', 'obj_emb_scale']
    
    # Find all keys related to memory attention components
    source_memory_keys = []
    for pattern in memory_patterns:
        keys = [k for k in source_state.keys() if pattern in k]
        source_memory_keys.extend(keys)
    
    target_memory_keys = []
    for pattern in memory_patterns:
        keys = [k for k in target_state.keys() if pattern in k]
        target_memory_keys.extend(keys)
    
    print(f"\nFound {len(source_memory_keys)} memory attention keys in source")
    print(f"Found {len(target_memory_keys)} memory attention keys in target")
    
    if not source_memory_keys:
        print("ERROR: No memory attention keys (cross_obj_attn, obj_embeddings, obj_emb_scale) found in source checkpoint!")
        print("These keys should be in the memory attention module.")
        # Help debug by showing memory-related keys
        memory_related = [k for k in source_state.keys() if 'memory' in k.lower() or 'mem' in k.lower()]
        if memory_related:
            print("\nFound these memory-related keys (first 10):")
            for key in memory_related[:10]:
                print(f"  {key}")
        return False
    
    print(f"\nFound {len(source_memory_keys)} memory attention keys in source")
    print(f"Found {len(target_memory_keys)} memory attention keys in target")
    
    if not source_memory_keys:
        print("ERROR: No memory attention keys (cross_obj_attn, obj_embeddings, obj_emb_scale) found in source checkpoint!")
        print("These keys should be in the memory attention module.")
        # Help debug by showing memory-related keys
        memory_related = [k for k in source_state.keys() if 'memory' in k.lower() or 'mem' in k.lower()]
        if memory_related:
            print("\nFound these memory-related keys (first 10):")
            for key in memory_related[:10]:
                print(f"  {key}")
        return False
    
    if not target_memory_keys:
        print("Note: No matching keys found in target - will add them as new keys.")
    
    # Print found keys for debugging
    if source_memory_keys:
        print("\nSource memory attention keys:")
        for key in sorted(source_memory_keys):
            print(f"  {key}: {source_state[key].shape if hasattr(source_state[key], 'shape') else 'scalar'}")
    
    if target_memory_keys:
        print("\nTarget memory attention keys (will be replaced):")
        for key in sorted(target_memory_keys):
            print(f"  {key}: {target_state[key].shape if hasattr(target_state[key], 'shape') else 'scalar'}")
    
    # Copy the memory attention weights
    copied_keys = 0
    added_keys = 0
    skipped_keys = 0
    
    print("\nCopying memory attention weights...")
    for source_key in source_memory_keys:
        if source_key in target_state:
            # Key exists in target - check if we can copy
            source_val = source_state[source_key]
            target_val = target_state[source_key]
            
            # Check if shapes match (for tensors)
            if hasattr(source_val, 'shape') and hasattr(target_val, 'shape'):
                if source_val.shape == target_val.shape:
                    target_state[source_key] = source_val.clone()
                    print(f"  Copied {source_key} (shape: {source_val.shape})")
                    copied_keys += 1
                else:
                    print(f"  WARNING: Shape mismatch for {source_key}: source {source_val.shape} vs target {target_val.shape}")
                    skipped_keys += 1
            else:
                # For scalars or other types
                target_state[source_key] = source_val
                print(f"  Copied {source_key} (scalar/other)")
                copied_keys += 1
        else:
            # Key doesn't exist in target - add it!
            source_val = source_state[source_key]
            target_state[source_key] = source_val.clone() if hasattr(source_val, 'clone') else source_val
            shape_info = f" (shape: {source_val.shape})" if hasattr(source_val, 'shape') else " (scalar)"
            print(f"  ADDED {source_key}{shape_info}")
            added_keys += 1
    
    print(f"\nSummary:")
    print(f"  Added (new): {added_keys} keys")
    print(f"  Copied (replaced): {copied_keys} keys")
    print(f"  Skipped: {skipped_keys} keys")
    
    # Save the modified checkpoint
    print(f"\nSaving modified checkpoint to: {output_path}")
    torch.save(target_ckpt, output_path)
    
    print("Successfully saved checkpoint!")
    return True

def list_checkpoint_keys(checkpoint_path, pattern=None):
    """
    Utility function to list keys in a checkpoint, optionally filtered by pattern.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state dict
    if 'model' in ckpt:
        state = ckpt['model']
    elif 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt
    
    keys = list(state.keys())
    
    if pattern:
        keys = [k for k in keys if pattern in k]
        print(f"\nKeys containing '{pattern}':")
    else:
        print(f"\nAll keys:")
    
    for key in sorted(keys):
        if hasattr(state[key], 'shape'):
            print(f"  {key}: {state[key].shape}")
        else:
            print(f"  {key}: scalar/other")
    
    print(f"\nTotal: {len(keys)} keys")

def main():
    parser = argparse.ArgumentParser(description='Copy memory attention weights between checkpoints')
    parser.add_argument('--source', '-s', required=True, help='Path to source checkpoint')
    parser.add_argument('--target', '-t', help='Path to target checkpoint (required unless using --list-keys)')
    parser.add_argument('--output', '-o', help='Path for output checkpoint (required unless using --list-keys)')
    parser.add_argument('--list-keys', action='store_true', help='List keys in checkpoints instead of copying')
    parser.add_argument('--filter', '-f', help='Filter pattern when listing keys')
    
    args = parser.parse_args()
    
    # Check required arguments based on mode
    if not args.list_keys and (not args.target or not args.output):
        parser.error("--target and --output are required when not using --list-keys")
    
    # Validate paths
    source_path = Path(args.source)
    target_path = Path(args.target) if args.target else None
    output_path = Path(args.output) if args.output else None
    
    if not source_path.exists():
        print(f"ERROR: Source checkpoint not found: {source_path}")
        sys.exit(1)
        
    if target_path and not target_path.exists() and not args.list_keys:
        print(f"ERROR: Target checkpoint not found: {target_path}")
        sys.exit(1)
    
    # If listing keys, do that instead
    if args.list_keys:
        print("=== SOURCE CHECKPOINT ===")
        list_checkpoint_keys(source_path, args.filter)
        if target_path and target_path.exists():
            print("\n=== TARGET CHECKPOINT ===")
            list_checkpoint_keys(target_path, args.filter)
        sys.exit(0)
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Perform the copy
    success = copy_memory_attention_weights(source_path, target_path, output_path)
    
    if success:
        print("\n✅ Checkpoint weights copied successfully!")
        sys.exit(0)
    else:
        print("\n❌ Failed to copy checkpoint weights!")
        sys.exit(1)

if __name__ == "__main__":
    main()