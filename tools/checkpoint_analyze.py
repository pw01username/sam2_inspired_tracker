import torch
import argparse
from collections import OrderedDict

def load_checkpoint(ckpt_path):
    """Load checkpoint and return state dict."""
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    return checkpoint

def analyze_checkpoint(ckpt_path):
    """Analyze checkpoint structure."""
    checkpoint = load_checkpoint(ckpt_path)
    
    print(f"Keys in checkpoint: {list(checkpoint.keys())}")
    
    if "model" in checkpoint:
        model_state = checkpoint["model"]
        print(f"\nModel state dict contains {len(model_state)} parameters")
        
        # Print a few example parameters
        print("\nSample parameters:")
        for i, (k, v) in enumerate(model_state.items()):
            if i >= 5:
                break
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    
    if "optimizer" in checkpoint:
        optimizer_state = checkpoint["optimizer"]
        print(f"\nOptimizer state dict keys: {list(optimizer_state.keys())}")
        
        if "param_groups" in optimizer_state:
            param_groups = optimizer_state["param_groups"]
            print(f"\nNumber of parameter groups: {len(param_groups)}")
            
            # Analyze first group
            if param_groups:
                first_group = param_groups[0]
                print(f"\nFirst parameter group keys: {list(first_group.keys())}")
                if "params" in first_group:
                    print(f"Number of parameters in first group: {len(first_group['params'])}")
    
    return checkpoint

def compare_checkpoints(ckpt_path1, ckpt_path2):
    """Compare two checkpoints and identify differences."""
    ckpt1 = load_checkpoint(ckpt_path1)
    ckpt2 = load_checkpoint(ckpt_path2)
    
    print(f"Checkpoint 1 keys: {list(ckpt1.keys())}")
    print(f"Checkpoint 2 keys: {list(ckpt2.keys())}")
    
    # Compare optimizer param groups
    if "optimizer" in ckpt1 and "optimizer" in ckpt2:
        param_groups1 = ckpt1["optimizer"]["param_groups"]
        param_groups2 = ckpt2["optimizer"]["param_groups"]
        
        print(f"\nCheckpoint 1 has {len(param_groups1)} parameter groups")
        print(f"Checkpoint 2 has {len(param_groups2)} parameter groups")
        
        if len(param_groups1) != len(param_groups2):
            print("\nDifferent number of parameter groups! This is likely the source of your error.")
            
            # Find names of parameters in each group
            for i, group in enumerate(param_groups1):
                param_indices = group["params"]
                if "param_names" in ckpt1["optimizer"]:
                    param_names = [ckpt1["optimizer"]["param_names"][idx] for idx in param_indices if idx < len(ckpt1["optimizer"]["param_names"])]
                    print(f"\nGroup {i} in checkpoint 1 contains parameters: {param_names[:5]}{'...' if len(param_names) > 5 else ''}")
                else:
                    print(f"\nGroup {i} in checkpoint 1 contains {len(param_indices)} parameters (names not available)")
            
            for i, group in enumerate(param_groups2):
                param_indices = group["params"]
                if "param_names" in ckpt2["optimizer"]:
                    param_names = [ckpt2["optimizer"]["param_names"][idx] for idx in param_indices if idx < len(ckpt2["optimizer"]["param_names"])]
                    print(f"\nGroup {i} in checkpoint 2 contains parameters: {param_names[:5]}{'...' if len(param_names) > 5 else ''}")
                else:
                    print(f"\nGroup {i} in checkpoint 2 contains {len(param_indices)} parameters (names not available)")
    
    # Compare model architectures
    if "model" in ckpt1 and "model" in ckpt2:
        model1 = ckpt1["model"]
        model2 = ckpt2["model"]
        
        print(f"\nModel 1 has {len(model1)} parameters")
        print(f"Model 2 has {len(model2)} parameters")
        
        # Find parameters in model1 but not in model2
        only_in_model1 = [k for k in model1.keys() if k not in model2]
        only_in_model2 = [k for k in model2.keys() if k not in model1]
        
        if only_in_model1:
            print(f"\nParameters in checkpoint 1 but not in checkpoint 2: {only_in_model1[:5]}{'...' if len(only_in_model1) > 5 else ''}")
        
        if only_in_model2:
            print(f"\nParameters in checkpoint 2 but not in checkpoint 1: {only_in_model2[:5]}{'...' if len(only_in_model2) > 5 else ''}")
        
        # Find parameters with different shapes
        common_keys = [k for k in model1.keys() if k in model2]
        diff_shapes = [k for k in common_keys if model1[k].shape != model2[k].shape]
        
        if diff_shapes:
            print("\nParameters with different shapes:")
            for k in diff_shapes[:5]:
                print(f"  {k}: {model1[k].shape} vs {model2[k].shape}")
            if len(diff_shapes) > 5:
                print("  ...")

def fix_checkpoint(src_ckpt_path, dst_ckpt_path, target_model):
    """
    Create a new checkpoint with optimizer state matching the target model structure.
    
    Args:
        src_ckpt_path: Path to source checkpoint
        dst_ckpt_path: Path to save fixed checkpoint
        target_model: Target model instance
    """
    checkpoint = load_checkpoint(src_ckpt_path)
    
    # Create a new optimizer state that matches the target model
    if "optimizer" in checkpoint:
        # This is where you'd need to reconstruct the optimizer state
        # based on your knowledge of the target model structure
        # ...
        pass
    
    # Save the fixed checkpoint
    torch.save(checkpoint, dst_ckpt_path)
    print(f"Fixed checkpoint saved to {dst_ckpt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze or compare SAM 2.1 checkpoints")
    parser.add_argument("command", choices=["analyze", "compare"], help="Command to execute")
    parser.add_argument("--ckpt1", required=True, help="Path to checkpoint 1")
    parser.add_argument("--ckpt2", help="Path to checkpoint 2 (for compare)")
    args = parser.parse_args()
    
    if args.command == "analyze":
        analyze_checkpoint(args.ckpt1)
    elif args.command == "compare":
        if not args.ckpt2:
            raise ValueError("--ckpt2 is required for compare command")
        compare_checkpoints(args.ckpt1, args.ckpt2)