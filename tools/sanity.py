import torch
import torch.nn as nn
import argparse
import numpy as np
import os

class CrossAttentionTest(nn.Module):
    """
    Simplified model that isolates the cross-object attention functionality
    for testing GPU differences.
    """
    def __init__(self, d_model=256, max_batch_items=50):
        super().__init__()
        self.d_model = d_model
        
        # Cross-object attention
        self.cross_obj_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=0.1,
            batch_first=False,
        )
        
        # Object embeddings to differentiate objects in the same image
        self.obj_embeddings = nn.Parameter(torch.zeros(max_batch_items, d_model))
        nn.init.normal_(self.obj_embeddings, mean=0.0, std=0.02)
        
        # Scaling factor for object embeddings
        self.obj_emb_scale = nn.Parameter(torch.ones(1) * 0.2)
    
    def cross_object_attention(self, batch_seq, img_ids):
        """
        Implements cross-object attention for objects from the same image.
        Args:
            batch_seq: Tensor of shape (B, seq, D)
            img_ids: Tensor of shape (B) containing image identifiers.
        Returns:
            Updated feature tensor of same shape as batch_seq.
        """
        B, seq, D = batch_seq.shape
        device = batch_seq.device
        
        # Skip cross-object attention if there's no need for it
        if img_ids is None or B <= 1:
            return batch_seq
            
        # Start with the original sequence as the output
        output = batch_seq.clone()
        
        # Process each image group separately
        for img_id in img_ids.unique():
            # Find indices of objects with the same image ID
            indices = (img_ids == img_id).nonzero(as_tuple=True)[0]
            num_objs = indices.size(0)
            
            if num_objs <= 1:
                continue  # Skip if only one object from this image
                
            # Get features for objects with this image ID
            group_features = batch_seq[indices]  # (num_objs, seq, D)
            
            # Add object embeddings to differentiate objects
            obj_indices = torch.arange(num_objs, device=device)
            obj_embs = self.obj_embeddings[obj_indices].unsqueeze(1)  # (num_objs, 1, D)
            obj_embs = obj_embs.expand(-1, seq, -1)  # (num_objs, seq, D)
            enhanced_group = group_features + self.obj_emb_scale * obj_embs
            
            # Use standard MultiheadAttention correctly
            # First transpose to (seq, num_objs, D) for non-batch_first format
            enhanced_group = enhanced_group.transpose(0, 1)
            
            # Apply self-attention within this group
            attended_group, _ = self.cross_obj_attn(
                enhanced_group, enhanced_group, enhanced_group,
                need_weights=False
            )
            
            # Transpose back to (num_objs, seq, D)
            attended_group = attended_group.transpose(0, 1)
            
            # Add residual connection and update the output
            attended_group = group_features + attended_group
            
            # Update the output tensor for these indices
            for i, idx in enumerate(indices):
                output[idx] = attended_group[i]
                
        return output
    
    def forward(self, batch_seq, img_ids):
        """Simple forward pass that just calls cross_object_attention"""
        return self.cross_object_attention(batch_seq, img_ids)


def run_test(device_str="cuda", seed=42, precision="float32", 
             save_path=None, debug=False):
    """Run test on specified device with controlled parameters"""
    # Set deterministic behavior for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create device
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    
    # Detect GPU info
    gpu_info = "CPU"
    if torch.cuda.is_available() and device.type == "cuda":
        gpu_info = torch.cuda.get_device_name(device)
    
    print(f"Running test on {device} - {gpu_info}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Set precision
    if precision == 'float16':
        torch_dtype = torch.float16
    elif precision == 'float64':
        torch_dtype = torch.float64
    else:  # default to float32
        torch_dtype = torch.float32
    
    print(f"Using precision: {precision}")
    
    # Parameters
    d_model = 256
    seq_len = 16
    batch_size = 10
    
    # Create model
    model = CrossAttentionTest(d_model=d_model).to(device).to(torch_dtype)
    
    # Generate deterministic input using linspace for reproducibility
    batch_seq_cpu = torch.linspace(-1, 1, batch_size * seq_len * d_model)
    batch_seq_cpu = batch_seq_cpu.reshape(batch_size, seq_len, d_model)
    batch_seq = batch_seq_cpu.to(device).to(torch_dtype)
    
    # Create image IDs: 3 objects from img1, 2 from img2, 5 from img3
    img_ids = torch.tensor([1, 1, 1, 2, 2, 3, 3, 3, 3, 3], device=device)
    
    print(f"Input shape: {batch_seq.shape}")
    
    # Store intermediate values for debugging
    intermediates = {}
    
    def debug_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                intermediates[name] = output[0].detach().cpu()
            else:
                intermediates[name] = output.detach().cpu()
        return hook_fn
    
    # Register hooks if debugging
    hooks = []
    if debug:
        hooks.append(model.cross_obj_attn.register_forward_hook(debug_hook('attn_output')))
    
    # Forward pass
    with torch.no_grad():
        output = model(batch_seq, img_ids)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Print output info
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    
    # Check for NaN or Inf values
    nan_count = torch.isnan(output).sum().item()
    inf_count = torch.isinf(output).sum().item()
    print(f"NaN values in output: {nan_count}")
    print(f"Inf values in output: {inf_count}")
    
    # Print some values for comparison
    print("\nSample output values for comparison:")
    positions = [(0, 0, 0), (0, 0, 1), (1, 5, 10), (5, 10, 200)]
    for pos in positions:
        i, j, k = pos
        try:
            print(f"Output[{i}, {j}, {k}] = {output[i, j, k].item():.10f}")
        except IndexError:
            print(f"Position {pos} is out of bounds for output shape {output.shape}")
    
    result = {
        'output': output.detach().cpu().numpy(),
        'info': {
            'device': str(device),
            'gpu_info': gpu_info,
            'precision': precision,
            'pytorch_version': torch.__version__,
        }
    }
    
    # Add intermediates if debugging
    if debug:
        for key, value in intermediates.items():
            result[key] = value.numpy()
    
    # Save results if path provided
    if save_path:
        np.save(save_path, result)
        print(f"Saved output to {save_path}")
    
    return result

def compare_results(file1, file2):
    """Compare two result files to identify differences"""
    result1 = np.load(file1, allow_pickle=True).item()
    result2 = np.load(file2, allow_pickle=True).item()
    
    output1 = result1['output']
    output2 = result2['output']
    
    # Calculate differences
    abs_diff = np.abs(output1 - output2)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()
    
    # Find indices of maximum difference
    max_idx = np.unravel_index(abs_diff.argmax(), abs_diff.shape)
    
    print(f"\nComparison between {file1} and {file2}:")
    print(f"File 1 info: {result1['info']}")
    print(f"File 2 info: {result2['info']}")
    print(f"Max difference: {max_diff} at index {max_idx}")
    print(f"Mean difference: {mean_diff}")
    print(f"Value in file 1 at max diff: {output1[max_idx]}")
    print(f"Value in file 2 at max diff: {output2[max_idx]}")
    
    # Count significant differences
    thresholds = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
    for threshold in thresholds:
        sig_diffs = (abs_diff > threshold).sum()
        percentage = 100.0 * sig_diffs / output1.size
        print(f"Differences > {threshold:.0e}: {sig_diffs} ({percentage:.4f}%)")
    
    # Compare intermediates if available
    common_keys = set(result1.keys()) & set(result2.keys()) - {'output', 'info'}
    for key in common_keys:
        interm1 = result1[key]
        interm2 = result2[key]
        interm_diff = np.abs(interm1 - interm2)
        max_interm_diff = interm_diff.max()
        mean_interm_diff = interm_diff.mean()
        
        print(f"\nDifference in {key}:")
        print(f"Max difference: {max_interm_diff}")
        print(f"Mean difference: {mean_interm_diff}")

def main():
    parser = argparse.ArgumentParser(description="Test cross-object attention on different GPUs")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save", type=str, default=None, help="Path to save output")
    parser.add_argument("--compare", nargs=2, metavar=('FILE1', 'FILE2'), help="Compare two saved outputs")
    parser.add_argument("--precision", type=str, default="float32", 
                        choices=["float16", "float32", "float64"], 
                        help="Precision to use for computation")
    parser.add_argument("--debug", action="store_true", help="Store intermediate values for debugging")
    
    args = parser.parse_args()
    
    # If comparison mode
    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return
    
    # Run the test
    result = run_test(args.device, args.seed, args.precision, args.save, args.debug)

if __name__ == "__main__":
    main()
