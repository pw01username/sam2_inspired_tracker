import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from matplotlib.gridspec import GridSpec

# Only modify the visualize_frame function to iterate through all predictions
def visualize_frame(
    outputs,
    targets,
    iteration,
    save_dir,
    frame_idx
):
    """
    Visualize a single frame with all its channels/masks.
    
    Args:
        outputs: Dictionary of model outputs for this frame
        targets: Target masks for this frame
        iteration: Current training iteration
        save_dir: Directory to save visualizations
        frame_idx: Index of this frame in the batch
    """
    try:
        # Determine the number of channels/masks in the targets
        num_channels = 1
        if len(targets.shape) > 2:
            num_channels = targets.shape[0]
        
        # Get predictions for this frame
        pred_masks = None
        if "multistep_pred_multimasks_high_res" in outputs and len(outputs["multistep_pred_multimasks_high_res"]) > 0:
            # Get the latest prediction
            pred_masks = outputs["multistep_pred_multimasks_high_res"][-1]
        
        # Create a directory for all predictions
        if pred_masks is not None and pred_masks.dim() == 4 and pred_masks.shape[1] > 1:
            all_preds_dir = os.path.join(save_dir, "all_predictions")
            os.makedirs(all_preds_dir, exist_ok=True)
            
            # Visualize each prediction separately
            for pred_idx in range(pred_masks.shape[1]):
                # Extract just this prediction
                single_pred = pred_masks[:, pred_idx:pred_idx+1, :, :]
                
                # Create a specific directory for this prediction
                pred_dir = os.path.join(all_preds_dir, f"pred_{pred_idx}")
                os.makedirs(pred_dir, exist_ok=True)
                
                # Process each channel with this prediction
                for channel_idx in range(num_channels):
                    # Get the target mask for this channel
                    if num_channels > 1:
                        target_mask = targets[channel_idx]
                    else:
                        target_mask = targets
                    
                    # Use the existing visualization functions
                    visualize_channel(
                        pred_mask=single_pred,
                        target_mask=target_mask,
                        iteration=iteration,
                        frame_idx=frame_idx,
                        channel_idx=channel_idx,
                        save_dir=os.path.join(pred_dir, f"channel_{channel_idx}")
                    )
                    
                    create_detailed_visualization(
                        pred_mask=single_pred,
                        target_mask=target_mask,
                        iteration=iteration,
                        frame_idx=frame_idx,
                        channel_idx=channel_idx,
                        save_dir=os.path.join(pred_dir, f"channel_{channel_idx}")
                    )
        
        # Keep original code for visualizing the first prediction
        # Process each channel/mask
        for channel_idx in range(num_channels):
            # Create a directory for this channel
            if num_channels > 1:
                channel_dir = os.path.join(save_dir, f"channel_{channel_idx}")
                os.makedirs(channel_dir, exist_ok=True)
            else:
                channel_dir = save_dir
            
            # Get the target mask for this channel
            if num_channels > 1:
                target_mask = targets[channel_idx]
            else:
                target_mask = targets
            
            # Get the prediction for this channel if available
            channel_pred = None
            if pred_masks is not None:
                if pred_masks.dim() == 4:
                    # For multiple predictions, just use the first one for the main visualization
                    if pred_masks.shape[1] > 0:
                        channel_pred = pred_masks[:, 0:1]
                elif pred_masks.dim() == 3 and pred_masks.shape[0] > channel_idx:
                    channel_pred = pred_masks[channel_idx:channel_idx+1].unsqueeze(0)
            
            # Create visualizations for this channel
            visualize_channel(
                pred_mask=channel_pred,
                target_mask=target_mask,
                iteration=iteration,
                frame_idx=frame_idx,
                channel_idx=channel_idx,
                save_dir=channel_dir
            )
            
            # Create a more detailed visualization
            create_detailed_visualization(
                pred_mask=channel_pred,
                target_mask=target_mask,
                iteration=iteration,
                frame_idx=frame_idx,
                channel_idx=channel_idx,
                save_dir=channel_dir
            )
        
        # If there are multiple channels, also create a combined visualization
        if num_channels > 1:
            # For multiple predictions, use only the first one for the combined view
            if pred_masks is not None and pred_masks.dim() == 4 and pred_masks.shape[1] > 1:
                first_pred = pred_masks[:, 0:1]
                create_multichannel_visualization(
                    pred_masks=first_pred,
                    targets=targets,
                    iteration=iteration,
                    frame_idx=frame_idx,
                    save_dir=save_dir
                )
            else:
                create_multichannel_visualization(
                    pred_masks=pred_masks,
                    targets=targets,
                    iteration=iteration,
                    frame_idx=frame_idx,
                    save_dir=save_dir
                )
        
    except Exception as e:
        print(f"Error in visualize_frame for frame {frame_idx}: {e}")
        import traceback
        traceback.print_exc()


def visualize_4d_tensor(tensor, filepath, cmap='gray', figsize=None, title=None, 
                       add_colorbar=False, normalize=True):
    """
    Visualize a 4D tensor as a grid of 2D visualizations.
    
    Args:
        tensor: A 4D tensor with shape [dim0, dim1, height, width]
              dim0 will be arranged as rows, dim1 as columns in the grid
        filepath: Path to save the visualization
        cmap: Colormap to use (default: 'gray')
        figsize: Custom figure size as (width, height) in inches
        title: Optional title for the plot
        add_colorbar: Whether to add a colorbar to each subplot
        normalize: Whether to normalize each subplot individually (default: True)
    """
    # Convert tensor to numpy if it's a PyTorch tensor
    if torch.is_tensor(tensor):
        tensor_numpy = tensor.detach().cpu().numpy()
    else:
        tensor_numpy = tensor
    
    # Check that input has 4 dimensions
    if len(tensor_numpy.shape) != 4:
        raise ValueError(f"Expected 4D tensor, got shape {tensor_numpy.shape}")
    
    dim0, dim1, height, width = tensor_numpy.shape
    
    # Calculate figure size if not provided
    if figsize is None:
        # Calculate size based on grid dimensions with some fixed aspect ratio
        base_size = 3
        figsize = (base_size * dim1, base_size * dim0)
    
    # Create figure and gridspec
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(dim0, dim1, figure=fig)
    
    # Add overall title if provided
    if title:
        fig.suptitle(title, fontsize=16)
    
    # Create subplots for each element in the grid
    for i in range(dim0):
        for j in range(dim1):
            # Get the 2D grid at position [i, j]
            grid = tensor_numpy[i, j]
            
            # Create subplot
            ax = fig.add_subplot(gs[i, j])
            
            # Normalize if requested
            if normalize:
                vmin, vmax = grid.min(), grid.max()
            else:
                vmin, vmax = None, None
            
            # Display the grid
            im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
            
            # Remove axis ticks
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add row/column indicators
            if j == 0:
                ax.set_ylabel(f"Row {i}")
            if i == 0:
                ax.set_title(f"Col {j}")
            
            # Add colorbar if requested
            if add_colorbar:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Adjust layout
    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.9)
    
    # Save and close
    try:
        import os
        
        # Check if filepath is empty or None
        if not filepath:
            filepath = "tensor_visualization.png"
            #print(f"Filepath was empty, using default: {filepath}")
        
        # Get the directory part of the path
        dirpath = os.path.dirname(filepath)
        
        # If dirpath is not empty, create the directories
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        
        # Check if filepath is a directory
        if os.path.isdir(filepath):
            # If it is, append a filename
            filepath = os.path.join(filepath, "tensor_visualization.png")
            #print(f"Filepath was a directory, saving to {filepath} instead")
        
        # Add extension if missing
        if not os.path.splitext(filepath)[1]:
            filepath += ".png"
            #print(f"Added missing extension: {filepath}")
        
        plt.savefig(filepath, bbox_inches='tight')
        #print(f"Successfully saved visualization to {filepath}")
    except Exception as e:
        print(f"Error saving visualization: {e}")
        # Fallback to current directory with a timestamp
        import time
        fallback_path = f"tensor_viz_{int(time.time())}.png"
        try:
            plt.savefig(fallback_path, bbox_inches='tight')
            print(f"Saved to fallback path instead: {fallback_path}")
        except:
            print(f"Could not save to fallback path either.")
    finally:
        plt.close()

def quick_visualize_rgb(r_tensor, g_tensor, b_tensor, filepath, normalize=True):
    """
    Create and save an RGB image from three separate channel tensors.
    
    Args:
        r_tensor: Red channel tensor with shape [1, H, W] or [H, W]
        g_tensor: Green channel tensor with shape [1, H, W] or [H, W]
        b_tensor: Blue channel tensor with shape [1, H, W] or [H, W]
        filepath: Path to save the output image
        normalize: Whether to normalize each channel to [0, 1] range
    """
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Convert tensors to numpy if they're PyTorch tensors
    def process_channel(tensor):
        if torch.is_tensor(tensor):
            tensor = tensor.detach().cpu().numpy()
        
        # Handle different dimensions
        if len(tensor.shape) > 2:
            if tensor.shape[0] == 1:
                tensor = tensor[0]  # Get the single channel
            else:
                tensor = tensor[0]  # Use the first channel
                
        # Normalize if requested
        if normalize:
            if tensor.max() > tensor.min():  # Avoid division by zero
                tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
                
        return tensor
    
    # Process each channel
    r = process_channel(r_tensor)
    g = process_channel(g_tensor)
    b = process_channel(b_tensor)
    
    # Ensure all channels have the same shape
    assert r.shape == g.shape == b.shape, "All channels must have the same shape"
    
    # Stack channels to create RGB
    rgb_image = np.stack([r, g, b], axis=2)
    
    # Clip values to be in valid range [0, 1]
    rgb_image = np.clip(rgb_image, 0, 1)
    
    # Create figure
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_image)
    plt.axis('off')
    plt.tight_layout()
    
    # Save and close
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
    plt.close()

def quick_visualize_mask(mask_tensor, filepath):
    # Convert tensor to numpy if it's a PyTorch tensor
    if torch.is_tensor(mask_tensor):
        mask_numpy = mask_tensor.detach().cpu().numpy()
    else:
        mask_numpy = mask_tensor
    
    # Handle different dimensions
    if len(mask_numpy.shape) > 2:
        # If it's a 3D tensor with shape [1, H, W] or [C, H, W]
        if mask_numpy.shape[0] == 1:
            mask_numpy = mask_numpy[0]  # Get the single channel
        else:
            # If multiple channels, just use the first one for quick viz
            mask_numpy = mask_numpy[0]
    
    # Create figure (with no border/axis)
    plt.figure(figsize=(8, 8))
    plt.imshow(mask_numpy, cmap='gray')  # Use 'viridis' or 'plasma' for color visualization
    plt.axis('off')
    plt.tight_layout()
    
    # Save and close
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
    plt.close()

def visualize_channel(
    pred_mask,
    target_mask,
    iteration,
    frame_idx,
    channel_idx,
    save_dir
):
    """
    Visualize a single channel/mask.
    
    Args:
        pred_mask: Predicted mask tensor or None
        target_mask: Ground truth mask tensor
        iteration: Current training iteration
        frame_idx: Index of the frame in the batch
        channel_idx: Index of the channel/mask
        save_dir: Directory to save visualizations
    """
    try:
        # Create a figure for visualization
        plt.figure(figsize=(15, 5))
        
        # 1. Ground truth mask
        plt.subplot(1, 3, 1)
        gt_mask = target_mask.detach().cpu().numpy()
        if len(gt_mask.shape) == 3 and gt_mask.shape[0] == 1:
            gt_mask = gt_mask[0]  # Squeeze out singleton dimension
        plt.imshow(gt_mask, cmap='gray')
        plt.title(f'Frame {frame_idx}, Channel {channel_idx}\nGround Truth')
        plt.colorbar()
        plt.axis('off')
        
        # 2. Predicted mask (if available)
        plt.subplot(1, 3, 2)
        if pred_mask is not None:
            # Try different shapes to find a valid prediction
            if pred_mask.dim() == 4 and pred_mask.shape[0] > 0 and pred_mask.shape[1] > 0:
                # Shape [B, C, H, W]
                pred = pred_mask[0, 0].detach().cpu().numpy()
                if len(pred.shape) == 3 and pred.shape[0] == 1:
                    pred = pred[0]  # Squeeze out singleton dimension
            elif pred_mask.dim() == 3 and pred_mask.shape[0] > 0:
                # Shape [C, H, W]
                pred = pred_mask[0].detach().cpu().numpy()
            else:
                pred = None
            
            if pred is not None:
                # Apply sigmoid if needed
                if pred.min() < 0 or pred.max() > 1:
                    pred = 1 / (1 + np.exp(-pred))  # sigmoid
                
                plt.imshow(pred, cmap='plasma')
                plt.title(f'Frame {frame_idx}, Channel {channel_idx}\nPrediction')
                plt.colorbar()
            else:
                plt.text(0.5, 0.5, "No valid prediction available", 
                         ha='center', va='center', transform=plt.gca().transAxes)
        else:
            plt.text(0.5, 0.5, "No prediction available", 
                     ha='center', va='center', transform=plt.gca().transAxes)
        plt.axis('off')
        
        # Save figure
        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'frame{frame_idx}_channel{channel_idx}_iter{iteration}.png'), dpi=150)
        plt.close()
        
    except Exception as e:
        print(f"Error in visualize_channel for frame {frame_idx}, channel {channel_idx}: {e}")
        import traceback
        traceback.print_exc()

def create_detailed_visualization(
    pred_mask,
    target_mask,
    iteration,
    frame_idx,
    channel_idx,
    save_dir
):
    """
    Create a more detailed visualization for a single channel/mask.
    
    Args:
        pred_mask: Predicted mask tensor or None
        target_mask: Ground truth mask tensor
        iteration: Current training iteration
        frame_idx: Index of the frame in the batch
        channel_idx: Index of the channel/mask
        save_dir: Directory to save visualizations
    """
    try:
        # Create a larger figure with more subplots
        plt.figure(figsize=(15, 12))
        
        # 1. Ground truth mask
        plt.subplot(2, 2, 1)
        gt_mask = target_mask.detach().cpu().numpy()
        if len(gt_mask.shape) == 3 and gt_mask.shape[0] == 1:
            gt_mask = gt_mask[0]  # Squeeze out singleton dimension
        plt.imshow(gt_mask, cmap='gray')
        plt.title(f'Frame {frame_idx}, Channel {channel_idx}\nGround Truth Mask')
        plt.colorbar()
        plt.axis('off')
        
        # 2. Predicted mask
        plt.subplot(2, 2, 2)
        if pred_mask is not None:
            # Try different shapes to find a valid prediction
            if pred_mask.dim() == 4 and pred_mask.shape[0] > 0 and pred_mask.shape[1] > 0:
                # Shape [B, C, H, W]
                pred = pred_mask[0, 0].detach().cpu().numpy()
                if len(pred.shape) == 3 and pred.shape[0] == 1:
                    pred = pred[0]  # Squeeze out singleton dimension
            elif pred_mask.dim() == 3 and pred_mask.shape[0] > 0:
                # Shape [C, H, W]
                pred = pred_mask[0].detach().cpu().numpy()
            else:
                pred = None
            
            if pred is not None:
                # Apply sigmoid if needed
                if pred.min() < 0 or pred.max() > 1:
                    pred = 1 / (1 + np.exp(-pred))  # sigmoid
                
                plt.imshow(pred, cmap='plasma')
                plt.title(f'Frame {frame_idx}, Channel {channel_idx}\nPrediction (min={pred.min():.2f}, max={pred.max():.2f})')
                plt.colorbar()
            else:
                plt.text(0.5, 0.5, "No valid prediction available", 
                         ha='center', va='center', transform=plt.gca().transAxes)
        else:
            plt.text(0.5, 0.5, "No prediction available", 
                     ha='center', va='center', transform=plt.gca().transAxes)
        plt.axis('off')
        
        # 3. Binary prediction (threshold > 0.5)
        plt.subplot(2, 2, 3)
        if pred_mask is not None and 'pred' in locals() and pred is not None:
            binary_pred = (pred > 0.5).astype(float)
            plt.imshow(binary_pred, cmap='gray')
            plt.title(f'Frame {frame_idx}, Channel {channel_idx}\nBinary Prediction (Threshold > 0.5)')
            plt.colorbar()
        else:
            plt.text(0.5, 0.5, "No prediction available for binary thresholding", 
                     ha='center', va='center', transform=plt.gca().transAxes)
        plt.axis('off')
        
        # 4. Difference map or overlay
        plt.subplot(2, 2, 4)
        if pred_mask is not None and 'pred' in locals() and pred is not None:
            # Create an overlay or difference map
            if np.sum(gt_mask) > 0:  # If ground truth has positive pixels
                # Create an RGB overlay
                overlay = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3))
                
                # Normalize for visualization
                gt_norm = gt_mask / np.max(gt_mask) if np.max(gt_mask) > 0 else gt_mask
                pred_norm = binary_pred
                
                # Red channel: ground truth
                overlay[:, :, 0] = gt_norm
                # Green channel: prediction
                overlay[:, :, 1] = pred_norm
                
                plt.imshow(overlay)
                plt.title(f'Frame {frame_idx}, Channel {channel_idx}\nOverlay (Red: GT, Green: Pred)')
            else:
                # Or show difference map if GT is all zeros
                diff = binary_pred - gt_mask
                plt.imshow(diff, cmap='coolwarm', vmin=-1, vmax=1)
                plt.title(f'Frame {frame_idx}, Channel {channel_idx}\nDifference (Pred - GT)')
                plt.colorbar()
        else:
            plt.text(0.5, 0.5, "No prediction available for comparison", 
                     ha='center', va='center', transform=plt.gca().transAxes)
        plt.axis('off')
        
        # Save the detailed figure
        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'frame{frame_idx}_channel{channel_idx}_detailed_iter{iteration}.png'), dpi=150)
        plt.close()
        
    except Exception as e:
        print(f"Error in create_detailed_visualization for frame {frame_idx}, channel {channel_idx}: {e}")
        import traceback
        traceback.print_exc()

def create_multichannel_visualization(
    pred_masks,
    targets,
    iteration,
    frame_idx,
    save_dir
):
    """
    Create a visualization showing all channels of a frame together.
    
    Args:
        pred_masks: Predicted masks tensor or None
        targets: Target masks tensor with multiple channels
        iteration: Current training iteration
        frame_idx: Index of the frame in the batch
        save_dir: Directory to save visualizations
    """
    try:
        # Determine number of channels
        num_channels = targets.shape[0]
        
        # Create a figure with one row per channel
        fig_height = 3 * num_channels
        plt.figure(figsize=(15, fig_height))
        
        # Process each channel
        for channel_idx in range(num_channels):
            # Get target mask for this channel
            target_mask = targets[channel_idx].detach().cpu().numpy()
            
            # Get prediction for this channel if available
            pred = None
            if pred_masks is not None:
                if pred_masks.dim() == 4 and pred_masks.shape[1] > channel_idx:  # [B, C, H, W]
                    pred = pred_masks[0, channel_idx].detach().cpu().numpy()
                    # Apply sigmoid if needed
                    if pred.min() < 0 or pred.max() > 1:
                        pred = 1 / (1 + np.exp(-pred))  # sigmoid
                elif pred_masks.dim() == 3 and pred_masks.shape[0] > channel_idx:  # [C, H, W]
                    pred = pred_masks[channel_idx].detach().cpu().numpy()
                    # Apply sigmoid if needed
                    if pred.min() < 0 or pred.max() > 1:
                        pred = 1 / (1 + np.exp(-pred))  # sigmoid
            
            # Row for this channel: GT, Pred, Binary Pred
            row_start = 3 * channel_idx + 1
            
            # Ground truth
            plt.subplot(num_channels, 3, row_start)
            plt.imshow(target_mask, cmap='gray')
            plt.title(f'Channel {channel_idx} - Ground Truth')
            plt.colorbar()
            plt.axis('off')
            
            # Prediction
            plt.subplot(num_channels, 3, row_start + 1)
            if pred is not None:
                plt.imshow(pred, cmap='plasma')
                plt.title(f'Channel {channel_idx} - Prediction')
                plt.colorbar()
            else:
                plt.text(0.5, 0.5, "No prediction", ha='center', va='center', transform=plt.gca().transAxes)
            plt.axis('off')
            
            # Binary prediction
            plt.subplot(num_channels, 3, row_start + 2)
            if pred is not None:
                binary_pred = (pred > 0.5).astype(float)
                plt.imshow(binary_pred, cmap='gray')
                plt.title(f'Channel {channel_idx} - Binary Prediction')
                plt.colorbar()
            else:
                plt.text(0.5, 0.5, "No binary prediction", ha='center', va='center', transform=plt.gca().transAxes)
            plt.axis('off')
        
        # Save the multichannel figure
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'frame{frame_idx}_all_channels_iter{iteration}.png'), dpi=150)
        plt.close()
        
        # Also create a combined view of all channels
        plt.figure(figsize=(15, 5))
        
        # 1. Combined ground truth
        plt.subplot(1, 3, 1)
        combined_gt = np.zeros_like(targets[0].detach().cpu().numpy())
        for i in range(num_channels):
            channel_gt = targets[i].detach().cpu().numpy()
            # Add each channel with a different intensity
            weight = (i + 1) / num_channels
            combined_gt = np.maximum(combined_gt, channel_gt * weight)
        
        plt.imshow(combined_gt, cmap='viridis')
        plt.title(f'Frame {frame_idx}\nCombined Ground Truth')
        plt.colorbar()
        plt.axis('off')
        
        # 2. Combined predictions
        plt.subplot(1, 3, 2)
        if pred_masks is not None:
            combined_pred = np.zeros_like(targets[0].detach().cpu().numpy())
            for i in range(min(num_channels, pred_masks.shape[1] if pred_masks.dim() == 4 else pred_masks.shape[0])):
                if pred_masks.dim() == 4:  # [B, C, H, W]
                    channel_pred = pred_masks[0, i].detach().cpu().numpy()
                else:  # [C, H, W]
                    channel_pred = pred_masks[i].detach().cpu().numpy()
                
                # Apply sigmoid if needed
                if channel_pred.min() < 0 or channel_pred.max() > 1:
                    channel_pred = 1 / (1 + np.exp(-channel_pred))
                
                # Add with different intensity
                weight = (i + 1) / num_channels
                combined_pred = np.maximum(combined_pred, channel_pred * weight)
            
            plt.imshow(combined_pred, cmap='plasma')
            plt.title(f'Frame {frame_idx}\nCombined Predictions')
            plt.colorbar()
        else:
            plt.text(0.5, 0.5, "No predictions available", 
                     ha='center', va='center', transform=plt.gca().transAxes)
        plt.axis('off')
        
        # Save the combined figure
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'frame{frame_idx}_combined_iter{iteration}.png'), dpi=150)
        plt.close()
        
    except Exception as e:
        print(f"Error in create_multichannel_visualization for frame {frame_idx}: {e}")
        import traceback
        traceback.print_exc()

def visualize_masks(
    pred_masks,
    target_masks,
    iteration,
    save_dir,
    frame_idx=0
):
    """
    Visualize predicted masks vs ground truth.
    """
    # Handle different shapes
    if pred_masks.dim() == 4 and pred_masks.shape[1] > 1:  # [B, C, H, W]
        # Multiple mask predictions, show the first one
        pred_masks = pred_masks[:, 0:1]
    
    if target_masks.dim() == 3:  # [B, H, W]
        target_masks = target_masks.unsqueeze(1)  # [B, 1, H, W]
    
    # Determine valid batch size by checking all inputs
    batch_size = min(pred_masks.shape[0], target_masks.shape[0], 4)  # Limit to 4 images
    if instance_ids is not None:
        batch_size = min(batch_size, instance_ids.shape[0])
    
    for b in range(batch_size):
        # Create figure with 3 subplots
        plt.figure(figsize=(18, 6))
        
        # 1. Show ground truth binary mask
        plt.subplot(1, 3, 1)
        gt_mask = target_masks[b, 0].detach().cpu().numpy()
        plt.imshow(gt_mask, cmap='gray')
        plt.title(f'Frame {frame_idx} - GT Mask (sum: {gt_mask.sum():.0f} px)')
        plt.axis('off')
        
        # 2. Show predicted mask (logits or probabilities)
        plt.subplot(1, 3, 2)
        pred_mask = pred_masks[b, 0].detach().cpu().numpy()
        
        # Apply sigmoid if needed (if values are outside [0,1])
        if pred_mask.min() < 0 or pred_mask.max() > 1:
            pred_probs = 1 / (1 + np.exp(-pred_mask))  # sigmoid
            plt.imshow(pred_probs, cmap='plasma')
            plt.title(f'Frame {frame_idx} - Pred Probs (min: {pred_probs.min():.2f}, max: {pred_probs.max():.2f})')
        else:
            plt.imshow(pred_mask, cmap='plasma')
            plt.title(f'Frame {frame_idx} - Pred Mask (min: {pred_mask.min():.2f}, max: {pred_mask.max():.2f})')
        plt.axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'masks_iter{iteration}_batch{b}.png'), dpi=150)
        plt.close()
    """
    Visualize embedding channels and PCA projection.
    """
    # Check actual batch sizes and adjust
    batch_size = min(pred_embeddings.shape[0], 4)  # Limit to 4 images
    
    for b in range(batch_size):
        # 1. Visualize individual embedding channels
        embedding = pred_embeddings[b].detach().cpu()  # [C, H, W]
        
        # Number of channels to show (limit to 8)
        n_channels = min(embedding.shape[0], max_channels)
        
        plt.figure(figsize=(16, 8))
        for i in range(n_channels):
            plt.subplot(2, 4, i+1)
            channel = embedding[i].numpy()
            plt.imshow(channel, cmap='coolwarm')
            plt.colorbar()
            plt.title(f'Frame {frame_idx} - Embed Ch {i} (min={channel.min():.2f}, max={channel.max():.2f})')
            plt.axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'embed_channels_iter{iteration}_batch{b}.png'), dpi=150)
        plt.close()
        
        # 2. Create a colored visualization using first 3 channels as RGB
        plt.figure(figsize=(12, 12))
        
        if embedding.shape[0] >= 3:
            # Normalize each channel to [0,1]
            rgb = []
            for i in range(3):
                channel = embedding[i].numpy()
                if channel.max() > channel.min():
                    channel = (channel - channel.min()) / (channel.max() - channel.min())
                rgb.append(channel)
                
            # Stack as RGB
            rgb_img = np.stack(rgb, axis=2)
            plt.imshow(rgb_img)
            plt.title(f'Frame {frame_idx} - Embeddings RGB Visualization (first 3 channels)')
        else:
            # Just show first channel if fewer than 3
            channel = embedding[0].numpy()
            plt.imshow(channel, cmap='viridis')
            plt.title(f'Frame {frame_idx} - Embedding Channel 0')
            
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'embed_rgb_iter{iteration}_batch{b}.png'), dpi=150)
        plt.close()