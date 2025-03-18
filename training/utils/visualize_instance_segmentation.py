import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

def visualize_instance_predictions(
    pred_masks,
    pred_instance_ids,
    target_masks,
    target_instance_ids,
    iteration,
    save_dir,
    frame_idx=0
):
    """
    Create a specialized visualization that focuses on instance predictions.
    
    Args:
        pred_masks: Predicted binary masks [B, C, H, W]
        pred_instance_ids: Predicted instance ID embeddings [B, E, H, W]
        target_masks: Target binary masks [B, C, H, W] or [B, H, W]
        target_instance_ids: Target instance IDs [B, H, W] or [B, 1, H, W]
        iteration: Current training iteration
        save_dir: Directory to save visualizations
        frame_idx: Index of this frame in the batch
    """
    import os
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from sklearn.cluster import KMeans
    
    # Check shape of inputs
    if target_masks.dim() == 3:  # [B, H, W]
        target_masks = target_masks.unsqueeze(1)  # [B, 1, H, W]
        
    if target_instance_ids.dim() == 3:  # [B, H, W]
        target_instance_ids = target_instance_ids.unsqueeze(1)  # [B, 1, H, W]
    
    # Get valid batch size
    batch_size = min(pred_masks.shape[0], target_masks.shape[0], 
                    pred_instance_ids.shape[0], target_instance_ids.shape[0], 4)  # Limit to 4 samples
    
    os.makedirs(save_dir, exist_ok=True)
    
    for b in range(batch_size):
        # 1. Extract all tensors for this batch
        pred_mask = pred_masks[b]  # [C, H, W]
        pred_embed = pred_instance_ids[b]  # [E, H, W]
        target_mask = target_masks[b]  # [C, H, W] or [1, H, W]
        target_ids = target_instance_ids[b]  # [1, H, W] or [H, W]
        
        # 2. Convert all to numpy arrays
        if pred_mask.dim() == 3 and pred_mask.shape[0] > 0:
            # Multiple prediction channels, use first one
            pred_mask_np = pred_mask[0].detach().cpu().numpy()
        else:
            pred_mask_np = pred_mask.detach().cpu().numpy()
            
        # Apply sigmoid if needed for prediction
        if pred_mask_np.min() < 0 or pred_mask_np.max() > 1:
            pred_mask_np = 1 / (1 + np.exp(-pred_mask_np))
            
        # Binarize the mask
        binary_pred = (pred_mask_np > 0.5).astype(float)
        
        # Get target mask as numpy
        if target_mask.dim() == 3 and target_mask.shape[0] > 0:
            target_mask_np = target_mask[0].detach().cpu().numpy()
        else:
            target_mask_np = target_mask.detach().cpu().numpy()
            
        # Get target instance IDs as numpy
        if target_ids.dim() == 3 and target_ids.shape[0] > 0:
            target_ids_np = target_ids[0].detach().cpu().numpy()
        else:
            target_ids_np = target_ids.detach().cpu().numpy()
        
        # 3. Process embedding to create predicted instance ID map
        # Get embedding dimensions
        E, H, W = pred_embed.shape
        
        # Convert the embeddings to numpy
        pred_embed_np = pred_embed.detach().cpu().numpy()
        
        # Only process areas where predicted mask is positive (binary_pred > 0)
        # This helps to visualize instance IDs only within predicted objects
        masked_pixels = np.where(binary_pred > 0)
        
        if len(masked_pixels[0]) > 0:  # If we have any positive predictions
            # Extract embedding vectors for masked pixels
            pixel_embeddings = np.zeros((len(masked_pixels[0]), E))
            for i in range(len(masked_pixels[0])):
                y, x = masked_pixels[0][i], masked_pixels[1][i]
                pixel_embeddings[i] = pred_embed_np[:, y, x]
            
            # Use K-means to cluster the embeddings into instance IDs
            # First, determine number of clusters from ground truth
            unique_gt_ids = np.unique(target_ids_np)
            unique_gt_ids = unique_gt_ids[unique_gt_ids > 0]  # Remove background (0)
            
            # Set number of clusters (n_clusters) to match ground truth if possible
            n_clusters = max(1, len(unique_gt_ids))
            n_clusters = min(n_clusters, 10)  # Cap at 10 clusters to avoid too many
            
            # Apply K-means
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(pixel_embeddings)
            cluster_labels = kmeans.labels_
            
            # Create a predicted instance ID map
            pred_instance_map = np.zeros_like(binary_pred)
            for i, (y, x) in enumerate(zip(masked_pixels[0], masked_pixels[1])):
                # Add 1 to avoid 0 (background)
                pred_instance_map[y, x] = cluster_labels[i] + 1
        else:
            # No positive predictions, empty instance map
            pred_instance_map = np.zeros_like(binary_pred)
        
        # 4. Create visualizations
        plt.figure(figsize=(20, 5))
        
        # 4.1 Ground truth binary mask
        plt.subplot(1, 5, 1)
        plt.imshow(target_mask_np, cmap='gray')
        plt.title(f'Frame {frame_idx} - GT Mask')
        plt.axis('off')
        
        # 4.2 Predicted binary mask
        plt.subplot(1, 5, 2)
        plt.imshow(binary_pred, cmap='gray')
        plt.title(f'Frame {frame_idx} - Pred Mask')
        plt.axis('off')
        
        # 4.3 Ground truth instance IDs
        plt.subplot(1, 5, 3)
        unique_ids = np.unique(target_ids_np)
        if len(unique_ids) > 1:  # If we have any non-zero IDs
            # Create a colormap
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_ids)))
            cmap = ListedColormap(colors)
            plt.imshow(target_ids_np, cmap=cmap)
            plt.title(f'Frame {frame_idx} - GT Instance IDs\n({len(unique_ids)-1} instances)')
        else:
            plt.imshow(np.zeros_like(target_mask_np), cmap='gray')
            plt.title(f'Frame {frame_idx} - No GT Instances')
        plt.axis('off')
        
        # 4.4 Predicted instance IDs
        plt.subplot(1, 5, 4)
        unique_pred_ids = np.unique(pred_instance_map)
        if len(unique_pred_ids) > 1:  # If we have any predicted instances
            # Create a colormap
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_pred_ids)))
            cmap = ListedColormap(colors)
            plt.imshow(pred_instance_map, cmap=cmap)
            plt.title(f'Frame {frame_idx} - Pred Instance IDs\n({len(unique_pred_ids)-1} clusters)')
        else:
            plt.imshow(np.zeros_like(binary_pred), cmap='gray')
            plt.title(f'Frame {frame_idx} - No Pred Instances')
        plt.axis('off')
        
        # 4.5 Overlay of predicted instances on binary mask
        plt.subplot(1, 5, 5)
        # Create an RGB image
        overlay = np.zeros((H, W, 3))
        
        # Set red channel to binary prediction
        overlay[:, :, 0] = binary_pred
        
        # Set green channel to predicted instance map (normalized)
        if pred_instance_map.max() > 0:
            overlay[:, :, 1] = pred_instance_map / pred_instance_map.max()
            
        # Set blue channel to target instance map (normalized)
        if target_ids_np.max() > 0:
            overlay[:, :, 2] = target_ids_np / target_ids_np.max()
            
        plt.imshow(overlay)
        plt.title(f'Frame {frame_idx} - Overlay\nRed: Pred Mask, Green: Pred Instances, Blue: GT Instances')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'instance_vis_frame{frame_idx}_batch{b}_iter{iteration}.png'), dpi=150)
        plt.close()
        
        # 5. Create a 3D visualization (first 3 embedding dimensions as RGB)
        if E >= 3:
            plt.figure(figsize=(15, 5))
            
            # 5.1 Embedding Channel 0
            plt.subplot(1, 4, 1)
            embed_0 = pred_embed_np[0]
            # Only show embeddings where mask is positive
            masked_embed_0 = embed_0.copy()
            masked_embed_0[binary_pred == 0] = 0
            plt.imshow(masked_embed_0, cmap='coolwarm')
            plt.title(f'Frame {frame_idx} - Embed Ch 0')
            plt.colorbar()
            plt.axis('off')
            
            # 5.2 Embedding Channel 1
            plt.subplot(1, 4, 2)
            embed_1 = pred_embed_np[1]
            masked_embed_1 = embed_1.copy()
            masked_embed_1[binary_pred == 0] = 0
            plt.imshow(masked_embed_1, cmap='coolwarm')
            plt.title(f'Frame {frame_idx} - Embed Ch 1')
            plt.colorbar()
            plt.axis('off')
            
            # 5.3 Embedding Channel 2
            plt.subplot(1, 4, 3)
            embed_2 = pred_embed_np[2]
            masked_embed_2 = embed_2.copy()
            masked_embed_2[binary_pred == 0] = 0
            plt.imshow(masked_embed_2, cmap='coolwarm')
            plt.title(f'Frame {frame_idx} - Embed Ch 2')
            plt.colorbar()
            plt.axis('off')
            
            # 5.4 RGB visualization of the first 3 embedding channels
            plt.subplot(1, 4, 4)
            rgb = np.zeros((H, W, 3))
            
            # Normalize each channel to [0,1] for RGB
            for i in range(3):
                ch = pred_embed_np[i].copy()
                if ch.max() > ch.min():
                    ch = (ch - ch.min()) / (ch.max() - ch.min())
                # Only set RGB values where binary mask is positive
                rgb[:, :, i] = ch * binary_pred
            
            plt.imshow(rgb)
            plt.title(f'Frame {frame_idx} - Embedding RGB\n(first 3 channels)')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'embed_vis_frame{frame_idx}_batch{b}_iter{iteration}.png'), dpi=150)
            plt.close()

def visualize_frame(
    outputs,
    targets,
    instance_ids,
    iteration,
    save_dir,
    frame_idx
):
    """
    Visualize a single frame with all its channels/masks.
    
    Args:
        outputs: Dictionary of model outputs for this frame
        targets: Target masks for this frame
        instance_ids: Instance ID maps for this frame
        iteration: Current training iteration
        save_dir: Directory to save visualizations
        frame_idx: Index of this frame in the batch
    """
    try:
        # Print debugging information
        #print(f"Frame {frame_idx} - Targets shape: {targets.shape}")
        #if instance_ids is not None:
        #    print(f"Frame {frame_idx} - Instance IDs shape: {instance_ids.shape}")
        
        # Determine the number of channels/masks in the targets
        num_channels = 1
        if len(targets.shape) > 2:
            num_channels = targets.shape[0]
        
        #print(f"Frame {frame_idx} - Number of channels/masks: {num_channels}")
        
        # Get predictions for this frame
        pred_masks = None
        if "multistep_pred_multimasks_high_res" in outputs and len(outputs["multistep_pred_multimasks_high_res"]) > 0:
            # Get the latest prediction
            pred_masks = outputs["multistep_pred_multimasks_high_res"][-1]
            #print(f"Frame {frame_idx} - Prediction shape: {pred_masks.shape}")
        
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
                if pred_masks.dim() == 4 and pred_masks.shape[1] > channel_idx:  # [B, C, H, W]
                    channel_pred = pred_masks[:, channel_idx:channel_idx+1]
                elif pred_masks.dim() == 3 and pred_masks.shape[0] > channel_idx:  # [C, H, W]
                    channel_pred = pred_masks[channel_idx:channel_idx+1].unsqueeze(0)
            
            # Create visualizations for this channel
            visualize_channel(
                pred_mask=channel_pred,
                target_mask=target_mask,
                instance_ids=instance_ids,
                iteration=iteration,
                frame_idx=frame_idx,
                channel_idx=channel_idx,
                save_dir=channel_dir
            )
            
            # Create a more detailed visualization
            create_detailed_visualization(
                pred_mask=channel_pred,
                target_mask=target_mask,
                instance_ids=instance_ids,
                iteration=iteration,
                frame_idx=frame_idx,
                channel_idx=channel_idx,
                save_dir=channel_dir
            )
        
        # If there are multiple channels, also create a combined visualization
        if num_channels > 1:
            create_multichannel_visualization(
                pred_masks=pred_masks,
                targets=targets,
                instance_ids=instance_ids,
                iteration=iteration,
                frame_idx=frame_idx,
                save_dir=save_dir
            )
        
    except Exception as e:
        print(f"Error in visualize_frame for frame {frame_idx}: {e}")
        import traceback
        traceback.print_exc()

def visualize_channel(
    pred_mask,
    target_mask,
    instance_ids,
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
        instance_ids: Instance ID maps or None
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
        
        # 3. Instance IDs (if available) or comparison
        plt.subplot(1, 3, 3)
        if instance_ids is not None:
            inst_map = instance_ids.detach().cpu().numpy()
            if len(inst_map.shape) == 3 and inst_map.shape[0] == 1:
                inst_map = inst_map[0]  # Squeeze out singleton dimension
            
            # Check if inst_map is 3D and we need to select the right channel
            if len(inst_map.shape) == 3 and inst_map.shape[0] > channel_idx:
                inst_map = inst_map[channel_idx]
            
            # Create a colormap
            unique_ids = np.unique(inst_map)
            if len(unique_ids) > 1:  # Only show if there are actual instance IDs
                colors = plt.cm.tab20(np.linspace(0, 1, len(unique_ids)))
                cmap = ListedColormap(colors)
                plt.imshow(inst_map, cmap=cmap)
                plt.title(f'Frame {frame_idx}, Channel {channel_idx}\nInstance IDs ({len(unique_ids)} unique)')
                plt.colorbar()
            else:
                # No actual instance IDs, show prediction comparison
                if pred_mask is not None and 'pred' in locals() and pred is not None:
                    binary_pred = (pred > 0.5).astype(float)
                    plt.imshow(binary_pred, cmap='gray')
                    plt.title(f'Frame {frame_idx}, Channel {channel_idx}\nBinary Prediction')
                    plt.colorbar()
                else:
                    plt.text(0.5, 0.5, "No instance IDs or prediction for comparison", 
                             ha='center', va='center', transform=plt.gca().transAxes)
        else:
            # No instance IDs, show binary prediction if available
            if pred_mask is not None and 'pred' in locals() and pred is not None:
                binary_pred = (pred > 0.5).astype(float)
                plt.imshow(binary_pred, cmap='gray')
                plt.title(f'Frame {frame_idx}, Channel {channel_idx}\nBinary Prediction')
                plt.colorbar()
            else:
                plt.text(0.5, 0.5, "No instance IDs or prediction for comparison", 
                         ha='center', va='center', transform=plt.gca().transAxes)
        plt.axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'frame{frame_idx}_channel{channel_idx}_iter{iteration}.png'), dpi=150)
        plt.close()
        
    except Exception as e:
        print(f"Error in visualize_channel for frame {frame_idx}, channel {channel_idx}: {e}")
        import traceback
        traceback.print_exc()

def create_detailed_visualization(
    pred_mask,
    target_mask,
    instance_ids,
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
        instance_ids: Instance ID maps or None
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
        plt.savefig(os.path.join(save_dir, f'frame{frame_idx}_channel{channel_idx}_detailed_iter{iteration}.png'), dpi=150)
        plt.close()
        
    except Exception as e:
        print(f"Error in create_detailed_visualization for frame {frame_idx}, channel {channel_idx}: {e}")
        import traceback
        traceback.print_exc()

def create_multichannel_visualization(
    pred_masks,
    targets,
    instance_ids,
    iteration,
    frame_idx,
    save_dir
):
    """
    Create a visualization showing all channels of a frame together.
    
    Args:
        pred_masks: Predicted masks tensor or None
        targets: Target masks tensor with multiple channels
        instance_ids: Instance ID maps or None
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
        
        # 3. Instance IDs or combined binary predictions
        plt.subplot(1, 3, 3)
        if instance_ids is not None:
            inst_map = instance_ids.detach().cpu().numpy()
            if len(inst_map.shape) == 3 and inst_map.shape[0] == 1:
                inst_map = inst_map[0]  # Squeeze out singleton dimension
            
            # Handle different instance_ids shapes
            if len(inst_map.shape) == 3:
                # Multiple channels, combine them
                combined_inst = np.zeros_like(targets[0].detach().cpu().numpy())
                for i in range(inst_map.shape[0]):
                    channel_inst = inst_map[i]
                    # Add each channel with a different intensity
                    weight = (i + 1) / inst_map.shape[0]
                    combined_inst = np.maximum(combined_inst, channel_inst * weight)
                inst_map = combined_inst
            
            # Create a colormap
            unique_ids = np.unique(inst_map)
            if len(unique_ids) > 1:  # Only show if there are actual instance IDs
                colors = plt.cm.tab20(np.linspace(0, 1, len(unique_ids)))
                cmap = ListedColormap(colors)
                plt.imshow(inst_map, cmap=cmap)
                plt.title(f'Frame {frame_idx}\nInstance IDs ({len(unique_ids)} unique)')
                plt.colorbar()
            else:
                # Just show combined binary predictions
                if pred_masks is not None:
                    combined_binary = np.zeros_like(targets[0].detach().cpu().numpy())
                    for i in range(min(num_channels, pred_masks.shape[1] if pred_masks.dim() == 4 else pred_masks.shape[0])):
                        if pred_masks.dim() == 4:  # [B, C, H, W]
                            channel_pred = pred_masks[0, i].detach().cpu().numpy()
                        else:  # [C, H, W]
                            channel_pred = pred_masks[i].detach().cpu().numpy()
                        
                        # Apply sigmoid if needed
                        if channel_pred.min() < 0 or channel_pred.max() > 1:
                            channel_pred = 1 / (1 + np.exp(-channel_pred))
                        
                        # Binarize
                        binary_pred = (channel_pred > 0.5).astype(float)
                        
                        # Add with different intensity
                        weight = (i + 1) / num_channels
                        combined_binary = np.maximum(combined_binary, binary_pred * weight)
                    
                    plt.imshow(combined_binary, cmap='gray')
                    plt.title(f'Frame {frame_idx}\nCombined Binary Predictions')
                    plt.colorbar()
                else:
                    plt.text(0.5, 0.5, "No predictions available", 
                             ha='center', va='center', transform=plt.gca().transAxes)
        else:
            # Just show combined binary predictions
            if pred_masks is not None:
                combined_binary = np.zeros_like(targets[0].detach().cpu().numpy())
                for i in range(min(num_channels, pred_masks.shape[1] if pred_masks.dim() == 4 else pred_masks.shape[0])):
                    if pred_masks.dim() == 4:  # [B, C, H, W]
                        channel_pred = pred_masks[0, i].detach().cpu().numpy()
                    else:  # [C, H, W]
                        channel_pred = pred_masks[i].detach().cpu().numpy()
                    
                    # Apply sigmoid if needed
                    if channel_pred.min() < 0 or channel_pred.max() > 1:
                        channel_pred = 1 / (1 + np.exp(-channel_pred))
                    
                    # Binarize
                    binary_pred = (channel_pred > 0.5).astype(float)
                    
                    # Add with different intensity
                    weight = (i + 1) / num_channels
                    combined_binary = np.maximum(combined_binary, binary_pred * weight)
                
                plt.imshow(combined_binary, cmap='gray')
                plt.title(f'Frame {frame_idx}\nCombined Binary Predictions')
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
    instance_ids,
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
        
        # 3. Show instance IDs
        plt.subplot(1, 3, 3)
        if instance_ids is not None and b < instance_ids.shape[0]:
            inst_ids = instance_ids[b].detach().cpu().numpy()
            unique_ids = np.unique(inst_ids)
            colors = plt.cm.tab20(np.linspace(0, 1, max(len(unique_ids), 10)))
            cmap = ListedColormap(colors)
            plt.imshow(inst_ids, cmap=cmap)
            plt.title(f'Frame {frame_idx} - Instance IDs ({len(unique_ids)-1} instances)')
        else:
            # Create binary pred mask if no instance IDs
            if pred_mask.min() < 0 or pred_mask.max() > 1:
                binary_pred = (pred_probs > 0.5).astype(float)
            else:
                binary_pred = (pred_mask > 0.5).astype(float)
            plt.imshow(binary_pred, cmap='gray')
            plt.title(f'Frame {frame_idx} - Binary Pred (sum: {binary_pred.sum():.0f} px)')
        plt.axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'masks_iter{iteration}_batch{b}.png'), dpi=150)
        plt.close()


def visualize_embeddings(
    pred_embeddings,
    instance_ids,
    iteration,
    save_dir,
    max_channels=8,
    frame_idx=0  # Add this parameter
):
    """
    Visualize embedding channels and PCA projection.
    """
    # Check actual batch sizes and adjust
    batch_size = min(pred_embeddings.shape[0], 4)  # Limit to 4 images
    
    # Check if instance_ids has a smaller batch size than pred_embeddings
    if instance_ids is not None:
        batch_size = min(batch_size, instance_ids.shape[0])
    
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
        
        # 3. Try PCA visualization if instance IDs are available
        if instance_ids is not None and b < instance_ids.shape[0] and instance_ids[b].sum() > 0:
            try:
                inst_ids = instance_ids[b].cpu().numpy()
                
                # Get unique non-zero instance IDs
                unique_ids = np.unique(inst_ids)
                unique_ids = unique_ids[unique_ids > 0]
                
                if len(unique_ids) > 0:
                    plt.figure(figsize=(10, 10))
                    
                    # Reshape embeddings to [C, H*W]
                    C, H, W = embedding.shape
                    flat_embed = embedding.reshape(C, -1)
                    
                    # Process each instance ID
                    for inst_id in unique_ids:
                        # Get pixels for this instance
                        inst_mask = (inst_ids == inst_id).flatten()
                        
                        # Skip if too few pixels
                        if inst_mask.sum() < 10:
                            continue
                            
                        # Get embeddings for these pixels
                        inst_embed = flat_embed[:, inst_mask].T  # [N, C]
                        
                        # Sample if too many points
                        if inst_embed.shape[0] > 1000:
                            indices = np.random.choice(inst_embed.shape[0], 1000, replace=False)
                            inst_embed = inst_embed[indices]
                        
                        # Apply PCA if we have enough points and dimensions
                        if inst_embed.shape[0] > 2 and inst_embed.shape[1] > 2:
                            pca = PCA(n_components=2)
                            inst_embed_2d = pca.fit_transform(inst_embed)
                            
                            plt.scatter(
                                inst_embed_2d[:, 0],
                                inst_embed_2d[:, 1],
                                alpha=0.5,
                                label=f'Instance {inst_id}'
                            )
                    
                    plt.title('PCA of Instance Embeddings')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f'embed_pca_iter{iteration}_batch{b}.png'), dpi=150)
                    plt.close()
            except Exception as e:
                print(f"Error in PCA visualization: {e}")