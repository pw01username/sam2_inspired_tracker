# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
import copy
import torch
from torch import nn, Tensor

from sam2.modeling.sam.transformer import RoPEAttention

from sam2.modeling.sam2_utils import get_activation_fn, get_clones


class MemoryAttentionLayer(nn.Module):

    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        self_attention: nn.Module,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation_str = activation
        self.activation = get_activation_fn(activation)

        # Where to add pos enc
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _forward_sa(self, tgt, query_pos):
        # Self-Attention
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        # Cross-Attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            **kwds,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:

        # Self-Attn, Cross-Attn
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)

        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class MemoryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: nn.Module,
        num_layers: int,
        batch_first: bool = True,  # Do layers expect batch first input?
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first

        # Additional modules for cross-object mixing
        # self.cross_obj_attn = nn.MultiheadAttention(
        #     embed_dim=d_model,
        #     num_heads=4,
        #     dropout=0.1,
        #     batch_first=False,
        # )
        
        # # This is the 1.1 mem_ca that got 0.79 val loss on davis. 90.2
        # max_batch_items = 50 # num images in a image * num images (batch size defined in yaml)
        # # Object embeddings to differentiate objects in the same image
        # # Shape is (max_objects, d_model) - no need for middle dimension
        # self.obj_embeddings = nn.Parameter(torch.zeros(max_batch_items, d_model))
        # nn.init.normal_(self.obj_embeddings, mean=0.0, std=0.02)
        # # Scaling factor for object embeddings
        # self.obj_emb_scale = nn.Parameter(torch.ones(1) * 0.1)

        # # Simple cross-object mixing with object identity
        # self.obj_embedding = nn.Embedding(100, d_model)  # Support up to 100 objects
        # self.obj_embedding_scale = nn.Parameter(torch.ones(1) * 0.1)
        # self.cross_obj_proj = nn.Linear(d_model, d_model)

    def _cross_object_attention(self, batch_seq: torch.Tensor, img_ids: torch.Tensor) -> torch.Tensor:
        """
        Implements cross-object attention for objects from the same image.
        Processes each image group separately for stability.
        
        Args:
            batch_seq: Tensor of shape (B, seq, D) where B is batch size, seq is sequence length,
                      and D is embedding dimension.
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
    
    def _cross_object_attention_batched(self, batch_seq: torch.Tensor, img_ids: torch.Tensor) -> torch.Tensor:
        """
        Efficiently implements cross-object attention for all images in parallel
        using attention masking instead of loops.
        
        Args:
            batch_seq: Tensor of shape (B, seq, D) where B is batch size, seq is sequence length,
                      and D is embedding dimension.
            img_ids: Tensor of shape (B) containing image identifiers.
            
        Returns:
            Updated feature tensor of same shape as batch_seq.
        """
        B, seq, D = batch_seq.shape
        device = batch_seq.device
        
        # Skip cross-object attention if there's no need for it
        if img_ids is None or B <= 1:
            return batch_seq
        
        # Add object embeddings to differentiate objects
        # Get the first B object embeddings and add them to each sequence element
        obj_embs = self.obj_embeddings[:B].unsqueeze(1)  # (B, 1, D)
        obj_embs = obj_embs.expand(-1, seq, -1)  # (B, seq, D)
        
        # Add object embeddings to input (using input directly, no norm needed)
        enhanced_seq = batch_seq + self.obj_emb_scale * obj_embs
        
        # Reshape to (1, B*seq, D) to process in one pass
        flat_seq = enhanced_seq.reshape(1, B * seq, D)
        
        # Create key padding mask instead of attn_mask
        # Start with a mask where all positions are invalid
        key_padding_mask = torch.ones(1, B * seq, device=device, dtype=torch.bool)
        
        # For each image ID, create pairwise valid attention between objects with same ID
        for img_id in img_ids.unique():
            # Find indices of objects with the same image ID
            indices = (img_ids == img_id).nonzero(as_tuple=True)[0]
            num_objs = indices.size(0)
            
            if num_objs <= 1:
                continue  # Skip if only one object from this image
                
            # For each token-block belonging to objects with same image_id
            for obj_idx in indices:
                # Get token range for this object
                start_idx = obj_idx * seq
                end_idx = start_idx + seq
                
                # Mark these positions as valid in the key padding mask
                # This allows these tokens to be attended to
                key_padding_mask[0, start_idx:end_idx] = False
        
        # Perform self-attention with the mask
        attended_flat, _ = self.cross_obj_attn(
            flat_seq, flat_seq, flat_seq,
            key_padding_mask=key_padding_mask
        )
        
        # Reshape back to (B, seq, D)
        attended_seq = attended_flat.reshape(B, seq, D)
        
        # Add residual connection
        output = batch_seq + attended_seq
        
        return output

    def _cross_object_attention_batched_v1(self, batch_seq: torch.Tensor, img_ids: torch.Tensor) -> torch.Tensor:
        """
        Efficiently implements cross-object attention for all images in parallel
        using attention masking instead of loops.
        
        Args:
            batch_seq: Tensor of shape (B, seq, D) where B is batch size, seq is sequence length,
                      and D is embedding dimension.
            img_ids: Tensor of shape (B) containing image identifiers.
            
        Returns:
            Updated feature tensor of same shape as batch_seq.
        """
        B, seq, D = batch_seq.shape
        device = batch_seq.device
        dtype = batch_seq.dtype
        
        # Apply layer norm before attention (pre-norm architecture)
        normed_seq = self.pre_cross_obj_norm(batch_seq)
        
        # Create object positional encodings for all objects
        obj_pos_encodings = self.obj_pos_enc[:B].to(device=device, dtype=dtype)
        obj_pos_encodings = obj_pos_encodings.expand(-1, seq, -1)  # (B, seq, D)
        
        # Add positional encoding to input
        pos_enhanced_seq = normed_seq + self.pos_scale * obj_pos_encodings
        
        # Reshape to (1, B*seq, D) to process in one pass
        flat_seq = pos_enhanced_seq.reshape(1, B * seq, D)
        
        # Create attention mask to enforce within-image attention only
        # Start with a mask that blocks all attention (filled with -inf)
        attn_mask = torch.full(
            (B * seq, B * seq), 
            float('-inf'),
            device=device,
            dtype=torch.float
        )
        
        # Allow attention between tokens from the same image
        for img_id in img_ids.unique():
            # Find indices of objects with the same image ID
            indices = (img_ids == img_id).nonzero(as_tuple=True)[0]
            num_objs = indices.size(0)
            if num_objs > 10:
                print("ERROR IN ASSUMPTION OF MAX NUMBER OF OBJECTS IN MOSE AND DAVIS", img_ids, num_objs)
                raise Exception("go to memory_attention.py")
            
            if num_objs <= 1:
                continue  # Skip if only one object from this image
                
            # For each object in this image
            for i, obj_idx in enumerate(indices):
                # Calculate start and end indices in the flattened sequence
                start_i = obj_idx * seq
                end_i = start_i + seq
                
                # Allow this object to attend to all objects in same image
                for j, other_idx in enumerate(indices):
                    start_j = other_idx * seq
                    end_j = start_j + seq
                    
                    # Allow attention between these objects
                    attn_mask[start_i:end_i, start_j:end_j] = 0.0
        
        # Perform self-attention with the mask
        attended_flat, _ = self.cross_obj_attn(
            flat_seq, flat_seq, flat_seq,
            attn_mask=attn_mask
        )
        
        # Reshape back to (B, seq, D)
        attended_seq = attended_flat.reshape(B, seq, D)
        
        # Add residual connection
        output = batch_seq + attended_seq
        
        return output

    def _cross_object_mixing(self, batch_seq: torch.Tensor, img_ids: torch.Tensor) -> torch.Tensor:
        """
        Simple cross-object mixing that avoids complex attention mechanisms.
        Instead, uses a simple identity-enhanced projection for objects from the same image.
        
        Args:
            batch_seq: Tensor of shape (B, seq, D) where B is batch size
            img_ids: Tensor of shape (B) containing image identifiers
            
        Returns:
            Modified sequence with cross-object information
        """
        B, seq, D = batch_seq.shape
        
        # Skip if no cross-object mixing needed
        if img_ids is None or B <= 1:
            return batch_seq
            
        # Create object indices for the embedding
        obj_indices = torch.arange(B, device=batch_seq.device)
        
        # Get object embeddings and broadcast to sequence length
        obj_embs = self.obj_embedding(obj_indices).unsqueeze(1)  # (B, 1, D)
        obj_embs = obj_embs.expand(-1, seq, -1)  # (B, seq, D)
        
        # Create identity-enhanced features
        enhanced_seq = batch_seq + self.obj_embedding_scale * obj_embs
        projected_seq = self.cross_obj_proj(enhanced_seq)
        
        # Start with the original sequence as the output
        output = batch_seq.clone()
        
        # For each unique image ID, mix information between objects
        for img_id in img_ids.unique():
            # Find all objects with this image ID
            indices = (img_ids == img_id).nonzero(as_tuple=True)[0]
            
            if len(indices) <= 1:
                continue  # Skip if only one object for this image ID
                
            # Average the projected features for this image ID
            group_features = projected_seq[indices].mean(dim=0, keepdim=True)  # (1, seq, D)
            
            # Add a portion of the averaged features to each object
            for idx in indices:
                # Add the averaged features (residual connection built in)
                output[idx] = output[idx] + 0.1 * group_features[0]
        
        return output

    def _cross_object_all_full(self, batch_seq: Tensor, img_ids: Tensor) -> Tensor:
        """
        batch_seq: (B, seq, D).  For each image-group in B we:
         1) select that group's G objects → (G, seq, D)
         2) flatten to (1, G*seq, D)
         3) self-attend → (1, G*seq, D)
         4) reshape back to (G, seq, D)
        and scatter into out.
        """
        B, seq, D = batch_seq.shape
        out = batch_seq.clone().bfloat16()

        for img_id in img_ids.unique():
            idx = (img_ids == img_id).nonzero(as_tuple=True)[0]
            G = idx.numel()
            if G <= 1:
                continue

            # 1) gather → (G, seq, D)
            grp = batch_seq[idx]  

            obj_emb = self.object_embeddings[:G, :, :]
            obj_emb = obj_emb.expand(-1, seq, -1)

            scaled_obj_emb = obj_emb * self.obj_embedding_scale

            grp_with_identity = self.obj_embedding_proj(grp + scaled_obj_emb)

            # 2) flatten to (1, G*seq, D)
            flat = grp_with_identity.reshape(1, G * seq, D)

            # 3) cross-object self-attend over all G*seq tokens
            attn_flat, _ = self.cross_obj_attn(flat, flat, flat)

            # 4) reshape back to (G, seq, D)
            attn_grp = attn_flat.view(G, seq, D)

            # scatter into output
            out[idx] = attn_grp

        return out

    def forward(
        self,
        curr: torch.Tensor,  # self-attention inputs
        memory: torch.Tensor,  # cross-attention inputs
        curr_pos: Optional[Tensor] = None,  # pos_enc for self-attention inputs
        memory_pos: Optional[Tensor] = None,  # pos_enc for cross-attention inputs
        num_obj_ptr_tokens: int = 0,  # number of object pointer *tokens*
        img_ids=None,
    ):
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = (
                curr[0],
                curr_pos[0],
            )

        assert (
            curr.shape[1] == memory.shape[1]
        ), "Batch size must be the same for curr and memory"

        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        if self.batch_first:
            # Convert to batch first
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)

        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}

            output = layer(
                tgt=output,
                memory=memory,
                pos=memory_pos,
                query_pos=curr_pos,
                **kwds,
            )

        # Cross-object attention
        # if img_ids is not None:
        #     output = self._cross_object_attention(output, img_ids)
        # else:
        #     print("WARNING ERROR IMG IDS NOT PROVIDED IN MEM ATTENTION FORWARD ----------------")

        normed_output = self.norm(output)

        if self.batch_first:
            # Convert back to seq first
            normed_output = normed_output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)

        return normed_output
