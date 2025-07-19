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
        crs_obj_attention: nn.Module, # NEW
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

        # Object embeddings for cross-object attention
        # self.max_objects_per_group = 15
        # self.object_embed = nn.Embedding(self.max_objects_per_group, d_model)
        # nn.init.normal_(self.object_embed.weight, mean=0.0, std=0.02)
        
        # self.cross_obj_self_attn = crs_obj_attention
        # self.norm_cross_obj = nn.LayerNorm(d_model)
        # self.dropout_cross_obj = nn.Dropout(dropout)

    def _forward_cross_obj_sa(self, tgt, query_pos, img_ids):
        """
        Cross-object self-attention with both spatial PE (query_pos) and object embeddings.
        Follows the same pattern as _forward_sa and _forward_ca.
        """
        B, S, C = tgt.shape  # B objects, S spatial tokens, C channels
        device = tgt.device
        
        if img_ids is None or B <= 1:
            return tgt
        
        # Process each image group
        for img_id in img_ids.unique():
            idx = (img_ids == img_id).nonzero(as_tuple=True)[0]
            G = idx.numel()
            
            if G <= 1:
                continue
            
            # Extract features for this group
            group_tgt = tgt[idx]  # [G, C]
            
            # Normalize the group
            tgt2 = self.norm_cross_obj(group_tgt)
            
            # Get spatial positions for this group
            group_query_pos = query_pos[idx] if query_pos is not None else None
            
            # Create object embeddings
            obj_positions = torch.arange(G, device=device)
            obj_pe = self.object_embed(obj_positions)  # [G, C]
            obj_pe = obj_pe.unsqueeze(1).expand(-1, tgt.shape[1], -1)  # [G, S, C]

            # Combine both position encodings:
            # - query_pos: spatial position (from RoPE or other spatial encoding)
            # - obj_pe: object identity within the image
            combined_pos = obj_pe
            
            #if self.pos_enc_at_attn:
            if group_query_pos is not None:
                combined_pos = combined_pos + group_query_pos # We always add this when not using ROPE attention here.
            else:
                raise Exception("GROUP QUERY POS NONE")

            # Add pos enc to image features
            tgt2_pos_enc = tgt2 + combined_pos # [G, S, C]

            # Merge into single sequence for cross-object attention
            tgt2_merged = tgt2_pos_enc.view(1, G * S, C)  # [1, G*S, C]
            tgt2_v_merged = tgt2.view(1, G * S, C)  # values without position encoding

            # Apply cross-object self-attention
            tgt2_merged = self.cross_obj_self_attn(
                q=tgt2_merged,
                k=tgt2_merged,
                v=tgt2_v_merged
            ) # [1, G*S, C]
            
            # Unmerge back
            tgt2 = tgt2_merged.view(G, S, C)  # [G, S, C]

            # Update with residual and dropout
            tgt[idx] = group_tgt + self.dropout_cross_obj(tgt2)
        
        return tgt

    # 91.35 version
    def _forward_ca_nm_sa(self, tgt, query_pos, img_ids):
        """
        Self-attention with both spatial PE (query_pos) and object embeddings.
        Follows the same pattern as _forward_sa and _forward_ca.
        """
        B, S, C = tgt.shape  # B objects, S spatial tokens, C channels
        device = tgt.device
        
        if img_ids is None or B <= 1:
            return tgt
        
        # Process each image group
        for img_id in img_ids.unique():
            idx = (img_ids == img_id).nonzero(as_tuple=True)[0]
            G = idx.numel()
            
            if G <= 1:
                continue
            
            # Extract features for this group
            group_tgt = tgt[idx]  # [G, C]
            
            # Normalize the group
            tgt2 = self.norm_cross_obj(group_tgt)
            
            # Get spatial positions for this group
            group_query_pos = query_pos[idx] if query_pos is not None else None
            
            # Create object embeddings
            obj_positions = torch.arange(G, device=device)
            obj_pe = self.object_embed(obj_positions)  # [G, C]
            obj_pe = obj_pe.unsqueeze(1).expand(-1, tgt.shape[1], -1)  # [G, S, C]

            # Combine both position encodings:
            # - query_pos: spatial position (from RoPE or other spatial encoding)
            # - obj_pe: object identity within the image
            combined_pos = obj_pe
            
            if self.pos_enc_at_attn:
                combined_pos = combined_pos + group_query_pos

            # Add pos enc to image features
            q = k = tgt2 + combined_pos # [G, S, C]

            # Apply cross-object self-attention
            tgt2_merged = self.cross_obj_self_attn(
                q=q,
                k=k,
                v=tgt2
            ) # [1, G*S, C]

            # Update with residual and dropout
            tgt[idx] = group_tgt + self.dropout_cross_obj(tgt2)
        
        return tgt

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
        enable_cross_obj_attn=False, # NEW
        img_ids: Optional[Tensor] = None,
    ) -> torch.Tensor:

        # Self-Attn, Cross-Attn
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)

        # Cross-object self-attention (before MLP)
        # if enable_cross_obj_attn:
        #     #tgt = self._forward_ca_nm_sa(tgt, query_pos, img_ids)
        #     tgt = self._forward_cross_obj_sa(tgt, query_pos, img_ids)

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

        # mem ca 1.0
        # Additional modules for cross-object mixing
        # self.cross_obj_attn = nn.MultiheadAttention(
        #     embed_dim=d_model,        
        #     num_heads=4,      
        #     dropout=0.1,      
        #     batch_first=True,     
        # )     
        # self.object_embeddings = nn.Parameter(        
        #     torch.zeros(10, 1, d_model)       
        # )     
        # nn.init.normal_(self.object_embeddings, mean=0.0, std=0.02)
        # self.obj_embedding_scale = nn.Parameter(torch.ones(1) * 0.1)
        # self.obj_embedding_proj = nn.Linear(d_model, d_model)

        
        # This is the 1.1 mem_ca that got 0.79 val loss on davis. 90.2
        # self.cross_obj_attn = nn.MultiheadAttention(
        #     embed_dim=d_model,
        #     num_heads=4,
        #     dropout=layer.dropout_value,
        #     batch_first=batch_first,
        # )
        # if batch_first != True:
        #     raise Exception("batch first not true in mem attention")
        # # 50 was used here originally.
        # max_batch_items = 30 # num objects in a video * num videos (batch size defined in yaml)
        # # Object embeddings to differentiate objects in the same image
        # # Shape is (max_objects, d_model) - no need for middle dimension
        # self.obj_embeddings = nn.Parameter(torch.zeros(max_batch_items, d_model))
        # nn.init.normal_(self.obj_embeddings, mean=0.0, std=0.02)
        # # Scaling factor for object embeddings
        # self.obj_emb_scale = nn.Parameter(torch.ones(1) * 0.1)

        # # Simple cross-object mixing with object identity
        # self.obj_embedding = nn.Embedding(100, d_model)  # Support up to 100 objects
        #self.obj_embedding_scale = nn.Parameter(torch.ones(1) * 0.1)
        # self.cross_obj_proj = nn.Linear(d_model, d_model)

    def _cross_object_attention_fully_batched(self, batch_seq: torch.Tensor, img_ids: torch.Tensor) -> torch.Tensor:
        B, seq, D = batch_seq.shape
        device = batch_seq.device
        
        # Skip cross-object attention if there's no need for it
        if img_ids is None or B <= 1:
            return batch_seq
            
        # Ensure img_ids is on the correct device
        img_ids = img_ids.to(device)
        
        # 1. Create attention mask based on image IDs
        same_image_mask = (img_ids.unsqueeze(1) == img_ids.unsqueeze(0)).float()
        
        # 2. Count objects per image for each object
        objects_per_image = same_image_mask.sum(dim=1)
        
        # 3. Find objects that have other objects from the same image
        has_other_objects = (objects_per_image > 1)
        
        # If no objects have others from the same image, return the input unchanged
        if not has_other_objects.any():
            return batch_seq
        
        # 4. Select only objects that have others from the same image
        active_indices = has_other_objects.nonzero().squeeze(-1)
        active_batch = batch_seq[active_indices]
        active_mask = same_image_mask[active_indices][:, active_indices]
        
        # 5. Add object embeddings to differentiate objects
        num_active = len(active_indices)
        obj_embs = self.obj_embeddings[:num_active].unsqueeze(1).expand(-1, seq, -1)
        enhanced_active_batch = active_batch + self.obj_emb_scale * obj_embs
        
        # 6. Prepare attention mask for the MultiheadAttention module
        # We need to convert our same_image_mask to an attention mask format
        # Reshape to match attention requirements: (num_active * seq, num_active * seq)
        attn_mask = torch.zeros(num_active * seq, num_active * seq, device=device)
        
        # Fill in the attention mask - allow attention only between tokens of objects from the same image
        for i in range(num_active):
            # For each object, find all other objects from the same image
            other_objects = active_mask[i].nonzero().squeeze(-1)
            
            # Allow attention between this object and all others from same image
            for j in other_objects:
                # Set all positions between these two objects to attend
                attn_mask[i*seq:(i+1)*seq, j*seq:(j+1)*seq] = 1.0
        
        # 7. Set mask values to -inf where attention is not allowed
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf'))
        
        # For batch_first=True
        flat_active_batch = enhanced_active_batch.reshape(1, num_active * seq, D)
            
        # Apply attention in one batched operation
        attended_active_batch, _ = self.cross_obj_attn(
            flat_active_batch, flat_active_batch, flat_active_batch,
            attn_mask=attn_mask
        )
        
        # Reshape back to (num_active, seq, D)
        attended_active_batch = attended_active_batch.reshape(num_active, seq, D)
        
        # 9. Add residual connection
        attended_active_batch = active_batch + attended_active_batch
        
        # 10. Create the output tensor directly - no cloning needed
        # We can modify batch_seq directly since it's not needed anymore
        batch_seq[active_indices] = attended_active_batch
        
        return batch_seq

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

        img_ids = img_ids.to(device)
        
        # Start with the original sequence as the output
        output = batch_seq.clone()
        
        # Process each image group separately
        for img_id in img_ids.unique():
            # Find indices of objects with the same image ID
            indices = (img_ids == img_id).nonzero(as_tuple=True)[0]
            num_objs = indices.size(0)
            
            if num_objs <= 1:
                continue  # Skip if only one object from this image
                
            # Get normalized features for objects with this image ID
            group_features = batch_seq[indices] #self.cross_obj_norm(batch_seq[indices])  # (num_objs, seq, D)
            
            # Add object embeddings to differentiate objects
            obj_indices = torch.arange(num_objs, device=device)
            obj_embs = self.obj_embeddings[obj_indices].unsqueeze(1)  # (num_objs, 1, D)
            obj_embs = obj_embs.expand(-1, seq, -1)  # (num_objs, seq, D)
            
            enhanced_group = group_features + self.obj_emb_scale * obj_embs
            
            # Apply self-attention within this group
            attended_group, _ = self.cross_obj_attn(
                enhanced_group, enhanced_group, enhanced_group,
                need_weights=False
            )
            
            # Add residual connection and update the output
            attended_group = group_features + attended_group
            
            # Update the output tensor for these indices
            output.index_copy_(0, indices.to(output.device), attended_group)
        
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
                enable_cross_obj_attn=True,
                img_ids=img_ids,
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
