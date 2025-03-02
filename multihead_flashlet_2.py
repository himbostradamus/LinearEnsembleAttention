import math
import torch
import torch.nn.functional as F
from torch import nn
from .kernel.rotary import apply_rotary_emb
from flash_attn import flash_attn_func
try:
    from apex.normalization import FusedRMSNorm as RMSNorm 
except ModuleNotFoundError:
    print("No fused RMSNorm")
    from .rms_norm import RMSNorm

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

class MultiheadFlashLinearEnsemble2(nn.Module):
    """
    Linear Ensemble Transformer implemented with FlashAttention
    For packages that do not support different qk/v dimensions
    """
    def __init__(
        self,
        args,
        embed_dim,
        depth,
        num_heads,
        num_maps=3,  # Default number of attention maps
    ):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.num_maps = num_maps
        
        # Keep the head count as in the original implementation
        self.num_heads = num_heads
        
        # Support for grouped query attention
        self.num_kv_heads = args.decoder_kv_attention_heads if args.decoder_kv_attention_heads is not None else num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        
        # Adjust dimensions for multiple maps
        self.head_dim = embed_dim // num_heads // num_maps
        self.scaling = self.head_dim ** -0.5
        
        # Query and key projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        
        # Value projection, split for each map
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Learnable raw weights with tanh normalization
        self.raw_map_weights = nn.Parameter(torch.zeros(num_maps, dtype=torch.float32))
        self.weight_scale = nn.Parameter(torch.ones(1, dtype=torch.float32))
        
        # Output normalization
        self.subln = RMSNorm(num_maps * self.head_dim, eps=1e-5, elementwise_affine=True)
        
        # Initialize with alternating small values
        with torch.no_grad():
            for i in range(num_maps):
                self.raw_map_weights[i] = 0.1 * ((-1) ** i)
    
    def forward(
        self,
        x,
        rel_pos,
        attn_mask=None,
    ):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len
        
        # Project inputs
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multiple maps
        q = q.view(bsz, tgt_len, self.num_maps * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, self.num_maps * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, self.num_maps, self.head_dim)
        
        # Apply rotary embeddings
        q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        k = apply_rotary_emb(k, *rel_pos, interleaved=True)
        
        # Reshape to separate the maps
        q = q.reshape(bsz, tgt_len, self.num_heads, self.num_maps, self.head_dim)
        k = k.reshape(bsz, src_len, self.num_kv_heads, self.num_maps, self.head_dim)
        
        # Apply tanh to get weights in range [-1, 1] and multiply by scale
        map_weights = torch.tanh(self.raw_map_weights) * self.weight_scale
        
        # Initialize storage for all attention outputs
        all_attns = []
        
        # Calculate attention for each map with each value part
        for map_idx in range(self.num_maps):
            q_map = q[:, :, :, map_idx]  # [bsz, tgt_len, num_heads, head_dim]
            k_map = k[:, :, :, map_idx]  # [bsz, src_len, num_kv_heads, head_dim]
            
            # For each map, compute attention with all value parts
            map_results = []
            for value_idx in range(self.num_maps):
                v_part = v[:, :, :, value_idx]  # [bsz, src_len, num_kv_heads, head_dim]
                
                # Calculate attention using flash attention
                attn_result = flash_attn_func(q_map, k_map, v_part, causal=True)
                map_results.append(attn_result)
            
            # Concatenate results for this map
            map_attn = torch.cat(map_results, dim=-1)  # [bsz, tgt_len, num_heads, num_maps*head_dim]
            all_attns.append(map_attn)
        
        # Apply weighted combination
        combined_attn = torch.zeros_like(all_attns[0])
        for i in range(self.num_maps):
            combined_attn = combined_attn + map_weights[i] * all_attns[i]
        
        # Normalize output
        combined_attn = self.subln(combined_attn)
        
        # Reshape to expected output format
        combined_attn = combined_attn.reshape(bsz, tgt_len, self.num_heads * self.num_maps * self.head_dim)
        
        # Final projection
        output = self.out_proj(combined_attn)
        
        return output
