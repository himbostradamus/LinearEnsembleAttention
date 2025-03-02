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

class MultiheadLinearEnsembleAttn(nn.Module):
    def __init__(
        self,
        args,
        embed_dim,
        depth,
        num_heads,
        num_maps=3,  # Default number of attention maps in the ensemble
    ):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.num_maps = num_maps
        
        # arg num_heads set to preserve transformer size
        self.num_heads = num_heads
        
        # arg decoder_kv_attention_heads for GQA support
        self.num_kv_heads = args.decoder_kv_attention_heads if args.decoder_kv_attention_heads is not None else num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        
        # Adjust head_dim for multiple maps
        self.head_dim = embed_dim // num_heads // num_maps
        self.scaling = self.head_dim ** -0.5
        
        # Create projections for each map
        self.q_projs = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim // num_maps, bias=False)
            for _ in range(num_maps)
        ])
        self.k_projs = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim // num_maps // self.n_rep, bias=False)
            for _ in range(num_maps)
        ])
        
        # Single value projection (shared across maps)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Raw attention weights with learnable initialization
        self.raw_map_weights = nn.Parameter(torch.zeros(num_maps, dtype=torch.float32))
        # Scale for controlling weight impact
        self.weight_scale = nn.Parameter(torch.ones(1, dtype=torch.float32))
        
        # Normalization for outputs
        self.subln = RMSNorm(self.head_dim * num_maps, eps=1e-5, elementwise_affine=True)
        
        # Initialize with varying weights
        with torch.no_grad():
            for i in range(num_maps):
                # Initialize with alternating positive/negative small values
                self.raw_map_weights[i] = 0.1 * ((-1) ** i)

    def forward(
        self,
        x,
        rel_pos,
        attn_mask=None,
    ):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len
        
        # Apply tanh to get weights in range [-1, 1] and multiply by scale
        map_weights = torch.tanh(self.raw_map_weights) * self.weight_scale
        
        # Process queries and keys for each map
        all_qs = []
        all_ks = []
        for i in range(self.num_maps):
            q_i = self.q_projs[i](x)
            k_i = self.k_projs[i](x)
            
            q_i = q_i.view(bsz, tgt_len, self.num_heads, self.head_dim)
            k_i = k_i.view(bsz, src_len, self.num_kv_heads, self.head_dim)
            
            # Apply rotary embeddings
            q_i = apply_rotary_emb(q_i, *rel_pos, interleaved=True)
            k_i = apply_rotary_emb(k_i, *rel_pos, interleaved=True)
            
            q_i = q_i.transpose(1, 2)  # [bsz, num_heads, tgt_len, head_dim]
            k_i = repeat_kv(k_i.transpose(1, 2), self.n_rep)  # [bsz, num_heads, src_len, head_dim]
            
            all_qs.append(q_i * self.scaling)
            all_ks.append(k_i)
        
        # Process values (shared across maps)
        v = self.v_proj(x)
        v = v.view(bsz, src_len, self.num_kv_heads, self.num_maps * self.head_dim)
        v = repeat_kv(v.transpose(1, 2), self.n_rep)  # [bsz, num_heads, src_len, num_maps*head_dim]
                
        # Prepare attention mask
        offset = src_len - tgt_len
        if attn_mask is None:
            attn_mask = torch.triu(
                torch.zeros([tgt_len, src_len])
                .float()
                .fill_(float("-inf"))
                .type_as(all_qs[0]),
                1 + offset,
            )
        
        # Initialize combined attention
        combined_attn = 0
        
        # Calculate attention for each map and combine them
        for i in range(self.num_maps):
            # Calculate attention scores
            attn_scores = torch.matmul(all_qs[i], all_ks[i].transpose(-1, -2))
            attn_scores = torch.nan_to_num(attn_scores)
            attn_scores += attn_mask
            
            # Apply softmax to get attention weights
            attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).type_as(attn_scores)
            
            # Add this map with its learned weight
            combined_attn = combined_attn + map_weights[i] * attn_weights
        
        # Apply combined attention to values
        attn = torch.matmul(combined_attn, v)
        
        # Normalize and process
        attn = self.subln(attn)
        
        # Transform back to output shape
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * self.num_maps * self.head_dim)
        attn = self.out_proj(attn)
        
        return attn
