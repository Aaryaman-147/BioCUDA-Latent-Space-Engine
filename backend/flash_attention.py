import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

# --------------------------------------------------------
# 1. The Triton Kernel (Skeleton)
# --------------------------------------------------------
@triton.jit
def bio_flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    cu_seqlens_ptr,               # Pointer to our protein boundary indices
    sm_scale,                     # Scaling factor (1 / sqrt(head_dim))
    stride_q_tok, stride_q_head, stride_q_dim,
    stride_k_tok, stride_k_head, stride_k_dim,
    stride_v_tok, stride_v_head, stride_v_dim,
    stride_o_tok, stride_o_head, stride_o_dim,
    BLOCK_M: tl.constexpr,        # Size of the query block
    BLOCK_N: tl.constexpr,        # Size of the key/value block
    BLOCK_DMODEL: tl.constexpr,   # Head dimension (e.g., 64)
):
    # 1. Identify where we are in the grid
    seq_idx = tl.program_id(0)    # Which protein in the batch are we processing?
    head_idx = tl.program_id(1)   # Which attention head?
    start_m = tl.program_id(2)    # Which chunk of the protein?

    # 2. Read the "brick walls" for this specific protein
    start_tok = tl.load(cu_seqlens_ptr + seq_idx)
    end_tok = tl.load(cu_seqlens_ptr + seq_idx + 1)
    seq_len = end_tok - start_tok

    # If this block is outside the current protein's length, do nothing
    block_start_tok = start_m * BLOCK_M
    if block_start_tok >= seq_len:
        return

    # 3. Calculate memory offsets for this specific protein and head
    # We shift all pointers forward by 'start_tok' so we only look at THIS protein
    q_offset = (start_tok + block_start_tok) * stride_q_tok + head_idx * stride_q_head
    k_offset = start_tok * stride_k_tok + head_idx * stride_k_head
    v_offset = start_tok * stride_v_tok + head_idx * stride_v_head

    # Set up pointers to SRAM
    offs_m = block_start_tok + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = q_ptr + q_offset + offs_m[:, None] * stride_q_tok + offs_d[None, :] * stride_q_dim
    k_ptrs = k_ptr + k_offset + offs_n[:, None] * stride_k_tok + offs_d[None, :] * stride_k_dim
    v_ptrs = v_ptr + v_offset + offs_n[:, None] * stride_v_tok + offs_d[None, :] * stride_v_dim

    # 4. Load Q block into fast SRAM
    mask_m = offs_m < seq_len
    q = tl.load(q_ptrs, mask=mask_m[:, None] & (offs_d[None, :] < BLOCK_DMODEL), other=0.0)

    # Initialize running totals for the softmax calculation (FlashAttention magic)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # 5. Inner Loop: Iterate through K and V blocks of THIS protein only
    for start_n in range(0, seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < seq_len

        # Load K and V blocks into SRAM
        k = tl.load(k_ptrs + start_n * stride_k_tok, mask=mask_n[:, None] & (offs_d[None, :] < BLOCK_DMODEL), other=0.0)
        v = tl.load(v_ptrs + start_n * stride_v_tok, mask=mask_n[:, None] & (offs_d[None, :] < BLOCK_DMODEL), other=0.0)

        # Compute Q * K^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= sm_scale

        # Apply causal/padding mask (don't attend outside the protein length)
        qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, float("-inf"))

        # Compute safe softmax
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp(m_i - m_i_new)
        p = tl.math.exp(qk - m_i_new[:, None])
        
        # Update running tallies
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(tl.float16), v)
        
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # 6. Normalize and write back to HBM
    acc = acc / l_i[:, None]
    
    out_offset = (start_tok + block_start_tok) * stride_o_tok + head_idx * stride_o_head
    out_ptrs = out_ptr + out_offset + offs_m[:, None] * stride_o_tok + offs_d[None, :] * stride_o_dim
    tl.store(out_ptrs, acc.to(tl.float16), mask=mask_m[:, None] & (offs_d[None, :] < BLOCK_DMODEL))

def apply_rotary_pos_emb(q, k, position_ids, head_dim):
    # ESM-2 RoPE Base is 10000
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=q.device).float() / head_dim))
    
    # Calculate frequencies based on our local position IDs
    # position_ids shape: [total_tokens] -> unsqueeze to [total_tokens, 1]
    freqs = torch.einsum("i,j->ij", position_ids.float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1) # Shape: [total_tokens, head_dim]
    
    # Reshape for broadcasting with Q and K: [total_tokens, 1, head_dim]
    cos = emb.cos().unsqueeze(1)
    sin = emb.sin().unsqueeze(1)
    
    # Rotate Q and K
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
        
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed.to(q.dtype), k_embed.to(k.dtype)

class BioCudaAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        
        # ESM-2 standard projections
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, packed_hidden_states, cu_seqlens, max_seqlen, position_ids): # Added position_ids
        total_tokens = packed_hidden_states.shape[0]
        batch_size = cu_seqlens.shape[0] - 1
        
        # 1. Project to Q, K, V
        qkv = self.qkv_proj(packed_hidden_states)
        qkv = qkv.view(total_tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=1)
        
        # 2. NEW: Apply Rotary Position Embeddings
        q, k = apply_rotary_pos_emb(q, k, position_ids, self.head_dim)
        
        # 3. Ensure contiguous memory for Triton
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        out = torch.empty_like(q)

        # 2. Define Triton Block Sizes 
        # (Tuning these optimizes for different GPU architectures, 64x64 is a safe default)
        BLOCK_M = 64
        BLOCK_N = 64
        
        # 3. Define the Grid
        # Dimension 0: Batch Size (Number of distinct proteins)
        # Dimension 1: Number of Attention Heads
        # Dimension 2: The sequence chopped into BLOCK_M sized chunks
        grid = lambda META: (
            batch_size, 
            self.num_heads, 
            triton.cdiv(max_seqlen, META['BLOCK_M'])
        )

        # 4. Launch the Kernel!
        bio_flash_attention_kernel[grid](
            q, k, v, out,
            cu_seqlens,
            self.scale,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=self.head_dim,
        )
        
        # 5. Final Output Projection
        out = out.view(total_tokens, self.hidden_size)
        return self.out_proj(out)
    
if __name__ == "__main__":
    print("Testing Bio-FlashAttention...")
    
    # Fake configuration based on ESM-2 8M
    hidden_size = 320
    num_heads = 20
    
    # Fake packed data (e.g., two proteins: lengths 10 and 20)
    cu_seqlens = torch.tensor([0, 10, 30], dtype=torch.int32).cuda()
    max_seqlen = 20
    total_tokens = 30
    
    # Fake hidden states coming from our embedding layer
    packed_hidden_states = torch.randn(
        (total_tokens, hidden_size), 
        dtype=torch.float16, 
        device="cuda"
    )
    
    # Initialize and run
    attention = BioCudaAttention(hidden_size, num_heads).cuda().half()
    
    try:
        output = attention(packed_hidden_states, cu_seqlens, max_seqlen)
        print(f"✅ Success! Output shape: {output.shape} (Expected: [{total_tokens}, {hidden_size}])")
    except Exception as e:
        print(f"❌ Error: {e}")