import torch
import torch.nn as nn
from transformers import AutoTokenizer, EsmModel
from fast_embedding import BioCudaEmbedding
from flash_attention import BioCudaAttention
from data_utils import pack_sequences
import time

class BioCudaEsmLayer(nn.Module):
    def __init__(self, hf_layer, config):
        super().__init__()
        hidden_size = config.hidden_size
        
        # 1. Initialize our ultra-fast Triton Attention
        self.attention = BioCudaAttention(hidden_size, config.num_attention_heads)
        
        # --- THE TRANSPLANT (Fusing Q, K, V weights) ---
        # HF uses 3 separate matrices. We fuse them into 1 for better memory bandwidth.
        q_w = hf_layer.attention.self.query.weight
        k_w = hf_layer.attention.self.key.weight
        v_w = hf_layer.attention.self.value.weight
        self.attention.qkv_proj.weight = nn.Parameter(torch.cat([q_w, k_w, v_w], dim=0).contiguous())
        
        q_b = hf_layer.attention.self.query.bias
        k_b = hf_layer.attention.self.key.bias
        v_b = hf_layer.attention.self.value.bias
        self.attention.qkv_proj.bias = nn.Parameter(torch.cat([q_b, k_b, v_b], dim=0).contiguous())
        
        # Inject Output Projection weights
        self.attention.out_proj.weight = nn.Parameter(hf_layer.attention.output.dense.weight.contiguous())
        self.attention.out_proj.bias = nn.Parameter(hf_layer.attention.output.dense.bias.contiguous())
        
        # 2. Extract standard LayerNorms and FFN components
        self.attn_layer_norm = hf_layer.attention.LayerNorm
        self.ffn_layer_norm = hf_layer.LayerNorm
        self.ffn_up = hf_layer.intermediate.dense
        self.ffn_out = hf_layer.output.dense

    def forward(self, hidden_states, cu_seqlens, max_seqlen, position_ids):
        # 1. Attention Block (Using our Triton Kernel)
        residual = hidden_states
        hidden_states = self.attn_layer_norm(hidden_states)
        hidden_states = self.attention(hidden_states, cu_seqlens, max_seqlen, position_ids)
        hidden_states = residual + hidden_states
        
        # 2. Feed Forward Block (Native PyTorch, fully compatible with 1D packed tensors)
        residual = hidden_states
        hidden_states = self.ffn_layer_norm(hidden_states)
        hidden_states = self.ffn_up(hidden_states)
        hidden_states = torch.nn.functional.gelu(hidden_states)
        hidden_states = self.ffn_out(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states.float() # Cast to FP32 right before the final average

class BioCudaEngine(nn.Module):
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D"):
        super().__init__()
        print(f"Downloading/Loading base HF model: {model_name}...")
        hf_model = EsmModel.from_pretrained(model_name).cuda().half()
        self.config = hf_model.config
        
        print("Transplanting Embedding Weights...")
        vocab_size, hidden_size = hf_model.embeddings.word_embeddings.weight.shape
        self.embeddings = BioCudaEmbedding(
            vocab_size, hidden_size, hf_model.embeddings.word_embeddings.weight
        )
        
        print(f"Transplanting {self.config.num_hidden_layers} Attention Layers...")
        self.layers = nn.ModuleList([
            BioCudaEsmLayer(hf_layer, self.config) 
            for hf_layer in hf_model.encoder.layer
        ])
        
        self.final_norm = hf_model.encoder.emb_layer_norm_after
        
        # Initialize Graph state variables
        self.is_graph_captured = False
        self.graph = None

    def forward(self, packed_input_ids, cu_seqlens, max_seqlen, position_ids):
        # If we have captured a graph, hijack the forward pass and run the graph instead
        if self.is_graph_captured:
            return self._run_graph(packed_input_ids, cu_seqlens, max_seqlen, position_ids)
            
        # Standard unrecorded forward pass
        hidden_states = self.embeddings(packed_input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cu_seqlens, max_seqlen, position_ids)
        hidden_states = self.final_norm(hidden_states)
        return hidden_states

    # --- CUDA GRAPH OPTIMIZATIONS ---
    def setup_cuda_graph(self, max_tokens, max_batch_size, max_seqlen):
        """Pre-records the engine execution to eliminate CPU overhead."""
        print(f"Recording CUDA Graph (Max Tokens: {max_tokens}, Max Batch: {max_batch_size})...")
        
        # 1. Create static input buffers (Memory that never moves)
        self.static_input_ids = torch.zeros(max_tokens, dtype=torch.long, device="cuda")
        self.static_cu_seqlens = torch.zeros(max_batch_size + 1, dtype=torch.int32, device="cuda") 
        self.static_position_ids = torch.zeros(max_tokens, dtype=torch.long, device="cuda")
        
        # 2. Warm up the JIT Compiler
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                # We call the standard math logic, bypassing the 'forward' hijack
                hidden_states = self.embeddings(self.static_input_ids)
                for layer in self.layers:
                    hidden_states = layer(hidden_states, self.static_cu_seqlens, max_seqlen, self.static_position_ids)
                self.static_output = self.final_norm(hidden_states)
        torch.cuda.current_stream().wait_stream(s)

        # 3. Capture the Graph
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            hidden_states = self.embeddings(self.static_input_ids)
            for layer in self.layers:
                hidden_states = layer(hidden_states, self.static_cu_seqlens, max_seqlen, self.static_position_ids)
            self.static_output = self.final_norm(hidden_states)
            
        self.is_graph_captured = True
        print("✅ CUDA Graph captured successfully.")

    def _run_graph(self, input_ids, cu_seqlens, max_seqlen, position_ids):
        """Replays the recorded graph."""
        n_tokens = input_ids.numel()
        n_batch = cu_seqlens.numel()
        
        # Copy real data into the static buffers
        self.static_input_ids[:n_tokens].copy_(input_ids)
        self.static_cu_seqlens[:n_batch].copy_(cu_seqlens)
        self.static_position_ids[:n_tokens].copy_(position_ids)
        
        # Replay! (This happens entirely on the GPU)
        self.graph.replay()
        
        # Return only the valid part of the output (ignoring the empty static buffer space)
        return self.static_output[:n_tokens, :]

# ==========================================
# FINAL VALIDATION: Hugging Face vs. BioCUDA
# ==========================================
if __name__ == "__main__":
    model_name = "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    test_batch = [
        "MALW",                                      
        "MKTLLLTLVVVTIVCLDLGYT",                     
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK"  
    ]
    
    # 1. Run Baseline Hugging Face (The Slow Way)
    print("\n--- Running Hugging Face Baseline ---")
    hf_model = EsmModel.from_pretrained(model_name).cuda().half()
    hf_inputs = tokenizer(test_batch, return_tensors="pt", padding=True).to("cuda")
    
    start_time = time.time()
    with torch.no_grad():
        hf_outputs = hf_model(**hf_inputs).last_hidden_state
    torch.cuda.synchronize()
    print(f"HF Time: {(time.time() - start_time)*1000:.2f} ms")
    
    # 2. Run BioCUDA Engine (The Fast Way)
    print("\n--- Running BioCUDA Engine ---")
    packed_data = pack_sequences(test_batch, tokenizer, device="cuda")
    engine = BioCudaEngine(model_name).cuda().half()
    
    # Run once to warm up the Triton JIT Compiler
    with torch.no_grad():
        _ = engine(packed_data["packed_input_ids"], packed_data["cu_seqlens"], packed_data["max_seqlen"], packed_data["position_ids"])
    
    # Run again for true speed measurement
    start_time = time.time()
    with torch.no_grad():
        biocuda_outputs = engine(
            packed_data["packed_input_ids"],
            packed_data["cu_seqlens"],
            packed_data["max_seqlen"],
            packed_data["position_ids"]
        )
    torch.cuda.synchronize()
    print(f"BioCUDA Time: {(time.time() - start_time)*1000:.2f} ms")

    # 3. The Math Match
    print("\n--- Validation ---")
    # We must extract the unpadded tokens from HF to compare with our 1D tensor
    hf_1d_list = []
    lens = packed_data["cu_seqlens"][1:] - packed_data["cu_seqlens"][:-1]
    
    for i, seq_len in enumerate(lens):
        # Slice out only the real tokens (ignoring the padding at the end of the row)
        hf_1d_list.append(hf_outputs[i, :seq_len, :])
        
    hf_packed_baseline = torch.cat(hf_1d_list, dim=0)

    # Calculate Cosine Similarity across the entire multi-dimensional space
    cos_sim = torch.nn.functional.cosine_similarity(
        hf_packed_baseline.view(-1).float(), 
        biocuda_outputs.view(-1).float(), 
        dim=0
    )
    
    if cos_sim.item() > 0.99:
        print(f"✅ PASSED: Perfect alignment. Cosine Similarity: {cos_sim.item():.6f}")
    else:
        print(f"❌ FAILED: Math mismatch. Cosine Similarity: {cos_sim.item():.6f}")