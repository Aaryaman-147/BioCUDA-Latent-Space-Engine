import torch

def pack_sequences(sequences, tokenizer, device="cuda"):
    # 1. Tokenize (no padding)
    tokenized_batch = [
        tokenizer(seq, return_tensors="pt")["input_ids"].squeeze(0) 
        for seq in sequences
    ]
    
    # 2. Get sequence lengths
    seq_lens = torch.tensor([len(t) for t in tokenized_batch], dtype=torch.int32, device=device)
    
    # 3. Concatenate tokens
    packed_input_ids = torch.cat(tokenized_batch).to(device)
    
    # 4. NEW: Generate Local Position IDs that reset for each protein
    # Example: [0, 1, 2, 3, 0, 1, 2, 3, 4, 5...]
    position_ids = torch.cat([
        torch.arange(length, dtype=torch.long, device=device) 
        for length in seq_lens
    ])
    
    # 5. Calculate Cumulative Sequence Lengths
    zero_pad = torch.zeros(1, dtype=torch.int32, device=device)
    cu_seqlens = torch.cat([zero_pad, torch.cumsum(seq_lens, dim=0)])
    max_seqlen = torch.max(seq_lens).item()
    
    return {
        "packed_input_ids": packed_input_ids,  
        "position_ids": position_ids,          # <-- New!
        "cu_seqlens": cu_seqlens,              
        "max_seqlen": max_seqlen,              
        "total_tokens": packed_input_ids.shape[0]
    }

# --- Quick Test ---
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    
    # A tiny peptide and a longer synthetic sequence
    test_batch = [
        "MALW",                 # Length 4 (plus cls/eos = 6)
        "MKTLLLTLVVVTIVCLDLGYT" # Length 21 (plus cls/eos = 23)
    ]
    
    packed_data = pack_sequences(test_batch, tokenizer, device="cpu") # using CPU just for a quick print test
    
    print("\n--- Packing Results ---")
    print(f"Original Sequences: {len(test_batch)}")
    print(f"Packed 1D Shape: {packed_data['packed_input_ids'].shape}")
    print(f"Cumulative Indices: {packed_data['cu_seqlens']}")
    print(f"Padding Tokens Used: 0 ✅")