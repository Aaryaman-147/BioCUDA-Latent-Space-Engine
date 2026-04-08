# ==============================================================================
# GPU CONFIGURATION NOTES:
# For A100/H100 (80GB+ VRAM): max_tokens=65536, batch_size=64+
# For T4/L4 Edge Deployments (15GB VRAM): max_tokens=2048, batch_size=8
# ==============================================================================

import torch
import time
import os
from transformers import AutoTokenizer

torch.set_default_dtype(torch.float16)
from esm_model import BioCudaEngine
from data_utils import pack_sequences

def read_fasta(file_path):
    sequences = []
    with open(file_path, 'r') as f:
        seq_id, seq_data = "", []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id: sequences.append((seq_id, "".join(seq_data)))
                seq_id = line[1:]
                seq_data = []
            else:
                seq_data.append(line)
        if seq_id: sequences.append((seq_id, "".join(seq_data)))
    return sequences

def run_speed_extraction(fasta_path, output_dir, batch_size=8): # 🛑 LOWERED BATCH SIZE
    model_name = "facebook/esm2_t36_3B_UR50D"
    print(f"🚀 Initializing BioCUDA SPEED ENGINE (Autocast Shield Active)...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    engine = BioCudaEngine(model_name).cuda()
    engine.eval()

    torch.set_default_dtype(torch.float32)

    print(f"Recording CUDA Graph (Max Tokens: 2048, Max Batch: {batch_size})...")

    # 🛑 FORCE GARBAGE COLLECTION BEFORE ALLOCATING THE GRAPH
    torch.cuda.empty_cache()

    # 🛑 Wrap the graph trace in no_grad() so it doesn't hoard memory for backpropagation
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        engine.setup_cuda_graph(
            max_tokens=2048,
            max_batch_size=batch_size,
            max_seqlen=1024
        )

    os.makedirs(output_dir, exist_ok=True)
    fasta_data = read_fasta(fasta_path)
    total_seqs = len(fasta_data)
    print(f"Processing {total_seqs} proteins...")

    start_time = time.time()
    results_db = {}

    for i in range(0, total_seqs, batch_size):
        batch = fasta_data[i : i + batch_size]
        batch_ids = [item[0] for item in batch]
        batch_seqs = [item[1] for item in batch]

        packed_data = pack_sequences(batch_seqs, tokenizer, device="cuda")

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = engine(
                packed_data["packed_input_ids"],
                packed_data["cu_seqlens"],
                packed_data["max_seqlen"],
                packed_data["position_ids"]
            )

        cu_seqlens = packed_data["cu_seqlens"].cpu().numpy()
        for j, seq_id in enumerate(batch_ids):
            start_idx, end_idx = cu_seqlens[j], cu_seqlens[j+1]

            protein_representation = outputs[start_idx:end_idx, :].float().mean(dim=0)
            results_db[seq_id] = protein_representation.cpu().clone()

        print(f"Processed batch {i//batch_size + 1} ({min(i+batch_size, total_seqs)}/{total_seqs})")

    torch.cuda.synchronize()
    total_time = time.time() - start_time

    print("\n" + "="*40)
    print(f"✅ Speed Extraction Complete in {total_time:.2f} seconds!")
    print(f"⚡ Processing Speed: {total_seqs / total_time:.2f} proteins/sec")
    print("="*40)

if __name__ == "__main__":
    run_speed_extraction("test_proteins.fasta", output_dir="./speed_db", batch_size=8)