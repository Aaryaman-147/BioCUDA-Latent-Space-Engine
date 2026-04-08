import torch
import time
import os
from transformers import AutoTokenizer, AutoModelForMaskedLM, BitsAndBytesConfig

# ---------------------------------------------------------
# 1. Simple FASTA Parser
# ---------------------------------------------------------
def read_fasta(file_path):
    sequences = []
    with open(file_path, 'r') as f:
        seq_id = ""
        seq_data = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id:
                    sequences.append((seq_id, "".join(seq_data)))
                seq_id = line[1:]
                seq_data = []
            else:
                seq_data.append(line)
        if seq_id:
            sequences.append((seq_id, "".join(seq_data)))
    return sequences

# ---------------------------------------------------------
# 2. The 8-Bit Bulk Extraction Pipeline
# ---------------------------------------------------------
def run_bulk_extraction(fasta_path, output_dir, batch_size=16): 
    model_name = "facebook/esm2_t36_3B_UR50D"
    print(f"Initializing BioCUDA Engine in 8-BIT MODE with {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0 
    )
    
    engine = AutoModelForMaskedLM.from_pretrained(
        model_name, 
        quantization_config=quant_config,
        device_map="auto"
    )
    engine.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nReading {fasta_path}...")
    fasta_data = read_fasta(fasta_path)
    total_seqs = len(fasta_data)
    print(f"Found {total_seqs} proteins.")
    
    print("\nStarting high-throughput 8-bit extraction...")
    start_time = time.time()
    
    results_db = {} 
    
    for i in range(0, total_seqs, batch_size):
        batch = fasta_data[i : i + batch_size]
        batch_ids = [item[0] for item in batch]
        batch_seqs = [item[1] for item in batch]
        
        # Standard HuggingFace tokenization with dynamic padding for the batch
        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True).to("cuda")
        
        with torch.no_grad():
            outputs = engine(**inputs, output_hidden_states=True)
            
        # THE FP32 HACK: Cast to 32-bit before averaging to prevent NaN corruption!
        hidden_states = outputs.hidden_states[-1].float()
        attention_mask = inputs['attention_mask']
        
        for j in range(len(batch_seqs)):
            # Calculate actual sequence length ignoring padding and special tokens
            seq_len = sum(attention_mask[j]) - 2 
            
            # Extract the specific protein and calculate its global average representation
            protein_representation = hidden_states[j, 1:seq_len+1, :].mean(dim=0)
            results_db[batch_ids[j]] = protein_representation.cpu().clone()
            
        print(f"Processed batch {i//batch_size + 1} ({min(i+batch_size, total_seqs)}/{total_seqs})")

    torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    output_file = os.path.join(output_dir, "protein_embeddings.pt")
    torch.save(results_db, output_file)
    
    print("\n=========================================")
    print(f"✅ Extraction Complete in {total_time:.2f} seconds!")
    print(f"⚡ Processing Speed: {total_seqs / total_time:.2f} proteins / second")
    print(f"💾 Saved database to: {output_file}")
    print("=========================================")

# ---------------------------------------------------------
# 3. Create a Dummy Dataset and Run
# ---------------------------------------------------------
if __name__ == "__main__":
    test_fasta = "test_proteins.fasta"
    if not os.path.exists(test_fasta):
        print("Generating a test FASTA file with 100 dummy proteins...")
        with open(test_fasta, "w") as f:
            for i in range(100):
                length = 50 + (i % 50) 
                f.write(f">Protein_{i}\n")
                f.write("A" * length + "\n")
                
    # Running with batch size of 16 thanks to 8-bit memory savings
    run_bulk_extraction(test_fasta, output_dir="./embeddings_db", batch_size=16)