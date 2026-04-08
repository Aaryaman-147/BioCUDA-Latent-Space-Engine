from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, BitsAndBytesConfig
from fastapi.responses import RedirectResponse
import numpy as np
from sklearn.decomposition import PCA
from typing import List, Dict
import re

# 1. Initialize Server & CORS
app = FastAPI(title="BioCUDA API", description="8-Bit Quantized ESM-2 3B Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load the 3B Leviathan in INT8 (8-bit) Mode
model_name = "facebook/esm2_t36_3B_UR50D"
print(f"Booting up {model_name} in 8-BIT MODE...")
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

# 3. Data Models
class ProteinRequest(BaseModel):
    sequence: str

class BatchProteinRequest(BaseModel):
    proteins: List[Dict[str, str]]

# --- THE GUARD: Sequence Validator ---
def validate_sequence(seq: str):
    seq = seq.upper().strip()
    if not re.match(r"^[ACDEFGHIKLMNPQRSTVWY]+$", seq):
        raise HTTPException(status_code=400, detail="Invalid sequence. Only standard amino acids allowed.")
    return seq

@app.get("/")
def home():
    return RedirectResponse(url="/docs")

# --- ROUTE 1: Single Embedding ---
@app.post("/embed")
def get_embedding(request: ProteinRequest):
    valid_seq = validate_sequence(request.sequence)

    # Use standard HuggingFace tokenizer for the 8-bit model
    inputs = tokenizer(valid_seq, return_tensors="pt", add_special_tokens=True).to("cuda")

    with torch.no_grad():
        outputs = engine(**inputs, output_hidden_states=True)

    # THE FP32 HACK: Cast to 32-bit float BEFORE calculating the mean to prevent NaNs!
    hidden_states = outputs.hidden_states[-1].float()
    protein_representation = hidden_states[0, 1:-1, :].mean(dim=0)

    return {
        "sequence": valid_seq,
        "dimensions": protein_representation.shape[0],
        "vector": protein_representation.cpu().tolist()
    }

# --- ROUTE 2: Batch Clustering Engine ---
@app.post("/cluster")
def cluster_proteins(request: BatchProteinRequest):
    embeddings = []

    for p in request.proteins:
        valid_seq = validate_sequence(p["sequence"])
        inputs = tokenizer(valid_seq, return_tensors="pt", add_special_tokens=True).to("cuda")

        with torch.no_grad():
            outputs = engine(**inputs, output_hidden_states=True)

        # THE FP32 HACK (Batch version)
        hidden_states = outputs.hidden_states[-1].float()
        protein_representation = hidden_states[0, 1:-1, :].mean(dim=0)
        embeddings.append(protein_representation.cpu().numpy())

    # Squash 2560 dimensions down to 2 dimensions using PCA
    X = np.array(embeddings)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)

    # Attach coordinates back to metadata
    results = []
    for i, p in enumerate(request.proteins):
        results.append({
            "id": p["id"],
            "family": p["family"],
            "sequence": p["sequence"],
            "x": float(coords[i][0]),
            "y": float(coords[i][1])
        })

    return {"clusters": results}