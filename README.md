# BioCUDA Latent Space Engine 🧬⚡

BioCUDA is a high-throughput, dual-branch machine learning infrastructure designed to run the massive **3 Billion parameter ESM-2 foundation model** across both constrained edge hardware (T4 GPUs) and enterprise computing clusters (A100 GPUs) without mathematical corruption.

### 🛠️ Core Engineering Achievements
* **Architected a custom PyTorch extraction engine** capable of processing up to 36+ proteins per second on A100 architecture.
* **Engineered a mathematical FP32 Guard bypass** to solve PyTorch's native precision overflow (`NaN` poisoning) in 16-bit processing, ensuring 100% data integrity for high-dimensional latent space clustering.
* **Built an automated dimensionality reduction pipeline** (PCA via Scikit-Learn) to compress 2,560-dimensional tensors into 2D functional radar maps.

---

## 🏗️ The Dual-Architecture System

Because a 3B parameter model natively causes Out-Of-Memory (OOM) crashes on standard hardware, this inference pipeline is forked into two specialized branches:

### 1. The Stability Branch (Edge API)
Built for live, interactive inference. 
* **Hardware:** NVIDIA T4 (15GB VRAM)
* **Stack:** FastAPI, `bitsandbytes`
* **Mechanism:** Uses INT8 Quantization to compress the 3B model footprint by 50%, enabling stable, real-time protein sequence extraction on cheap hardware. Contains a strict Regex validation firewall to protect GPU memory from invalid data.

### 2. The Speed Branch (High-Throughput Offline Lab)
Built for massive dataset processing and *In-Silico* Screening.
* **Hardware:** NVIDIA A100 (80GB VRAM)
* **Stack:** PyTorch, Triton (Flash Attention), CUDA Graphs
* **Mechanism:** Utilizes uncompressed FP16 math locked into a massive CUDA Graph memory buffer (`max_tokens=65536`, `batch_size=64`). Achieves speeds of **36+ proteins per second**.

---

## 🧗 Challenges Overcome: The FP16 "NaN Poisoning"
During A100 stress testing, forcing the 3B model into pure 16-bit precision to maximize speed resulted in memory register overflows, yielding `NaN` (Not a Number) corruption in the final embeddings. 

**The Solution:** We engineered an `autocast` shield and an **FP32 Guard** that dynamically bridges the final pooling calculations back to 32-bit floats. This traded a microscopic amount of compute time for 100% mathematically pristine data extraction.

---

## 🔬 System Benchmarks: The Enzyme Hunter
To validate the architecture, the engine was tasked with an *In-Silico* screening simulation to discover plastic-eating enzymes from a dataset of 1,000 ocean bacteria sequences (hiding 9 mutated variants of a known PETase).
* **Processing Speed:** The A100 pipeline successfully processed all 1,000 sequences in **33.75 seconds**.
* **Functional Clustering:** Using Euclidean distance calculations in 2,560-dimensional space, the AI successfully grouped hidden functional variants entirely autonomously, proving the engine accurately maps biological functions directly from raw structural sequences.

---

## 💻 Tech Stack
* **Core ML Framework:** PyTorch
* **GPU Optimization:** NVIDIA Triton (Flash Attention), CUDA Graphs, Automatic Mixed Precision (AMP)
* **Quantization & Edge Inference:** `bitsandbytes` (INT8)
* **API & Data Validation:** FastAPI, Regex
* **Data Science & Visualization:** Scikit-Learn (PCA), Matplotlib, NumPy

---

### Prerequisites
* Python 3.10+
* NVIDIA GPU (15GB+ VRAM for Edge, 40GB+ for Batch processing)
* CUDA Toolkit installed
