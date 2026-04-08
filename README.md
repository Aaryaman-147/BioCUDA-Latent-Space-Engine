# BioCUDA Latent Space Engine 🧬⚡

BioCUDA is a high-throughput machine learning infrastructure designed to run the massive **3 Billion parameter ESM-2 foundation model** without mathematical corruption. 

Because a 3B parameter model natively causes Out-Of-Memory (OOM) crashes on standard hardware, this system was architected with a **Dual-Branch Pipeline**:
* **The Edge Branch (T4 GPUs):** Utilizes INT8 quantization (`bitsandbytes`) and a FastAPI validation firewall to compress the model footprint by 50%, enabling stable, real-time protein sequence extraction on cheap 15GB Colab GPUs.
* **The Lab Branch (A100 GPUs):** A high-throughput offline pipeline utilizing uncompressed FP16 math, Triton optimizations, and massive CUDA Graphs (`max_tokens=65536`) to flood the A100 Tensor Cores.

### 🛠️ Core Engineering Achievements
* **Architected a custom PyTorch extraction engine** capable of processing up to 36+ proteins per second on A100 architecture.
* **Engineered a mathematical FP32 Guard bypass** to solve PyTorch's native precision overflow (`NaN` poisoning) in 16-bit processing, ensuring 100% data integrity for high-dimensional latent space clustering.
* **Built an automated dimensionality reduction pipeline** (PCA via Scikit-Learn) to compress 2,560-dimensional tensors into 2D functional radar maps.

---

### 📊 System Results & Benchmarks
To validate the architecture, the engine was tasked with an *In-Silico* screening simulation to discover plastic-eating enzymes from a dataset of 1,000 ocean bacteria sequences.
* **Processing Speed:** The A100 pipeline successfully processed all 1,000 sequences in **33.75 seconds**.
* **Functional Clustering:** Using Euclidean distance calculations, the AI successfully grouped hidden functional variants of the PETase enzyme entirely autonomously, proving the engine accurately maps biological functions directly from raw structural sequences.

---
