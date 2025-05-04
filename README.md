# TACSCAN-AI 🫁🔬  

**TACSCAN-AI** is a student project developed for the **Sistemi Digitali M** course at the University of Bologna.
It focuses on binary classification of chest X-ray images (ChestMNIST dataset) to identify cases of cardiomegaly and analyze computational performance between CPU and GPU implementations.


## 🎯 Project Goals

- Implement a **simple classifier** based on weighted sum of pixels and a decision threshold.
- Develop:
  - a **sequential C version** (CPU baseline),
  - a **naïve CUDA version** (basic parallelization),
  - an **optimized CUDA version** applying advanced techniques (shared memory, occupancy, coalesced memory access).
- Perform **performance profiling** using Nsight Compute.
- Analyze **scaling behavior** as input image size increases (28×28, 56×56, up to 224×224).


This work is exploratory and educational in nature — not intended for clinical deployment.

## 🛠️ Tools & Libraries

- Python  
- PyTorch  
- NumPy / Pandas  
- (Planned) SimpleITK or similar for image handling  
- Jupyter Notebooks  

## 📍 Project Status

🧾 Dataset exploration (in progress)  
🔬 Preprocessing pipeline (planned)  
🧠 Model design and training (coming next)  
📊 Evaluation and reporting (to follow)

## 📁 Repository Structure (planned)

```
- `/data` → datasets and preprocessing scripts
- `/src`
  - `/sequential` → sequential C implementation
  - `/cuda_naive` → initial CUDA implementation
  - `/cuda_opt` → optimized CUDA implementation
- `/results` → profiling reports, performance plots
- `/report` → slides, final report
```

## 📊 Dataset

This project uses the [ChestMNIST dataset](https://medmnist.com/), a collection of preprocessed medical images available in multiple 2D and 3D resolutions.

For simplicity, the focus is solely on the **inference phase** (no training), using approximately 512 images at varying resolutions.

## ⚙ CUDA Techniques
The project will apply several CUDA-specific optimization techniques, including:
- Use of **shared memory** to reduce global memory access latency,
- Designing **coalesced memory accesses** to maximize throughput,
- Tuning **grid and block configurations** to optimize SM occupancy,
- Analyzing and minimizing **warp divergence** where necessary.

## 📜 License

This project is released under the MIT License.

## 👨‍💻 Author

**Enrico Strangio**  
MSc student in Computer Engineering, University of Bologna  
[linkedin.com/in/enrico-strangio](https://www.linkedin.com/in/enrico-strangio)

## 📬 Contact

Feel free to reach out if you're working on similar topics or interested in collaboration!

