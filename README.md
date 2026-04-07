# **Project Eisen: Zero-Dependency Rust LLM Training Engine**

**Objective:** Build a custom, bare-metal autograd engine and train a German-optimized LLM from scratch. The engine must operate entirely without heavy ML framework dependencies, run within a strict 6-8GB GPU VRAM constraint, utilize custom-written CUDA kernels inspired by state-of-the-art implementations, and output models native to the Hugging Face ecosystem.

## **Phase 1: The Foundation (Math & Autograd on CPU) ✅**

*Focus: Memory-safe abstractions and the Wengert List.*

* ✅ **The Tensor Struct:** 1D flat-buffer data container with multidimensional stride, shape, and broadcasting logic.  
* ✅ **The Graph Context:** Graph struct owning the Gradient Tape (Wengert List) and tensor lifecycles.  
* ✅ **Core Math Ops (CPU):** Implementation of Add, Mul, MatMul, Transpose, Sum, and Max.  
* ✅ **Autograd Dispatch:** Reverse-mode differentiation closures for all core operations.  
* ✅ **Milestone 1:** Gradient verification via complex matrix chains matching analytical expectations.

## **Phase 2: Deep Learning Primitives ✅**

*Focus: Assembling the building blocks of a neural network.*

* ✅ **Activations & Loss:** SiLU, numerically stable Softmax, and CrossEntropyLoss.  
* ✅ **Neural Modules:** Linear layer, Embedding layer, and RMSNorm.  
* ✅ **Optimizer:** AdamW implementation for flat parameter buffer updates.  
* ✅ **Milestone 2:** Successfully overfitted a tiny MLP on a classification task with loss reaching zero.

## **Phase 3: Hardware Acceleration (The GPU Leap) ✅**

*Focus: Moving from RAM to VRAM and writing blazing-fast compute kernels via cudarc.*

* ✅ **The Backend Abstraction:** Refactored Tensor to support Device::Cpu and Device::Gpu(Arc\<CudaContext\>, Arc\<CudaStream\>).  
* ✅ **PTX Pipeline:** Automated .cu to .ptx compilation via build.rs and module loading in Graph.  
* ✅ **Custom CUDA Kernels:** All Core Math Operations ported to VRAM\!  
* ✅ **VRAM Allocator:** Implemented a size-bucketed free list (Graph::vram\_pool). Activation buffers are intelligently stripped and recycled at the end of each step.  
* ✅ **Milestone 3:** Run the XOR and Bengio LM tests entirely on the GPU with zero allocations inside the loop.

## **Phase 4: The Transformer Architecture & VRAM Strategy ✅**

*Focus: Scaled dot-product attention and strict memory survival.*

* ✅ **Attention Primitives:** Added memory-efficient Batched MatMul (bmm\_f32 with inline transpose flags) and standalone softmax\_f32 kernels.  
* ✅ **Multi-Head Attention:** Constructed MultiHeadAttention. Implemented a highly specific transpose\_0213\_f32 kernel to slice matrices into multiple heads dynamically without breaking VRAM.  
* ✅ **Positional Encodings:** Implemented Rotary Positional Embeddings (RoPE). The frequencies are computed entirely on-the-fly in registers.  
* ✅ **Transformer Block:** Assembled the TransformerBlock utilizing the Pre-Norm architecture with dual residual connections.  
* ✅ **Gradient Checkpointing:** Added no\_grad, mark\_save\_point, and restore\_save\_point to the Graph. Fixed VRAM allocation provenance (is\_pooled) to ensure a zero-leak environment.  
* ✅ **Milestone 4:** Full Transformer block forward/backward pass on GPU with zero memory leaks and fully verified gradient routing.

## **Phase 5: German Optimization & Data Pipeline ✅**

*Focus: Language sovereignty and keeping the CPU fed.*

* ✅ **Custom BPE Tokenizer:** Byte-pair encoding tailored for German morphology.  
* ✅ **Dataset Extraction Bridge:** Python utility utilizing datasets streaming to pipe Hugging Face datasets into flat .txt.  
* ✅ **Parallelized BPE Trainer:** Built a multi-threaded Rust pool with statistical sampling for memory-safe BPE training.  
* ✅ **High-Throughput I/O Dataloader:** BinaryDataLoader uses massive sequential buffered chunking to sustain maximum GPU utilization.  
* ✅ **EisenBoard (Telemetry):** Built a zero-dependency, raw std::net::TcpListener background thread serving a real-time dark-mode web dashboard (HTML5 Canvas). It shares loss and TPS state via Arc\<RwLock\> without blocking the GPU.  
* ✅ **Milestone 5:** The engine is a complete research suite, fully abstracted from data bottlenecks with real-time browser monitoring.

## **Phase 6: Hyper-Optimization & Engine Tuning 🚀**

*Focus: Pushing the limits of VRAM, custom kernels, and training stability.*

* \[x\] **Cosine Learning Rate Decay:** Implement a dynamic learning rate scheduler to smoothly converge the model to its minimum.  
* \[x\] **Fused AdamW (GPU):** Write a custom CUDA kernel for the optimizer to keep weights and gradients strictly in VRAM, eliminating the PCIe bus bottleneck.  
* \[x\] **MatMul Tiling:** Utilize CUDA shared memory (SRAM) for matrix multiplication to bypass global memory bandwidth limits.  
* \[x\] **Gradient Accumulation:** Simulate massive batch sizes (e.g., effective batch 64\) without increasing VRAM footprint to stabilize optimization noise.  
* \[x\] **Native BF16 Mixed Precision:** Leverage 4th-gen Tensor Cores (RTX 4070\) for 2x memory reduction and massive TFLOPS acceleration without the need for loss scaling.  
* \[x\] **Flash Attention:** Implement a custom Triton-style fused kernel for Attention to bypass the $O(N^2)$ memory bottleneck.  
* \[x\] **Milestone 6:** A hyper-optimized, stable engine training a 14M+ parameter model at peak GPU saturation, ready for full convergence.

## **Phase 7: The Hugging Face Bridge 🚧**

*Focus: Ecosystem interoperability.*

* \[ \] **Weight Exporter:** Write a .safetensors binary writer to export Eisen parameters.  
* \[ \] **Config Generator:** Script to output config.json compatible with Llama-style architectures.  
* \[ \] **Milestone 7:** Load the custom Rust model natively in Python transformers for inference.

Quick smoke workflow:
1. `cargo run --example train_tiny_hf_smoke`
2. `python scripts/validate_hf_export.py --export-dir data/hf_export_tiny_smoke`

## **Phase 8: Stability measures**

*Focus: Training stability*

* \[ \] **Improved EisenBoard:** Make the Eisenboard a dedicated crate. Add more metrics and information to EisenBoard.
* \[ \] **Gradient Clipping:** To protect against occasional exploding gradients
* \[ \] **Determinism/Reproducibility:** Add reproducibility controls and run manifest logging (rng seed, determinism, run manifests)
* \[ \] **Fuse OPs:** Fuse attention scale and masking operations to remove per-step large tensor allocations
* \[ \] **Improved Checkpointing:** Expand checkpoints to include optimizer, and training progress state

## **Phase 9: Advanced Inference & Fine-Tuning 🔮**

*Focus: Post-training capabilities.*

* \[ \] **KV Caching:** Fast inference kernel for autoregressive generation.  
* \[ \] **LoRA (Low-Rank Adaptation):** Implement $A$ and $B$ adapter matrices for parameter-efficient fine-tuning on our own tape.
