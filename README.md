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

* ✅ **Custom BPE Tokenizer:** Byte-pair encoding tailored for German morphology. Replaces naive word splits and enables sub-word semantics.  
* ✅ **Dataset Extraction Bridge:** Python utility utilizing datasets streaming to pipe ungated Hugging Face datasets (e.g., German Wikipedia) into flat .txt files.  
* ✅ **BPE Sampling & Progress Tracking:** Implemented statistical chunk sampling for memory-safe BPE training on massive corpora, alongside custom zero-dependency ANSI progress bars.  
* ✅ **Parallelized Data Preprocessor:** Built a Producer-Consumer thread pool in Rust to encode massive 100GB+ text files into binary formats at maximum CPU utilization, complete with instant byte-offset state recovery.  
* ✅ **High-Throughput I/O Dataloader:** Upgraded the BinaryDataLoader to use massive sequential buffered chunking. Eliminated backward disk seeks, bypassing OS page-cache thrashing and perfectly sustaining maximum GPU utilization.  
* ✅ **Milestone 5:** The engine is completely abstracted from data loading bottlenecks, capable of saturating the GPU by streaming massive German datasets with zero runtime tokenization overhead.

## **Phase 6: The Hugging Face Bridge 🚧**

*Focus: Ecosystem interoperability.*

* \[ \] **Weight Exporter:** Write a .safetensors binary writer to export Eisen parameters.  
* \[ \] **Config Generator:** Script to output config.json compatible with Llama-style architectures.  
* \[ \] **Milestone 6:** Load the custom Rust model natively in Python transformers for inference.

## **Phase 7: Hyper-Optimization & Research Features 🚀**

*Focus: Pushing the limits of 6GB VRAM and custom kernels.*

* \[ \] **Flash Attention:** Implement a custom Triton-style fused kernel for Attention to bypass the $O(N^2)$ memory bottleneck.  
* \[ \] **KV Caching:** Fast inference kernel that only computes the new token's attention.  
* \[ \] **LoRA (Low-Rank Adaptation):** Implement $A$ and $B$ adapter matrices for parameter-efficient fine-tuning on our own tape.
