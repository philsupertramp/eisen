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

## **Phase 3: Hardware Acceleration (The GPU Leap) 🚧**

*Focus: Moving from RAM to VRAM and writing blazing-fast compute kernels via cudarc.*

* ✅ **The Backend Abstraction:** Refactored Tensor to support Device::Cpu and Device::Gpu(Arc\<CudaContext\>, Arc\<CudaStream\>).  
* ✅ **PTX Pipeline:** Automated .cu to .ptx compilation via build.rs and module loading in Graph.  
* 🚧 **Custom CUDA Kernels:**  
  * ✅ add\_f32: Forward element-wise addition (Broadcast-aware).  
  * ✅ fill\_f32: Buffer initialization (seed gradients).  
  * ✅ accumulate\_f32: Gradient accumulation (a \+= b with atomicAdd).  
  * ✅ mul\_f32: Element-wise multiplication (Forward/Backward).  
  * ✅ matmul\_f32: The core GEMM kernel (Naive 2D implementation).  
  * ✅ silu\_f32 / silu\_backward\_f32: Fused activation pass.  
  * \[ \] gather\_f32: Row-based indexing for Embeddings.  
  * \[ \] rms\_norm\_f32: Fused normalization kernel.  
  * \[ \] cross\_entropy\_f32: Fused loss and softmax kernel.  
  * \[ \] reductions: Sum and Max kernels for dimension reduction.  
* \[ \] **VRAM Allocator:** Implement a memory pool in the Graph to strictly manage the 6-8GB VRAM limit and avoid allocation overhead.  
* \[ \] **Milestone 3:** Run the XOR and Bengio LM tests entirely on the GPU. Assert bit-perfect gradient matches with massive speedups.

## **Phase 4: The Transformer Architecture & VRAM Strategy**

*Focus: Scaled dot-product attention and strict memory survival.*

* **Positional Encodings:** Implement RoPE (Rotary Positional Embeddings).  
* **Multi-Head Attention:** QKV projections, causal masking, and optimized attention.  
* **Gradient Checkpointing:** Intentional activation dropping and recomputation to survive 6-8GB VRAM limits.  
* **Milestone 4:** Full Transformer block forward/backward pass on GPU without OOM panics.

## **Phase 5: German Optimization & Data Pipeline**

*Focus: Language sovereignty and keeping the CPU fed.*

* **Custom BPE Tokenizer:** Byte-pair encoding tailored for German compound words.  
* **hf-mount Integration:** Use Hugging Face's mounting tool to stream large-scale German datasets (OSCAR, Wikipedia, Gutenberg) directly into the training loop, bypassing local storage bottlenecks.  
* **Streaming Dataloader:** Multi-threaded Rust dataloader to read from the hf-mount virtual filesystem and prep batches in CPU RAM before DMA transfer to VRAM.  
* **Milestone 5:** Train a prototype model (10M-50M parameters) until it generates coherent German phrases.

## **Phase 6: The Hugging Face Bridge**

*Focus: Ecosystem interoperability.*

* **Parameter Naming:** Map Tensor names to standard transformers keys (Llama/Mistral).  
* **Binary Exporter:** Write .safetensors binary format.  
* **Milestone 6:** Load the custom Rust model natively in Python transformers for inference.
