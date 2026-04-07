use crate::tape::{Tape, TapeNode};
use crate::tensor::{Device, Storage, Tensor};
#[cfg(feature = "bf16")]
use cudarc::driver::CudaSlice;
use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use std::collections::{HashMap, HashSet};

// ── StreamingReport ────────────────────────────────────────────────────────────

/// Returned by `Graph::plan_streaming`. Describes what ended up where.
pub struct StreamingReport {
    /// Param tensor IDs kept in VRAM (data + grad + adam moments all GPU).
    pub resident_param_ids: Vec<usize>,
    /// Param tensor IDs moved to CPU RAM (data + grad + adam moments all CPU).
    pub streamed_param_ids: Vec<usize>,
    /// Total bytes of resident params (×4: data+grad+m+v per param).
    pub resident_bytes: usize,
    /// Total bytes of streamed param data (just the weight bytes; moments
    /// are additional ×3 in CPU RAM).
    pub streamed_bytes: usize,
    /// Peak VRAM consumed by a single streaming temp buffer during a
    /// forward or backward pass (= size of the largest streamed param).
    pub streaming_headroom_bytes: usize,
}

impl std::fmt::Display for StreamingReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn gb(b: usize) -> f64 {
            b as f64 / 1024f64.powi(3)
        }
        fn mb(b: usize) -> f64 {
            b as f64 / 1024f64.powi(2)
        }
        writeln!(f, "=== Streaming Layout ===")?;
        writeln!(
            f,
            "  VRAM-resident params:  {} tensors  ({:.2} GB × 4 = {:.2} GB VRAM)",
            self.resident_param_ids.len(),
            gb(self.resident_bytes / 4),
            gb(self.resident_bytes)
        )?;
        writeln!(
            f,
            "  CPU-streamed params:   {} tensors  ({:.2} GB weights + {:.2} GB moments)",
            self.streamed_param_ids.len(),
            gb(self.streamed_bytes),
            gb(self.streamed_bytes * 3)
        )?;
        writeln!(
            f,
            "  Peak streaming temp:   {:.0} MB  (one block at a time)",
            mb(self.streaming_headroom_bytes)
        )?;
        write!(f, "========================")
    }
}

pub struct Graph {
    pub tensors: Vec<Tensor>,
    pub tape: Tape,
    pub device: Device,
    pub functions: HashMap<String, CudaFunction>,

    pub vram_pool: HashMap<usize, Vec<Storage>>,
    pub num_params: usize,
    pub no_grad: bool,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new(Device::Cpu)
    }
}

impl Graph {
    pub fn new(device: Device) -> Self {
        let mut functions = HashMap::new();

        if let Device::Gpu(ctx, _) = &device {
            let ptx = include_str!(concat!(env!("OUT_DIR"), "/ops.ptx"));
            let module = ctx
                .load_module(ptx.into())
                .expect("Failed to load PTX module");

            let mut names = vec![
                "add_f32",
                "fill_f32",
                "accumulate_f32",
                "mul_f32",
                "mul_backward_f32",
                "matmul_f32",
                "matmul_backward_a_f32",
                "matmul_backward_b_f32",
                "silu_f32",
                "silu_backward_f32",
                "gather_f32",
                "gather_backward_f32",
                "rmsnorm_f32",
                "rmsnorm_backward_f32",
                "copy_f32",
                "cross_entropy_f32",
                "cross_entropy_backward_f32",
                "sum_f32",
                "sum_backward_f32",
                "max_f32",
                "max_backward_f32",
                "bmm_f32",
                "bmm_backward_a_f32",
                "bmm_backward_b_f32",
                "bmm_backward_a_transb_f32",
                "bmm_backward_b_transb_f32",
                "softmax_f32",
                "softmax_backward_f32",
                "flash_attention_f32",
                "transpose_0213_f32",
                "transpose_0213_backward_f32",
                "rope_f32",
                "rope_backward_f32",
                "adamw_step_f32",
            ];

            // BF16 specific kernels
            #[cfg(feature = "bf16")]
            names.extend_from_slice(&[
                "cast_f32_to_bf16",
                "cast_bf16_to_f32",
                "cast_bf16_to_f32_accumulate",
                "matmul_bf16_f32",
                "matmul_f32_bf16accum_f32",
                "bmm_f32_bf16accum_f32",
            ]);

            for name in names {
                let f = module
                    .load_function(name)
                    .expect(&format!("Failed to load {} kernel", name));
                functions.insert(name.to_string(), f);
            }
        }

        Self {
            tensors: Vec::new(),
            tape: Tape::default(),
            device,
            functions,
            vram_pool: HashMap::new(),
            num_params: 0,
            no_grad: false,
        }
    }

    pub fn mark_params(&mut self) {
        self.num_params = self.tensors.len();
    }

    pub fn mark_save_point(&self) -> usize {
        self.tensors.len()
    }

    pub fn restore_save_point(&mut self, save_point: usize) {
        while self.tensors.len() > save_point {
            let t = self.tensors.pop().unwrap();

            // ARCHITECTURAL FIX: Only recycle tensors that were specifically allocated by internal ops.
            // If `is_pooled` is false (e.g. explicitly injected batches), it simply drops and `cudaFree`s natively!
            if t.is_pooled {
                let size = if t.shape.is_empty() {
                    1
                } else {
                    t.shape.iter().product()
                };
                self.vram_pool.entry(size).or_default().push(t.data);
                self.vram_pool.entry(size).or_default().push(t.grad);
            }
        }
    }

    // --- Helper Methods ---
    pub fn load_tensor_data(&mut self, id: usize, host_data: &[f32]) {
        let tensor = &mut self.tensors[id];
        let size = if tensor.shape.is_empty() {
            1
        } else {
            tensor.shape.iter().product::<usize>()
        };

        // Sanity check to prevent catastrophic memory corruption
        assert_eq!(
            size,
            host_data.len(),
            "Shape mismatch: Tensor {} expects {} elements, but got {}.",
            id,
            size,
            host_data.len()
        );

        match &mut tensor.data {
            // CPU fallback: fast memory copy
            Storage::Cpu(cpu_vec) => {
                cpu_vec.copy_from_slice(host_data);
            }
            // GPU path: push via PCIe bus to VRAM
            Storage::Gpu(gpu_slice) => {
                if let Device::Gpu(ctx, stream) = &self.device {
                    stream
                        .memcpy_htod(host_data, gpu_slice)
                        .expect("Failed to copy weights from Host RAM to VRAM!");
                } else {
                    panic!("Graph device mismatch: Tensor is GPU but Graph is not.");
                }
            }
            #[cfg(feature = "bf16")]
            Storage::GpuBf16(gpu_slice) => {
                if let Device::Gpu(ctx, stream) = &self.device {
                    // Convert f32 to bf16 by shifting off the lower 16 bits of the mantissa
                    let u16_data: Vec<u16> = host_data
                        .iter()
                        .map(|&f| (f.to_bits() >> 16) as u16)
                        .collect();
                    stream
                        .memcpy_htod(u16_data.as_slice(), gpu_slice)
                        .expect("Failed to copy BF16 weights from Host RAM to VRAM!");
                } else {
                    panic!("Graph device mismatch: Tensor is GPU but Graph is not.");
                }
            }
        }
    }

    pub fn clear_activations(&mut self) {
        self.tape.nodes.clear();
        self.restore_save_point(self.num_params);
    }

    pub fn alloc_pooled(&mut self, shape: Vec<usize>) -> usize {
        let size = if shape.is_empty() {
            1
        } else {
            shape.iter().product::<usize>()
        };
        let device = self.device.clone();

        let mut get_block = || -> Storage {
            if let Some(blocks) = self.vram_pool.get_mut(&size) {
                if let Some(block) = blocks.pop() {
                    return block;
                }
            }
            match &device {
                Device::Cpu => Storage::Cpu(vec![0.0; size]),
                Device::Gpu(_, stream) => Storage::Gpu(stream.alloc_zeros::<f32>(size).unwrap()),
            }
        };

        let data_storage = get_block();
        let mut grad_storage = get_block();

        match &device {
            Device::Cpu => {
                if let Storage::Cpu(v) = &mut grad_storage {
                    v.fill(0.0);
                }
            }
            Device::Gpu(_, stream) => {
                if let Storage::Gpu(s) = &mut grad_storage {
                    let f = self.functions.get("fill_f32").unwrap().clone();
                    let n = size as u64;
                    let val = 0.0f32;
                    let mut builder = stream.launch_builder(&f);
                    builder.arg(s).arg(&val).arg(&n);
                    unsafe { builder.launch(LaunchConfig::for_num_elems(size as u32)) }.unwrap();
                }
            }
        }

        let mut strides = vec![1; shape.len()];
        if !shape.is_empty() {
            for i in (1..shape.len()).rev() {
                strides[i - 1] = strides[i] * shape[i];
            }
        }

        let id = self.tensors.len();
        self.tensors.push(Tensor {
            id,
            shape: shape.clone(),
            strides,
            data: data_storage,
            grad: grad_storage,
            device,
            name: None,
            is_pooled: true, // This tensor belongs to the internal engine pool
        });
        id
    }

    pub fn alloc(&mut self, shape: Vec<usize>, data: Vec<f32>) -> usize {
        let id = self.tensors.len();
        // Tensor::new sets is_pooled = false automatically
        self.tensors
            .push(Tensor::new(id, shape, data, self.device.clone()));
        id
    }

    /// Allocate a tensor that permanently lives in CPU RAM, even when the
    /// Graph device is GPU. Use this for weights that should be streamed
    /// rather than resident in VRAM.
    ///
    /// In practice you rarely call this directly — `plan_streaming` converts
    /// existing GPU-resident params to CPU storage automatically.
    pub fn alloc_cpu_homed(&mut self, shape: Vec<usize>, data: Vec<f32>) -> usize {
        let id = self.tensors.len();
        let size = if shape.is_empty() {
            1
        } else {
            shape.iter().product::<usize>()
        };
        let strides = Tensor::compute_strides(&shape);

        self.tensors.push(Tensor {
            id,
            shape,
            strides,
            data: Storage::Cpu(data),
            grad: Storage::Cpu(vec![0.0; size]),
            device: self.device.clone(), // graph device (GPU) — sync_to_cpu still works
            name: None,
            is_pooled: false,
        });
        id
    }

    /// Move an existing tensor's data+grad storage to CPU RAM.
    ///
    /// Useful when building very large models on GPU without first holding all
    /// parameters resident in VRAM. If the tensor is already CPU-homed this is
    /// a no-op.
    pub fn demote_tensor_to_cpu(&mut self, tensor_id: usize) {
        let size = self.tensors[tensor_id].size();
        let stream = match &self.device {
            Device::Gpu(_, s) => Some(s.clone()),
            Device::Cpu => None,
        };

        let cpu_data = match &self.tensors[tensor_id].data {
            Storage::Cpu(v) => v.clone(),
            Storage::Gpu(s) => {
                let st = stream.as_ref().expect("GPU storage without GPU stream");
                st.clone_dtoh(s)
                    .expect("demote_tensor_to_cpu: dtoh data failed")
            }
            #[cfg(feature = "bf16")]
            Storage::GpuBf16(_) => {
                panic!("demote_tensor_to_cpu: BF16 tensor demotion not supported")
            }
        };

        self.tensors[tensor_id].data = Storage::Cpu(cpu_data);
        self.tensors[tensor_id].grad = Storage::Cpu(vec![0.0; size]);

        // Ensure dtoh and subsequent drops are fully retired so VRAM is
        // actually reclaimed during large model initialization.
        if let Some(st) = stream {
            st.synchronize()
                .expect("demote_tensor_to_cpu: stream sync failed");
        }
    }

    /// Decide which parameters stay in VRAM and which stream from CPU RAM,
    /// then convert the overflow params in place.
    ///
    /// # Arguments
    /// * `vram_budget_bytes`      — total VRAM budget (e.g. `7 * 1024_usize.pow(3)`)
    /// * `activation_reserve_bytes` — VRAM to reserve for activations and
    ///                               checkpointing overhead (400–600 MB is safe)
    /// * `pinned_ids`             — param IDs that MUST stay in VRAM regardless of budget
    ///                             (typically: embedding weights, lm_head, final norm)
    ///
    /// # Returns
    /// A `StreamingReport` describing the final layout. Print it to verify.
    ///
    /// # Panics
    /// Panics if called on a CPU graph (streaming is GPU-only).
    pub fn plan_streaming(
        &mut self,
        vram_budget_bytes: usize,
        activation_reserve_bytes: usize,
        pinned_ids: &[usize],
    ) -> StreamingReport {
        assert!(
            matches!(&self.device, Device::Gpu(_, _)),
            "plan_streaming requires a GPU graph"
        );
        assert!(
            self.num_params > 0,
            "call mark_params() before plan_streaming()"
        );

        let pinned_set: HashSet<usize> = pinned_ids.iter().cloned().collect();

        // ── Phase 1: VRAM consumed by pinned params (data + grad + m + v = ×4) ──
        let pinned_vram: usize = pinned_ids
            .iter()
            .map(|&pid| self.tensors[pid].size() * 4 * 4)
            .sum();

        let budget_after_fixed = vram_budget_bytes
            .saturating_sub(activation_reserve_bytes)
            .saturating_sub(pinned_vram);

        // ── Phase 2: greedily assign non-pinned params ─────────────────────────
        // We also need to reserve headroom for the largest single streamed
        // param (the peak temp buffer). We compute this after the first pass
        // and check that the headroom fits — if not, bump one more param to CPU.

        let candidate_ids: Vec<usize> = (0..self.num_params)
            .filter(|id| !pinned_set.contains(id))
            .collect();

        let mut resident: Vec<usize> = Vec::new();
        let mut streamed: Vec<usize> = Vec::new();
        let mut used = 0usize;

        for pid in &candidate_ids {
            let cost = self.tensors[*pid].size() * 4 * 4; // ×4 buffers ×4 bytes
            if used + cost <= budget_after_fixed {
                used += cost;
                resident.push(*pid);
            } else {
                streamed.push(*pid);
            }
        }

        // Streaming headroom = size of largest single streamed param.
        // This must fit in the remaining VRAM alongside the resident params.
        let max_stream_bytes = streamed
            .iter()
            .map(|&pid| self.tensors[pid].size() * 4)
            .max()
            .unwrap_or(0);

        // If the headroom doesn't fit, demote the largest resident param.
        // This keeps the invariant: at any moment, only 1 streaming temp
        // is alive in VRAM (because we sync-free before returning from matmul).
        while max_stream_bytes > budget_after_fixed.saturating_sub(used) && !resident.is_empty() {
            // Demote the largest resident param to streaming
            let largest_idx = resident
                .iter()
                .enumerate()
                .max_by_key(|&(_, &pid)| self.tensors[pid].size())
                .map(|(i, _)| i)
                .unwrap();
            let pid = resident.remove(largest_idx);
            used -= self.tensors[pid].size() * 4 * 4;
            streamed.push(pid);
        }

        // ── Phase 3: dtoh + convert streamed params to Storage::Cpu ────────────
        let stream = match &self.device {
            Device::Gpu(_, s) => s.clone(),
            Device::Cpu => unreachable!(),
        };

        let mut streamed_bytes = 0usize;
        for &pid in &streamed {
            let size = self.tensors[pid].size();
            streamed_bytes += size * 4;

            let cpu_data = match &self.tensors[pid].data {
                Storage::Gpu(s) => stream
                    .clone_dtoh(s)
                    .expect("plan_streaming: dtoh weight failed"),
                Storage::Cpu(_) => continue, // already CPU — nothing to do
                #[cfg(feature = "bf16")]
                Storage::GpuBf16(_) => panic!("plan_streaming: BF16 streaming not yet supported"),
            };

            // Replacing data/grad drops the old CudaSlices → cudaFree
            self.tensors[pid].data = Storage::Cpu(cpu_data);
            self.tensors[pid].grad = Storage::Cpu(vec![0.0; size]);
        }

        // ── Report ──────────────────────────────────────────────────────────────
        let resident_bytes: usize = pinned_ids
            .iter()
            .chain(resident.iter())
            .map(|&pid| self.tensors[pid].size() * 4 * 4)
            .sum();

        StreamingReport {
            resident_param_ids: pinned_ids.iter().cloned().chain(resident).collect(),
            streamed_param_ids: streamed,
            resident_bytes,
            streamed_bytes,
            streaming_headroom_bytes: max_stream_bytes,
        }
    }

    /// BF16 mixed-precision matrix multiplication for CPU-homed (streamed) weights.
    ///
    /// ## Forward VRAM profile
    ///
    /// ```
    ///   a_fp32 (GPU, resident)
    ///   b_cpu  (CPU RAM, home)
    ///   │
    ///   ├─ htod ──────────────► b_fp32_temp  (GPU temp, 1 block)
    ///   │
    ///   matmul_f32_bf16accum_f32(a_fp32, b_fp32_temp) ──► out_fp32
    ///   │
    ///   stream.synchronize()
    ///   drop(b_fp32_temp)   ← temp freed before next streamed matmul
    ///   │
    ///   peak ≈ a_size + b_size bytes (plus output/activations)
    /// ```
    ///
    /// For a single transformer block (hidden=1536, ffn=4096):
    ///   Largest streamed weight tile is 1536×4096 = 24MB FP32 temp.
    ///   No additional full BF16 staging buffers are required.
    ///
    /// ## Backward VRAM profile
    ///
    /// Backward uses FP32 kernels on the CPU master weights (re-htod'd),
    /// giving full-precision gradients to AdamW — identical to matmul_streamed.
    /// BF16 quantization is forward-only and happens per multiply in-kernel.
    ///
    /// ```
    ///   htod b_cpu ──► b_fp32_bwd  (GPU temp)
    ///   matmul_backward_a_f32(grad_out, b_fp32_bwd) → grad_a  (GPU, accumulate)
    ///   matmul_backward_b_f32(a_fp32, grad_out)     → grad_b_temp  (GPU temp)
    ///   sync → dtoh grad_b_temp → accumulate into b_cpu_grad (CPU Vec)
    ///   drop(b_fp32_bwd, grad_b_temp)
    /// ```
    #[cfg(feature = "bf16")]
    fn matmul_bf16_streamed(&mut self, a_id: usize, b_id: usize) -> usize {
        use cudarc::driver::CudaSlice;

        let a_shape = self.tensors[a_id].shape.clone();
        let b_shape = self.tensors[b_id].shape.clone();
        assert!(
            a_shape.len() >= 2,
            "matmul_bf16_streamed: lhs must have rank >= 2"
        );
        assert_eq!(
            b_shape.len(),
            2,
            "matmul_bf16_streamed: rhs must have rank 2 [k, n]"
        );
        let k = *a_shape.last().unwrap();
        let m = a_shape[..a_shape.len() - 1].iter().product::<usize>();
        let n = b_shape[1];
        assert_eq!(
            b_shape[0], k,
            "matmul_bf16_streamed: lhs last dim must equal rhs first dim"
        );

        let stream = match &self.device {
            Device::Gpu(_, s) => s.clone(),
            Device::Cpu => unreachable!("matmul_bf16_streamed called on CPU graph"),
        };

        // Kernels
        let f_matmul_bf16 = self
            .functions
            .get("matmul_f32_bf16accum_f32")
            .unwrap()
            .clone();
        let f_bwd_a = self.functions.get("matmul_backward_a_f32").unwrap().clone();
        let f_bwd_b = self.functions.get("matmul_backward_b_f32").unwrap().clone();
        let stream_bwd = stream.clone();

        // ── Forward ────────────────────────────────────────────────────────────

        // 1. htod b (CPU f32) → b_fp32_temp (GPU f32)
        let b_cpu_data = self.tensors[b_id].data.as_cpu().clone();
        let b_fp32_temp: CudaSlice<f32> = stream
            .clone_htod(b_cpu_data.as_slice())
            .expect("bf16_streamed: forward htod failed");

        // 2. Allocate output first (mutable borrow of self ends here).
        let out_id = self.alloc_pooled(vec![m, n]);

        // 3. Borrow tensor storages for launch arguments.
        let a_fp32 = match &self.tensors[a_id].data {
            Storage::Gpu(s) => s,
            _ => unreachable!("matmul_bf16_streamed: a must be GPU storage"),
        };
        let o_fp32 = match &self.tensors[out_id].data {
            Storage::Gpu(s) => s,
            _ => unreachable!(),
        };
        let m_u64 = m as u64;
        let k_u64 = k as u64;
        let n_u64 = n as u64;
        let cfg_fwd = LaunchConfig {
            grid_dim: ((n as u32 + 15) / 16, (m as u32 + 15) / 16, 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&f_matmul_bf16);
        builder
            .arg(a_fp32)
            .arg(&b_fp32_temp)
            .arg(o_fp32)
            .arg(&m_u64)
            .arg(&k_u64)
            .arg(&n_u64);
        unsafe { builder.launch(cfg_fwd) }.unwrap();

        // 4. Sync then free the streamed weight temp.
        stream
            .synchronize()
            .expect("bf16_streamed: forward sync failed");
        drop(b_fp32_temp);

        // ── Backward closure ───────────────────────────────────────────────────
        //
        // The BF16 cast is forward-only. Backward uses full FP32 master weights
        // (re-htod'd from CPU), giving accurate gradients to AdamW.
        // This is the same pattern as matmul_streamed's backward.
        let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
            // Re-htod the CPU master weight for backward kernels
            let b_cpu_bwd = tensors[b_id].data.as_cpu();
            let b_fp32_bwd: CudaSlice<f32> = stream_bwd
                .clone_htod(b_cpu_bwd)
                .expect("bf16_streamed: backward htod failed");

            let out_grad = match &tensors[out_id].grad {
                Storage::Gpu(s) => s,
                _ => unreachable!(),
            };
            let a_grad = match &tensors[a_id].grad {
                Storage::Gpu(s) => s,
                _ => unreachable!(),
            };
            let a_data = match &tensors[a_id].data {
                Storage::Gpu(s) => s,
                _ => unreachable!(),
            };

            // grad_a = grad_out @ b^T  (FP32, accumulates in-place into GPU grad)
            let cfg_a = LaunchConfig {
                grid_dim: ((k as u32 + 15) / 16, (m as u32 + 15) / 16, 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };
            let mut b1 = stream_bwd.launch_builder(&f_bwd_a);
            b1.arg(out_grad)
                .arg(&b_fp32_bwd)
                .arg(a_grad)
                .arg(&m_u64)
                .arg(&k_u64)
                .arg(&n_u64);
            unsafe { b1.launch(cfg_a) }.unwrap();

            // grad_b = a^T @ grad_out  (FP32, into a fresh GPU temp)
            let grad_b_temp: CudaSlice<f32> = stream_bwd
                .alloc_zeros::<f32>(k * n)
                .expect("bf16_streamed: grad_b_temp alloc failed");
            let cfg_b = LaunchConfig {
                grid_dim: ((n as u32 + 15) / 16, (k as u32 + 15) / 16, 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };
            let mut b2 = stream_bwd.launch_builder(&f_bwd_b);
            b2.arg(a_data)
                .arg(out_grad)
                .arg(&grad_b_temp)
                .arg(&m_u64)
                .arg(&k_u64)
                .arg(&n_u64);
            unsafe { b2.launch(cfg_b) }.unwrap();

            // Sync before dtoh: both backward kernels must be complete
            stream_bwd
                .synchronize()
                .expect("bf16_streamed: backward sync failed");

            let grad_b_cpu = stream_bwd
                .clone_dtoh(&grad_b_temp)
                .expect("bf16_streamed: grad dtoh failed");

            // Free GPU temporaries as early as possible
            drop(b_fp32_bwd);
            drop(grad_b_temp);

            // Accumulate into CPU grad buffer.
            // Correct under gradient accumulation: zero_grad resets this between
            // optimizer steps; multiple micro-batch backward passes add up here.
            let b_grad = tensors[b_id].grad.as_cpu_mut();
            for (acc, delta) in b_grad.iter_mut().zip(grad_b_cpu.iter()) {
                *acc += delta;
            }
        });

        if !self.no_grad {
            self.tape.nodes.push(TapeNode {
                inputs: vec![a_id, b_id],
                output: out_id,
                backward_fn,
            });
        }

        out_id
    }

    // --- Helper Methods ---
    pub fn get_data(&self, tensor_id: usize) -> &Vec<f32> {
        self.tensors[tensor_id].data.as_cpu()
    }
    pub fn get_grad(&self, tensor_id: usize) -> &Vec<f32> {
        self.tensors[tensor_id].grad.as_cpu()
    }
    pub fn sync_grad_to_cpu(&self, tensor_id: usize) -> Vec<f32> {
        match &self.tensors[tensor_id].grad {
            Storage::Cpu(v) => v.clone(),
            Storage::Gpu(s) => {
                let (_, stream) = match &self.device {
                    Device::Gpu(_, s) => (None::<f32>, s),
                    _ => unreachable!(),
                };
                stream.clone_dtoh(s).unwrap()
            }
            #[cfg(feature = "bf16")]
            Storage::GpuBf16(s) => {
                let (_, stream) = match &self.device {
                    Device::Gpu(_, s) => (None::<f32>, s),
                    _ => unreachable!(),
                };
                let u16_data = stream.clone_dtoh(s).unwrap();
                u16_data
                    .into_iter()
                    .map(|b| f32::from_bits((b as u32) << 16))
                    .collect()
            }
        }
    }

    // --- Core Math Ops ---
    pub fn add(&mut self, a_id: usize, b_id: usize) -> usize {
        let a_shape = self.tensors[a_id].shape.clone();
        let b_shape = self.tensors[b_id].shape.clone();
        let out_shape = Tensor::get_broadcasted_shape(&a_shape, &b_shape);
        let out_size = out_shape.iter().product::<usize>();
        let device = self.device.clone();

        match &device {
            Device::Gpu(_, stream) => {
                let a_strides = Tensor::get_broadcasted_strides(
                    &a_shape,
                    &self.tensors[a_id].strides,
                    &out_shape,
                );
                let b_strides = Tensor::get_broadcasted_strides(
                    &b_shape,
                    &self.tensors[b_id].strides,
                    &out_shape,
                );
                let rank = out_shape.len() as u64;

                let mut s = [1u64; 3];
                let mut a_str = [0u64; 3];
                let mut b_str = [0u64; 3];
                for i in 0..out_shape.len() {
                    s[i] = out_shape[i] as u64;
                    a_str[i] = a_strides[i] as u64;
                    b_str[i] = b_strides[i] as u64;
                }

                let f_forward = self.functions.get("add_f32").unwrap().clone();
                let f_accumulate = self.functions.get("accumulate_f32").unwrap().clone();
                let stream_clone = stream.clone();

                let out_id = self.alloc_pooled(out_shape.clone());

                let a_s = match &self.tensors[a_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
                let b_s = match &self.tensors[b_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
                let o_s = match &self.tensors[out_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };

                let n = out_size as u64;
                let mut builder = stream.launch_builder(&f_forward);
                builder
                    .arg(a_s)
                    .arg(b_s)
                    .arg(o_s)
                    .arg(&n)
                    .arg(&rank)
                    .arg(&s[0])
                    .arg(&s[1])
                    .arg(&s[2])
                    .arg(&a_str[0])
                    .arg(&a_str[1])
                    .arg(&a_str[2])
                    .arg(&b_str[0])
                    .arg(&b_str[1])
                    .arg(&b_str[2]);
                unsafe { builder.launch(LaunchConfig::for_num_elems(out_size as u32)) }.unwrap();

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = match &tensors[out_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let a_grad = match &tensors[a_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let mut b1 = stream_clone.launch_builder(&f_accumulate);
                    b1.arg(a_grad)
                        .arg(out_grad)
                        .arg(&n)
                        .arg(&rank)
                        .arg(&s[0])
                        .arg(&s[1])
                        .arg(&s[2])
                        .arg(&a_str[0])
                        .arg(&a_str[1])
                        .arg(&a_str[2]);
                    unsafe { b1.launch(LaunchConfig::for_num_elems(out_size as u32)) }.unwrap();

                    let b_grad = match &tensors[b_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let mut b2 = stream_clone.launch_builder(&f_accumulate);
                    b2.arg(b_grad)
                        .arg(out_grad)
                        .arg(&n)
                        .arg(&rank)
                        .arg(&s[0])
                        .arg(&s[1])
                        .arg(&s[2])
                        .arg(&b_str[0])
                        .arg(&b_str[1])
                        .arg(&b_str[2]);
                    unsafe { b2.launch(LaunchConfig::for_num_elems(out_size as u32)) }.unwrap();
                });

                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id, b_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
            Device::Cpu => {
                let a_strides = Tensor::get_broadcasted_strides(
                    &a_shape,
                    &self.tensors[a_id].strides,
                    &out_shape,
                );
                let b_strides = Tensor::get_broadcasted_strides(
                    &b_shape,
                    &self.tensors[b_id].strides,
                    &out_shape,
                );
                let mut out_data = vec![0.0; out_size];
                let a_data = self.tensors[a_id].data.as_cpu();
                let b_data = self.tensors[b_id].data.as_cpu();
                for i in 0..out_size {
                    let nd = Tensor::flat_to_nd(i, &out_shape);
                    out_data[i] = a_data[Tensor::nd_to_flat(&nd, &a_strides)]
                        + b_data[Tensor::nd_to_flat(&nd, &b_strides)];
                }
                let out_id = self.alloc(out_shape.clone(), out_data);
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = tensors[out_id].grad.as_cpu().clone();
                    for i in 0..out_size {
                        let nd = Tensor::flat_to_nd(i, &out_shape);
                        tensors[a_id].grad.as_cpu_mut()[Tensor::nd_to_flat(&nd, &a_strides)] +=
                            out_grad[i];
                        tensors[b_id].grad.as_cpu_mut()[Tensor::nd_to_flat(&nd, &b_strides)] +=
                            out_grad[i];
                    }
                });
                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id, b_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
        }
    }

    pub fn mul(&mut self, a_id: usize, b_id: usize) -> usize {
        let a_shape = self.tensors[a_id].shape.clone();
        let b_shape = self.tensors[b_id].shape.clone();
        let out_shape = Tensor::get_broadcasted_shape(&a_shape, &b_shape);
        let out_size: usize = out_shape.iter().product();
        let device = self.device.clone();

        match &device {
            Device::Gpu(_, stream) => {
                let f_fwd = self.functions.get("mul_f32").unwrap().clone();
                let f_bwd = self.functions.get("mul_backward_f32").unwrap().clone();
                let stream_clone = stream.clone();

                let out_id = self.alloc_pooled(out_shape.clone());

                let a_s = match &self.tensors[a_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
                let b_s = match &self.tensors[b_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
                let o_s = match &self.tensors[out_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };

                let n = out_size as u64;
                let mut builder = stream.launch_builder(&f_fwd);
                builder.arg(a_s).arg(b_s).arg(o_s).arg(&n);
                unsafe { builder.launch(LaunchConfig::for_num_elems(out_size as u32)) }.unwrap();

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let a_data = match &tensors[a_id].data {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let b_data = match &tensors[b_id].data {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let out_grad = match &tensors[out_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let a_grad = match &tensors[a_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let b_grad = match &tensors[b_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };

                    let mut b1 = stream_clone.launch_builder(&f_bwd);
                    b1.arg(a_data)
                        .arg(b_data)
                        .arg(out_grad)
                        .arg(a_grad)
                        .arg(b_grad)
                        .arg(&n);
                    unsafe { b1.launch(LaunchConfig::for_num_elems(out_size as u32)) }.unwrap();
                });

                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id, b_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
            Device::Cpu => {
                let a_strides = Tensor::get_broadcasted_strides(
                    &a_shape,
                    &self.tensors[a_id].strides,
                    &out_shape,
                );
                let b_strides = Tensor::get_broadcasted_strides(
                    &b_shape,
                    &self.tensors[b_id].strides,
                    &out_shape,
                );
                let mut out_data = vec![0.0; out_size];
                let a_fwd = self.tensors[a_id].data.as_cpu().clone();
                let b_fwd = self.tensors[b_id].data.as_cpu().clone();
                for i in 0..out_size {
                    let nd = Tensor::flat_to_nd(i, &out_shape);
                    out_data[i] = a_fwd[Tensor::nd_to_flat(&nd, &a_strides)]
                        * b_fwd[Tensor::nd_to_flat(&nd, &b_strides)];
                }
                let out_id = self.alloc(out_shape.clone(), out_data);
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = tensors[out_id].grad.as_cpu().clone();
                    for i in 0..out_size {
                        let nd = Tensor::flat_to_nd(i, &out_shape);
                        tensors[a_id].grad.as_cpu_mut()[Tensor::nd_to_flat(&nd, &a_strides)] +=
                            b_fwd[Tensor::nd_to_flat(&nd, &b_strides)] * out_grad[i];
                        tensors[b_id].grad.as_cpu_mut()[Tensor::nd_to_flat(&nd, &b_strides)] +=
                            a_fwd[Tensor::nd_to_flat(&nd, &a_strides)] * out_grad[i];
                    }
                });
                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id, b_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
        }
    }

    pub fn matmul(&mut self, a_id: usize, b_id: usize) -> usize {
        // Streaming dispatch: if b is CPU-homed (weight streaming), use the
        // streaming path which htods b on demand and frees it immediately.
        let b_is_cpu = matches!(&self.tensors[b_id].data, Storage::Cpu(_));
        let a_is_gpu = matches!(&self.tensors[a_id].data, Storage::Gpu(_));
        if a_is_gpu && b_is_cpu {
            return self.matmul_streamed(a_id, b_id);
        }

        let a_shape = self.tensors[a_id].shape.clone();
        let b_shape = self.tensors[b_id].shape.clone();
        assert!(a_shape.len() >= 2, "matmul: lhs must have rank >= 2");
        assert_eq!(b_shape.len(), 2, "matmul: rhs must have rank 2 [k, n]");
        let k = *a_shape.last().unwrap();
        let m = a_shape[..a_shape.len() - 1].iter().product::<usize>();
        let n = b_shape[1];
        assert_eq!(
            b_shape[0], k,
            "matmul: lhs last dim must equal rhs first dim"
        );
        let device = self.device.clone();

        match &device {
            Device::Gpu(_, stream) => {
                let f_fwd = self.functions.get("matmul_f32").unwrap().clone();
                let f_bwd_a = self.functions.get("matmul_backward_a_f32").unwrap().clone();
                let f_bwd_b = self.functions.get("matmul_backward_b_f32").unwrap().clone();
                let stream_clone = stream.clone();

                let out_id = self.alloc_pooled(vec![m, n]);

                let a_s = match &self.tensors[a_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
                let b_s = match &self.tensors[b_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
                let o_s = match &self.tensors[out_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };

                let m_u64 = m as u64;
                let k_u64 = k as u64;
                let n_u64 = n as u64;

                let mut builder = stream.launch_builder(&f_fwd);
                builder
                    .arg(a_s)
                    .arg(b_s)
                    .arg(o_s)
                    .arg(&m_u64)
                    .arg(&k_u64)
                    .arg(&n_u64);

                let cfg = LaunchConfig {
                    grid_dim: ((n as u32 + 15) / 16, (m as u32 + 15) / 16, 1),
                    block_dim: (16, 16, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { builder.launch(cfg) }.unwrap();

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = match &tensors[out_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let a_grad = match &tensors[a_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let b_grad = match &tensors[b_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let a_data = match &tensors[a_id].data {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let b_data = match &tensors[b_id].data {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };

                    let cfg_a = LaunchConfig {
                        grid_dim: ((k as u32 + 15) / 16, (m as u32 + 15) / 16, 1),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut b1 = stream_clone.launch_builder(&f_bwd_a);
                    b1.arg(out_grad)
                        .arg(b_data)
                        .arg(a_grad)
                        .arg(&m_u64)
                        .arg(&k_u64)
                        .arg(&n_u64);
                    unsafe { b1.launch(cfg_a) }.unwrap();

                    let cfg_b = LaunchConfig {
                        grid_dim: ((n as u32 + 15) / 16, (k as u32 + 15) / 16, 1),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut b2 = stream_clone.launch_builder(&f_bwd_b);
                    b2.arg(a_data)
                        .arg(out_grad)
                        .arg(b_grad)
                        .arg(&m_u64)
                        .arg(&k_u64)
                        .arg(&n_u64);
                    unsafe { b2.launch(cfg_b) }.unwrap();
                });

                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id, b_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
            Device::Cpu => {
                let mut out_data = vec![0.0; m * n];
                let a_fwd = self.tensors[a_id].data.as_cpu().clone();
                let b_fwd = self.tensors[b_id].data.as_cpu().clone();
                for r in 0..m {
                    for c in 0..n {
                        let mut sum = 0.0;
                        for i in 0..k {
                            sum += a_fwd[r * k + i] * b_fwd[i * n + c];
                        }
                        out_data[r * n + c] = sum;
                    }
                }
                let out_id = self.alloc(vec![m, n], out_data);
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = tensors[out_id].grad.as_cpu().clone();
                    let a_grad = tensors[a_id].grad.as_cpu_mut();
                    for r in 0..m {
                        for i in 0..k {
                            let mut sum = 0.0;
                            for c in 0..n {
                                sum += out_grad[r * n + c] * b_fwd[i * n + c];
                            }
                            a_grad[r * k + i] += sum;
                        }
                    }
                    let b_grad = tensors[b_id].grad.as_cpu_mut();
                    for i in 0..k {
                        for c in 0..n {
                            let mut sum = 0.0;
                            for r in 0..m {
                                sum += a_fwd[r * k + i] * out_grad[r * n + c];
                            }
                            b_grad[i * n + c] += sum;
                        }
                    }
                });
                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id, b_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
        }
    }

    /// Mixed-precision matrix multiplication.
    ///
    /// Forward:  A_fp32 → BF16  |  B_fp32 → BF16  |  BF16 × BF16 → FP32 out
    /// Backward: standard FP32 matmul gradients (no loss scaling required for BF16)
    ///
    /// Master weights (B) always stay FP32, so the AdamW optimizer and the
    /// rest of the graph are completely unmodified.
    ///
    /// BF16 quantization happens on-the-fly in the forward kernel, so
    /// no full-size BF16 staging tensors are allocated in VRAM.
    #[cfg(feature = "bf16")]
    pub fn matmul_bf16(&mut self, a_id: usize, b_id: usize) -> usize {
        // Streaming dispatch: if b is CPU-homed (weight streaming), use the
        // streaming path which htods b on demand and frees it immediately.
        let b_is_cpu = matches!(&self.tensors[b_id].data, Storage::Cpu(_));
        let a_is_gpu = matches!(&self.tensors[a_id].data, Storage::Gpu(_));
        if a_is_gpu && b_is_cpu {
            return self.matmul_bf16_streamed(a_id, b_id);
        }
        let a_shape = self.tensors[a_id].shape.clone();
        let b_shape = self.tensors[b_id].shape.clone();
        assert!(a_shape.len() >= 2, "matmul_bf16: lhs must have rank >= 2");
        assert_eq!(b_shape.len(), 2, "matmul_bf16: rhs must have rank 2 [k, n]");
        let k = *a_shape.last().unwrap();
        let m = a_shape[..a_shape.len() - 1].iter().product::<usize>();
        let n = b_shape[1];
        assert_eq!(
            b_shape[0], k,
            "matmul_bf16: lhs last dim must equal rhs first dim"
        );
        let device = self.device.clone();

        match &device {
            Device::Gpu(_, stream) => {
                let f_matmul = self
                    .functions
                    .get("matmul_f32_bf16accum_f32")
                    .unwrap()
                    .clone();
                // Backward uses the existing tiled FP32 kernels — no new kernel needed.
                let f_bwd_a = self.functions.get("matmul_backward_a_f32").unwrap().clone();
                let f_bwd_b = self.functions.get("matmul_backward_b_f32").unwrap().clone();
                let stream_clone = stream.clone();

                // Allocate output first to avoid aliasing immutable tensor borrows
                // with a mutable borrow of `self`.
                let out_id = self.alloc_pooled(vec![m, n]);

                let a_fp32 = match &self.tensors[a_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
                let b_fp32 = match &self.tensors[b_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
                // ── BF16-style compute (on-the-fly quantization) → FP32 output ─────
                let o_fp32 = match &self.tensors[out_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
                let m_u64 = m as u64;
                let k_u64 = k as u64;
                let n_u64 = n as u64;
                let cfg_fwd = LaunchConfig {
                    grid_dim: ((n as u32 + 15) / 16, (m as u32 + 15) / 16, 1),
                    block_dim: (16, 16, 1),
                    shared_mem_bytes: 0,
                };
                let mut builder = stream.launch_builder(&f_matmul);
                builder
                    .arg(a_fp32)
                    .arg(b_fp32)
                    .arg(o_fp32)
                    .arg(&m_u64)
                    .arg(&k_u64)
                    .arg(&n_u64);
                unsafe { builder.launch(cfg_fwd) }.unwrap();

                // ── Backward closure ─────────────────────────────────────────────────
                //
                // Gradients are computed entirely in FP32 using the existing tiled
                // kernels (matmul_backward_a_f32 / matmul_backward_b_f32).
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = match &tensors[out_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let a_grad = match &tensors[a_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let b_grad = match &tensors[b_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    // For the backward pass we use the original FP32 weight data
                    // (master weights), giving full-precision gradients for AdamW.
                    let a_data = match &tensors[a_id].data {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let b_data = match &tensors[b_id].data {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };

                    // grad_a = grad_out @ B^T
                    let cfg_a = LaunchConfig {
                        grid_dim: ((k as u32 + 15) / 16, (m as u32 + 15) / 16, 1),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut b1 = stream_clone.launch_builder(&f_bwd_a);
                    b1.arg(out_grad)
                        .arg(b_data)
                        .arg(a_grad)
                        .arg(&m_u64)
                        .arg(&k_u64)
                        .arg(&n_u64);
                    unsafe { b1.launch(cfg_a) }.unwrap();

                    // grad_b = A^T @ grad_out
                    let cfg_b = LaunchConfig {
                        grid_dim: ((n as u32 + 15) / 16, (k as u32 + 15) / 16, 1),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut b2 = stream_clone.launch_builder(&f_bwd_b);
                    b2.arg(a_data)
                        .arg(out_grad)
                        .arg(b_grad)
                        .arg(&m_u64)
                        .arg(&k_u64)
                        .arg(&n_u64);
                    unsafe { b2.launch(cfg_b) }.unwrap();
                });

                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id, b_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }

            // CPU fallback: BF16 is a GPU-only optimisation.
            // On CPU we silently use the standard FP32 path so tests still pass.
            Device::Cpu => self.matmul(a_id, b_id),
        }
    }

    /// Matrix multiplication where `b` is a CPU-homed weight that must be
    /// streamed to VRAM for the forward kernel, then freed immediately.
    ///
    /// Called automatically by `matmul` when it detects Storage::Cpu on b.
    ///
    /// ## VRAM profile
    /// ```
    /// forward:  htod(b) → kernel → sync → FREE b_temp     [peak = 1 block]
    /// backward: htod(b) → kernel    (grad_a)
    ///           alloc grad_b_temp → kernel → sync → dtoh → accumulate → FREE
    /// ```
    /// Peak VRAM at any point = resident params + 1 streaming temp + activations.
    ///
    /// ## Why `stream.synchronize()` before `drop`
    /// `cuMemFree` is not stream-ordered. Without a sync, the matmul kernel
    /// may still be reading `b_temp` when the Rust drop fires the free.
    /// For large matmuls (the common case) the kernel finishes long before
    /// the next CPU instruction, so the sync is effectively free in practice.
    fn matmul_streamed(&mut self, a_id: usize, b_id: usize) -> usize {
        let a_shape = self.tensors[a_id].shape.clone();
        let b_shape = self.tensors[b_id].shape.clone();
        assert!(
            a_shape.len() >= 2,
            "matmul_streamed: lhs must have rank >= 2"
        );
        assert_eq!(
            b_shape.len(),
            2,
            "matmul_streamed: rhs must have rank 2 [k, n]"
        );
        let k = *a_shape.last().unwrap();
        let m = a_shape[..a_shape.len() - 1].iter().product::<usize>();
        let n = b_shape[1];
        assert_eq!(
            b_shape[0], k,
            "matmul_streamed: lhs last dim must equal rhs first dim"
        );

        let stream = match &self.device {
            Device::Gpu(_, s) => s.clone(),
            Device::Cpu => unreachable!("matmul_streamed called on CPU graph"),
        };

        let f_fwd = self.functions.get("matmul_f32").unwrap().clone();
        let f_bwd_a = self.functions.get("matmul_backward_a_f32").unwrap().clone();
        let f_bwd_b = self.functions.get("matmul_backward_b_f32").unwrap().clone();
        let stream_bwd = stream.clone();

        // ── Forward: htod b → kernel → SYNC → FREE ─────────────────────────────
        let b_cpu = self.tensors[b_id].data.as_cpu().clone(); // cheap clone of the Vec ref
        let b_temp_fwd = stream
            .clone_htod(b_cpu.as_slice())
            .expect("matmul_streamed: forward htod failed");

        let out_id = self.alloc_pooled(vec![m, n]);

        let a_s = match &self.tensors[a_id].data {
            Storage::Gpu(s) => s,
            _ => unreachable!("matmul_streamed: input a must be GPU storage"),
        };
        let o_s = match &self.tensors[out_id].data {
            Storage::Gpu(s) => s,
            _ => unreachable!(),
        };

        let m_u64 = m as u64;
        let k_u64 = k as u64;
        let n_u64 = n as u64;
        let cfg_fwd = LaunchConfig {
            grid_dim: ((n as u32 + 15) / 16, (m as u32 + 15) / 16, 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = stream.launch_builder(&f_fwd);
        builder
            .arg(a_s)
            .arg(&b_temp_fwd)
            .arg(o_s)
            .arg(&m_u64)
            .arg(&k_u64)
            .arg(&n_u64);
        unsafe { builder.launch(cfg_fwd) }.unwrap();

        // Sync before free: guarantees the matmul kernel has finished reading
        // b_temp_fwd before cudaFree is called. For large matmuls this is
        // effectively zero-cost — the kernel is already done.
        stream
            .synchronize()
            .expect("matmul_streamed: forward sync failed");
        drop(b_temp_fwd); // cudaFree — now safe

        // ── Backward closure ────────────────────────────────────────────────────
        // NOTE: b_cpu (the Vec<f32>) is moved into the closure. This is just
        // a Vec on the CPU heap — not VRAM. The closure re-htods it during
        // backward to get a fresh GPU buffer for the backward kernels.
        let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
            // Re-htod the CPU weight for backward kernels.
            // We read directly from tensors[b_id].data rather than the captured
            // b_cpu so that any weight update applied between forward and backward
            // (which shouldn't happen but is safe to handle) is reflected.
            let b_cpu_bwd = tensors[b_id].data.as_cpu();
            let b_temp_bwd = stream_bwd
                .clone_htod(b_cpu_bwd)
                .expect("matmul_streamed: backward htod failed");

            let out_grad = match &tensors[out_id].grad {
                Storage::Gpu(s) => s,
                _ => unreachable!(),
            };
            let a_grad = match &tensors[a_id].grad {
                Storage::Gpu(s) => s,
                _ => unreachable!(),
            };
            let a_data = match &tensors[a_id].data {
                Storage::Gpu(s) => s,
                _ => unreachable!(),
            };

            // grad_a = grad_out @ b^T   (GPU → GPU, accumulate in-place)
            let cfg_a = LaunchConfig {
                grid_dim: ((k as u32 + 15) / 16, (m as u32 + 15) / 16, 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };
            let mut b1 = stream_bwd.launch_builder(&f_bwd_a);
            b1.arg(out_grad)
                .arg(&b_temp_bwd)
                .arg(a_grad)
                .arg(&m_u64)
                .arg(&k_u64)
                .arg(&n_u64);
            unsafe { b1.launch(cfg_a) }.unwrap();

            // grad_b = a^T @ grad_out  (GPU temp → dtoh → accumulate into CPU grad)
            //
            // We cannot atomicAdd directly into CPU RAM from a CUDA kernel, so we
            // compute into a fresh GPU buffer, dtoh it, then add on the CPU.
            // alloc_zeros initialises to 0, and the backward kernel uses atomicAdd,
            // so grad_b_temp is the exact gradient for this micro-batch.
            let grad_b_temp = stream_bwd
                .alloc_zeros::<f32>(k * n)
                .expect("matmul_streamed: grad_b_temp alloc failed");

            let cfg_b = LaunchConfig {
                grid_dim: ((n as u32 + 15) / 16, (k as u32 + 15) / 16, 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };
            let mut b2 = stream_bwd.launch_builder(&f_bwd_b);
            b2.arg(a_data)
                .arg(out_grad)
                .arg(&grad_b_temp)
                .arg(&m_u64)
                .arg(&k_u64)
                .arg(&n_u64);
            unsafe { b2.launch(cfg_b) }.unwrap();

            // Sync before dtoh: the backward kernel must be done before we read
            stream_bwd
                .synchronize()
                .expect("matmul_streamed: backward sync failed");

            let grad_b_gpu = stream_bwd
                .clone_dtoh(&grad_b_temp)
                .expect("matmul_streamed: grad dtoh failed");

            // Free GPU temporaries now that data is on CPU
            drop(b_temp_bwd);
            drop(grad_b_temp);

            // Accumulate into the CPU grad buffer (supports gradient accumulation:
            // multiple backward passes add up here, zero_grad resets between steps)
            let b_grad = tensors[b_id].grad.as_cpu_mut();
            for (acc, delta) in b_grad.iter_mut().zip(grad_b_gpu.iter()) {
                *acc += delta;
            }
        });

        if !self.no_grad {
            self.tape.nodes.push(TapeNode {
                inputs: vec![a_id, b_id],
                output: out_id,
                backward_fn,
            });
        }

        out_id
    }

    pub fn bmm(&mut self, a_id: usize, b_id: usize, trans_b: bool) -> usize {
        let a_shape = self.tensors[a_id].shape.clone();
        let b_shape = self.tensors[b_id].shape.clone();
        let batch = a_shape[0];
        let m = a_shape[1];
        let k = a_shape[2];
        let n = if trans_b { b_shape[1] } else { b_shape[2] };
        let device = self.device.clone();

        match &device {
            Device::Gpu(_, stream) => {
                #[cfg(feature = "bf16")]
                let f_fwd = self.functions.get("bmm_f32_bf16accum_f32").unwrap().clone();
                #[cfg(not(feature = "bf16"))]
                let f_fwd = self.functions.get("bmm_f32").unwrap().clone();
                let f_bwd_a = self
                    .functions
                    .get(if trans_b {
                        "bmm_backward_a_transb_f32"
                    } else {
                        "bmm_backward_a_f32"
                    })
                    .unwrap()
                    .clone();
                let f_bwd_b = self
                    .functions
                    .get(if trans_b {
                        "bmm_backward_b_transb_f32"
                    } else {
                        "bmm_backward_b_f32"
                    })
                    .unwrap()
                    .clone();
                let stream_clone = stream.clone();

                let out_id = self.alloc_pooled(vec![batch, m, n]);

                let a_s = match &self.tensors[a_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
                let b_s = match &self.tensors[b_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
                let o_s = match &self.tensors[out_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };

                let batch_u64 = batch as u64;
                let m_u64 = m as u64;
                let k_u64 = k as u64;
                let n_u64 = n as u64;

                let mut builder = stream.launch_builder(&f_fwd);
                builder
                    .arg(a_s)
                    .arg(b_s)
                    .arg(o_s)
                    .arg(&batch_u64)
                    .arg(&m_u64)
                    .arg(&k_u64)
                    .arg(&n_u64)
                    .arg(&trans_b);
                unsafe {
                    builder.launch(LaunchConfig {
                        grid_dim: ((n as u32 + 15) / 16, (m as u32 + 15) / 16, batch as u32),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .unwrap();

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = match &tensors[out_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let a_grad = match &tensors[a_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let b_grad = match &tensors[b_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let a_data = match &tensors[a_id].data {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let b_data = match &tensors[b_id].data {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };

                    let mut b1 = stream_clone.launch_builder(&f_bwd_a);
                    b1.arg(out_grad)
                        .arg(b_data)
                        .arg(a_grad)
                        .arg(&batch_u64)
                        .arg(&m_u64)
                        .arg(&k_u64)
                        .arg(&n_u64);
                    unsafe {
                        b1.launch(LaunchConfig {
                            grid_dim: ((k as u32 + 15) / 16, (m as u32 + 15) / 16, batch as u32),
                            block_dim: (16, 16, 1),
                            shared_mem_bytes: 0,
                        })
                    }
                    .unwrap();

                    let mut b2 = stream_clone.launch_builder(&f_bwd_b);
                    b2.arg(a_data)
                        .arg(out_grad)
                        .arg(b_grad)
                        .arg(&batch_u64)
                        .arg(&m_u64)
                        .arg(&k_u64)
                        .arg(&n_u64);
                    let cfg_b = if trans_b {
                        LaunchConfig {
                            grid_dim: ((k as u32 + 15) / 16, (n as u32 + 15) / 16, batch as u32),
                            block_dim: (16, 16, 1),
                            shared_mem_bytes: 0,
                        }
                    } else {
                        LaunchConfig {
                            grid_dim: ((n as u32 + 15) / 16, (k as u32 + 15) / 16, batch as u32),
                            block_dim: (16, 16, 1),
                            shared_mem_bytes: 0,
                        }
                    };
                    unsafe { b2.launch(cfg_b) }.unwrap();
                });

                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id, b_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
            Device::Cpu => panic!("BMM not fully implemented for CPU in this phase"),
        }
    }

    pub fn softmax(&mut self, a_id: usize) -> usize {
        let device = self.device.clone();
        match &device {
            Device::Gpu(_, stream) => {
                let a = &self.tensors[a_id];
                let out_shape = a.shape.clone();
                let n = *out_shape.last().unwrap() as u64;
                let b = (a.data.len() as u64) / n;

                let f_fwd = self.functions.get("softmax_f32").unwrap().clone();
                let f_bwd = self.functions.get("softmax_backward_f32").unwrap().clone();
                let stream_clone = stream.clone();

                let out_id = self.alloc_pooled(out_shape);

                let a_s = match &self.tensors[a_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
                let o_s = match &self.tensors[out_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };

                let mut builder = stream.launch_builder(&f_fwd);
                builder.arg(a_s).arg(o_s).arg(&b).arg(&n);
                unsafe { builder.launch(LaunchConfig::for_num_elems(b as u32)) }.unwrap();

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_data = match &tensors[out_id].data {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let out_grad = match &tensors[out_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let a_grad = match &tensors[a_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };

                    let mut b1 = stream_clone.launch_builder(&f_bwd);
                    b1.arg(out_data).arg(out_grad).arg(a_grad).arg(&b).arg(&n);
                    unsafe { b1.launch(LaunchConfig::for_num_elems(b as u32)) }.unwrap();
                });

                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
            Device::Cpu => panic!("Softmax not fully implemented for CPU in this phase"),
        }
    }

    pub fn flash_attention(
        &mut self,
        q_id: usize,
        k_id: usize,
        v_id: usize,
        scale: f32,
        causal: bool,
    ) -> usize {
        assert!(
            self.no_grad,
            "flash_attention currently supports no_grad/inference path only"
        );
        let q_shape = self.tensors[q_id].shape.clone();
        let k_shape = self.tensors[k_id].shape.clone();
        let v_shape = self.tensors[v_id].shape.clone();
        assert_eq!(q_shape.len(), 3, "flash_attention expects q=[B,M,D]");
        assert_eq!(k_shape.len(), 3, "flash_attention expects k=[B,N,D]");
        assert_eq!(v_shape.len(), 3, "flash_attention expects v=[B,N,D]");

        let batch = q_shape[0];
        let m = q_shape[1];
        let d = q_shape[2];
        let n = k_shape[1];

        assert_eq!(k_shape[0], batch, "flash_attention batch mismatch q/k");
        assert_eq!(v_shape[0], batch, "flash_attention batch mismatch q/v");
        assert_eq!(k_shape[2], d, "flash_attention head dim mismatch q/k");
        assert_eq!(v_shape[1], n, "flash_attention sequence mismatch k/v");
        assert_eq!(v_shape[2], d, "flash_attention head dim mismatch q/v");
        assert!(
            d <= 256,
            "flash_attention kernel currently supports head_dim <= 256"
        );

        let device = self.device.clone();
        match &device {
            Device::Gpu(_, stream) => {
                let f_fwd = self.functions.get("flash_attention_f32").unwrap().clone();
                let out_id = self.alloc_pooled(vec![batch, m, d]);

                let q_s = match &self.tensors[q_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
                let k_s = match &self.tensors[k_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
                let v_s = match &self.tensors[v_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
                let o_s = match &self.tensors[out_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };

                let batch_u64 = batch as u64;
                let m_u64 = m as u64;
                let n_u64 = n as u64;
                let d_u64 = d as u64;
                let total_rows = (batch * m) as u32;

                let mut builder = stream.launch_builder(&f_fwd);
                builder
                    .arg(q_s)
                    .arg(k_s)
                    .arg(v_s)
                    .arg(o_s)
                    .arg(&batch_u64)
                    .arg(&m_u64)
                    .arg(&n_u64)
                    .arg(&d_u64)
                    .arg(&scale)
                    .arg(&causal);
                unsafe { builder.launch(LaunchConfig::for_num_elems(total_rows)) }.unwrap();
                out_id
            }
            Device::Cpu => panic!("flash_attention CPU fallback not supported"),
        }
    }

    pub fn rope(&mut self, a_id: usize, head_dim: usize) -> usize {
        let shape = self.tensors[a_id].shape.clone();
        let seq_len = shape[1];
        let hidden_dim = shape[2];
        let size = shape.iter().product::<usize>();
        let num_pairs = size / 2;
        let device = self.device.clone();

        match &device {
            Device::Gpu(_, stream) => {
                let f_fwd = self.functions.get("rope_f32").unwrap().clone();
                let f_bwd = self.functions.get("rope_backward_f32").unwrap().clone();
                let stream_clone = stream.clone();

                let out_id = self.alloc_pooled(shape);

                let a_s = match &self.tensors[a_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
                let o_s = match &self.tensors[out_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };

                let s_u64 = seq_len as u64;
                let hd_u64 = hidden_dim as u64;
                let hdim_u64 = head_dim as u64;
                let np_u64 = num_pairs as u64;

                let mut builder = stream.launch_builder(&f_fwd);
                builder
                    .arg(a_s)
                    .arg(o_s)
                    .arg(&s_u64)
                    .arg(&hd_u64)
                    .arg(&hdim_u64)
                    .arg(&np_u64);
                unsafe { builder.launch(LaunchConfig::for_num_elems(num_pairs as u32)) }.unwrap();

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = match &tensors[out_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let a_grad = match &tensors[a_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };

                    let mut b1 = stream_clone.launch_builder(&f_bwd);
                    b1.arg(out_grad)
                        .arg(a_grad)
                        .arg(&s_u64)
                        .arg(&hd_u64)
                        .arg(&hdim_u64)
                        .arg(&np_u64);
                    unsafe { b1.launch(LaunchConfig::for_num_elems(num_pairs as u32)) }.unwrap();
                });

                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
            Device::Cpu => panic!("RoPE not fully implemented for CPU"),
        }
    }

    pub fn transpose_0213(&mut self, a_id: usize) -> usize {
        let shape = self.tensors[a_id].shape.clone();
        let b = shape[0];
        let s = shape[1];
        let h = shape[2];
        let d = shape[3];
        let out_shape = vec![b, h, s, d];
        let size = b * s * h * d;
        let device = self.device.clone();

        match &device {
            Device::Gpu(_, stream) => {
                let f_fwd = self.functions.get("transpose_0213_f32").unwrap().clone();
                let f_bwd = self
                    .functions
                    .get("transpose_0213_backward_f32")
                    .unwrap()
                    .clone();
                let stream_clone = stream.clone();

                let out_id = self.alloc_pooled(out_shape);

                let a_s = match &self.tensors[a_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
                let o_s = match &self.tensors[out_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };

                let b_u64 = b as u64;
                let s_u64 = s as u64;
                let h_u64 = h as u64;
                let d_u64 = d as u64;

                let mut builder = stream.launch_builder(&f_fwd);
                builder
                    .arg(a_s)
                    .arg(o_s)
                    .arg(&b_u64)
                    .arg(&s_u64)
                    .arg(&h_u64)
                    .arg(&d_u64);
                unsafe { builder.launch(LaunchConfig::for_num_elems(size as u32)) }.unwrap();

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = match &tensors[out_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let a_grad = match &tensors[a_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let mut b1 = stream_clone.launch_builder(&f_bwd);
                    b1.arg(out_grad)
                        .arg(a_grad)
                        .arg(&b_u64)
                        .arg(&s_u64)
                        .arg(&h_u64)
                        .arg(&d_u64);
                    unsafe { b1.launch(LaunchConfig::for_num_elems(size as u32)) }.unwrap();
                });

                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
            Device::Cpu => panic!("transpose_0213 CPU fallback not supported"),
        }
    }

    pub fn silu(&mut self, a_id: usize) -> usize {
        let device = self.device.clone();
        match &device {
            Device::Gpu(_, stream) => {
                let a_shape = self.tensors[a_id].shape.clone();
                let out_size = a_shape.iter().product::<usize>();

                let f_fwd = self.functions.get("silu_f32").unwrap().clone();
                let f_bwd = self.functions.get("silu_backward_f32").unwrap().clone();
                let stream_clone = stream.clone();

                let out_id = self.alloc_pooled(a_shape);

                let a_s = match &self.tensors[a_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
                let o_s = match &self.tensors[out_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };

                let n = out_size as u64;
                let mut builder = stream.launch_builder(&f_fwd);
                builder.arg(a_s).arg(o_s).arg(&n);
                unsafe { builder.launch(LaunchConfig::for_num_elems(out_size as u32)) }.unwrap();

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let a_data = match &tensors[a_id].data {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let out_grad = match &tensors[out_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let a_grad = match &tensors[a_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };

                    let mut b1 = stream_clone.launch_builder(&f_bwd);
                    b1.arg(a_data).arg(out_grad).arg(a_grad).arg(&n);
                    unsafe { b1.launch(LaunchConfig::for_num_elems(out_size as u32)) }.unwrap();
                });

                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
            Device::Cpu => {
                let a_data = self.tensors[a_id].data.as_cpu().clone();
                let mut out_data = vec![0.0; a_data.len()];
                for i in 0..a_data.len() {
                    let x = a_data[i];
                    out_data[i] = x * (1.0 / (1.0 + (-x).exp()));
                }
                let out_id = self.alloc(self.tensors[a_id].shape.clone(), out_data);
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = tensors[out_id].grad.as_cpu().clone();
                    let a_grad = tensors[a_id].grad.as_cpu_mut();
                    for i in 0..a_grad.len() {
                        let x = a_data[i];
                        let sig = 1.0 / (1.0 + (-x).exp());
                        let silu = x * sig;
                        a_grad[i] += out_grad[i] * (silu + sig * (1.0 - silu));
                    }
                });
                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
        }
    }

    pub fn gather(&mut self, weights_id: usize, indices_id: usize) -> usize {
        let device = self.device.clone();
        match &device {
            Device::Gpu(_, stream) => {
                let w = &self.tensors[weights_id];
                let idx = &self.tensors[indices_id];
                let hidden_dim = w.shape[1];
                let num_indices = idx.data.len();
                let out_size = num_indices * hidden_dim;

                let mut out_shape = idx.shape.clone();
                out_shape.push(hidden_dim);

                let f_fwd = self.functions.get("gather_f32").unwrap().clone();
                let f_bwd = self.functions.get("gather_backward_f32").unwrap().clone();
                let stream_clone = stream.clone();

                let out_id = self.alloc_pooled(out_shape);

                let w_s = match &self.tensors[weights_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
                let idx_s = match &self.tensors[indices_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
                let o_s = match &self.tensors[out_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };

                let hidden_u64 = hidden_dim as u64;
                let out_size_u64 = out_size as u64;

                let mut builder = stream.launch_builder(&f_fwd);
                builder
                    .arg(w_s)
                    .arg(idx_s)
                    .arg(o_s)
                    .arg(&hidden_u64)
                    .arg(&out_size_u64);
                unsafe { builder.launch(LaunchConfig::for_num_elems(out_size as u32)) }.unwrap();

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let idx_data = match &tensors[indices_id].data {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let out_grad = match &tensors[out_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let w_grad = match &tensors[weights_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };

                    let mut b1 = stream_clone.launch_builder(&f_bwd);
                    b1.arg(idx_data)
                        .arg(out_grad)
                        .arg(w_grad)
                        .arg(&hidden_u64)
                        .arg(&out_size_u64);
                    unsafe { b1.launch(LaunchConfig::for_num_elems(out_size as u32)) }.unwrap();
                });

                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![weights_id, indices_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
            Device::Cpu => {
                let w = &self.tensors[weights_id];
                let idx = &self.tensors[indices_id];
                let hidden_dim = w.shape[1];
                let num_indices = idx.data.as_cpu().len();
                let mut out_data = vec![0.0; num_indices * hidden_dim];
                let idx_data: Vec<usize> = idx.data.as_cpu().iter().map(|&x| x as usize).collect();

                let w_data = w.data.as_cpu();
                for i in 0..num_indices {
                    let row = idx_data[i];
                    for d in 0..hidden_dim {
                        out_data[i * hidden_dim + d] = w_data[row * hidden_dim + d];
                    }
                }

                let mut out_shape = idx.shape.clone();
                out_shape.push(hidden_dim);
                let out_id = self.alloc(out_shape, out_data);
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let o_grad = tensors[out_id].grad.as_cpu().clone();
                    let w_grad = tensors[weights_id].grad.as_cpu_mut();
                    for i in 0..num_indices {
                        let row = idx_data[i];
                        for d in 0..hidden_dim {
                            w_grad[row * hidden_dim + d] += o_grad[i * hidden_dim + d];
                        }
                    }
                });
                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![weights_id, indices_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
        }
    }

    pub fn cross_entropy(&mut self, logits_id: usize, targets: &[usize]) -> usize {
        let device = self.device.clone();
        match &device {
            Device::Gpu(_, stream) => {
                let logits = &self.tensors[logits_id];
                let batch_size = logits.shape[0];
                let num_classes = logits.shape[1];
                let out_id = self.alloc_pooled(vec![]);

                let f_fwd = self.functions.get("cross_entropy_f32").unwrap().clone();
                let f_bwd = self
                    .functions
                    .get("cross_entropy_backward_f32")
                    .unwrap()
                    .clone();
                let stream_clone = stream.clone();

                let targets_f32: Vec<f32> = targets.iter().map(|&x| x as f32).collect();
                let targets_d = stream.clone_htod(targets_f32.as_slice()).unwrap();

                let l_s = match &self.tensors[logits_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
                let o_s = match &self.tensors[out_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };

                let b_u64 = batch_size as u64;
                let c_u64 = num_classes as u64;

                let mut builder = stream.launch_builder(&f_fwd);
                builder
                    .arg(l_s)
                    .arg(&targets_d)
                    .arg(o_s)
                    .arg(&b_u64)
                    .arg(&c_u64);
                unsafe { builder.launch(LaunchConfig::for_num_elems(batch_size as u32)) }.unwrap();

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let l_data = match &tensors[logits_id].data {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let out_grad = match &tensors[out_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let l_grad = match &tensors[logits_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };

                    let mut b1 = stream_clone.launch_builder(&f_bwd);
                    b1.arg(l_data)
                        .arg(&targets_d)
                        .arg(out_grad)
                        .arg(l_grad)
                        .arg(&b_u64)
                        .arg(&c_u64);
                    unsafe { b1.launch(LaunchConfig::for_num_elems(batch_size as u32)) }.unwrap();
                });

                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![logits_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
            Device::Cpu => {
                let logits = &self.tensors[logits_id];
                let batch_size = logits.shape[0];
                let num_classes = logits.shape[1];
                let mut out_loss = 0.0;
                let mut probs = vec![0.0; batch_size * num_classes];
                let logits_data = logits.data.as_cpu();

                for b in 0..batch_size {
                    let mut max_val = f32::NEG_INFINITY;
                    for c in 0..num_classes {
                        max_val = max_val.max(logits_data[b * num_classes + c]);
                    }
                    let mut sum_exp = 0.0;
                    for c in 0..num_classes {
                        let exp_val = (logits_data[b * num_classes + c] - max_val).exp();
                        probs[b * num_classes + c] = exp_val;
                        sum_exp += exp_val;
                    }
                    for c in 0..num_classes {
                        probs[b * num_classes + c] /= sum_exp;
                    }
                    out_loss += -(probs[b * num_classes + targets[b]] + 1e-8).ln();
                }

                let out_id = self.alloc(vec![], vec![out_loss / batch_size as f32]);
                let targets_cap = targets.to_vec();
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let o_grad = tensors[out_id].grad.as_cpu()[0];
                    let l_grad = tensors[logits_id].grad.as_cpu_mut();
                    for b in 0..batch_size {
                        for c in 0..num_classes {
                            let idx = b * num_classes + c;
                            let mut g = probs[idx];
                            if c == targets_cap[b] {
                                g -= 1.0;
                            }
                            l_grad[idx] += (g / batch_size as f32) * o_grad;
                        }
                    }
                });
                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![logits_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
        }
    }

    pub fn rms_norm(&mut self, x_id: usize, weight_id: usize, eps: f32) -> usize {
        let device = self.device.clone();
        match &device {
            Device::Gpu(_, stream) => {
                let x = &self.tensors[x_id];
                let dim = *x.shape.last().unwrap();
                let num_vecs = x.data.len() / dim;

                let f_fwd = self.functions.get("rmsnorm_f32").unwrap().clone();
                let f_bwd = self.functions.get("rmsnorm_backward_f32").unwrap().clone();
                let stream_clone = stream.clone();

                let out_id = self.alloc_pooled(x.shape.clone());

                let x_s = match &self.tensors[x_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
                let w_s = match &self.tensors[weight_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
                let o_s = match &self.tensors[out_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };

                let dim_u64 = dim as u64;
                let num_vecs_u64 = num_vecs as u64;

                let mut builder = stream.launch_builder(&f_fwd);
                builder
                    .arg(x_s)
                    .arg(w_s)
                    .arg(o_s)
                    .arg(&dim_u64)
                    .arg(&eps)
                    .arg(&num_vecs_u64);
                unsafe { builder.launch(LaunchConfig::for_num_elems(num_vecs as u32)) }.unwrap();

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let x_data = match &tensors[x_id].data {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let w_data = match &tensors[weight_id].data {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let out_grad = match &tensors[out_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let x_grad = match &tensors[x_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let w_grad = match &tensors[weight_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };

                    let mut b1 = stream_clone.launch_builder(&f_bwd);
                    b1.arg(x_data)
                        .arg(w_data)
                        .arg(out_grad)
                        .arg(x_grad)
                        .arg(w_grad)
                        .arg(&dim_u64)
                        .arg(&eps)
                        .arg(&num_vecs_u64);
                    unsafe { b1.launch(LaunchConfig::for_num_elems(num_vecs as u32)) }.unwrap();
                });

                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![x_id, weight_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
            Device::Cpu => {
                let x = &self.tensors[x_id];
                let dim = *x.shape.last().unwrap();
                let num_vecs = x.data.as_cpu().len() / dim;
                let mut out_data = vec![0.0; x.data.as_cpu().len()];
                let mut rrms_cache = vec![0.0; num_vecs];
                let x_fwd = x.data.as_cpu().clone();
                let w_fwd = self.tensors[weight_id].data.as_cpu().clone();

                for n in 0..num_vecs {
                    let off = n * dim;
                    let mut ss = 0.0;
                    for d in 0..dim {
                        ss += x_fwd[off + d].powi(2);
                    }
                    let rrms = 1.0 / (ss / dim as f32 + eps).sqrt();
                    rrms_cache[n] = rrms;
                    for d in 0..dim {
                        out_data[off + d] = x_fwd[off + d] * rrms * w_fwd[d];
                    }
                }

                let out_id = self.alloc(x.shape.clone(), out_data);
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let o_grad = tensors[out_id].grad.as_cpu().clone();
                    for n in 0..num_vecs {
                        let off = n * dim;
                        let rrms = rrms_cache[n];
                        let mut gdxw = 0.0;
                        for d in 0..dim {
                            gdxw += o_grad[off + d] * x_fwd[off + d] * w_fwd[d];
                        }
                        let rrc_d = (rrms.powi(3)) / dim as f32;
                        for d in 0..dim {
                            let dx =
                                rrms * (o_grad[off + d] * w_fwd[d]) - x_fwd[off + d] * rrc_d * gdxw;
                            tensors[x_id].grad.as_cpu_mut()[off + d] += dx;
                            tensors[weight_id].grad.as_cpu_mut()[d] +=
                                o_grad[off + d] * x_fwd[off + d] * rrms;
                        }
                    }
                });
                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![x_id, weight_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
        }
    }

    pub fn reshape(&mut self, a_id: usize, new_shape: Vec<usize>) -> usize {
        let device = self.device.clone();
        match &device {
            Device::Gpu(_, stream) => {
                let old_size = self.tensors[a_id].data.len();
                let f_fwd = self.functions.get("copy_f32").unwrap().clone();
                let f_bwd = self.functions.get("accumulate_f32").unwrap().clone();
                let stream_clone = stream.clone();

                let out_id = self.alloc_pooled(new_shape);

                let a_s = match &self.tensors[a_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
                let o_s = match &self.tensors[out_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };

                let n = old_size as u64;
                let mut builder = stream.launch_builder(&f_fwd);
                builder.arg(a_s).arg(o_s).arg(&n);
                unsafe { builder.launch(LaunchConfig::for_num_elems(old_size as u32)) }.unwrap();

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = match &tensors[out_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let a_grad = match &tensors[a_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };

                    let rank = 1u64;
                    let s0 = n;
                    let s1 = 1u64;
                    let s2 = 1u64;
                    let a_str0 = 1u64;
                    let a_str1 = 1u64;
                    let a_str2 = 1u64;

                    let mut b1 = stream_clone.launch_builder(&f_bwd);
                    b1.arg(a_grad)
                        .arg(out_grad)
                        .arg(&n)
                        .arg(&rank)
                        .arg(&s0)
                        .arg(&s1)
                        .arg(&s2)
                        .arg(&a_str0)
                        .arg(&a_str1)
                        .arg(&a_str2);
                    unsafe { b1.launch(LaunchConfig::for_num_elems(old_size as u32)) }.unwrap();
                });

                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
            Device::Cpu => {
                let old_size = self.tensors[a_id].data.as_cpu().len();
                let out_id = self.alloc(new_shape, self.tensors[a_id].data.as_cpu().clone());
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let o_grad = tensors[out_id].grad.as_cpu().clone();
                    let a_grad = tensors[a_id].grad.as_cpu_mut();
                    for i in 0..old_size {
                        a_grad[i] += o_grad[i];
                    }
                });
                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
        }
    }

    /// Reinterpret tensor metadata as a new contiguous shape without allocating.
    /// Caller must ensure the element count matches.
    pub fn reinterpret_shape(&mut self, tensor_id: usize, new_shape: Vec<usize>) {
        let old_elems = self.tensors[tensor_id].shape.iter().product::<usize>();
        let new_elems = if new_shape.is_empty() {
            1
        } else {
            new_shape.iter().product::<usize>()
        };
        assert_eq!(
            old_elems, new_elems,
            "reinterpret_shape: element count mismatch"
        );
        self.tensors[tensor_id].shape = new_shape.clone();
        self.tensors[tensor_id].strides = Tensor::compute_strides(&new_shape);
    }

    pub fn transpose(&mut self, a_id: usize, dim0: usize, dim1: usize) -> usize {
        match &self.device {
            Device::Gpu(_, _) => {
                panic!("GPU Transpose not natively implemented - use reshape fallback or CPU")
            }
            Device::Cpu => {
                let a = &self.tensors[a_id];
                let mut out_shape = a.shape.clone();
                out_shape.swap(dim0, dim1);
                let mut out_strides = a.strides.clone();
                out_strides.swap(dim0, dim1);
                let mut out_data = vec![0.0; a.data.as_cpu().len()];
                let a_data = a.data.as_cpu();
                for i in 0..out_data.len() {
                    let nd = Tensor::flat_to_nd(i, &out_shape);
                    out_data[i] = a_data[Tensor::nd_to_flat(&nd, &out_strides)];
                }
                let out_id = self.alloc(out_shape, out_data);
                let out_strides_cap = out_strides.clone();
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let o_grad = tensors[out_id].grad.as_cpu().clone();
                    for i in 0..o_grad.len() {
                        let nd = Tensor::flat_to_nd(i, &tensors[out_id].shape);
                        tensors[a_id].grad.as_cpu_mut()
                            [Tensor::nd_to_flat(&nd, &out_strides_cap)] += o_grad[i];
                    }
                });
                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
        }
    }

    pub fn sum(&mut self, a_id: usize, dim: usize) -> usize {
        let device = self.device.clone();
        match &device {
            Device::Gpu(_, stream) => {
                let a = &self.tensors[a_id];
                let mut out_shape = a.shape.clone();
                let mut in_strides = a.strides.clone();
                let reduced_dim_size = a.shape[dim] as u64;
                let reduced_dim_stride = a.strides[dim] as u64;
                out_shape.remove(dim);
                in_strides.remove(dim);
                let out_size = if out_shape.is_empty() {
                    1
                } else {
                    out_shape.iter().product()
                };
                let rank = out_shape.len() as u64;

                let mut os = [1u64; 3];
                let mut is = [0u64; 3];
                for i in 0..out_shape.len() {
                    os[i] = out_shape[i] as u64;
                    is[i] = in_strides[i] as u64;
                }

                let f_fwd = self.functions.get("sum_f32").unwrap().clone();
                let f_bwd = self.functions.get("sum_backward_f32").unwrap().clone();
                let stream_clone = stream.clone();

                let out_id = self.alloc_pooled(out_shape.clone());

                let a_s = match &self.tensors[a_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
                let o_s = match &self.tensors[out_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };

                let out_size_u64 = out_size as u64;
                let mut builder = stream.launch_builder(&f_fwd);
                builder
                    .arg(a_s)
                    .arg(o_s)
                    .arg(&out_size_u64)
                    .arg(&reduced_dim_size)
                    .arg(&reduced_dim_stride)
                    .arg(&rank)
                    .arg(&os[0])
                    .arg(&os[1])
                    .arg(&os[2])
                    .arg(&is[0])
                    .arg(&is[1])
                    .arg(&is[2]);
                unsafe { builder.launch(LaunchConfig::for_num_elems(out_size as u32)) }.unwrap();

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let out_grad = match &tensors[out_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let a_grad = match &tensors[a_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };

                    let mut b1 = stream_clone.launch_builder(&f_bwd);
                    b1.arg(out_grad)
                        .arg(a_grad)
                        .arg(&out_size_u64)
                        .arg(&reduced_dim_size)
                        .arg(&reduced_dim_stride)
                        .arg(&rank)
                        .arg(&os[0])
                        .arg(&os[1])
                        .arg(&os[2])
                        .arg(&is[0])
                        .arg(&is[1])
                        .arg(&is[2]);
                    unsafe { b1.launch(LaunchConfig::for_num_elems(out_size as u32)) }.unwrap();
                });

                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
            Device::Cpu => {
                let a = &self.tensors[a_id];
                let mut out_shape = a.shape.clone();
                out_shape.remove(dim);
                let out_size = if out_shape.is_empty() {
                    1
                } else {
                    out_shape.iter().product()
                };
                let mut out_data = vec![0.0; out_size];
                let os = Tensor::compute_strides(&out_shape);
                let a_shape = a.shape.clone();
                let a_data = a.data.as_cpu();
                for i in 0..a_data.len() {
                    let mut nd = Tensor::flat_to_nd(i, &a_shape);
                    nd.remove(dim);
                    let idx = if out_shape.is_empty() {
                        0
                    } else {
                        Tensor::nd_to_flat(&nd, &os)
                    };
                    out_data[idx] += a_data[i];
                }
                let out_id = self.alloc(out_shape, out_data);
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let o_grad = tensors[out_id].grad.as_cpu().clone();
                    for i in 0..tensors[a_id].data.as_cpu().len() {
                        let mut nd = Tensor::flat_to_nd(i, &a_shape);
                        nd.remove(dim);
                        let idx = if tensors[out_id].shape.is_empty() {
                            0
                        } else {
                            Tensor::nd_to_flat(&nd, &os)
                        };
                        tensors[a_id].grad.as_cpu_mut()[i] += o_grad[idx];
                    }
                });
                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
        }
    }

    pub fn max(&mut self, a_id: usize, dim: usize) -> usize {
        let device = self.device.clone();
        match &device {
            Device::Gpu(_, stream) => {
                let a = &self.tensors[a_id];
                let mut out_shape = a.shape.clone();
                let mut in_strides = a.strides.clone();
                let reduced_dim_size = a.shape[dim] as u64;
                let reduced_dim_stride = a.strides[dim] as u64;
                out_shape.remove(dim);
                in_strides.remove(dim);
                let out_size = if out_shape.is_empty() {
                    1
                } else {
                    out_shape.iter().product()
                };
                let rank = out_shape.len() as u64;

                let mut os = [1u64; 3];
                let mut is = [0u64; 3];
                for i in 0..out_shape.len() {
                    os[i] = out_shape[i] as u64;
                    is[i] = in_strides[i] as u64;
                }

                let f_fwd = self.functions.get("max_f32").unwrap().clone();
                let f_bwd = self.functions.get("max_backward_f32").unwrap().clone();
                let stream_clone = stream.clone();

                let out_id = self.alloc_pooled(out_shape.clone());

                let a_s = match &self.tensors[a_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
                let o_s = match &self.tensors[out_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };

                let out_size_u64 = out_size as u64;
                let mut builder = stream.launch_builder(&f_fwd);
                builder
                    .arg(a_s)
                    .arg(o_s)
                    .arg(&out_size_u64)
                    .arg(&reduced_dim_size)
                    .arg(&reduced_dim_stride)
                    .arg(&rank)
                    .arg(&os[0])
                    .arg(&os[1])
                    .arg(&os[2])
                    .arg(&is[0])
                    .arg(&is[1])
                    .arg(&is[2]);
                unsafe { builder.launch(LaunchConfig::for_num_elems(out_size as u32)) }.unwrap();

                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let a_data = match &tensors[a_id].data {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let out_grad = match &tensors[out_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let a_grad = match &tensors[a_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };

                    let mut b1 = stream_clone.launch_builder(&f_bwd);
                    b1.arg(a_data)
                        .arg(out_grad)
                        .arg(a_grad)
                        .arg(&out_size_u64)
                        .arg(&reduced_dim_size)
                        .arg(&reduced_dim_stride)
                        .arg(&rank)
                        .arg(&os[0])
                        .arg(&os[1])
                        .arg(&os[2])
                        .arg(&is[0])
                        .arg(&is[1])
                        .arg(&is[2]);
                    unsafe { b1.launch(LaunchConfig::for_num_elems(out_size as u32)) }.unwrap();
                });

                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
            Device::Cpu => {
                let a = &self.tensors[a_id];
                let mut out_shape = a.shape.clone();
                out_shape.remove(dim);
                let out_size = if out_shape.is_empty() {
                    1
                } else {
                    out_shape.iter().product()
                };
                let mut out_data = vec![f32::NEG_INFINITY; out_size];
                let mut argmax = vec![0; out_size];
                let os = Tensor::compute_strides(&out_shape);
                let a_data = a.data.as_cpu();
                for i in 0..a_data.len() {
                    let mut nd = Tensor::flat_to_nd(i, &a.shape);
                    nd.remove(dim);
                    let idx = if out_shape.is_empty() {
                        0
                    } else {
                        Tensor::nd_to_flat(&nd, &os)
                    };
                    if a_data[i] > out_data[idx] {
                        out_data[idx] = a_data[i];
                        argmax[idx] = i;
                    }
                }
                let out_id = self.alloc(out_shape, out_data);
                let backward_fn = Box::new(move |tensors: &mut [Tensor]| {
                    let o_grad = tensors[out_id].grad.as_cpu().clone();
                    for i in 0..o_grad.len() {
                        tensors[a_id].grad.as_cpu_mut()[argmax[i]] += o_grad[i];
                    }
                });
                if !self.no_grad {
                    self.tape.nodes.push(TapeNode {
                        inputs: vec![a_id],
                        output: out_id,
                        backward_fn,
                    });
                }
                out_id
            }
        }
    }

    pub fn backward(&mut self, loss_id: usize) {
        match &self.device {
            Device::Cpu => {
                let loss_grad = self.tensors[loss_id].grad.as_cpu_mut();
                for g in loss_grad.iter_mut() {
                    *g = 1.0;
                }
            }
            Device::Gpu(_, stream) => {
                let grad_slice = match &self.tensors[loss_id].grad {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
                let f = self.functions.get("fill_f32").unwrap().clone();
                let n = grad_slice.len() as u64;
                let val = 1.0f32;
                let mut builder = stream.launch_builder(&f);
                builder.arg(grad_slice).arg(&val).arg(&n);
                unsafe { builder.launch(LaunchConfig::for_num_elems(n as u32)) }.unwrap();
            }
        }
        for node in self.tape.nodes.iter().rev() {
            (node.backward_fn)(&mut self.tensors);
        }
    }
}
