use crate::graph::{Graph, TapeNode, is_bf16};
use crate::safe_bf16_temp;
use crate::tensor::{Tensor, Device, Storage, f32_to_bf16u, bf16u_to_f32};
use cudarc::driver::{PushKernelArg, LaunchConfig, CudaSlice, CudaFunction};
use std::collections::HashSet;

// ── StreamingReport ────────────────────────────────────────────────────────────

pub struct StreamingReport {
    pub resident_param_ids: Vec<usize>,
    pub streamed_param_ids: Vec<usize>,
    pub resident_bytes: usize,
    pub resident_headroom_bytes: usize,
    pub streamed_bytes: usize,
    pub streaming_headroom_bytes: usize,
}

impl std::fmt::Display for StreamingReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn gb(b: usize) -> f64 { b as f64 / 1024f64.powi(3) }
        fn mb(b: usize) -> f64 { b as f64 / 1024f64.powi(2) }
        writeln!(f, "=== Streaming Layout ===")?;
        writeln!(
            f,
            "  VRAM-resident params:  {} tensors  ({:.2} GB × 4 = {:.2} GB VRAM) [+ {:.2} MB Headroom]",
            self.resident_param_ids.len(),
            gb(self.resident_bytes / 4),
            gb(self.resident_bytes),
            mb(self.resident_headroom_bytes)
        )?;
        #[cfg(feature = "bf16")]
        writeln!(
            f,
            "  CPU-streamed params:   {} tensors  ({:.2} GB BF16 weights + {:.2} GB FP32 moments)",
            self.streamed_param_ids.len(),
            gb(self.streamed_bytes / 2),  // BF16 = 2 bytes per element
            gb(self.streamed_bytes * 3),  // moments still FP32
        )?;
        #[cfg(not(feature = "bf16"))]
        writeln!(
            f,
            "  CPU-streamed params:   {} tensors  ({:.2} GB weights + {:.2} GB moments)",
            self.streamed_param_ids.len(),
            gb(self.streamed_bytes),
            gb(self.streamed_bytes * 3),
        )?;
        writeln!(
            f,
            "  Peak streaming temp:   {:.0} MB  (one block at a time)",
            mb(self.streaming_headroom_bytes)
        )?;
        write!(f, "========================")
    }
}

// ─── helpers ──────────────────────────────────────────────────────────────────

/// True if this tensor should be compressed to CpuBf16 when streamed to RAM.
/// Only applies to 2D weight matrices (large); 1D norms stay FP32.
#[cfg(feature = "bf16")]
fn should_compress_to_bf16(shape: &[usize]) -> bool {
    shape.len() == 2
}

impl Graph {
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
        assert!(self.num_params > 0, "call mark_params() before plan_streaming()");

        let pinned_set: HashSet<usize> = pinned_ids.iter().cloned().collect();

        let pinned_vram: usize = pinned_ids
            .iter()
            .map(|&pid| self.tensors[pid].size() * 4 * 4)
            .sum();

        let budget_after_fixed = vram_budget_bytes
            .saturating_sub(activation_reserve_bytes)
            .saturating_sub(pinned_vram);

        let candidate_ids: Vec<usize> =
            (0..self.num_params).filter(|id| !pinned_set.contains(id)).collect();

        let mut resident: Vec<usize> = Vec::new();
        let mut streamed: Vec<usize> = Vec::new();
        let mut used = 0usize;
        let headroom = used;

        for pid in &candidate_ids {
            let cost = self.tensors[*pid].size() * 4 * 4;
            if used + cost <= budget_after_fixed {
                used += cost;
                resident.push(*pid);
            } else {
                streamed.push(*pid);
            }
        }

        let max_stream_bytes = streamed
            .iter()
            .map(|&pid| self.tensors[pid].size() * 4)
            .max()
            .unwrap_or(0);

        while max_stream_bytes > budget_after_fixed.saturating_sub(used) && !resident.is_empty() {
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

        let stream = match &self.device {
            Device::Gpu(_, s) => s.clone(),
            Device::Cpu => unreachable!(),
        };

        let mut streamed_bytes = 0usize;
        for &pid in &streamed {
            let size = self.tensors[pid].size();
            streamed_bytes += size * 4;

            // Fetch FP32 data from GPU -------------------------------------------
            let cpu_f32: Vec<f32> = match &self.tensors[pid].data {
                Storage::Gpu(s) => stream
                    .clone_dtoh(s)
                    .expect("plan_streaming: dtoh weight failed"),
                Storage::Cpu(_) => continue,
                #[cfg(feature = "bf16")]
                Storage::CpuBf16(_) => continue,
                #[cfg(feature = "bf16")]
                Storage::GpuBf16(s) => {
                    let bf16 = stream
                        .clone_dtoh(s)
                        .expect("plan_streaming: dtoh BF16 weight failed");
                    bf16.into_iter().map(bf16u_to_f32).collect()
                }
            };

            // Store as CpuBf16 if feature enabled and tensor is a 2D matrix ------
            #[cfg(feature = "bf16")]
            {
                if should_compress_to_bf16(&self.tensors[pid].shape) {
                    let bf16_data: Vec<u16> = cpu_f32.iter().map(|&f| f32_to_bf16u(f)).collect();
                    self.tensors[pid].data = Storage::CpuBf16(bf16_data);
                } else {
                    self.tensors[pid].data = Storage::Cpu(cpu_f32);
                }
            }
            #[cfg(not(feature = "bf16"))]
            {
                self.tensors[pid].data = Storage::Cpu(cpu_f32);
            }

            self.tensors[pid].grad = Storage::Cpu(vec![0.0; size]);
        }

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
            resident_headroom_bytes: headroom,
        }
    }

    pub fn safe_alloc_zeros<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        &mut self,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        size: usize,
    ) -> cudarc::driver::CudaSlice<T> {
        if let Ok(slice) = stream.alloc_zeros::<T>(size) {
            return slice;
        }

        self.vram_pool.clear();
        #[cfg(feature = "bf16")]
        self.vram_pool_bf16.clear();
        stream.synchronize().unwrap();

        if let Ok(slice) = stream.alloc_zeros::<T>(size) {
            return slice;
        }

        let protected_ids: HashSet<usize> = if self.active_node_tensors.is_empty() {
            let mut ids = HashSet::new();
            for n in &self.tape.nodes {
                ids.extend(n.inputs.iter().copied());
                ids.insert(n.output);
            }
            ids
        } else {
            self.active_node_tensors.iter().copied().collect()
        };

        #[cfg(feature = "bf16")]
        let candidate_ids: Vec<usize> = self.tensors.iter()
            .filter(|t| t.is_pooled && matches!(t.data, Storage::Gpu(_) | Storage::GpuBf16(_)))
            .filter(|t| !protected_ids.contains(&t.id))
            .map(|t| t.id)
            .collect();
        #[cfg(not(feature = "bf16"))]
        let candidate_ids: Vec<usize> = self.tensors.iter()
            .filter(|t| t.is_pooled && matches!(t.data, Storage::Gpu(_)))
            .filter(|t| !protected_ids.contains(&t.id))
            .map(|t| t.id)
            .collect();

        for id in candidate_ids {
            self.demote_tensor_to_cpu(id);
            stream.synchronize().unwrap();
            if let Ok(slice) = stream.alloc_zeros::<T>(size) {
                return slice;
            }
        }

        panic!("FATAL OOM: Exhausted VRAM even after full activation eviction!");
    }

    pub fn ensure_on_gpu(&mut self, tensor_id: usize) {
        let stream = match &self.device {
            Device::Gpu(_, s) => s.clone(),
            _ => return,
        };

        // ── Restore data ──────────────────────────────────────────────────────
        match &self.tensors[tensor_id].data {
            Storage::Cpu(cpu_data) => {
                let cpu_data_clone = cpu_data.clone();
                #[cfg(feature = "bf16")]
                if self.uses_bf16_mixed_precision() && self.tensors[tensor_id].is_pooled {
                    let u16_data: Vec<u16> = cpu_data_clone.iter()
                        .map(|&f| f32_to_bf16u(f))
                        .collect();
                    let mut gpu_slice = self.safe_alloc_zeros::<u16>(&stream, u16_data.len());
                    stream.memcpy_htod(&u16_data, &mut gpu_slice).unwrap();
                    self.tensors[tensor_id].data = Storage::GpuBf16(gpu_slice);
                    return; // grad handled below
                }
                let mut gpu_slice = self.safe_alloc_zeros::<f32>(&stream, cpu_data_clone.len());
                stream.memcpy_htod(&cpu_data_clone, &mut gpu_slice).unwrap();
                self.tensors[tensor_id].data = Storage::Gpu(gpu_slice);
            }
            #[cfg(feature = "bf16")]
            Storage::CpuBf16(bf16_data) => {
                // Streamed BF16 weight: decompress to FP32 for the GPU compute
                let f32_data: Vec<f32> = bf16_data.iter().map(|&b| bf16u_to_f32(b)).collect();
                let mut gpu_slice = self.safe_alloc_zeros::<f32>(&stream, f32_data.len());
                stream.memcpy_htod(&f32_data, &mut gpu_slice).unwrap();
                self.tensors[tensor_id].data = Storage::Gpu(gpu_slice);
            }
            _ => {} // already on GPU
        }

        // ── Restore gradients (always FP32) ───────────────────────────────────
        if let Storage::Cpu(cpu_grad) = &self.tensors[tensor_id].grad {
            let cpu_grad_clone = cpu_grad.clone();
            let mut gpu_grad_slice = self.safe_alloc_zeros::<f32>(&stream, cpu_grad_clone.len());
            stream.memcpy_htod(&cpu_grad_clone, &mut gpu_grad_slice).unwrap();
            self.tensors[tensor_id].grad = Storage::Gpu(gpu_grad_slice);
        }
    }

    pub fn demote_tensor_to_cpu(&mut self, tensor_id: usize) {
        let size = self.tensors[tensor_id].size();
        let stream = match &self.device {
            Device::Gpu(_, s) => Some(s.clone()),
            Device::Cpu => None,
        };

        let cpu_f32: Vec<f32> = match &self.tensors[tensor_id].data {
            Storage::Cpu(_) => return,
            #[cfg(feature = "bf16")]
            Storage::CpuBf16(_) => return, // already CPU
            Storage::Gpu(s) => {
                let st = stream.as_ref().expect("GPU storage without GPU stream");
                st.clone_dtoh(s).expect("demote: dtoh data failed")
            }
            #[cfg(feature = "bf16")]
            Storage::GpuBf16(s) => {
                let st = stream.as_ref().expect("GPU storage without GPU stream");
                let bf16 = st.clone_dtoh(s).expect("demote: dtoh BF16 data failed");
                bf16.into_iter().map(bf16u_to_f32).collect()
            }
        };

        // Compress if this is a large 2D weight and BF16 feature is on
        #[cfg(feature = "bf16")]
        {
            if should_compress_to_bf16(&self.tensors[tensor_id].shape) {
                let bf16_data: Vec<u16> = cpu_f32.iter().map(|&f| f32_to_bf16u(f)).collect();
                self.tensors[tensor_id].data = Storage::CpuBf16(bf16_data);
            } else {
                self.tensors[tensor_id].data = Storage::Cpu(cpu_f32);
            }
        }
        #[cfg(not(feature = "bf16"))]
        {
            self.tensors[tensor_id].data = Storage::Cpu(cpu_f32);
        }

        self.tensors[tensor_id].grad = Storage::Cpu(vec![0.0; size]);

        if let Some(st) = stream {
            st.synchronize().expect("demote: stream sync failed");
        }
    }

    pub fn print_vram_state(&self, context: &str) {
        if let Device::Gpu(ctx, _stream) = &self.device {
            let (free, total) = ctx.mem_get_info().unwrap();
            let used = total - free;

            let mut active_data_mb = 0usize;
            let mut active_grad_mb = 0usize;
            let mut offloaded_mb = 0usize;

            for t in &self.tensors {
                match &t.data {
                    Storage::Gpu(s) => active_data_mb += (s.len() * 4) / 1024 / 1024,
                    #[cfg(feature = "bf16")]
                    Storage::GpuBf16(s) => active_data_mb += (s.len() * 2) / 1024 / 1024,
                    Storage::Cpu(s) => offloaded_mb += (s.len() * 4) / 1024 / 1024,
                    #[cfg(feature = "bf16")]
                    Storage::CpuBf16(s) => offloaded_mb += (s.len() * 2) / 1024 / 1024,
                }
                match &t.grad {
                    Storage::Gpu(s) => active_grad_mb += (s.len() * 4) / 1024 / 1024,
                    #[cfg(feature = "bf16")]
                    Storage::GpuBf16(s) => active_grad_mb += (s.len() * 2) / 1024 / 1024,
                    Storage::Cpu(s) => offloaded_mb += (s.len() * 4) / 1024 / 1024,
                    #[cfg(feature = "bf16")]
                    Storage::CpuBf16(s) => offloaded_mb += (s.len() * 2) / 1024 / 1024,
                }
            }

            println!("--- VRAM STATE [{}] ---", context);
            println!("Driver Used:   {:>5} MB / {} MB", used / 1024 / 1024, total / 1024 / 1024);
            println!("Data on GPU:   {:>5} MB", active_data_mb);
            println!("Grads on GPU:  {:>5} MB", active_grad_mb);
            println!("CPU Offloaded: {:>5} MB  (BF16-compressed where applicable)", offloaded_mb);
            println!("---------------------------------");
        }
    }
}
