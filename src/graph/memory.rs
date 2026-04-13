use crate::graph::{Graph, TapeNode, is_bf16};
use crate::safe_bf16_temp;
use crate::tensor::{Tensor, Device, Storage};
use cudarc::driver::{PushKernelArg, LaunchConfig, CudaSlice, CudaFunction};

use std::collections::HashSet;


// ── StreamingReport ────────────────────────────────────────────────────────────

/// Returned by `Graph::plan_streaming`. Describes what ended up where.
pub struct StreamingReport {
    /// Param tensor IDs kept in VRAM (data + grad + adam moments all GPU).
    pub resident_param_ids: Vec<usize>,
    /// Param tensor IDs moved to CPU RAM (data + grad + adam moments all CPU).
    pub streamed_param_ids: Vec<usize>,
    /// Total bytes of resident params (×4: data+grad+m+v per param).
    pub resident_bytes: usize,
    /// 
    pub resident_headroom_bytes: usize,
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
            "  VRAM-resident params:  {} tensors  ({:.2} GB × 4 = {:.2} GB VRAM) [+ {:.2} MB Headroom]",
            self.resident_param_ids.len(),
            gb(self.resident_bytes / 4),
            gb(self.resident_bytes),
            mb(self.resident_headroom_bytes)
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


impl Graph {

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
        // leave some headroom for calculations (~10%)
        let mut used = 0usize;//(vram_budget_bytes - budget_after_fixed) / 2;
        let headroom = used;

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
                Storage::GpuBf16(s) => {
                    let bf16 = stream
                        .clone_dtoh(s)
                        .expect("plan_streaming: dtoh BF16 weight failed");
                    bf16.into_iter()
                        .map(|b| f32::from_bits((b as u32) << 16))
                        .collect()
                }
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
            resident_headroom_bytes: headroom,
        }
    }

    /// Helper to safely allocate with an emergency pool flush
    pub fn safe_alloc_zeros<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        &mut self, 
        stream: &std::sync::Arc<cudarc::driver::CudaStream>, 
        size: usize
    ) -> cudarc::driver::CudaSlice<T> {
        // 1. Try standard allocation
        if let Ok(slice) = stream.alloc_zeros::<T>(size) {
            return slice;
        }

        // 2. Try flushing caches
        self.vram_pool.clear();
        #[cfg(feature = "bf16")]
        self.vram_pool_bf16.clear();
        stream.synchronize().unwrap();

        if let Ok(slice) = stream.alloc_zeros::<T>(size) {
            return slice;
        }

        // 3. EMERGENCY EVICTION: Offload activations to CPU RAM
        // We collect the IDs first to avoid borrowing `self` during the loop
        // We will not evict tensors that are currently used.
        let candidate_ids: Vec<usize> = self.tensors.iter()
            .filter(|t| {
                t.is_pooled && (
                    matches!(t.data, Storage::Gpu(_) | Storage::GpuBf16(_)) || 
                    matches!(t.grad, Storage::Gpu(_)) // 🚨 NEW: Catch stranded gradients!
                )
            })
            .filter(|t| !self.active_node_tensors.contains(&t.id))
            .map(|t| t.id)
            .collect();

        for id in candidate_ids {
            self.demote_tensor_to_cpu(id); // Use your existing demote logic
            stream.synchronize().unwrap(); // Ensure the memory is actually freed

            if let Ok(slice) = stream.alloc_zeros::<T>(size) {
                return slice;
            }
        }

        panic!("FATAL OOM: Exhausted VRAM even after full activation eviction!");
    }

    /// Ensures a tensor is on the GPU device
    pub fn ensure_on_gpu(&mut self, tensor_id: usize) {
        if !self.tensors[tensor_id].is_pooled {
            return; 
        }

        let (is_gpu, stream) = match &self.device {
            Device::Gpu(_, s) => (true, Some(s.clone())),
            _ => (false, None),
        };
        if !is_gpu { return; }
        let stream = stream.unwrap();

        // 1. Restore Data
        if let Storage::Cpu(cpu_data) = &self.tensors[tensor_id].data {
            let cpu_data_clone = cpu_data.clone();
            
            #[cfg(feature = "bf16")]
            if self.uses_bf16_mixed_precision() && self.tensors[tensor_id].is_pooled {
                // Restore as BF16
                let u16_data: Vec<u16> = cpu_data_clone.iter().map(|&f| (f.to_bits() >> 16) as u16).collect();
                let mut gpu_slice = self.safe_alloc_zeros::<u16>(&stream, u16_data.len());
                stream.memcpy_htod(&u16_data, &mut gpu_slice).unwrap();
                self.tensors[tensor_id].data = Storage::GpuBf16(gpu_slice);
            } else {
                // Restore as FP32
                let mut gpu_slice = self.safe_alloc_zeros::<f32>(&stream, cpu_data_clone.len());
                stream.memcpy_htod(&cpu_data_clone, &mut gpu_slice).unwrap();
                self.tensors[tensor_id].data = Storage::Gpu(gpu_slice);
            }
            
            #[cfg(not(feature = "bf16"))]
            {
                let mut gpu_slice = self.safe_alloc_zeros::<f32>(&stream, cpu_data_clone.len());
                stream.memcpy_htod(&cpu_data_clone, &mut gpu_slice).unwrap();
                self.tensors[tensor_id].data = Storage::Gpu(gpu_slice);
            }
        }

        // 2. Restore Gradients (Always FP32)
        if let Storage::Cpu(cpu_grad) = &self.tensors[tensor_id].grad {
            let cpu_grad_clone = cpu_grad.clone();
            let mut gpu_grad_slice = self.safe_alloc_zeros::<f32>(&stream, cpu_grad_clone.len());
            stream.memcpy_htod(&cpu_grad_clone, &mut gpu_grad_slice).unwrap();
            self.tensors[tensor_id].grad = Storage::Gpu(gpu_grad_slice);
        }
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
            Storage::GpuBf16(s) => {
                let st = stream.as_ref().expect("GPU storage without GPU stream");
                let bf16 = st
                    .clone_dtoh(s)
                    .expect("demote_tensor_to_cpu: dtoh BF16 data failed");
                bf16.into_iter()
                    .map(|b| f32::from_bits((b as u32) << 16))
                    .collect()
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

    pub fn print_vram_state(&self, context: &str) {
        if let Device::Gpu(ctx, stream) = &self.device {
            let (free, total) = ctx.mem_get_info().unwrap();
            let used = total - free;
            
            let mut active_data_mb = 0;
            let mut active_grad_mb = 0;
            let mut offloaded_mb = 0;

            for t in &self.tensors {
                match &t.data {
                    Storage::Gpu(s) => active_data_mb += (s.len() * 4) / 1024 / 1024,
                    #[cfg(feature = "bf16")]
                    Storage::GpuBf16(s) => active_data_mb += (s.len() * 2) / 1024 / 1024,
                    Storage::Cpu(s) => offloaded_mb += (s.len() * 4) / 1024 / 1024,
                }
                match &t.grad {
                    Storage::Gpu(s) => active_grad_mb += (s.len() * 4) / 1024 / 1024,
                    #[cfg(feature = "bf16")]
                    Storage::GpuBf16(s) => active_grad_mb += (s.len() * 2) / 1024 / 1024,
                    Storage::Cpu(s) => offloaded_mb += (s.len() * 4) / 1024 / 1024,
                }
            }

            println!("--- VRAM STATE [{}] ---", context);
            println!("Driver Used:   {:>5} MB / {} MB", used / 1024 / 1024, total / 1024 / 1024);
            println!("Data on GPU:   {:>5} MB", active_data_mb);
            println!("Grads on GPU:  {:>5} MB", active_grad_mb);
            println!("CPU Offloaded: {:>5} MB", offloaded_mb);
            println!("---------------------------------");
        }
    }

}
