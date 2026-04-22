use crate::graph::{Graph};
use crate::tensor::{Device, Storage, f32_to_bf16u, bf16u_to_f32};
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

        // ensure gpu resident tensors are on the device
        for &pid in &resident {
            self.ensure_on_gpu(pid);
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
        let alloc_bytes = size * std::mem::size_of::<T>();
        
        // Pass `&Graph` explicitly so the borrow drops immediately after evaluation
        let check_budget = |g: &Graph| {
            g.vram_budget_bytes.map_or(true, |budget| {
                g.current_vram_usage() + alloc_bytes <= budget
            })
        };

        // 1. Initial attempt
        if check_budget(self) {
            if let Ok(slice) = stream.alloc_zeros::<T>(size) {
                return slice;
            }
        }

        // 2. Try again after dropping the cache pool
        self.vram_pool.clear();
        #[cfg(feature = "bf16")]
        self.vram_pool_bf16.clear();
        stream.synchronize().unwrap();

        if check_budget(self) {
            if let Ok(slice) = stream.alloc_zeros::<T>(size) {
                return slice;
            }
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

        // 3. Fallback: Evict candidates to CPU sequentially until we fit in the budget
        for id in candidate_ids {
            self.demote_tensor_to_cpu(id);
            stream.synchronize().unwrap();
            
            // Re-evaluate budget safely
            if check_budget(self) {
                if let Ok(slice) = stream.alloc_zeros::<T>(size) {
                    return slice;
                }
            }
        }

        self.print_vram_state("safe_alloc_zeros");
        let requested_mb = (size * std::mem::size_of::<T>()) as f64 / 1024.0 / 1024.0;
        panic!(
            "FATAL OOM: Exhausted VRAM even after full activation eviction! \n\
            Attempted to allocate {:.2} MB ({} elements).", 
            requested_mb, size
        );
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
                }
                #[cfg(feature = "bf16")]
                if !(self.uses_bf16_mixed_precision() && self.tensors[tensor_id].is_pooled) {
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
            if !cpu_grad.is_empty() { // Prevent zero-size allocations
                let cpu_grad_clone = cpu_grad.clone();
                let mut gpu_grad_slice = self.safe_alloc_zeros::<f32>(&stream, cpu_grad_clone.len());
                stream.memcpy_htod(&cpu_grad_clone, &mut gpu_grad_slice).unwrap();
                self.tensors[tensor_id].grad = Storage::Gpu(gpu_grad_slice);
            } else {
                println!("IS EMPTY TENSOR!");
            }
        }
    }

    pub fn demote_tensor_to_cpu(&mut self, tensor_id: usize) {
        let stream = match &self.device {
            Device::Gpu(_, s) => Some(s.clone()),
            Device::Cpu => None,
        };

        // ── 1. Demote Data Safely ──────────────────────────────────────────────
        let cpu_data_opt: Option<Vec<f32>> = match &self.tensors[tensor_id].data {
            Storage::Cpu(_) => None, // Already on CPU
            #[cfg(feature = "bf16")]
            Storage::CpuBf16(_) => None,
            Storage::Gpu(s) => {
                let st = stream.as_ref().expect("GPU storage without stream");
                Some(st.clone_dtoh(s).expect("demote: dtoh data failed"))
            }
            #[cfg(feature = "bf16")]
            Storage::GpuBf16(s) => {
                let st = stream.as_ref().expect("GPU storage without stream");
                let bf16 = st.clone_dtoh(s).expect("demote: dtoh BF16 data failed");
                Some(bf16.into_iter().map(bf16u_to_f32).collect())
            }
        };

        if let Some(cpu_f32) = cpu_data_opt {
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
        }

        // ── 2. Demote Grad Safely (Preserve computed gradients!) ───────────────
        if let Storage::Gpu(s) = &self.tensors[tensor_id].grad {
            let st = stream.as_ref().expect("GPU storage without stream");
            let cpu_grad = st.clone_dtoh(s).expect("demote: dtoh grad failed");
            self.tensors[tensor_id].grad = Storage::Cpu(cpu_grad);
        }

        if let Some(st) = stream {
            st.synchronize().expect("demote: stream sync failed");
        }
    }

    pub fn current_vram_usage(&self) -> usize {
        let mut used = 0;
        for t in &self.tensors {
            match &t.data {
                Storage::Gpu(s) => used += s.len() * 4,
                #[cfg(feature = "bf16")]
                Storage::GpuBf16(s) => used += s.len() * 2,
                _ => {}
            }
            match &t.grad {
                Storage::Gpu(s) => used += s.len() * 4,
                _ => {}
            }
        }
        for (size, blocks) in &self.vram_pool {
            used += size * 4 * blocks.len();
        }
        #[cfg(feature = "bf16")]
        for (size, blocks) in &self.vram_pool_bf16 {
            used += size * 2 * blocks.len();
        }
        used
    }

    /// Ensures that a tensor's gradient is allocated on the GPU.
    /// This should be called inside backward closures right before writing to a gradient.
    pub fn ensure_grad_allocated(&mut self, id: usize) {
        let size = {
            let t = &self.tensors[id];
            // Already allocated?
            if matches!(t.grad, Storage::Gpu(_)) { return; }
            t.shape.iter().product::<usize>()
        };

        let stream = match &self.device {
            Device::Gpu(_, s) => s.clone(),
            _ => panic!("ensure_grad_allocated only supported on GPU"),
        };

        // Pull from the appropriate pool
        let grad_slice = self.safe_alloc_zeros::<f32>(&stream, size);
        self.tensors[id].grad = Storage::Gpu(grad_slice);
    }

    pub fn print_vram_state(&self, context: &str) {
        if let Device::Gpu(ctx, _stream) = &self.device {
            let (total_free, _total) = ctx.mem_get_info().unwrap();
            let total = self.vram_budget_bytes.expect("No budget set.");
            let free = total_free.min(total);
            let used = total - free;
            let device_used = _total - total_free;

            let mut param_data_bytes = 0usize;
            let mut param_grad_bytes = 0usize;
            let mut pooled_data_bytes = 0usize;
            let mut pooled_grad_bytes = 0usize;

            for t in &self.tensors {
                let mut d = 0;
                match &t.data {
                    Storage::Gpu(s) => d = s.len() * 4,
                    #[cfg(feature = "bf16")]
                    Storage::GpuBf16(s) => d = s.len() * 2,
                    _ => {}
                }

                let mut g = 0;
                match &t.grad {
                    Storage::Gpu(s) => g = s.len() * 4,
                    _ => {}
                }

                if t.is_pooled {
                    pooled_data_bytes += d;
                    pooled_grad_bytes += g;
                } else {
                    param_data_bytes += d;
                    param_grad_bytes += g;
                }
            }

            let mut idle_pool_bytes = 0usize;
            for (size, blocks) in &self.vram_pool { idle_pool_bytes += size * 4 * blocks.len(); }
            #[cfg(feature = "bf16")]
            for (size, blocks) in &self.vram_pool_bf16 { idle_pool_bytes += size * 2 * blocks.len(); }

            let total_accounted = param_data_bytes + param_grad_bytes + pooled_data_bytes + pooled_grad_bytes + idle_pool_bytes;
            let unaccounted = device_used as i64 - total_accounted as i64;

            println!("--- VRAM STATE [{}] ---", context);
            println!("Driver Used:   {:>8.2} MB / {:.2} MB", used as f32 / 1024.0 / 1024.0, total as f32 / 1024.0 / 1024.0);
            println!("Params (D+G):  {:>8.2} MB (Data: {:.1}, Grad: {:.1})", (param_data_bytes+param_grad_bytes) as f32/1024./1024., param_data_bytes as f32/1024./1024., param_grad_bytes as f32/1024./1024.);
            println!("Pooled (D+G):  {:>8.2} MB (Data: {:.1}, Grad: {:.1})", (pooled_data_bytes+pooled_grad_bytes) as f32/1024./1024., pooled_data_bytes as f32/1024./1024., pooled_grad_bytes as f32/1024./1024.);
            println!("Idle Pool:     {:>8.2} MB", idle_pool_bytes as f32 / 1024.0 / 1024.0);
            println!("Unaccounted:   {:>8.2} MB (Fragmentation / Workspace / Overhead)", unaccounted as f32 / 1024.0 / 1024.0);
            
            // Efficiency warning for researchers
            if pooled_grad_bytes > (pooled_data_bytes / 2) && pooled_data_bytes > 0 {
                println!("⚠️ WARNING: High Pooled Grad/Data Ratio ({:.2}).", pooled_grad_bytes as f32 / pooled_data_bytes as f32);
                println!("   Intermediate activations are carrying gradients during forward pass.");
            }
            println!("---------------------------------");

            self.print_tensor_breakdown();
        }
    }

    pub fn print_tensor_breakdown(&self) {
        use std::collections::HashMap;
        
        struct Usage {
            data_bytes: usize,
            grad_bytes: usize,
            count: usize,
            is_pooled: bool,
        }

        let mut stats: HashMap<String, Usage> = HashMap::new();

        for t in &self.tensors {
            let mut d_bytes = 0;
            match &t.data {
                Storage::Gpu(s) => d_bytes += s.len() * 4,
                #[cfg(feature = "bf16")]
                Storage::GpuBf16(s) => d_bytes += s.len() * 2,
                _ => {}
            }
            
            let mut g_bytes = 0;
            match &t.grad {
                Storage::Gpu(s) => g_bytes += s.len() * 4,
                _ => {}
            }
            
            if d_bytes > 0 || g_bytes > 0 {
                let tag = if t.is_pooled { "[P]" } else { "[S]" };
                let identity = t.name.clone().unwrap_or_else(|| format!("ID {} {:?}", t.id, t.shape));
                let key = format!("{} {}", tag, identity);
                
                let entry = stats.entry(key).or_insert(Usage {
                    data_bytes: 0,
                    grad_bytes: 0,
                    count: 0,
                    is_pooled: t.is_pooled,
                });
                entry.data_bytes += d_bytes;
                entry.grad_bytes += g_bytes;
                entry.count += 1;
            }
        }

        let mut sorted: Vec<_> = stats.into_iter().collect();
        sorted.sort_by(|a, b| (b.1.data_bytes + b.1.grad_bytes).cmp(&(a.1.data_bytes + a.1.grad_bytes))); 

        println!("--- Top GPU Tensor Consumers ---");
        println!("  TOTAL MEM  |  DATA MEM  |  GRAD MEM  | G/D | COUNT | IDENTITY");
        for (name, usage) in sorted.into_iter().take(25) {
            let total_mb = (usage.data_bytes + usage.grad_bytes) as f32 / 1024.0 / 1024.0;
            let data_mb = usage.data_bytes as f32 / 1024.0 / 1024.0;
            let grad_mb = usage.grad_bytes as f32 / 1024.0 / 1024.0;
            
            let ratio = if usage.data_bytes > 0 {
                format!("{:.1}", grad_mb / data_mb)
            } else {
                "inf".to_string()
            };
            
            println!(
                "  {:>8.2} MB | {:>8.2} MB | {:>8.2} MB | {:>3} | {:>5} | {}",
                total_mb, data_mb, grad_mb, ratio, usage.count, name
            );
        }
        println!("---------------------------------");
    }
}
