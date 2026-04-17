pub mod ops;
pub mod memory;

use crate::tape::{Tape, TapeNode};
use crate::tensor::{Device, Storage, Tensor};
use cudarc::driver::CudaSlice;
use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use std::collections::{HashMap, HashSet};
use std::env;


#[cfg(feature = "bf16")]
#[inline]
fn is_bf16(s: &crate::tensor::Storage) -> bool {
    matches!(s, crate::tensor::Storage::GpuBf16(_))
}
#[cfg(not(feature = "bf16"))]
#[inline]
fn is_bf16(_s: &crate::tensor::Storage) -> bool { false }


#[macro_export] macro_rules! safe_bf16_temp {
    ($self:ident, $id:expr, $size:expr, $stream:expr, $cast_fn:expr) => {{
        // 1. Check if it's BF16 (short immutable borrow, drops immediately)
        if is_bf16(&$self.tensors[$id].data) {
            
            // 2. Safely allocate the temp buffer (mutable borrow of self)
            let tmp = $self.safe_alloc_zeros::<f32>($stream, $size);
            
            // 3. Borrow the tensor again to run the cast
            if let Storage::GpuBf16(s) = &$self.tensors[$id].data {
                let n = $size as u64;
                let mut b = $stream.launch_builder($cast_fn);
                b.arg(s).arg(&tmp).arg(&n);
                unsafe { b.launch(LaunchConfig::for_num_elems($size as u32)) }.unwrap();
            }
            Some(tmp)
        } else {
            None
        }
    }};
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionMode {
    Fp32,
    #[cfg(feature = "bf16")]
    Bf16Mixed,
}

pub struct Graph {
    pub tensors: Vec<Tensor>,
    pub tape: Tape,
    pub device: Device,
    pub functions: HashMap<String, CudaFunction>,

    pub vram_pool: HashMap<usize, Vec<Storage>>,
    #[cfg(feature = "bf16")]
    pub vram_pool_bf16: HashMap<usize, Vec<cudarc::driver::CudaSlice<u16>>>,

    pub num_params: usize,

    pub no_grad: bool,
    pub precision_mode: PrecisionMode,

    /// Protects active node tensors from being evicted
    pub active_node_tensors: Vec<usize>,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new(Device::Cpu)
    }
}

impl Graph {
    #[cfg(feature = "bf16")]
    fn bf16_requested_by_env() -> bool {
        match env::var("EISEN_PRECISION") {
            Ok(v) => {
                let normalized = v.trim().to_ascii_lowercase();
                normalized == "bf16" || normalized == "auto"
            }
            Err(_) => true,
        }
    }

    #[cfg(feature = "bf16")]
    fn force_fp32_by_env() -> bool {
        match env::var("EISEN_FORCE_FP32") {
            Ok(v) => {
                let normalized = v.trim().to_ascii_lowercase();
                normalized == "1" || normalized == "true" || normalized == "yes"
            }
            Err(_) => false,
        }
    }

    #[cfg(feature = "bf16")]
    fn probe_bf16_matmul_kernel(
        device: &Device,
        functions: &HashMap<String, CudaFunction>,
    ) -> bool {
        let f = match functions.get("matmul_f32_bf16accum_f32") {
            Some(f) => f,
            None => return false,
        };
        let stream = match device {
            Device::Gpu(_, s) => s,
            Device::Cpu => return false,
        };

        let a = match stream.alloc_zeros::<f32>(1) {
            Ok(v) => v,
            Err(_) => return false,
        };
        let b = match stream.alloc_zeros::<f32>(1) {
            Ok(v) => v,
            Err(_) => return false,
        };
        let out = match stream.alloc_zeros::<f32>(1) {
            Ok(v) => v,
            Err(_) => return false,
        };

        let m = 1u64;
        let k = 1u64;
        let n = 1u64;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(f);
        builder.arg(&a).arg(&b).arg(&out).arg(&m).arg(&k).arg(&n);
        if unsafe { builder.launch(cfg) }.is_err() {
            return false;
        }
        stream.synchronize().is_ok()
    }

    pub fn new(device: Device) -> Self {
        let mut functions = HashMap::new();

        if let Device::Gpu(ctx, _) = &device {
            let ptx = include_str!(concat!(env!("OUT_DIR"), "/ops_f32.ptx"));
            let module = ctx
                .load_module(ptx.into())
                .expect("Failed to load f32 PTX module");

            #[allow(unused_mut)]
            let mut names = vec![
                "add_f32",
                "fill_f32",
                "scale_f32",
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
            for name in names {
                let f = module
                    .load_function(name)
                    .expect(&format!("Failed to load {} kernel", name));
                functions.insert(name.to_string(), f);
            }

            // BF16 specific kernels
            #[cfg(feature = "bf16")]
            {
                let ptx = include_str!(concat!(env!("OUT_DIR"), "/ops_bf16.ptx"));
                let module = ctx
                    .load_module(ptx.into())
                    .expect("Failed to load bf16 PTX module");

                names = vec![
                    "cast_f32_to_bf16",
                    "cast_bf16_to_f32",
                    "cast_bf16_to_f32_accumulate",
                    "matmul_bf16_f32",
                    "matmul_f32_bf16accum_f32",
                    "matmul_f32_bf16rhsaccum_f32",
                    "matmul_backward_a_bf16b_f32",
                    "bmm_f32_bf16accum_f32",
                    "gather_bf16_f32",
                    "rmsnorm_f32_bf16w",
                    "rmsnorm_backward_bf16w_f32",
                    "adamw_step_bf16mom_f32",
                    "adamw_step_bf16w_bf16mom_f32",
                    "add_bf16",
                    "add_bf16lhs_f32rhs_bf16out",
                    "accumulate_bf16out",
                    "mul_bf16",
                    "mul_bf16lhs_f32rhs_bf16out",
                    "mul_backward_bf16in_f32",
                    "mul_backward_bf16lhs_f32rhs",
                    "silu_bf16",
                    "silu_backward_bf16in_f32",
                    "softmax_bf16",
                    "softmax_backward_bf16in_f32",
                    "copy_bf16",
                    "transpose_0213_bf16",
                    "rope_bf16",
                    "rmsnorm_bf16",
                    "rmsnorm_backward_bf16in_f32",
                    "gather_bf16_bf16out",
                    "bmm_f32_bf16out",
                    "matmul_f32_bf16out",
                ];

                for name in names {
                    let f = module
                        .load_function(name)
                        .expect(&format!("Failed to load {} kernel", name));
                    functions.insert(name.to_string(), f);
                }
            }
        }

        #[cfg(feature = "bf16")]
        let precision_mode = {
            let requested = Self::bf16_requested_by_env() && !Self::force_fp32_by_env();
            if requested
                && matches!(device, Device::Gpu(_, _))
                && Self::probe_bf16_matmul_kernel(&device, &functions)
            {
                PrecisionMode::Bf16Mixed
            } else {
                PrecisionMode::Fp32
            }
        };
        #[cfg(not(feature = "bf16"))]
        let precision_mode = PrecisionMode::Fp32;

        Self {
            tensors: Vec::new(),
            tape: Tape::default(),
            device,
            functions,
            vram_pool: HashMap::new(),
            #[cfg(feature = "bf16")]
            vram_pool_bf16: HashMap::new(),

            num_params: 0,
            no_grad: false,
            precision_mode,

            active_node_tensors: Vec::new(),
        }
    }

    pub fn precision_mode(&self) -> PrecisionMode {
        self.precision_mode
    }

    pub fn uses_bf16_mixed_precision(&self) -> bool {
        #[cfg(feature = "bf16")]
        {
            self.precision_mode == PrecisionMode::Bf16Mixed
        }
        #[cfg(not(feature = "bf16"))]
        {
            false
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
            if t.is_pooled {
                let size = if t.shape.is_empty() { 1 } else { t.shape.iter().product() };

                // Return grad block (always FP32) to the main pool
                self.vram_pool.entry(size).or_default().push(t.grad);

                // Return data block to the correct pool based on its storage type
                match t.data {
                    #[cfg(feature = "bf16")]
                    Storage::GpuBf16(slice) => {
                        self.vram_pool_bf16.entry(size).or_default().push(slice);
                    }
                    other => {
                        self.vram_pool.entry(size).or_default().push(other);
                    }
                }
            }
            // non-pooled tensors drop here → cudaFree via RAII
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
                if let Device::Gpu(_ctx, stream) = &self.device {
                    stream
                        .memcpy_htod(host_data, gpu_slice)
                        .expect("Failed to copy weights from Host RAM to VRAM!");
                } else {
                    panic!("Graph device mismatch: Tensor is GPU but Graph is not.");
                }
            }
            #[cfg(feature = "bf16")]
            Storage::GpuBf16(gpu_slice) => {
                if let Device::Gpu(_ctx, stream) = &self.device {
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
        let size = if shape.is_empty() { 1 } else { shape.iter().product::<usize>() };
        let device = self.device.clone();

        // ── Data buffer: BF16 in Bf16Mixed mode, FP32 otherwise ───────────────
        #[cfg(feature = "bf16")]
        let data_storage: Storage = if self.uses_bf16_mixed_precision() {
            match &device {
                Device::Gpu(_, stream) => {
                    // 1. Try to pop from the cache (borrows self.vram_pool mutably, then drops)
                    let mut cached_slice = None;
                    if let Some(blocks) = self.vram_pool_bf16.get_mut(&size) {
                        cached_slice = blocks.pop();
                    }

                    // 2. Allocate if nothing was found (borrows self mutably, totally safe now!)
                    let slice = cached_slice.unwrap_or_else(|| self.safe_alloc_zeros::<u16>(stream, size));
                    Storage::GpuBf16(slice)
                }
                Device::Cpu => Storage::Cpu(vec![0.0; size]), // CPU always FP32
            }
        } else {
            // FP32 path — same as original
            if let Some(blocks) = self.vram_pool.get_mut(&size) {
                if let Some(block) = blocks.pop() {
                    block
                } else {
                    match &device {
                        Device::Cpu => Storage::Cpu(vec![0.0; size]),
                        Device::Gpu(_, stream) => Storage::Gpu(self.safe_alloc_zeros::<f32>(&stream, size)),
                    }
                }
            } else {
                match &device {
                    Device::Cpu => Storage::Cpu(vec![0.0; size]),
                    Device::Gpu(_, stream) => Storage::Gpu(self.safe_alloc_zeros::<f32>(&stream, size)),
                }
            }
        };

        #[cfg(not(feature = "bf16"))]
        let data_storage: Storage = {
            if let Some(blocks) = self.vram_pool.get_mut(&size) {
                if let Some(block) = blocks.pop() { block }
                else {
                    match &device {
                        Device::Cpu => Storage::Cpu(vec![0.0; size]),
                        Device::Gpu(_, stream) => Storage::Gpu(self.safe_alloc_zeros::<f32>(&stream, size)),
                    }
                }
            } else {
                match &device {
                    Device::Cpu => Storage::Cpu(vec![0.0; size]),
                    Device::Gpu(_, stream) => Storage::Gpu(self.safe_alloc_zeros::<f32>(&stream, size)),
                }
            }
        };

        // ── Grad buffer: always FP32 (optimizer stability requirement) ─────────
        let mut grad_storage = {
            let mut reused_block = None;
            if let Some(blocks) = self.vram_pool.get_mut(&size) {
                if let Some(block) = blocks.pop() {
                    // only reuse FP32 blocks for grad
                    if matches!(block, Storage::Gpu(_)) {
                        reused_block = Some(block);
                    } else {
                        // wrong type ended up here — discard and allocate fresh
                    }
                }
            }
            if let Some(block) = reused_block {
                block
            } else {
                match &device {
                    Device::Cpu => Storage::Cpu(vec![0.0; size]),
                    Device::Gpu(_, stream) => Storage::Gpu(self.safe_alloc_zeros::<f32>(&stream, size)),
                }
            }
        };

        // Zero the grad block before use
        match &device {
            Device::Cpu => {
                if let Storage::Cpu(v) = &mut grad_storage { v.fill(0.0); }
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

        let mut strides = vec![1usize; shape.len()];
        if !shape.is_empty() {
            for i in (1..shape.len()).rev() {
                strides[i - 1] = strides[i] * shape[i];
            }
        }

        let id = self.tensors.len();
        self.tensors.push(Tensor {
            id,
            shape,
            strides,
            data: data_storage,
            grad: grad_storage,
            device,
            name: None,
            is_pooled: true,
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

    #[cfg(feature = "bf16")]
    pub fn alloc_param_bf16(&mut self, shape: Vec<usize>, data: Vec<f32>) -> usize {
        let id = self.tensors.len();
        let size = if shape.is_empty() {
            1
        } else {
            shape.iter().product::<usize>()
        };
        let strides = Tensor::compute_strides(&shape);

        match &self.device {
            Device::Gpu(_, stream) => {
                let u16_data: Vec<u16> = data.iter().map(|&f| (f.to_bits() >> 16) as u16).collect();
                let d_data = stream
                    .clone_htod(u16_data.as_slice())
                    .expect("alloc_param_bf16: htod failed");
                let d_grad = stream
                    .alloc_zeros::<f32>(size)
                    .expect("OOM in alloc_param_bf16");
                self.tensors.push(Tensor {
                    id,
                    shape,
                    strides,
                    data: Storage::GpuBf16(d_data),
                    grad: Storage::Gpu(d_grad),
                    device: self.device.clone(),
                    name: None,
                    is_pooled: false,
                });
                id
            }
            Device::Cpu => self.alloc(shape, data),
        }
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




    pub fn backward(&mut self, loss_id: usize) {
        if self.no_grad {
            return;
        }

        // --- 1. INITIALIZE LOSS GRADIENT TO 1.0 ---
        match &self.device {
            Device::Cpu => {
                if let Storage::Cpu(g) = &mut self.tensors[loss_id].grad {
                    g.iter_mut().for_each(|x| *x = 1.0);
                }
            }
            Device::Gpu(_, stream) => {
                if let Storage::Gpu(g) = &mut self.tensors[loss_id].grad {
                    let host_ones = vec![1.0f32; g.len()];
                    stream.memcpy_htod(&host_ones, g).unwrap();
                }
            }
        }
        
        let (ctx, stream_opt) = match &self.device {
            Device::Gpu(ctx, s) => (Some(ctx.clone()), Some(s.clone())),
            _ => (None, None),
        };

        // --- 2. PREPARE VRAM FOR RAW CLOSURE ALLOCATIONS ---
        // The closures use raw `stream.alloc_zeros` and cannot access the pool.
        // We MUST flush the forward-pass pool back to the driver so the driver 
        // has space to fulfill those raw allocations.
        if let Some(stream) = &stream_opt {
            self.vram_pool.clear();
            #[cfg(feature = "bf16")]
            self.vram_pool_bf16.clear();
            stream.synchronize().unwrap();
        }

        // --- 3. EXECUTE THE TAPE ---
        let nodes = std::mem::take(&mut self.tape.nodes);

        for node in nodes.iter().rev() {
            // 🛡️ SHIELD: Protect the active tensors from being cannibalized
            self.active_node_tensors.clear();
            self.active_node_tensors.extend_from_slice(&node.inputs);
            self.active_node_tensors.push(node.output);

            // JIT Page-In
            for &input_id in &node.inputs {
                self.ensure_on_gpu(input_id);
            }
            self.ensure_on_gpu(node.output);

            // 🚀 PROACTIVE SCRATCH SPACE GENERATION
            // Ensure the CUDA driver has enough raw free memory for bf16_to_f32_temp
            if let Some(stream) = &stream_opt {
                let (free_vram, _) = ctx.clone().expect("Was not able to acquire context").mem_get_info().unwrap();
                let mut current_free = free_vram;
                let scratch_budget = 1024 * 1024 * 1024; // 128 MB safety buffer for temporaries

                if current_free < scratch_budget {
                    // Evict tensors (that are not shielded) until we reach the budget
                    #[cfg(feature = "bf16")]
                    let candidate_ids: Vec<usize> = self.tensors.iter()
                        .filter(|t| t.is_pooled && matches!(t.data, Storage::Gpu(_) | Storage::GpuBf16(_)))
                        .filter(|t| !self.active_node_tensors.contains(&t.id))
                        .map(|t| t.id)
                        .collect();

                    #[cfg(not(feature = "bf16"))]
                    let candidate_ids: Vec<usize> = self.tensors.iter()
                        .filter(|t| t.is_pooled && matches!(t.data, Storage::Gpu(_)))
                        .filter(|t| !self.active_node_tensors.contains(&t.id))
                        .map(|t| t.id)
                        .collect();

                    for id in candidate_ids {
                        self.demote_tensor_to_cpu(id);
                        stream.synchronize().unwrap();
                        
                        let (new_free, _) = ctx.clone().expect("Could not acquire context!").mem_get_info().unwrap();
                        current_free = new_free;
                        if current_free >= scratch_budget {
                            break; // We have enough scratch space!
                        }
                    }

                    if current_free < scratch_budget {
                        println!("Did not manage to clear enough space!!!");
                        self.print_vram_state("backward pass error");
                    }
                } 
                //else {
                //    println!("Currently available: {}, attempting to reclaim: {}", current_free, scratch_budget);
                //    self.print_vram_state("backward pass");
                //}
            }

            // Execute the backward closure safely
            (node.backward_fn)(&mut self.tensors);
        }

        // --- 4. CLEANUP ---
        self.active_node_tensors.clear();
        self.tape.nodes = nodes;
    }
}

