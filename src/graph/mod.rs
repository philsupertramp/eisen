//! # Core Computation Graph and Execution Engine
//!
//! This module defines the [`Graph`] structure, which serves as the central runtime
//! for tensor allocation, dynamic VRAM pooling, automatic differentiation (autodiff),
//! and GPU kernel dispatch.
//!
//! ## Key Architectural Concepts
//!
//! - **Dynamic Memory Pooling (`vram_pool` / `vram_pool_bf16`)**:
//!   Instead of invoking expensive driver allocations (`cudaMalloc` / `cudaFree`) on every
//!   intermediate operation, deallocated buffers are cached by byte size and reused
//!   across training steps.
//! - **Forward-Only Activation Tracking**:
//!   Tensors allocated specifically for the forward pass via [`Graph::alloc_forward`] are
//!   tracked and automatically reclaimed once the backward pass or activation cleanup finishes.
//! - **Mixed-Precision Execution**:
//!   Supports seamless switching between pure single-precision (FP32) and bfloat16 mixed-precision
//!   (BF16 data buffers with FP32 gradient accumulation for numerical stability).
//! - **Memory Eviction & Demotion**:
//!   When running close to hardware limits or an explicit `EISEN_VRAM_BUDGET_MB`, inactive
//!   intermediate tensors are demoted to host RAM to preserve scratchpad memory for CUDA kernels.

pub mod memory;
pub mod ops;

use crate::tape::{Tape, TapeNode};
use crate::tensor::{Device, Storage, Tensor};
use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use std::collections::HashMap;
use std::env;

/// Parses an unsigned integer environment variable, falling back to a default value
/// if the variable is unset or contains an invalid integer string.
///
/// # Arguments
///
/// * `name` - Name of the environment variable to query.
/// * `default` - Fallback value returned on missing variable or parse error.
///
/// # Returns
///
/// The parsed `usize` value, or `default`.
fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

/// Checks whether a tensor storage container holds bfloat16 elements on the GPU.
///
/// # Arguments
///
/// * `s` - Reference to the [`Storage`] enum to inspect.
///
/// # Returns
///
/// `true` if storage is [`Storage::GpuBf16`], otherwise `false`.
#[cfg(feature = "bf16")]
#[inline]
fn is_bf16(s: &crate::tensor::Storage) -> bool {
    matches!(s, crate::tensor::Storage::GpuBf16(_))
}

/// Checks whether a tensor storage container holds bfloat16 elements on the GPU.
///
/// Always returns `false` when the `bf16` Cargo feature is disabled at compile time.
#[cfg(not(feature = "bf16"))]
#[inline]
fn is_bf16(_s: &crate::tensor::Storage) -> bool {
    false
}

/// Allocates a temporary FP32 GPU buffer and casts a BF16 tensor into it if required.
///
/// If the tensor indexed by `$id` is stored in BF16 format, this macro allocates a zeroed
/// temporary FP32 device buffer, launches the specified `$cast_fn` CUDA kernel to populate it,
/// and returns `Some(tmp)`. If the tensor is already FP32, it returns `None`.
///
/// # Macro Parameters
///
/// * `$self` - Mutable reference to the [`Graph`].
/// * `$id` - Target tensor ID in `$self.tensors`.
/// * `$size` - Number of elements to allocate and cast.
/// * `$stream` - CUDA stream handle for asynchronous kernel launch.
/// * `$cast_fn` - Pointer or reference to the loaded CUDA casting kernel function.
#[macro_export]
macro_rules! safe_bf16_temp {
    ($self:ident, $id:expr, $size:expr, $stream:expr, $cast_fn:expr) => {{
        if is_bf16(&$self.tensors[$id].data) {
            let tmp = $self.safe_alloc_zeros::<f32>($stream, $size);
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

/// Numerical precision modes supported by the computation graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionMode {
    /// Full single-precision (FP32) for both weights and activations.
    Fp32,
    /// Mixed precision using 16-bit brain floating point (BF16) for activations/weights
    /// and 32-bit single precision (FP32) for gradients and optimizer states.
    #[cfg(feature = "bf16")]
    Bf16Mixed,
}

/// The central execution engine and state container for tensor operations and autodiff.
///
/// `Graph` coordinates tensor allocations, retains the operation tape for reverse-mode
/// automatic differentiation, maintains reusable GPU buffer pools, and dispatches CUDA kernels.
pub struct Graph {
    /// Contiguous registry of all tensors managed by this graph.
    pub tensors: Vec<Tensor>,

    /// Tape recording the history of forward operations for automatic differentiation.
    pub tape: Tape,

    /// Target compute device (CPU or CUDA GPU).
    pub device: Device,

    /// Map of compiled CUDA kernel functions keyed by kernel symbol name.
    pub functions: HashMap<String, CudaFunction>,

    /// Cache of free FP32 GPU storage buffers keyed by element capacity.
    pub vram_pool: HashMap<usize, Vec<Storage>>,

    /// Cache of free BF16 GPU device slices keyed by element capacity.
    #[cfg(feature = "bf16")]
    pub vram_pool_bf16: HashMap<usize, Vec<cudarc::driver::CudaSlice<u16>>>,

    /// Tensor IDs of temporary activations created during the forward pass.
    ///
    /// Reclaimed into the pool automatically upon calling [`Graph::clear_activations`].
    pub forward_tensor_ids: Vec<usize>,

    /// Number of model parameter tensors registered before forward execution.
    ///
    /// Tensors at indices `< num_params` are treated as persistent model weights.
    pub num_params: usize,

    /// Flag to disable tape recording and gradient tracking during inference or evaluation.
    pub no_grad: bool,

    /// Active precision configuration (FP32 vs BF16 Mixed Precision).
    pub precision_mode: PrecisionMode,

    /// Tensor IDs currently locked by the active backward tape node.
    ///
    /// Serves as an eviction pin to prevent offloading critical inputs/outputs during OOM recovery.
    pub active_node_tensors: Vec<usize>,

    /// User-defined ceiling on total GPU VRAM usage in bytes.
    ///
    /// If specified, triggers memory eviction when consumption approaches this threshold.
    pub vram_budget_bytes: Option<usize>,

    /// Minimum scratchpad VRAM buffer in megabytes reserved for kernel execution.
    pub scratch_budget_mb: Option<usize>,
}

impl Default for Graph {
    /// Creates a default `Graph` instance targeting the host CPU.
    fn default() -> Self {
        Self::new(Device::Cpu)
    }
}

impl Graph {
    /// Inspects the `EISEN_PRECISION` environment variable to determine if BF16 is requested.
    ///
    /// Defaults to `true` if unset, or if set to `"bf16"` or `"auto"`.
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

    /// Inspects the `EISEN_FORCE_FP32` environment variable to check if FP32 is explicitly forced.
    ///
    /// Returns `true` if set to `"1"`, `"true"`, or `"yes"`.
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

    /// Verifies hardware capability for BF16 matrix multiplication by launching a $1 \times 1$ test kernel.
    ///
    /// # Arguments
    ///
    /// * `device` - Compute device to test. Must be [`Device::Gpu`].
    /// * `functions` - Compiled CUDA function map containing `"matmul_f32_bf16accum_f32"`.
    ///
    /// # Returns
    ///
    /// `true` if the device successfully launches and synchronizes the kernel, otherwise `false`.
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

    /// Initializes a new computation graph configured for the specified device.
    ///
    /// If initialized with a CUDA device:
    /// - Compiles and loads default FP32 and BF16 PTX modules from the build output directory.
    /// - Probes the hardware for BF16 mixed-precision support.
    /// - Reads memory budget environment variables (`EISEN_VRAM_BUDGET_MB`, `EISEN_ACTIVATION_RESERVE_MB`).
    ///
    /// # Arguments
    ///
    /// * `device` - Target compute device ([`Device::Cpu`] or [`Device::Gpu`]).
    ///
    /// # Panics
    ///
    /// Panics if the GPU PTX module fails to load or required CUDA kernels are missing.
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
                "transpose_0213_f32",
                "transpose_0213_backward_f32",
                "gather_f32",
                "gather_backward_f32",
                "matmul_f32",
                "rmsnorm_f32",
                "rmsnorm_backward_f32",
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
                    "scale_bf16",
                    "matmul_bf16_f32",
                    "matmul_f32_bf16accum_f32",
                    "matmul_f32_bf16rhsaccum_f32",
                    "matmul_backward_a_bf16b_f32",
                    "matmul_trans_b_bf16",
                    "matmul_trans_a_bf16",
                    // matmul pure bf16
                    "matmul_bf16",
                    "matmul_backward_a_bf16",
                    "matmul_backward_b_bf16",
                    "bmm_bf16",
                    "bmm_f32_bf16accum_f32",
                    "bmm_backward_a_bf16",
                    "bmm_backward_a_f32_to_bf16",
                    "bmm_backward_a_transb_bf16",
                    "bmm_backward_a_transb_f32go_f32b_bf16ga",
                    "bmm_backward_b_transb_bf16a_f32go_f32gb",
                    "bmm_backward_b_bf16",
                    "bmm_backward_b_transb_bf16",
                    "bmm_backward_b_bf16a_f32go_f32gb",
                    "gather_bf16_f32",
                    "gather_bf16_bf16out",
                    "gather_backward_bf16",
                    // RMS Norm
                    "rmsnorm_f32_bf16w",
                    "rmsnorm_backward_bf16w_f32",
                    "rmsnorm_bf16",
                    "rmsnorm_backward_bf16in_f32",
                    "rmsnorm_backward_bf16",
                    "adamw_step_bf16mom_f32",
                    "adamw_step_bf16w_bf16mom_f32",
                    "adamw_step_bf16mom",
                    "add_bf16",
                    "add_bf16lhs_f32rhs_bf16out",
                    "accumulate_bf16out",
                    "accumulate_bf16",
                    "mul_bf16",
                    "mul_bf16lhs_f32rhs_bf16out",
                    "mul_backward_bf16in_f32",
                    "mul_backward_bf16lhs_f32rhs",
                    "mul_backward_bf16",
                    "silu_bf16",
                    "silu_backward_bf16in_f32",
                    "silu_backward_bf16",
                    "sum_bf16",
                    "sum_backward_bf16",
                    "max_bf16",
                    "max_backward_bf16",
                    "softmax_bf16",
                    "softmax_backward_bf16in_f32",
                    "softmax_backward_bf16",
                    "copy_bf16",
                    "transpose_0213_bf16",
                    "transpose_0213_backward_bf16",
                    "rope_bf16",
                    "rope_backward_bf16",
                    "bmm_f32_bf16out",
                    "matmul_f32_bf16out",
                    "repeat_kv_bf16",
                    "repeat_kv_backward_bf16",
                    "cross_entropy_bf16",
                    "cross_entropy_backward_bf16",
                    "fill_bf16",
                    "matmul_f32_bf16accum_f32",
                    "matmul_f32_bf16rhsaccum_f32",
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

        let vram_budget_bytes = env::var("EISEN_VRAM_BUDGET_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|mb| mb * 1024 * 1024);

        let scratch_budget_mb: usize = env_usize("EISEN_ACTIVATION_RESERVE_MB", 64);

        Self {
            tensors: Vec::new(),
            tape: Tape::default(),
            device,
            functions,
            vram_pool: HashMap::new(),
            #[cfg(feature = "bf16")]
            vram_pool_bf16: HashMap::new(),
            forward_tensor_ids: Vec::new(),
            num_params: 0,
            no_grad: false,
            precision_mode,
            active_node_tensors: Vec::new(),
            vram_budget_bytes,
            scratch_budget_mb: Some(scratch_budget_mb),
        }
    }

    /// Returns the active numerical precision mode configured for this graph.
    pub fn precision_mode(&self) -> PrecisionMode {
        self.precision_mode
    }

    /// Assigns a human-readable debug name to a tensor.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique index of the target tensor.
    /// * `name` - Descriptive label (e.g., `"attention_weights"` or `"layer_0_norm"`).
    pub fn name_tensor(&mut self, id: usize, name: &str) {
        self.tensors[id].name = Some(name.to_string());
    }

    /// Returns `true` if the graph is executing under bfloat16 mixed-precision mode.
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

    // ---------------------------------------------------------------------
    // Forward-only allocation helpers
    // ---------------------------------------------------------------------

    /// Allocates an uninitialized temporary activation tensor for the forward pass.
    ///
    /// Reclaims memory from the pool when possible. The resulting ID is tracked in
    /// `forward_tensor_ids` and will be recycled into the pool during [`Graph::clear_activations`].
    ///
    /// # Arguments
    ///
    /// * `shape` - Dimensions of the tensor to allocate.
    ///
    /// # Returns
    ///
    /// The unique tensor ID of the allocated activation tensor.
    pub fn alloc_forward(&mut self, shape: Vec<usize>) -> usize {
        let id = self.alloc_pooled(shape);
        self.forward_tensor_ids.push(id);
        id
    }

    /// Allocates a temporary forward activation tensor initialized with host data.
    ///
    /// Convenience wrapper combining [`Graph::alloc_forward`] and [`Graph::load_tensor_data`].
    ///
    /// # Arguments
    ///
    /// * `shape` - Dimensions of the tensor to allocate.
    /// * `host_data` - Flattened floating-point values to copy into the tensor.
    ///
    /// # Returns
    ///
    /// The unique tensor ID of the allocated activation tensor.
    pub fn alloc_forward_with_data(&mut self, shape: Vec<usize>, host_data: &Vec<f32>) -> usize {
        let id = self.alloc_forward(shape);
        self.load_tensor_data(id, &*host_data);
        id
    }

    // ---------------------------------------------------------------------
    // Existing allocation helpers
    // ---------------------------------------------------------------------

    /// Allocates a tensor using cached memory from the internal VRAM pool whenever available.
    ///
    /// - In mixed-precision mode, allocates a BF16 data buffer and an FP32 gradient buffer.
    /// - In full-precision mode, allocates FP32 buffers for both data and gradient.
    /// - If no cached buffer of the exact size exists in the pool, allocates fresh device memory.
    ///
    /// # Arguments
    ///
    /// * `shape` - Dimensions of the tensor.
    ///
    /// # Returns
    ///
    /// The unique tensor ID for the newly allocated pooled tensor.
    pub fn alloc_pooled(&mut self, shape: Vec<usize>) -> usize {
        let size = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        let device = self.device.clone();

        // ── Data buffer: BF16 in Bf16Mixed mode, FP32 otherwise ───────────────
        #[cfg(feature = "bf16")]
        let data_storage: Storage = if self.uses_bf16_mixed_precision() {
            match &device {
                Device::Gpu(_, stream) => {
                    let mut cached_slice = None;
                    if let Some(blocks) = self.vram_pool_bf16.get_mut(&size) {
                        cached_slice = blocks.pop();
                    }

                    let slice =
                        cached_slice.unwrap_or_else(|| self.safe_alloc_zeros::<u16>(&stream, size));
                    Storage::GpuBf16(slice)
                }
                Device::Cpu => Storage::CpuBf16(vec![0u16; size]),
            }
        } else {
            // FP32 path
            if let Some(blocks) = self.vram_pool.get_mut(&size) {
                if let Some(block) = blocks.pop() {
                    block
                } else {
                    match &device {
                        Device::Cpu => Storage::Cpu(vec![0.0; size]),
                        Device::Gpu(_, stream) => {
                            Storage::Gpu(self.safe_alloc_zeros::<f32>(&stream, size))
                        }
                    }
                }
            } else {
                match &device {
                    Device::Cpu => Storage::Cpu(vec![0.0; size]),
                    Device::Gpu(_, stream) => {
                        Storage::Gpu(self.safe_alloc_zeros::<f32>(&stream, size))
                    }
                }
            }
        };

        #[cfg(not(feature = "bf16"))]
        let data_storage: Storage = {
            if let Some(blocks) = self.vram_pool.get_mut(&size) {
                if let Some(block) = blocks.pop() {
                    block
                } else {
                    match &device {
                        Device::Cpu => Storage::Cpu(vec![0.0; size]),
                        Device::Gpu(_, stream) => {
                            Storage::Gpu(self.safe_alloc_zeros::<f32>(&stream, size))
                        }
                    }
                }
            } else {
                match &device {
                    Device::Cpu => Storage::Cpu(vec![0.0; size]),
                    Device::Gpu(_, stream) => {
                        Storage::Gpu(self.safe_alloc_zeros::<f32>(&stream, size))
                    }
                }
            }
        };

        // ── Grad buffer: always FP32 (optimizer stability requirement) ─────────
        let grad_storage = match &data_storage {
            #[cfg(feature = "bf16")]
            Storage::Gpu(_) | Storage::GpuBf16(_) => {
                match &device {
                    Device::Gpu(_, stream) => Storage::Gpu(self.safe_alloc_zeros::<f32>(&stream, size)),
                    Device::Cpu => Storage::Cpu(vec![0.0; size]),
                }
            }
            #[cfg(not(feature = "bf16"))]
            Storage::Gpu(_) => {
                match &device {
                    Device::Gpu(_, stream) => Storage::Gpu(self.safe_alloc_zeros::<f32>(&stream, size)),
                    Device::Cpu => Storage::Cpu(vec![0.0; size]),
                }
            }
            #[cfg(feature = "bf16")]
            Storage::Cpu(_) | Storage::CpuBf16(_) => Storage::Cpu(vec![0.0; size]),
            #[cfg(not(feature = "bf16"))]
            Storage::Cpu(_) => Storage::Cpu(vec![0.0; size]),
        };

        let strides = Tensor::compute_strides(&shape);
        let id = self.tensors.len();
        self.tensors.push(Tensor {
            id,
            shape,
            data: data_storage,
            grad: grad_storage,
            name: None,
            is_pooled: true,
            is_param: false,
            device: device.clone(),
            strides,
        });
        id
    }

    /// Allocates a pooled tensor and initializes its contents from a host slice.
    ///
    /// # Arguments
    ///
    /// * `shape` - Dimensions of the tensor.
    /// * `host_data` - Host float buffer whose element count must match `shape.iter().product()`.
    ///
    /// # Returns
    ///
    /// The unique tensor ID of the initialized tensor.
    pub fn alloc_pooled_with_data(&mut self, shape: Vec<usize>, host_data: &Vec<f32>) -> usize {
        let vec_id = self.alloc_pooled(shape);
        self.load_tensor_data(vec_id, &*host_data);
        vec_id
    }

    /// Directly allocates a non-pooled, unmanaged tensor initialized with host data.
    ///
    /// Bypasses the VRAM cache pool. Typically used for standalone inputs or evaluation buffers.
    ///
    /// # Arguments
    ///
    /// * `shape` - Dimensions of the tensor.
    /// * `data` - Initial flat host data vector.
    ///
    /// # Returns
    ///
    /// The unique tensor ID of the newly created tensor.
    pub fn alloc(&mut self, shape: Vec<usize>, data: Vec<f32>) -> usize {
        let id = self.tensors.len();
        self.tensors.push(Tensor::new(id, shape, data, self.device.clone()));
        id
    }

    /// Transfers host floating-point data into an existing tensor's storage buffer.
    ///
    /// Handles Host-to-Device memory copy and performs automatic FP32 $\to$ BF16 bitwise conversion
    /// when targeting BF16 storage backends.
    ///
    /// # Arguments
    ///
    /// * `id` - ID of the tensor to populate.
    /// * `host_data` - Slice of single-precision floats matching the tensor's total element count.
    ///
    /// # Panics
    ///
    /// - Panics if `host_data.len()` does not equal the tensor's total number of elements.
    /// - Panics on device mismatch or failed CUDA memory transfer.
    pub fn load_tensor_data(&mut self, id: usize, host_data: &[f32]) {
        let tensor = &mut self.tensors[id];
        let size = if tensor.shape.is_empty() { 1 } else { tensor.shape.iter().product::<usize>() };
        assert_eq!(size, host_data.len(), "Shape mismatch: Tensor {} expects {} elements, but got {}.", id, size, host_data.len());
        match &mut tensor.data {
            Storage::Cpu(cpu_vec) => {
                cpu_vec.copy_from_slice(host_data);
            }
            Storage::Gpu(gpu_slice) => {
                if let Device::Gpu(_ctx, stream) = &self.device {
                    stream.memcpy_htod(host_data, gpu_slice).expect("Failed to copy weights from Host RAM to VRAM!");
                } else {
                    panic!("Graph device mismatch: Tensor is GPU but Graph is not.");
                }
            }
            #[cfg(feature = "bf16")]
            Storage::CpuBf16(cpu_vec) => {
                if let Device::Gpu(_ctx, _stream) = &self.device {
                    let u16_data: Vec<u16> = host_data.iter().map(|&f| (f.to_bits() >> 16) as u16).collect();
                    cpu_vec.copy_from_slice(&u16_data);
                } else {
                    panic!("Graph device mismatch: Tensor is GPU but Graph is not.");
                }
            }
            #[cfg(feature = "bf16")]
            Storage::GpuBf16(gpu_slice) => {
                if let Device::Gpu(_ctx, stream) = &self.device {
                    let u16_data: Vec<u16> = host_data.iter().map(|&f| (f.to_bits() >> 16) as u16).collect();
                    stream.memcpy_htod(u16_data.as_slice(), gpu_slice).expect("Failed to copy BF16 weights from Host RAM to VRAM!");
                } else {
                    panic!("Graph device mismatch: Tensor is GPU but Graph is not.");
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // Miscellaneous helper methods
    // ---------------------------------------------------------------------

    /// Clears the autodiff tape and reclaims all intermediate activations allocated during the forward pass.
    ///
    /// Pops tensors added after parameter initialization back into their respective memory pools,
    /// recycles all `forward_tensor_ids`, and resets the forward activation registry.
    pub fn clear_activations(&mut self) {
        self.tape.nodes.clear();
        self.restore_save_point(self.num_params);
        // Free forward tensors
        for id in &self.forward_tensor_ids {
            if let Some(t) = self.tensors.get_mut(*id) {
                // Move the block back into the pool if it is on GPU
                if let Storage::Gpu(slice) = &t.data {
                    self.vram_pool.entry(t.shape.iter().product()).or_default().push(Storage::Gpu(slice.clone()));
                }
            }
        }
        self.forward_tensor_ids.clear();
    }

    /// Locks in the current number of allocated tensors as model parameters.
    ///
    /// Must be invoked after instantiating all learnable weights and biases, and prior to
    /// beginning the forward pass. Protects parameters from eviction and automatic cleanup.
    pub fn mark_params(&mut self) {
        self.num_params = self.tensors.len();
    }

    /// Captures the current total count of allocated tensors as a restoration checkpoint.
    ///
    /// # Returns
    ///
    /// The current number of tensors in the graph.
    pub fn mark_save_point(&self) -> usize {
        self.tensors.len()
    }

    /// Pops and returns all pooled tensors allocated after the specified checkpoint index to the VRAM pool.
    ///
    /// # Arguments
    ///
    /// * `save_point` - Index representing the maximum tensor ID to retain.
    pub fn restore_save_point(&mut self, save_point: usize) {
        while self.tensors.len() > save_point {
            let t = self.tensors.pop().unwrap();
            if t.is_pooled {
                let size = if t.shape.is_empty() {
                    1
                } else {
                    t.shape.iter().product()
                };

                if matches!(t.grad, Storage::Gpu(_)) {
                    self.vram_pool.entry(size).or_default().push(t.grad);
                }

                match t.data {
                    #[cfg(feature = "bf16")]
                    Storage::GpuBf16(slice) => {
                        self.vram_pool_bf16.entry(size).or_default().push(slice);
                    }
                    Storage::Gpu(slice) => {
                        self.vram_pool
                            .entry(size)
                            .or_default()
                            .push(Storage::Gpu(slice));
                    }
                    _ => {}
                }
            }
        }
    }

    /// Allocates a persistent model parameter tensor in BF16 format directly in GPU VRAM.
    ///
    /// Converts the provided FP32 host data into BF16 bit-patterns before uploading to device memory.
    ///
    /// # Arguments
    ///
    /// * `shape` - Tensor dimensions.
    /// * `data` - Initial floating-point parameter weights.
    ///
    /// # Returns
    ///
    /// The unique tensor ID of the newly allocated parameter.
    ///
    /// # Panics
    ///
    /// Panics if the current graph device is not [`Device::Gpu`].
    #[cfg(feature = "bf16")]
    pub fn alloc_param_bf16(&mut self, shape: Vec<usize>, data: Vec<f32>) -> usize {
        let id = self.tensors.len();
        let size = if shape.is_empty() {
            1
        } else {
            shape.iter().product::<usize>()
        };
        let strides = Tensor::compute_strides(&shape);
        let device = self.device.clone();

        match device {
            Device::Gpu(_, stream) => {
                let u16_data: Vec<u16> = data.iter().map(|&f| (f.to_bits() >> 16) as u16).collect();
                let d_data = stream
                    .clone_htod(u16_data.as_slice())
                    .expect("alloc_param_bf16: htod failed");
                let d_grad = self.safe_alloc_zeros::<u16>(&stream, size);
                self.tensors.push(Tensor {
                    id,
                    shape,
                    strides,
                    data: Storage::GpuBf16(d_data),
                    grad: Storage::GpuBf16(d_grad),
                    device: self.device.clone(),
                    name: None,
                    is_pooled: false,
                    is_param: true,
                });
                id
            }
            _ => panic!("Only supports GPU!"),
        }
    }

    /// Allocates a tensor whose primary residence remains host CPU RAM, even when the graph device is a GPU.
    ///
    /// Intended for parameter streaming and offloading weight matrices that are staged
    /// into VRAM only during execution of their respective layers.
    ///
    /// # Arguments
    ///
    /// * `shape` - Tensor dimensions.
    /// * `data` - Host floating-point values.
    ///
    /// # Returns
    ///
    /// The unique tensor ID of the CPU-resident tensor.
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
            device: self.device.clone(),
            name: None,
            is_pooled: false,
            is_param: false,
        });
        id
    }

    // --- Helper Methods ---

    /// Borrows the host CPU floating-point data buffer of a tensor.
    ///
    /// # Panics
    ///
    /// Panics if the tensor's data storage is not [`Storage::Cpu`].
    pub fn get_data(&self, tensor_id: usize) -> &Vec<f32> {
        self.tensors[tensor_id].data.as_cpu()
    }

    /// Borrows the host CPU floating-point gradient buffer of a tensor.
    ///
    /// # Panics
    ///
    /// Panics if the tensor's gradient storage is not [`Storage::Cpu`].
    pub fn get_grad(&self, tensor_id: usize) -> &Vec<f32> {
        self.tensors[tensor_id].grad.as_cpu()
    }

    /// Synchronously downloads gradient data from the device to host memory as an FP32 vector.
    ///
    /// Automatically performs device-to-host memory transfers and bitwise conversion from BF16
    /// when the underlying gradient is stored in 16-bit float format.
    ///
    /// # Arguments
    ///
    /// * `tensor_id` - ID of the tensor whose gradient should be downloaded.
    ///
    /// # Returns
    ///
    /// A newly allocated `Vec<f32>` containing the gradient values.
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
            #[cfg(feature = "bf16")]
            Storage::CpuBf16(s) => {
                let (_, _stream) = match &self.device {
                    Device::Gpu(_, s) => (None::<f32>, s),
                    _ => unreachable!(),
                };
                s.into_iter()
                    .map(|b| f32::from_bits((*b as u32) << 16))
                    .collect()
            }
        }
    }

    /// Executes reverse-mode automatic differentiation starting from the specified scalar loss tensor.
    ///
    /// # Execution Pipeline
    ///
    /// 1. Initializes the loss tensor's gradient to $1.0$ using a fast asynchronous CUDA fill kernel.
    /// 2. Iterates backward through recorded tape nodes in reverse topological order.
    /// 3. Inspects current VRAM usage against `vram_budget_bytes` and `scratch_budget_mb`,
    ///    dynamically demoting unpinned pooled tensors to CPU memory if scratch space is depleted.
    /// 4. Ensures the current node's input and output tensors reside in GPU memory prior to backward evaluation.
    /// 5. Executes the registered backward closure for each operation to accumulate upstream gradients.
    ///
    /// If [`Graph::no_grad`] is set to `true`, this function immediately returns without computing gradients.
    ///
    /// # Arguments
    ///
    /// * `loss_id` - Tensor ID representing the objective loss (typically a scalar).
    pub fn backward(&mut self, loss_id: usize) {
        if self.no_grad {
            return;
        }
        self.ensure_grad_allocated(loss_id);

        // --- 1. INITIALIZE LOSS GRADIENT TO 1.0 ---
        match &self.device {
            Device::Cpu => match &mut self.tensors[loss_id].grad {
                Storage::Cpu(g) => g.iter_mut().for_each(|x| *x = 1.0),
                #[cfg(feature = "bf16")]
                Storage::CpuBf16(g) => g.iter_mut().for_each(|x| *x = 0x3f80),
                _ => {}
            },
            Device::Gpu(_, stream) => {
                match &mut self.tensors[loss_id].grad {
                    Storage::Gpu(g) => {
                        let f = self.functions.get("fill_f32").unwrap().clone();
                        let val = 1.0f32;
                        let n = g.len() as u64;
                        let mut b = stream.launch_builder(&f);
                        let num_elems = g.len() as u32;
                        b.arg(g).arg(&val).arg(&n);
                        unsafe { b.launch(LaunchConfig::for_num_elems(num_elems)) }.unwrap();
                    }
                    #[cfg(feature = "bf16")]
                    Storage::GpuBf16(g) => {
                        let f = self.functions.get("fill_bf16").unwrap().clone();
                        let val = 0x3f80u16; // bf16(1.0)
                        let n = g.len() as u64;
                        let mut b = stream.launch_builder(&f);
                        let num_elems = g.len() as u32;
                        b.arg(g).arg(&val).arg(&n);
                        unsafe { b.launch(LaunchConfig::for_num_elems(num_elems)) }.unwrap();
                    }
                    _ => {}
                }
            }
        }

        let nodes = std::mem::take(&mut self.tape.nodes);
        let (ctx, stream_opt) = match &self.device {
            Device::Gpu(ctx, s) => (Some(ctx.clone()), Some(s.clone())),
            _ => (None, None),
        };

        for node in nodes.iter().rev() {
            self.active_node_tensors.clear();
            self.active_node_tensors.extend_from_slice(&node.inputs);
            self.active_node_tensors.push(node.output);

            // Lazy eviction logic without global pipeline halts
            if let Some(context) = &ctx {
                let scratch_budget = self.scratch_budget_mb.unwrap_or_else(|| 64) * 1024 * 1024;

                self.ensure_grad_allocated(node.output);
                for &input_id in &node.inputs {
                    self.ensure_grad_allocated(input_id);
                }

                // Calculate free VRAM based on the user budget if provided, else fall back to driver info
                let mut free_vram = if let Some(budget) = self.vram_budget_bytes {
                    budget.saturating_sub(self.current_vram_usage())
                } else {
                    context.mem_get_info().unwrap().0
                };

                if free_vram < scratch_budget {
                    // Only target tensors that are ACTUALLY currently taking up VRAM
                    #[cfg(feature = "bf16")]
                    let candidate_ids: Vec<usize> = self
                        .tensors
                        .iter()
                        .filter(|t| t.is_pooled && !self.active_node_tensors.contains(&t.id))
                        .filter(|t| {
                            matches!(t.data, Storage::Gpu(_) | Storage::GpuBf16(_))
                                || matches!(t.grad, Storage::Gpu(_))
                        })
                        .map(|t| t.id)
                        .collect();

                    #[cfg(not(feature = "bf16"))]
                    let candidate_ids: Vec<usize> = self
                        .tensors
                        .iter()
                        .filter(|t| t.is_pooled && !self.active_node_tensors.contains(&t.id))
                        .filter(|t| {
                            matches!(t.data, Storage::Gpu(_)) || matches!(t.grad, Storage::Gpu(_))
                        })
                        .map(|t| t.id)
                        .collect();

                    for id in candidate_ids {
                        self.demote_tensor_to_cpu(id);
                        if let Some(stream) = &stream_opt {
                            stream.synchronize().unwrap();
                        }

                        free_vram = if let Some(budget) = self.vram_budget_bytes {
                            budget.saturating_sub(self.current_vram_usage())
                        } else {
                            context.mem_get_info().unwrap().0
                        };
                        if free_vram >= scratch_budget {
                            break;
                        }
                    }

                    if free_vram < scratch_budget {
                        println!("Did not manage to clear enough space!!!");
                        self.print_vram_state("backward pass error");
                    }
                }
            }

            // Ensure tensors required for this backward step reside in GPU memory
            self.ensure_on_gpu(node.output);
            for &input_id in &node.inputs {
                self.ensure_on_gpu(input_id);
            }

            if let Some(stream) = &stream_opt {
                stream.synchronize().unwrap_or_else(|err| {
                    panic!("backward post-cast synchronize failed: {:?}", err)
                });
            }

            // Execute the backward closure safely
            (node.backward_fn)(&mut self.tensors);

            if let Some(stream) = &stream_opt {
                stream.synchronize().unwrap_or_else(|err| {
                    panic!("backward post-call synchronize failed: {:?}", err)
                });
            }
        }

        // --- 4. CLEANUP ---
        self.active_node_tensors.clear();
        self.tape.nodes = nodes;
    }
}
