use crate::graph::Graph;
use crate::tensor::{Device, Storage};
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use std::collections::HashMap;

pub struct AdamW {
    pub params: Vec<usize>,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    t: usize,

    // CPU moment storage — used for CPU-homed params regardless of graph device
    m_cpu: HashMap<usize, Vec<f32>>,
    v_cpu: HashMap<usize, Vec<f32>>,

    // GPU moment storage — used for VRAM-resident params
    m_gpu: HashMap<usize, CudaSlice<f32>>,
    v_gpu: HashMap<usize, CudaSlice<f32>>,
    #[cfg(feature = "bf16")]
    m_gpu_bf16: HashMap<usize, CudaSlice<u16>>,
    #[cfg(feature = "bf16")]
    v_gpu_bf16: HashMap<usize, CudaSlice<u16>>,
    grad_clip_norm: Option<f32>,
    last_grad_norm: f32,
    last_grad_scale: f32,
}

impl AdamW {
    pub fn new(params: Vec<usize>, lr: f32) -> Self {
        Self {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            t: 0,
            m_cpu: HashMap::new(),
            v_cpu: HashMap::new(),
            m_gpu: HashMap::new(),
            v_gpu: HashMap::new(),
            #[cfg(feature = "bf16")]
            m_gpu_bf16: HashMap::new(),
            #[cfg(feature = "bf16")]
            v_gpu_bf16: HashMap::new(),
            grad_clip_norm: None,
            last_grad_norm: 0.0,
            last_grad_scale: 1.0,
        }
    }

    pub fn set_grad_clip_norm(&mut self, max_norm: f32) {
        if max_norm > 0.0 {
            self.grad_clip_norm = Some(max_norm);
        }
    }

    pub fn last_grad_norm(&self) -> f32 {
        self.last_grad_norm
    }

    pub fn last_grad_scale(&self) -> f32 {
        self.last_grad_scale
    }

    /// Pre-allocates moment buffers in VRAM. Call this before the training loop
    /// to prevent VRAM fragmentation and out-of-memory errors during the first step.
    pub fn init_moments(&mut self, g: &mut Graph) {
        let stream_opt = match &g.device {
            Device::Gpu(_, s) => Some(s.clone()),
            Device::Cpu => None,
        };

        for &p_id in &self.params {
            let size = g.tensors[p_id].shape.iter().product::<usize>();

            #[cfg(feature = "bf16")]
            let is_gpu = matches!(&g.tensors[p_id].data, Storage::Gpu(_) | Storage::GpuBf16(_));
            #[cfg(not(feature = "bf16"))]
            let is_gpu = matches!(&g.tensors[p_id].data, Storage::Gpu(_));

            if is_gpu {
                let stream = stream_opt.as_ref().unwrap();

                #[cfg(feature = "bf16")]
                if g.uses_bf16_mixed_precision() {
                    if !self.m_gpu_bf16.contains_key(&p_id) {
                        self.m_gpu_bf16
                            .insert(p_id, g.safe_alloc_zeros::<u16>(stream, size));
                        self.v_gpu_bf16
                            .insert(p_id, g.safe_alloc_zeros::<u16>(stream, size));
                    }
                } else {
                    if !self.m_gpu.contains_key(&p_id) {
                        self.m_gpu
                            .insert(p_id, g.safe_alloc_zeros::<f32>(stream, size));
                        self.v_gpu
                            .insert(p_id, g.safe_alloc_zeros::<f32>(stream, size));
                    }
                }
                #[cfg(not(feature = "bf16"))]
                {
                    if !self.m_gpu.contains_key(&p_id) {
                        self.m_gpu
                            .insert(p_id, g.safe_alloc_zeros::<f32>(stream, size));
                        self.v_gpu
                            .insert(p_id, g.safe_alloc_zeros::<f32>(stream, size));
                    }
                }
            } else {
                self.m_cpu.entry(p_id).or_insert_with(|| vec![0.0; size]);
                self.v_cpu.entry(p_id).or_insert_with(|| vec![0.0; size]);
            }
        }
    }

    fn maybe_clip_gradients(&mut self, g: &mut Graph) {
        let Some(max_norm) = self.grad_clip_norm else {
            self.last_grad_norm = 0.0;
            self.last_grad_scale = 1.0;
            return;
        };

        let stream_opt = match &g.device {
            Device::Gpu(_, s) => Some(s.clone()),
            Device::Cpu => None,
        };

        let mut total_sq = 0.0f64;
        for &p_id in &self.params {
            match &g.tensors[p_id].grad {
                Storage::Cpu(v) => {
                    for &x in v {
                        let xf = x as f64;
                        total_sq += xf * xf;
                    }
                }
                Storage::Gpu(s) => {
                    let stream = stream_opt
                        .as_ref()
                        .expect("GPU grad but graph has no GPU stream");
                    let host = stream.clone_dtoh(s).expect("AdamW clip: dtoh failed");
                    for &x in &host {
                        let xf = x as f64;
                        total_sq += xf * xf;
                    }
                }
                #[cfg(feature = "bf16")]
                Storage::GpuBf16(_) => panic!("AdamW clip: BF16 grad storage is unsupported"),
            }
        }

        let grad_norm = (total_sq as f32).sqrt();
        let scale = if grad_norm > max_norm {
            max_norm / (grad_norm + 1e-6)
        } else {
            1.0
        };

        self.last_grad_norm = grad_norm;
        self.last_grad_scale = scale;

        if scale >= 1.0 {
            return;
        }

        let scale_fn_opt = stream_opt
            .as_ref()
            .map(|_| g.functions.get("scale_f32").unwrap().clone());

        for &p_id in &self.params {
            let size = g.tensors[p_id].shape.iter().product::<usize>();
            match &mut g.tensors[p_id].grad {
                Storage::Cpu(v) => {
                    for x in v.iter_mut() {
                        *x *= scale;
                    }
                }
                Storage::Gpu(s) => {
                    let stream = stream_opt
                        .as_ref()
                        .expect("GPU grad but graph has no GPU stream");
                    let f = scale_fn_opt.as_ref().unwrap().clone();
                    let mut builder = stream.launch_builder(&f);
                    let n = size as u64;
                    builder.arg(&*s).arg(&scale).arg(&n);
                    unsafe { builder.launch(LaunchConfig::for_num_elems(size as u32)) }.unwrap();
                }
                #[cfg(feature = "bf16")]
                Storage::GpuBf16(_) => panic!("AdamW clip: BF16 grad storage is unsupported"),
            }
        }
    }

    /// Zero all parameter gradients.
    ///
    /// Dispatches per-tensor: GPU-resident params use the `fill_f32` kernel;
    /// CPU-homed params (streaming) fill their Vec directly. The graph device
    /// is only used to obtain the CUDA stream for GPU params.
    pub fn zero_grad(&self, g: &mut Graph) {
        // We need the stream handle for GPU params. Clone it once to avoid
        // repeated borrows of g.device inside the loop.
        let stream_opt = match &g.device {
            Device::Gpu(_, s) => Some(s.clone()),
            Device::Cpu => None,
        };
        let fill_fn_opt = stream_opt
            .as_ref()
            .map(|_| g.functions.get("fill_f32").unwrap().clone());

        for &p_id in &self.params {
            let size = g.tensors[p_id].shape.iter().product::<usize>();

            // Inspect storage variant to decide which path to take
            let is_gpu_grad = matches!(&g.tensors[p_id].grad, Storage::Gpu(_));

            if is_gpu_grad {
                let stream = stream_opt.as_ref().expect("GPU grad but no GPU stream");
                let f = fill_fn_opt.as_ref().unwrap().clone();
                let grad = match &g.tensors[p_id].grad {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
                let n = size as u64;
                let val = 0.0f32;
                let mut builder = stream.launch_builder(&f);
                builder.arg(grad).arg(&val).arg(&n);
                unsafe { builder.launch(LaunchConfig::for_num_elems(size as u32)) }.unwrap();
            } else {
                // CPU-homed param (streaming) — fill directly
                let tensor = &mut g.tensors[p_id];
                match &mut tensor.grad {
                    Storage::Cpu(v) => v.fill(0.0),
                    _ => unreachable!(),
                }
            }
        }
    }

    /// Global gradient clipping by L2 norm across all parameters.
    ///
    /// Returns `(grad_norm, clip_coef_applied)` where `clip_coef_applied <= 1.0`.
    pub fn clip_grad_norm(&self, g: &mut Graph, max_norm: f32) -> (f32, f32) {
        let mut sum_sq = 0.0_f64;

        let stream_opt = match &g.device {
            Device::Gpu(_, s) => Some(s.clone()),
            Device::Cpu => None,
        };

        for &p_id in &self.params {
            match &g.tensors[p_id].grad {
                Storage::Cpu(v) => {
                    for &x in v {
                        let xf = x as f64;
                        sum_sq += xf * xf;
                    }
                }
                Storage::Gpu(s) => {
                    let stream = stream_opt.as_ref().expect("GPU grad but no GPU stream");
                    let host = stream
                        .clone_dtoh(s)
                        .expect("Failed to copy gradient from VRAM");
                    for &x in &host {
                        let xf = x as f64;
                        sum_sq += xf * xf;
                    }
                }
                #[cfg(feature = "bf16")]
                Storage::GpuBf16(_) => unreachable!("Gradient buffers are FP32 in Eisen"),
            };
        }

        let grad_norm = (sum_sq.sqrt()) as f32;
        let clip_coef = if grad_norm > 0.0 {
            (max_norm / (grad_norm + 1e-6)).min(1.0)
        } else {
            1.0
        };

        if clip_coef < 1.0 {
            let scale_fn_opt = stream_opt
                .as_ref()
                .map(|_| g.functions.get("scale_f32").unwrap().clone());

            for &p_id in &self.params {
                let size = g.tensors[p_id].shape.iter().product::<usize>();
                match &mut g.tensors[p_id].grad {
                    Storage::Cpu(v) => {
                        for val in v.iter_mut() {
                            *val *= clip_coef;
                        }
                    }
                    Storage::Gpu(s) => {
                        let stream = stream_opt.as_ref().expect("GPU grad but no GPU stream");
                        let f = scale_fn_opt.as_ref().unwrap().clone();
                        let n = size as u64;
                        let mut builder = stream.launch_builder(&f);
                        builder.arg(&*s).arg(&clip_coef).arg(&n);
                        unsafe { builder.launch(LaunchConfig::for_num_elems(size as u32)) }
                            .unwrap();
                    }
                    #[cfg(feature = "bf16")]
                    Storage::GpuBf16(_) => unreachable!("Gradient buffers are FP32 in Eisen"),
                }
            }
        }

        (grad_norm, clip_coef)
    }

    /// Perform one AdamW optimizer step.
    ///
    /// For each parameter, the update path is chosen based on the tensor's
    /// actual storage, not the graph device:
    ///
    ///   GPU storage → fused `adamw_step_f32` kernel (zero PCIe traffic)
    ///   CPU storage → in-place Vec update  (used by streamed weights)
    ///
    /// Moment buffers are lazily initialised on the first step and live in
    /// the same memory space as the parameter (GPU moments for GPU params,
    /// CPU Vecs for CPU-homed params).
    pub fn step(&mut self, g: &mut Graph) {
        self.maybe_clip_gradients(g);
        self.t += 1;
        let bc1 = 1.0_f32 / (1.0 - self.beta1.powi(self.t as i32));
        let bc2 = 1.0_f32 / (1.0 - self.beta2.powi(self.t as i32));

        let stream_opt = match &g.device {
            Device::Gpu(_, s) => Some(s.clone()),
            Device::Cpu => None,
        };
        let adamw_fn_opt = stream_opt
            .as_ref()
            .map(|_| g.functions.get("adamw_step_f32").unwrap().clone());
        #[cfg(feature = "bf16")]
        let adamw_bf16mom_fn_opt = if g.uses_bf16_mixed_precision() {
            stream_opt
                .as_ref()
                .map(|_| g.functions.get("adamw_step_bf16mom_f32").unwrap().clone())
        } else {
            None
        };
        #[cfg(feature = "bf16")]
        let adamw_bf16w_fn_opt = if g.uses_bf16_mixed_precision() {
            stream_opt.as_ref().map(|_| {
                g.functions
                    .get("adamw_step_bf16w_bf16mom_f32")
                    .unwrap()
                    .clone()
            })
        } else {
            None
        };

        for &p_id in &self.params {
            let size = g.tensors[p_id].shape.iter().product::<usize>();
            #[cfg(feature = "bf16")]
            let is_gpu = matches!(&g.tensors[p_id].data, Storage::Gpu(_) | Storage::GpuBf16(_));
            #[cfg(not(feature = "bf16"))]
            let is_gpu = matches!(&g.tensors[p_id].data, Storage::Gpu(_));

            if is_gpu {
                // ── Fused GPU kernel path ──────────────────────────────────────
                let stream = stream_opt.as_ref().expect("GPU param but no GPU stream");
                let n = size as u64;
                #[cfg(feature = "bf16")]
                if adamw_bf16mom_fn_opt.is_some() {
                    // Lazy-init BF16 moment buffers (if init_moments wasn't called)
                    if !self.m_gpu_bf16.contains_key(&p_id) {
                        self.m_gpu_bf16
                            .insert(p_id, g.safe_alloc_zeros::<u16>(stream, size));
                        self.v_gpu_bf16
                            .insert(p_id, g.safe_alloc_zeros::<u16>(stream, size));
                    }
                    let m_s = self.m_gpu_bf16.get(&p_id).unwrap();
                    let v_s = self.v_gpu_bf16.get(&p_id).unwrap();
                    let grads = match &g.tensors[p_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    match &g.tensors[p_id].data {
                        Storage::Gpu(weights) => {
                            let f = adamw_bf16mom_fn_opt.as_ref().unwrap().clone();
                            let mut builder = stream.launch_builder(&f);
                            builder
                                .arg(weights)
                                .arg(grads)
                                .arg(m_s)
                                .arg(v_s)
                                .arg(&self.lr)
                                .arg(&self.beta1)
                                .arg(&self.beta2)
                                .arg(&self.eps)
                                .arg(&self.weight_decay)
                                .arg(&bc1)
                                .arg(&bc2)
                                .arg(&n);
                            unsafe { builder.launch(LaunchConfig::for_num_elems(size as u32)) }
                                .unwrap();
                        }
                        Storage::GpuBf16(weights) => {
                            let f = adamw_bf16w_fn_opt.as_ref().unwrap().clone();
                            let mut builder = stream.launch_builder(&f);
                            builder
                                .arg(weights)
                                .arg(grads)
                                .arg(m_s)
                                .arg(v_s)
                                .arg(&self.lr)
                                .arg(&self.beta1)
                                .arg(&self.beta2)
                                .arg(&self.eps)
                                .arg(&self.weight_decay)
                                .arg(&bc1)
                                .arg(&bc2)
                                .arg(&n);
                            unsafe { builder.launch(LaunchConfig::for_num_elems(size as u32)) }
                                .unwrap();
                        }
                        _ => unreachable!(),
                    }
                } else {
                    // Lazy-init FP32 moment buffers (if init_moments wasn't called)
                    if !self.m_gpu.contains_key(&p_id) {
                        self.m_gpu
                            .insert(p_id, g.safe_alloc_zeros::<f32>(stream, size));
                        self.v_gpu
                            .insert(p_id, g.safe_alloc_zeros::<f32>(stream, size));
                    }
                    let weights = match &g.tensors[p_id].data {
                        Storage::Gpu(s) => s,
                        _ => panic!("FP32 AdamW kernel requires FP32 GPU weights"),
                    };
                    let grads = match &g.tensors[p_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let f = adamw_fn_opt.as_ref().unwrap().clone();
                    let m_s = self.m_gpu.get(&p_id).unwrap();
                    let v_s = self.v_gpu.get(&p_id).unwrap();
                    let mut builder = stream.launch_builder(&f);
                    builder
                        .arg(weights)
                        .arg(grads)
                        .arg(m_s)
                        .arg(v_s)
                        .arg(&self.lr)
                        .arg(&self.beta1)
                        .arg(&self.beta2)
                        .arg(&self.eps)
                        .arg(&self.weight_decay)
                        .arg(&bc1)
                        .arg(&bc2)
                        .arg(&n);
                    unsafe { builder.launch(LaunchConfig::for_num_elems(size as u32)) }.unwrap();
                }
                #[cfg(not(feature = "bf16"))]
                {
                    // Lazy-init FP32 moment buffers
                    if !self.m_gpu.contains_key(&p_id) {
                        self.m_gpu
                            .insert(p_id, g.safe_alloc_zeros::<f32>(stream, size));
                        self.v_gpu
                            .insert(p_id, g.safe_alloc_zeros::<f32>(stream, size));
                    }
                    let weights = match &g.tensors[p_id].data {
                        Storage::Gpu(s) => s,
                        _ => panic!("FP32 AdamW kernel requires FP32 GPU weights"),
                    };
                    let grads = match &g.tensors[p_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let f = adamw_fn_opt.as_ref().unwrap().clone();
                    let m_s = self.m_gpu.get(&p_id).unwrap();
                    let v_s = self.v_gpu.get(&p_id).unwrap();
                    let mut builder = stream.launch_builder(&f);
                    builder
                        .arg(weights)
                        .arg(grads)
                        .arg(m_s)
                        .arg(v_s)
                        .arg(&self.lr)
                        .arg(&self.beta1)
                        .arg(&self.beta2)
                        .arg(&self.eps)
                        .arg(&self.weight_decay)
                        .arg(&bc1)
                        .arg(&bc2)
                        .arg(&n);
                    unsafe { builder.launch(LaunchConfig::for_num_elems(size as u32)) }.unwrap();
                }
            } else {
                // ── CPU path (streaming weights, or pure CPU graph) ────────────
                let m = self.m_cpu.entry(p_id).or_insert_with(|| vec![0.0; size]);
                let v = self.v_cpu.entry(p_id).or_insert_with(|| vec![0.0; size]);

                let tensor = &mut g.tensors[p_id];

                // Borrow grad and weight robustly via field splitting
                let (data_storage, grad_storage) = (&mut tensor.data, &tensor.grad);
                let weight_data = match data_storage {
                    Storage::Cpu(vec) => vec,
                    _ => unreachable!(),
                };
                let grad_data = match grad_storage {
                    Storage::Cpu(vec) => vec,
                    _ => unreachable!(),
                };

                for i in 0..size {
                    let grad = grad_data[i];
                    let weight = weight_data[i];

                    // Decoupled weight decay (applied to weight, not gradient)
                    let decayed = weight * (1.0 - self.lr * self.weight_decay);

                    m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * grad;
                    v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * grad * grad;

                    let m_hat = m[i] * bc1;
                    let v_hat = v[i] * bc2;

                    weight_data[i] = decayed - self.lr * m_hat / (v_hat.sqrt() + self.eps);
                }
            }
        }
    }
}
