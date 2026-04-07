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
                    total_sq += v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>();
                }
                Storage::Gpu(s) => {
                    let stream = stream_opt
                        .as_ref()
                        .expect("GPU grad but graph has no GPU stream");
                    let host = stream.clone_dtoh(s).expect("AdamW clip: dtoh failed");
                    total_sq += host.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>();
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

        for &p_id in &self.params {
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
                    let mut host = stream.clone_dtoh(s).expect("AdamW clip: dtoh failed");
                    for x in host.iter_mut() {
                        *x *= scale;
                    }
                    stream
                        .memcpy_htod(host.as_slice(), s)
                        .expect("AdamW clip: htod failed");
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
                let grad = g.tensors[p_id].grad.as_cpu_mut();
                grad.fill(0.0);
            }
        }
    }

    /// Global gradient clipping by L2 norm across all parameters.
    ///
    /// Returns `(grad_norm, clip_coef_applied)` where `clip_coef_applied <= 1.0`.
    pub fn clip_grad_norm(&self, g: &mut Graph, max_norm: f32) -> (f32, f32) {
        let mut sum_sq = 0.0_f64;

        for &p_id in &self.params {
            let grad_cpu: Vec<f32> = match &g.tensors[p_id].grad {
                Storage::Cpu(v) => v.clone(),
                Storage::Gpu(s) => {
                    let stream = match &g.tensors[p_id].device {
                        Device::Gpu(_, stream) => stream,
                        Device::Cpu => unreachable!("GPU grad must belong to GPU tensor"),
                    };
                    stream
                        .clone_dtoh(s)
                        .expect("Failed to copy gradient from VRAM")
                }
                #[cfg(feature = "bf16")]
                Storage::GpuBf16(_) => unreachable!("Gradient buffers are FP32 in Eisen"),
            };

            for &x in &grad_cpu {
                let xf = x as f64;
                sum_sq += xf * xf;
            }
        }

        let grad_norm = (sum_sq.sqrt()) as f32;
        let clip_coef = if grad_norm > 0.0 {
            (max_norm / (grad_norm + 1e-6)).min(1.0)
        } else {
            1.0
        };

        if clip_coef < 1.0 {
            let stream_opt = match &g.device {
                Device::Gpu(_, s) => Some(s.clone()),
                Device::Cpu => None,
            };
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

        for &p_id in &self.params {
            let size = g.tensors[p_id].shape.iter().product::<usize>();
            let is_gpu = matches!(&g.tensors[p_id].data, Storage::Gpu(_));

            if is_gpu {
                // ── Fused GPU kernel path ──────────────────────────────────────
                let stream = stream_opt.as_ref().expect("GPU param but no GPU stream");
                let f = adamw_fn_opt.as_ref().unwrap().clone();

                // Lazy-init VRAM moment buffers
                if !self.m_gpu.contains_key(&p_id) {
                    self.m_gpu.insert(
                        p_id,
                        stream
                            .alloc_zeros::<f32>(size)
                            .expect("AdamW: failed to alloc m moment in VRAM"),
                    );
                    self.v_gpu.insert(
                        p_id,
                        stream
                            .alloc_zeros::<f32>(size)
                            .expect("AdamW: failed to alloc v moment in VRAM"),
                    );
                }

                let weights = match &g.tensors[p_id].data {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
                let grads = match &g.tensors[p_id].grad {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
                let m_s = self.m_gpu.get(&p_id).unwrap();
                let v_s = self.v_gpu.get(&p_id).unwrap();

                let n = size as u64;
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
            } else {
                // ── CPU path (streaming weights, or pure CPU graph) ────────────
                let m = self.m_cpu.entry(p_id).or_insert_with(|| vec![0.0; size]);
                let v = self.v_cpu.entry(p_id).or_insert_with(|| vec![0.0; size]);

                // Borrow grad first (immutable clone), then weight (mutable)
                let grad_data = g.tensors[p_id].grad.as_cpu().clone();
                let weight_data = g.tensors[p_id].data.as_cpu_mut();

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
