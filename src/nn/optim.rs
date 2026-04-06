use std::collections::HashMap;
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use crate::graph::Graph;
use crate::tensor::{Device, Storage};

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
            Device::Cpu       => None,
        };
        let fill_fn_opt = stream_opt.as_ref().map(|_| {
            g.functions.get("fill_f32").unwrap().clone()
        });

        for &p_id in &self.params {
            let size = g.tensors[p_id].shape.iter().product::<usize>();

            // Inspect storage variant to decide which path to take
            let is_gpu_grad = matches!(&g.tensors[p_id].grad, Storage::Gpu(_));

            if is_gpu_grad {
                let stream = stream_opt.as_ref().expect("GPU grad but no GPU stream");
                let f      = fill_fn_opt.as_ref().unwrap().clone();
                let grad   = match &g.tensors[p_id].grad {
                    Storage::Gpu(s) => s,
                    _ => unreachable!(),
                };
                let n   = size as u64;
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
        self.t += 1;
        let bc1 = 1.0_f32 / (1.0 - self.beta1.powi(self.t as i32));
        let bc2 = 1.0_f32 / (1.0 - self.beta2.powi(self.t as i32));

        let stream_opt = match &g.device {
            Device::Gpu(_, s) => Some(s.clone()),
            Device::Cpu       => None,
        };
        let adamw_fn_opt = stream_opt.as_ref().map(|_| {
            g.functions.get("adamw_step_f32").unwrap().clone()
        });

        for &p_id in &self.params {
            let size     = g.tensors[p_id].shape.iter().product::<usize>();
            let is_gpu   = matches!(&g.tensors[p_id].data, Storage::Gpu(_));

            if is_gpu {
                // ── Fused GPU kernel path ──────────────────────────────────────
                let stream = stream_opt.as_ref().expect("GPU param but no GPU stream");
                let f      = adamw_fn_opt.as_ref().unwrap().clone();

                // Lazy-init VRAM moment buffers
                if !self.m_gpu.contains_key(&p_id) {
                    self.m_gpu.insert(
                        p_id,
                        stream.alloc_zeros::<f32>(size)
                            .expect("AdamW: failed to alloc m moment in VRAM"),
                    );
                    self.v_gpu.insert(
                        p_id,
                        stream.alloc_zeros::<f32>(size)
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
                    let grad   = grad_data[i];
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
