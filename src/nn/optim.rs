use std::collections::HashMap;
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use crate::graph::Graph;
use crate::tensor::{Device, Storage};

// ============================================================
// AdamW Optimizer with Fused GPU Kernel
//
// On CPU: classic in-place update (unchanged behavior).
// On GPU: moments are stored entirely in VRAM as CudaSlices and
//         updated by `adamw_step_f32` — a single kernel launch
//         that reads weights+grads and writes new weights+moments
//         without ever touching the PCIe bus.
//
// PCIe bottleneck comparison for a 14M-param model (56 MB):
//   Old path: 56 MB htod + 56 MB dtoh per step ≈ 3-5 ms wasted
//   New path: 0 bytes copied, runs at VRAM bandwidth (~448 GB/s)
// ============================================================
pub struct AdamW {
    pub params: Vec<usize>,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    t: usize,

    // CPU moment storage (used when device == CPU)
    m_cpu: HashMap<usize, Vec<f32>>,
    v_cpu: HashMap<usize, Vec<f32>>,

    // GPU moment storage (lazily initialized on first step)
    // Stored as CudaSlice<f32> so they live entirely in VRAM.
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
    /// GPU path now uses the `fill_f32` kernel instead of allocating
    /// new zero buffers every step. This avoids repeated cudaMalloc/cudaFree
    /// pairs and keeps the same VRAM address stable across steps.
    pub fn zero_grad(&self, g: &mut Graph) {
        let device = g.device.clone();
        for &p_id in &self.params {
            let size = g.tensors[p_id].shape.iter().product::<usize>();
            match &device {
                Device::Cpu => {
                    let grad = g.tensors[p_id].grad.as_cpu_mut();
                    grad.fill(0.0);
                }
                Device::Gpu(_, stream) => {
                    // Use fill_f32 kernel to zero in-place — no allocation.
                    let f = g.functions.get("fill_f32").unwrap().clone();
                    let stream = stream.clone();
                    let grad = match &g.tensors[p_id].grad {
                        Storage::Gpu(s) => s,
                        _ => unreachable!(),
                    };
                    let n = size as u64;
                    let val = 0.0f32;
                    let mut builder = stream.launch_builder(&f);
                    builder.arg(grad).arg(&val).arg(&n);
                    unsafe { builder.launch(LaunchConfig::for_num_elems(size as u32)) }.unwrap();
                }
            }
        }
    }

    pub fn step(&mut self, g: &mut Graph) {
        self.t += 1;

        // Bias-correction scalars — computed once per step on CPU,
        // then passed as kernel arguments so every CUDA thread can use them.
        let bc1 = 1.0_f32 / (1.0 - self.beta1.powi(self.t as i32));
        let bc2 = 1.0_f32 / (1.0 - self.beta2.powi(self.t as i32));

        let device = g.device.clone();

        match &device {
            Device::Gpu(_, stream) => {
                // Clone stream + function handle outside the param loop so we
                // don't hold a borrow on `g` while also borrowing tensors.
                let stream = stream.clone();
                let f = g.functions.get("adamw_step_f32").unwrap().clone();

                for &p_id in &self.params {
                    let size = g.tensors[p_id].shape.iter().product::<usize>();

                    // Lazily initialise VRAM moment buffers on the first step.
                    if !self.m_gpu.contains_key(&p_id) {
                        self.m_gpu.insert(
                            p_id,
                            stream.alloc_zeros::<f32>(size).expect("Failed to alloc m moment in VRAM"),
                        );
                        self.v_gpu.insert(
                            p_id,
                            stream.alloc_zeros::<f32>(size).expect("Failed to alloc v moment in VRAM"),
                        );
                    }

                    // Borrow the tensor slices — all borrows are immutable at the
                    // Rust level; actual writes happen on the GPU side (unsafe).
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
                }
            }

            Device::Cpu => {
                for &p_id in &self.params {
                    let size = g.tensors[p_id].shape.iter().product::<usize>();
                    let m = self.m_cpu.entry(p_id).or_insert(vec![0.0; size]);
                    let v = self.v_cpu.entry(p_id).or_insert(vec![0.0; size]);

                    let tensor = &mut g.tensors[p_id];
                    let grad_data = tensor.grad.as_cpu().clone();
                    let weight_data = tensor.data.as_cpu_mut();

                    for i in 0..size {
                        let grad = grad_data[i];
                        let weight = weight_data[i];

                        // Decoupled weight decay
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
}
