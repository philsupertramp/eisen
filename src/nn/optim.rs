use std::collections::HashMap;
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
    m: HashMap<usize, Vec<f32>>,
    v: HashMap<usize, Vec<f32>>,
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
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    pub fn zero_grad(&self, g: &mut Graph) {
        // Clone device early to avoid borrowing `g` immutably during the loop
        let device = g.device.clone();
        
        for &p_id in &self.params {
            let size = g.tensors[p_id].shape.iter().product::<usize>();
            
            match &device {
                Device::Cpu => {
                    let grad = g.tensors[p_id].grad.as_cpu_mut();
                    for i in 0..size {
                        grad[i] = 0.0;
                    }
                },
                Device::Gpu(_, stream) => {
                    // Fast VRAM zeroing: Just allocate a new zeroed buffer on the device
                    g.tensors[p_id].grad = Storage::Gpu(stream.alloc_zeros::<f32>(size).expect("Failed to zero VRAM grad"));
                }
            }
        }
    }

    pub fn step(&mut self, g: &mut Graph) {
        self.t += 1;
        
        // Clone device early to avoid borrowing `g` immutably during the loop
        let device = g.device.clone();
        
        for &p_id in &self.params {
            let size = g.tensors[p_id].shape.iter().product::<usize>();
            
            let m = self.m.entry(p_id).or_insert(vec![0.0; size]);
            let v = self.v.entry(p_id).or_insert(vec![0.0; size]);
            
            match &device {
                Device::Cpu => {
                    // Fix: Index the vector ONCE to get a single mutable reference to the Tensor.
                    // Rust understands that `.grad` and `.data` are disjoint fields of the struct,
                    // allowing us to borrow them separately without overlapping vector locks.
                    let tensor = &mut g.tensors[p_id];
                    let grad_data = tensor.grad.as_cpu();
                    let weight_data = tensor.data.as_cpu_mut();
                    
                    for i in 0..size {
                        let grad = grad_data[i];
                        let weight = weight_data[i];
                        
                        let decayed_weight = weight - self.lr * self.weight_decay * weight;
                        
                        m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * grad;
                        v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * grad * grad;
                        
                        let m_hat = m[i] / (1.0 - self.beta1.powi(self.t as i32));
                        let v_hat = v[i] / (1.0 - self.beta2.powi(self.t as i32));
                        
                        weight_data[i] = decayed_weight - self.lr * m_hat / (v_hat.sqrt() + self.eps);
                    }
                },
                Device::Gpu(_, stream) => {
                    // 1. Pull current VRAM weights and gradients to Host RAM
                    let mut weight_data = g.tensors[p_id].sync_to_cpu();
                    let grad_data = g.sync_grad_to_cpu(p_id);
                    
                    // 2. Perform AdamW update on CPU
                    for i in 0..size {
                        let grad = grad_data[i];
                        let weight = weight_data[i];
                        
                        let decayed_weight = weight - self.lr * self.weight_decay * weight;
                        
                        m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * grad;
                        v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * grad * grad;
                        
                        let m_hat = m[i] / (1.0 - self.beta1.powi(self.t as i32));
                        let v_hat = v[i] / (1.0 - self.beta2.powi(self.t as i32));
                        
                        weight_data[i] = decayed_weight - self.lr * m_hat / (v_hat.sqrt() + self.eps);
                    }
                    
                    // 3. Push updated weights back to VRAM
                    g.tensors[p_id].data = Storage::Gpu(stream.clone_htod(weight_data.as_slice()).expect("Failed to push weights to VRAM"));
                }
            }
        }
    }
}
