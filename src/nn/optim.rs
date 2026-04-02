use std::collections::HashMap;
use crate::graph::Graph;

pub struct AdamW {
    pub params: Vec<usize>,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    
    // Optimizer State
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

    /// Zeroes out the gradients of all tracked parameters.
    pub fn zero_grad(&self, g: &mut Graph) {
        for &p_id in &self.params {
            let grad = &mut g.tensors[p_id].grad;
            for i in 0..grad.len() {
                grad[i] = 0.0;
            }
        }
    }

    /// Applies the AdamW parameter update using the accumulated gradients.
    pub fn step(&mut self, g: &mut Graph) {
        self.t += 1;
        
        for &p_id in &self.params {
            let tensor = &mut g.tensors[p_id];
            let size = tensor.data.len();
            
            // Initialize state lazily
            let m = self.m.entry(p_id).or_insert(vec![0.0; size]);
            let v = self.v.entry(p_id).or_insert(vec![0.0; size]);
            
            for i in 0..size {
                let grad = tensor.grad[i];
                let weight = tensor.data[i];
                
                // 1. Decoupled Weight Decay (AdamW specifically)
                let decayed_weight = weight - self.lr * self.weight_decay * weight;
                
                // 2. Update biased first moment estimate (Momentum)
                m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * grad;
                
                // 3. Update biased second raw moment estimate (Variance)
                v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * grad * grad;
                
                // 4. Compute bias-corrected estimates
                let m_hat = m[i] / (1.0 - self.beta1.powi(self.t as i32));
                let v_hat = v[i] / (1.0 - self.beta2.powi(self.t as i32));
                
                // 5. Update weights
                tensor.data[i] = decayed_weight - self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
        }
    }
}
