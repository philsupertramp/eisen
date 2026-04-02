use crate::graph::Graph;
use crate::nn::Module;

pub struct Linear {
    pub weight_id: usize,
    pub bias_id: Option<usize>,
}

impl Linear {
    /// Creates a new Linear layer: out = x @ W + b
    /// Automatically allocates the weights and biases inside the provided Graph.
    pub fn new(g: &mut Graph, in_features: usize, out_features: usize, use_bias: bool) -> Self {
        // Xavier/Glorot uniform initialization bounds
        let limit = (6.0f32 / (in_features as f32 + out_features as f32)).sqrt();
        
        let weight_len = in_features * out_features;
        let mut weight_data = vec![0.0; weight_len];
        
        // A minimal, zero-dependency pseudo-random number generator (LCG)
        // This guarantees reproducible initializations for testing!
        let mut seed: u32 = 42; 
        for i in 0..weight_len {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let rand_val = (seed as f32 / u32::MAX as f32) * 2.0 - 1.0; // Scale to [-1.0, 1.0]
            weight_data[i] = rand_val * limit;
        }
        
        let weight_id = g.alloc(vec![in_features, out_features], weight_data);
        
        let bias_id = if use_bias {
            // Biases are conventionally initialized to zero
            Some(g.alloc(vec![out_features], vec![0.0; out_features]))
        } else {
            None
        };

        Self { weight_id, bias_id }
    }
}

impl Module for Linear {
    /// Computes the forward pass: x @ W + b
    fn forward(&self, g: &mut Graph, x_id: usize) -> usize {
        let out_id = g.matmul(x_id, self.weight_id);
        
        if let Some(b_id) = self.bias_id {
            // Our graph's broadcasting rules will automatically expand the 
            // 1D bias vector [out_features] to match the 2D output [batch_size, out_features]!
            g.add(out_id, b_id)
        } else {
            out_id
        }
    }

    /// Returns the Graph IDs of the learnable parameters in this layer.
    fn params(&self) -> Vec<usize> {
        let mut p = vec![self.weight_id];
        if let Some(b) = self.bias_id {
            p.push(b);
        }
        p
    }
}
