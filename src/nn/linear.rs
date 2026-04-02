use crate::graph::Graph;
use crate::nn::Module;

pub struct Linear {
    pub weight_id: usize,
    pub bias_id: Option<usize>,
}

impl Linear {
    pub fn new(g: &mut Graph, in_features: usize, out_features: usize, use_bias: bool) -> Self {
        let limit = (6.0f32 / (in_features as f32 + out_features as f32)).sqrt();
        let weight_len = in_features * out_features;
        let mut weight_data = vec![0.0; weight_len];
        
        let mut seed: u32 = 42; 
        for i in 0..weight_len {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let rand_val = (seed as f32 / u32::MAX as f32) * 2.0 - 1.0;
            weight_data[i] = rand_val * limit;
        }
        
        let weight_id = g.alloc(vec![in_features, out_features], weight_data);
        
        let bias_id = if use_bias {
            Some(g.alloc(vec![out_features], vec![0.0; out_features]))
        } else {
            None
        };

        Self { weight_id, bias_id }
    }
}

impl Module for Linear {
    fn forward(&self, g: &mut Graph, x_id: usize) -> usize {
        let x_shape = g.tensors[x_id].shape.clone();
        let is_3d = x_shape.len() == 3;
        
        // BUG FIX: Flatten 3D inputs [Batch, Seq, Dim] to [Batch * Seq, Dim]
        // so our 2D matmul can process them correctly.
        let flat_x_id = if is_3d {
            g.reshape(x_id, vec![x_shape[0] * x_shape[1], x_shape[2]])
        } else {
            x_id
        };

        let out_id = g.matmul(flat_x_id, self.weight_id);
        
        let out_biased_id = if let Some(b_id) = self.bias_id {
            g.add(out_id, b_id)
        } else {
            out_id
        };

        // Restore the 3D shape: [Batch, Seq, OutDim]
        if is_3d {
            let out_dim = g.tensors[self.weight_id].shape[1];
            g.reshape(out_biased_id, vec![x_shape[0], x_shape[1], out_dim])
        } else {
            out_biased_id
        }
    }

    fn params(&self) -> Vec<usize> {
        let mut p = vec![self.weight_id];
        if let Some(b) = self.bias_id { p.push(b); }
        p
    }
}
