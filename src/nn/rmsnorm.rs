use crate::graph::Graph;
use crate::nn::Module;

pub struct RMSNorm {
    pub weight_id: usize,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(g: &mut Graph, dim: usize, eps: f32) -> Self {
        // RMSNorm weights (often called 'gamma' or 'gain') are conventionally initialized to 1.0
        #[cfg(feature = "bf16")]
        let weight_id = if g.uses_bf16_mixed_precision() {
            g.alloc_param_bf16(vec![dim], vec![1.0; dim])
        } else {
            g.alloc(vec![dim], vec![1.0; dim])
        };
        #[cfg(not(feature = "bf16"))]
        let weight_id = g.alloc(vec![dim], vec![1.0; dim]);
        Self { weight_id, eps }
    }
}

impl Module for RMSNorm {
    fn forward(&self, g: &mut Graph, x_id: usize) -> usize {
        g.rms_norm(x_id, self.weight_id, self.eps)
    }

    fn params(&self) -> Vec<usize> {
        vec![self.weight_id]
    }
}
