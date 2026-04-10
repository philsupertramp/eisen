use crate::graph::Graph;
use crate::nn::Module;

pub struct Embedding {
    pub weight_id: usize,
}

impl Embedding {
    pub fn new(g: &mut Graph, vocab_size: usize, hidden_dim: usize) -> Self {
        let weight_len = vocab_size * hidden_dim;
        let mut weight_data = vec![0.0; weight_len];

        // LCG for reproducible normal-ish initialization
        // We initialize embeddings with very small weights
        let mut seed: u32 = 84;
        for i in 0..weight_len {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let rand_val = (seed as f32 / u32::MAX as f32) * 2.0 - 1.0;
            weight_data[i] = rand_val * 0.02;
        }

        #[cfg(feature = "bf16")]
        let weight_id = if g.uses_bf16_mixed_precision() {
            g.alloc_param_bf16(vec![vocab_size, hidden_dim], weight_data)
        } else {
            g.alloc(vec![vocab_size, hidden_dim], weight_data)
        };
        #[cfg(not(feature = "bf16"))]
        let weight_id = g.alloc(vec![vocab_size, hidden_dim], weight_data);

        Self { weight_id }
    }
}

impl Module for Embedding {
    fn forward(&self, g: &mut Graph, x_id: usize) -> usize {
        g.gather(self.weight_id, x_id)
    }

    fn params(&self) -> Vec<usize> {
        vec![self.weight_id]
    }
}
