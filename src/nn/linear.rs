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

        // Instead of Xavier uniform:
        // let limit = (6.0f32 / (in_features + out_features)).sqrt();
        // Instead use:
        let std = 0.02;

        // Box-Muller to generate normal distribution
        let mut seed: u32 = 42;
        for i in 0..weight_len {
            let u1 = ((seed as f32 / u32::MAX as f32) + 1e-7).min(1.0 - 1e-7);
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let u2 = (seed as f32 / u32::MAX as f32).max(1e-7).min(1.0 - 1e-7);
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            
            let mag = (-2.0 * u1.ln()).sqrt();
            let z0 = mag * (2.0 * std::f32::consts::PI * u2).cos();
            weight_data[i] = z0 * std;
        }

        #[cfg(feature = "bf16")]
        let weight_id = if g.uses_bf16_mixed_precision() {
            g.alloc_param_bf16(vec![in_features, out_features], weight_data)
        } else {
            g.alloc(vec![in_features, out_features], weight_data)
        };
        #[cfg(not(feature = "bf16"))]
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

        // ---------------------------------------------------------------
        // Mixed-Precision Dispatch
        //
        // With the `bf16` feature enabled, the weight matrix multiplication
        // uses BF16 compute (forward) with FP32 accumulation and FP32
        // gradients (backward). Master weights always stay FP32, so the
        // optimizer and loss scaling are unchanged.
        //
        // Without the feature, plain FP32 matmul is used (identical to
        // the original behaviour).
        // ---------------------------------------------------------------
        let out_id = if g.uses_bf16_mixed_precision() {
            #[cfg(feature = "bf16")]
            {
                g.matmul_bf16(x_id, self.weight_id)
            }
            #[cfg(not(feature = "bf16"))]
            {
                g.matmul(x_id, self.weight_id)
            }
        } else {
            g.matmul(x_id, self.weight_id)
        };

        let out_biased_id = if let Some(b_id) = self.bias_id {
            g.add(out_id, b_id)
        } else {
            out_id
        };

        // Restore the 3D shape: [Batch, Seq, OutDim]
        if is_3d {
            let out_dim = g.tensors[self.weight_id].shape[1];
            g.reinterpret_shape(out_biased_id, vec![x_shape[0], x_shape[1], out_dim]);
            out_biased_id
        } else {
            out_biased_id
        }
    }

    fn params(&self) -> Vec<usize> {
        let mut p = vec![self.weight_id];
        if let Some(b) = self.bias_id {
            p.push(b);
        }
        p
    }
}
