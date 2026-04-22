use crate::graph::Graph;
use crate::nn::linear::Linear;
use crate::nn::Module;
use std::sync::RwLock;

/// Attention module.
/// Configured for Single-Head Attention (num_heads=1) to fit our 3D tensor limits
/// without requiring an expensive 4D VRAM transposition kernel.
pub struct Attention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub hidden_dim: usize,
    pub cached_scale: RwLock<Option<((usize, usize), usize)>>, // ((batch, seq_len), id)
    pub cached_mask: RwLock<Option<(usize, usize)>>,           // (seq_len, id)
}

impl Attention {
    pub fn new(g: &mut Graph, hidden_dim: usize) -> Self {
        Self {
            q_proj: Linear::new(g, hidden_dim, hidden_dim, false),
            k_proj: Linear::new(g, hidden_dim, hidden_dim, false),
            v_proj: Linear::new(g, hidden_dim, hidden_dim, false),
            out_proj: Linear::new(g, hidden_dim, hidden_dim, false),
            hidden_dim,
            cached_scale: RwLock::new(None),
            cached_mask: RwLock::new(None),
        }
    }

    /// Forward pass with an optional causal masking flag
    pub fn forward_with_mask(&self, g: &mut Graph, x_id: usize, causal: bool) -> usize {
        let shape = g.tensors[x_id].shape.clone();
        assert_eq!(
            shape.len(),
            3,
            "Attention requires 3D input [Batch, Seq, Dim]"
        );
        let batch = shape[0];
        let seq_len = shape[1];

        // 1. Projections -> [Batch, Seq, Dim]
        let q_id = self.q_proj.forward(g, x_id);
        let k_id = self.k_proj.forward(g, x_id);
        let v_id = self.v_proj.forward(g, x_id);

        let scale = 1.0 / (self.hidden_dim as f32).sqrt();
        let context_id = if g.no_grad {
            g.flash_attention(q_id, k_id, v_id, scale, causal)
        } else {
            // 2. Q @ K.T -> Scores [Batch, Seq, Seq]
            // trans_b = true avoids allocating a transposed VRAM buffer!
            let scores_id = g.bmm(q_id, k_id, true);

            // 3. Scale by 1 / sqrt(d_k)
            let mut current_scale = None;
            if let Some(((c_b, c_s), id)) = *self.cached_scale.read().unwrap() {
                if c_b == batch && c_s == seq_len {
                    current_scale = Some(id);
                }
            }
            let scale_id = if let Some(id) = current_scale {
                id
            } else {
                let scores_size = batch * seq_len * seq_len;
                let id = g.alloc(vec![batch, seq_len, seq_len], vec![scale; scores_size]);
                *self.cached_scale.write().unwrap() = Some(((batch, seq_len), id));
                id
            };
            
            let scaled_scores_id = g.mul(scores_id, scale_id);

            // 4. Causal Masking (for autoregressive generation)
            let masked_scores_id = if causal {
                let mut current_mask = None;
                if let Some((c_s, id)) = *self.cached_mask.read().unwrap() {
                    if c_s == seq_len {
                        current_mask = Some(id);
                    }
                }
                
                let mask_id = if let Some(id) = current_mask {
                    id
                } else {
                    let mut mask_data = vec![0.0; seq_len * seq_len];
                    for r in 0..seq_len {
                        for c in 0..seq_len {
                            // Tokens cannot look ahead into the future
                            if c > r {
                                mask_data[r * seq_len + c] = -1e20;
                            }
                        }
                    }
                    // Broadcast mask [1, Seq, Seq] over the Batch dimension
                    let id = g.alloc(vec![1, seq_len, seq_len], mask_data);
                    *self.cached_mask.write().unwrap() = Some((seq_len, id));
                    id
                };
                g.add(scaled_scores_id, mask_id)
            } else {
                scaled_scores_id
            };

            // 5. Softmax -> [Batch, Seq, Seq]
            let probs_id = g.softmax(masked_scores_id);

            // 6. Probs @ V -> Context [Batch, Seq, Dim]
            g.bmm(probs_id, v_id, false)
        };

        // 7. Output Projection
        self.out_proj.forward(g, context_id)
    }
}

impl Module for Attention {
    fn forward(&self, g: &mut Graph, x_id: usize) -> usize {
        self.forward_with_mask(g, x_id, true) // Default to causal attention
    }

    fn params(&self) -> Vec<usize> {
        let mut p = Vec::new();
        p.extend(self.q_proj.params());
        p.extend(self.k_proj.params());
        p.extend(self.v_proj.params());
        p.extend(self.out_proj.params());
        p
    }
}

pub struct MultiHeadAttention {
    pub num_heads: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub cached_scale: RwLock<Option<((usize, usize), usize)>>,
    pub cached_mask: RwLock<Option<(usize, usize)>>,
}

impl MultiHeadAttention {
    pub fn new(g: &mut Graph, hidden_dim: usize, num_heads: usize) -> Self {
        assert!(
            hidden_dim % num_heads == 0,
            "hidden_dim must be cleanly divisible by num_heads"
        );
        let head_dim = hidden_dim / num_heads;

        Self {
            num_heads,
            head_dim,
            hidden_dim,
            q_proj: Linear::new(g, hidden_dim, hidden_dim, false),
            k_proj: Linear::new(g, hidden_dim, hidden_dim, false),
            v_proj: Linear::new(g, hidden_dim, hidden_dim, false),
            out_proj: Linear::new(g, hidden_dim, hidden_dim, false),
            cached_scale: RwLock::new(None),
            cached_mask: RwLock::new(None),
        }
    }

    pub fn forward_with_mask(&self, g: &mut Graph, x_id: usize, causal: bool) -> usize {
        let shape = g.tensors[x_id].shape.clone();
        assert_eq!(shape.len(), 3, "MHA requires [Batch, Seq, Dim]");
        let batch = shape[0];
        let seq_len = shape[1];

        // 1. Projections -> [Batch, Seq, HiddenDim]
        let q_id = self.q_proj.forward(g, x_id);
        let k_id = self.k_proj.forward(g, x_id);
        let v_id = self.v_proj.forward(g, x_id);

        // 1.5 Apply Rotary Positional Embeddings (RoPE) to Q and K natively!
        let q_rope_id = g.rope(q_id, self.head_dim);
        let k_rope_id = g.rope(k_id, self.head_dim);

        // 2. Reshape for Heads -> [Batch, Seq, Heads, HeadDim]
        let q_4d = g.reshape(
            q_rope_id,
            vec![batch, seq_len, self.num_heads, self.head_dim],
        );
        let k_4d = g.reshape(
            k_rope_id,
            vec![batch, seq_len, self.num_heads, self.head_dim],
        );
        let v_4d = g.reshape(v_id, vec![batch, seq_len, self.num_heads, self.head_dim]);

        // 3. Transpose MHA -> [Batch, Heads, Seq, HeadDim]
        let q_t = g.transpose_0213(q_4d);
        let k_t = g.transpose_0213(k_4d);
        let v_t = g.transpose_0213(v_4d);

        // 4. Flatten Batch and Heads to use BMM -> [Batch * Heads, Seq, HeadDim]
        let bh = batch * self.num_heads;

        // ZERO-COPY reinterpret
        g.reinterpret_shape(q_t, vec![bh, seq_len, self.head_dim]);
        g.reinterpret_shape(k_t, vec![bh, seq_len, self.head_dim]);
        g.reinterpret_shape(v_t, vec![bh, seq_len, self.head_dim]);

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let context_flat = if g.no_grad {
            g.flash_attention(q_t, k_t, v_t, scale, causal)
        } else {
            // 5. Q @ K.T -> Scores [Batch * Heads, Seq, Seq]
            let scores_id = g.bmm(q_t, k_t, true);

            // 6. Scale by 1 / sqrt(d_k)
            let mut current_scale = None;
            if let Some(((c_bh, c_s), id)) = *self.cached_scale.read().unwrap() {
                if c_bh == bh && c_s == seq_len {
                    current_scale = Some(id);
                }
            }
            
            let scale_id = if let Some(id) = current_scale {
                id
            } else {
                let scale_m = vec![scale; bh * seq_len * seq_len];
                let id = g.alloc(vec![bh, seq_len, seq_len], scale_m);
                g.name_tensor(id, "attn_scale");
                *self.cached_scale.write().unwrap() = Some(((bh, seq_len), id));
                id
            };
            
            let scaled_scores_id = g.mul(scores_id, scale_id);

            // 7. Causal Masking
            let masked_scores_id = if causal {
                let mut current_mask = None;
                if let Some((c_s, id)) = *self.cached_mask.read().unwrap() {
                    if c_s == seq_len {
                        current_mask = Some(id);
                    }
                }

                let mask_id = if let Some(id) = current_mask {
                    id
                } else {
                    let mut mask_data = vec![0.0; seq_len * seq_len];
                    for r in 0..seq_len {
                        for c in 0..seq_len {
                            if c > r {
                                mask_data[r * seq_len + c] = -1e20;
                            }
                        }
                    }

                    #[cfg(feature = "bf16")]
                    let id = if g.uses_bf16_mixed_precision() {
                        g.alloc_param_bf16(vec![1, seq_len, seq_len], mask_data)
                    } else {
                        g.alloc(vec![1, seq_len, seq_len], mask_data)
                    };
                    #[cfg(not(feature = "bf16"))]
                    let id = g.alloc(vec![1, seq_len, seq_len], mask_data);
                    
                    *self.cached_mask.write().unwrap() = Some((seq_len, id));
                    id
                };

                g.add(scaled_scores_id, mask_id) // Add broadcasts across Batch*Heads
            } else {
                scaled_scores_id
            };

            // 8. Softmax -> [Batch * Heads, Seq, Seq]
            let probs_id = g.softmax(masked_scores_id);

            // 9. Probs @ V -> Context [Batch * Heads, Seq, HeadDim]
            g.bmm(probs_id, v_t, false)
        };

        // 10. Reshape -> [Batch, Heads, Seq, HeadDim]
        g.reinterpret_shape(
            context_flat,
            vec![batch, self.num_heads, seq_len, self.head_dim],
        );

        // 11. Transpose back -> [Batch, Seq, Heads, HeadDim]
        let context_t = g.transpose_0213(context_flat);

        // 12. Flatten heads -> [Batch, Seq, HiddenDim]
        let context_id = g.reshape(context_t, vec![batch, seq_len, self.hidden_dim]);

        // 13. Output Projection
        self.out_proj.forward(g, context_id)
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, g: &mut Graph, x_id: usize) -> usize {
        self.forward_with_mask(g, x_id, true)
    }

    fn params(&self) -> Vec<usize> {
        let mut p = Vec::new();
        p.extend(self.q_proj.params());
        p.extend(self.k_proj.params());
        p.extend(self.v_proj.params());
        p.extend(self.out_proj.params());
        p
    }
}

pub struct GroupedQueryAttention {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub cached_scale: RwLock<Option<((usize, usize), usize)>>,
    pub cached_mask: RwLock<Option<(usize, usize)>>,
}

impl GroupedQueryAttention {
    pub fn new(g: &mut Graph, hidden_dim: usize, num_heads: usize, num_kv_heads: usize) -> Self {
        assert!(
            hidden_dim % num_heads == 0,
            "hidden_dim must be cleanly divisible by num_heads"
        );
        assert!(
            num_heads % num_kv_heads == 0,
            "num_heads must be divisible by num_kv_heads"
        );

        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;

        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_dim,
            q_proj: Linear::new(g, hidden_dim, hidden_dim, false),
            // Look here: Projections are drastically smaller!
            k_proj: Linear::new(g, hidden_dim, kv_dim, false),
            v_proj: Linear::new(g, hidden_dim, kv_dim, false),
            out_proj: Linear::new(g, hidden_dim, hidden_dim, false),
            cached_scale: RwLock::new(None),
            cached_mask: RwLock::new(None),
        }
    }

    pub fn forward_with_mask(&self, g: &mut Graph, x_id: usize, causal: bool) -> usize {
        let shape = g.tensors[x_id].shape.clone();
        assert_eq!(shape.len(), 3, "GQA requires [Batch, Seq, Dim]");
        let batch = shape[0];
        let seq_len = shape[1];

        // 1. Projections -> Q is [B, S, HDim], K/V are [B, S, KVDim]
        let q_id = self.q_proj.forward(g, x_id);
        let k_id = self.k_proj.forward(g, x_id);
        let v_id = self.v_proj.forward(g, x_id);

        // 2. Rotary Embeddings applied natively
        let q_rope_id = g.rope(q_id, self.head_dim);
        let k_rope_id = g.rope(k_id, self.head_dim);

        // 3. Reshape for Heads
        let q_4d = g.reshape(
            q_rope_id,
            vec![batch, seq_len, self.num_heads, self.head_dim],
        );
        let k_4d = g.reshape(
            k_rope_id,
            vec![batch, seq_len, self.num_kv_heads, self.head_dim],
        );
        let v_4d = g.reshape(
            v_id, 
            vec![batch, seq_len, self.num_kv_heads, self.head_dim]
        );

        // 4. Transpose to [Batch, Heads, Seq, HeadDim]
        let q_t = g.transpose_0213(q_4d);
        let k_t = g.transpose_0213(k_4d);
        let v_t = g.transpose_0213(v_4d);

        // 5. Repeat KV Heads to match Query Heads
        let repeats = self.num_heads / self.num_kv_heads;
        let k_repeated = g.repeat_kv(k_t, repeats);
        let v_repeated = g.repeat_kv(v_t, repeats);

        // 6. Flatten Batch and Heads to use BMM -> [Batch * Heads, Seq, HeadDim]
        let bh = batch * self.num_heads;
        g.reinterpret_shape(q_t, vec![bh, seq_len, self.head_dim]);
        g.reinterpret_shape(k_repeated, vec![bh, seq_len, self.head_dim]);
        g.reinterpret_shape(v_repeated, vec![bh, seq_len, self.head_dim]);

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let context_flat = if g.no_grad {
            g.flash_attention(q_t, k_repeated, v_repeated, scale, causal)
        } else {
            // 7. Q @ K.T -> Scores [Batch * Heads, Seq, Seq]
            let scores_id = g.bmm(q_t, k_repeated, true);

            // 8. Scale
            let mut current_scale = None;
            if let Some(((c_bh, c_s), id)) = *self.cached_scale.read().unwrap() {
                if c_bh == bh && c_s == seq_len {
                    current_scale = Some(id);
                    println!("FOUND PREV MASK");
                }
            }
            
            let scale_id = if let Some(id) = current_scale {
                id
            } else {
                println!("ALLOC ATTN MASK");
                let scale_m = vec![scale; bh * seq_len * seq_len];
                let id = g.alloc_pooled(vec![bh, seq_len, seq_len]);
                g.load_tensor_data(id, scale_m.as_slice());
                g.name_tensor(id, "gqa_attn_scale");
                *self.cached_scale.write().unwrap() = Some(((bh, seq_len), id));
                id
            };
            
            let scaled_scores_id = g.mul(scores_id, scale_id);

            // 9. Masking
            let masked_scores_id = if causal {
                let mut current_mask = None;
                if let Some((c_s, id)) = *self.cached_mask.read().unwrap() {
                    if c_s == seq_len {
                        current_mask = Some(id);
                    }
                }

                let mask_id = if let Some(id) = current_mask {
                    id
                } else {
                    let mut mask_data = vec![0.0; seq_len * seq_len];
                    for r in 0..seq_len {
                        for c in 0..seq_len {
                            if c > r {
                                mask_data[r * seq_len + c] = -1e20;
                            }
                        }
                    }

                    #[cfg(feature = "bf16")]
                    let id = if g.uses_bf16_mixed_precision() {
                        g.alloc_param_bf16(vec![1, seq_len, seq_len], mask_data)
                    } else {
                        g.alloc(vec![1, seq_len, seq_len], mask_data)
                    };
                    #[cfg(not(feature = "bf16"))]
                    let id = g.alloc(vec![1, seq_len, seq_len], mask_data);
                    
                    *self.cached_mask.write().unwrap() = Some((seq_len, id));
                    id
                };

                g.add(scaled_scores_id, mask_id)
            } else {
                scaled_scores_id
            };

            // 10. Softmax & V
            let probs_id = g.softmax(masked_scores_id);
            g.bmm(probs_id, v_repeated, false)
        };

        // 11. Reshape & Transpose Back
        g.reinterpret_shape(
            context_flat,
            vec![batch, self.num_heads, seq_len, self.head_dim],
        );
        let context_t = g.transpose_0213(context_flat);
        let context_id = g.reshape(context_t, vec![batch, seq_len, self.hidden_dim]);

        // 12. Final Projection
        self.out_proj.forward(g, context_id)
    }
}

impl Module for GroupedQueryAttention {
    fn forward(&self, g: &mut Graph, x_id: usize) -> usize {
        self.forward_with_mask(g, x_id, true)
    }

    fn params(&self) -> Vec<usize> {
        let mut p = Vec::new();
        p.extend(self.q_proj.params());
        p.extend(self.k_proj.params());
        p.extend(self.v_proj.params());
        p.extend(self.out_proj.params());
        p
    }
}
