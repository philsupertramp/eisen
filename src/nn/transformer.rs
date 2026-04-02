use crate::graph::Graph;
use crate::nn::Module;
use crate::nn::attention::MultiHeadAttention;
use crate::nn::linear::Linear;
use crate::nn::rmsnorm::RMSNorm;

/// A standard Pre-Norm Transformer Block.
pub struct TransformerBlock {
    pub norm1: RMSNorm,
    pub attn: MultiHeadAttention,
    pub norm2: RMSNorm,
    pub ffn1: Linear,
    pub ffn2: Linear,
}

impl TransformerBlock {
    pub fn new(g: &mut Graph, hidden_dim: usize, num_heads: usize, ffn_dim: usize) -> Self {
        Self {
            norm1: RMSNorm::new(g, hidden_dim, 1e-5),
            attn: MultiHeadAttention::new(g, hidden_dim, num_heads),
            norm2: RMSNorm::new(g, hidden_dim, 1e-5),
            // No biases in modern LLM FFN layers
            ffn1: Linear::new(g, hidden_dim, ffn_dim, false), 
            ffn2: Linear::new(g, ffn_dim, hidden_dim, false),
        }
    }

    pub fn forward_with_mask(&self, g: &mut Graph, x_id: usize, causal: bool) -> usize {
        // --- 1. Attention Block (Pre-Norm) ---
        let norm1_out = self.norm1.forward(g, x_id);
        let attn_out = self.attn.forward_with_mask(g, norm1_out, causal);
        // Residual Connection 1
        let res1_out = g.add(x_id, attn_out);
        
        // --- 2. Feed-Forward Block (Pre-Norm) ---
        let norm2_out = self.norm2.forward(g, res1_out);
        let ffn1_out = self.ffn1.forward(g, norm2_out);
        let act_out = g.silu(ffn1_out);
        let ffn2_out = self.ffn2.forward(g, act_out);
        // Residual Connection 2
        let res2_out = g.add(res1_out, ffn2_out);
        
        res2_out
    }
}

impl Module for TransformerBlock {
    fn forward(&self, g: &mut Graph, x_id: usize) -> usize {
        self.forward_with_mask(g, x_id, true)
    }

    fn params(&self) -> Vec<usize> {
        let mut p = Vec::new();
        p.extend(self.norm1.params());
        p.extend(self.attn.params());
        p.extend(self.norm2.params());
        p.extend(self.ffn1.params());
        p.extend(self.ffn2.params());
        p
    }
}
