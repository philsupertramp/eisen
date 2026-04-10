use crate::graph::Graph;
use crate::nn::Module;
use crate::nn::attention::MultiHeadAttention;
use crate::nn::linear::Linear;
use crate::nn::rmsnorm::RMSNorm;
use crate::nn::embedding::Embedding;

/// A standard Pre-Norm Transformer Block.
pub struct TransformerBlock {
    pub norm1: RMSNorm,
    pub attn: MultiHeadAttention,
    pub norm2: RMSNorm,
    pub ffn1: Linear,
    pub ffn2: Linear,
}

impl TransformerBlock {
    pub fn new(g: &mut Graph, hidden_dim: usize, num_heads: usize, ffn_dim: usize, seq_len: usize) -> Self {
        Self {
            norm1: RMSNorm::new(g, hidden_dim, 1e-5),
            attn: MultiHeadAttention::new(g, hidden_dim, num_heads, seq_len),
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

pub struct TransformerLM {
    pub token_emb: Embedding,
    pub blocks:    Vec<TransformerBlock>,
    pub norm_f:    RMSNorm,
    pub lm_head:   Linear,
}

impl TransformerLM {
    pub fn new(
        g:          &mut Graph,
        vocab_size: usize,
        hidden_dim: usize,
        num_heads:  usize,
        ffn_dim:    usize,
        num_layers: usize,
        seq_len: usize,
    ) -> Self {
        let token_emb = Embedding::new(g, vocab_size, hidden_dim);

        // Build blocks one-by-one and immediately demote their parameters to
        // CPU so model init does not require full-model VRAM residency.
        let mut blocks = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let block = TransformerBlock::new(g, hidden_dim, num_heads, ffn_dim, seq_len);
            for pid in block.params() {
                // Keep tiny 1D norm scales resident; stream only matrix weights.
                // This avoids hitting GPU-only RMSNorm kernels with CPU weights.
                if g.tensors[pid].shape.len() == 2 {
                    g.demote_tensor_to_cpu(pid);
                }
            }
            blocks.push(block);
        }

        let norm_f  = RMSNorm::new(g, hidden_dim, 1e-5);
        let lm_head = Linear::new(g, hidden_dim, vocab_size, false);
        Self { token_emb, blocks, norm_f, lm_head }
    }

    pub fn named_params(&self) -> Vec<(String, usize)> {
        let mut out = Vec::new();
        out.push((
            "model.embed_tokens.weight".to_string(),
            self.token_emb.weight_id,
        ));
        for (i, b) in self.blocks.iter().enumerate() {
            out.push((
                format!("model.layers.{i}.input_layernorm.weight"),
                b.norm1.weight_id,
            ));
            out.push((
                format!("model.layers.{i}.self_attn.q_proj.weight"),
                b.attn.q_proj.weight_id,
            ));
            out.push((
                format!("model.layers.{i}.self_attn.k_proj.weight"),
                b.attn.k_proj.weight_id,
            ));
            out.push((
                format!("model.layers.{i}.self_attn.v_proj.weight"),
                b.attn.v_proj.weight_id,
            ));
            out.push((
                format!("model.layers.{i}.self_attn.o_proj.weight"),
                b.attn.out_proj.weight_id,
            ));
            out.push((
                format!("model.layers.{i}.post_attention_layernorm.weight"),
                b.norm2.weight_id,
            ));
            out.push((
                format!("model.layers.{i}.mlp.gate_proj.weight"),
                b.ffn1.weight_id,
            ));

            out.push((
                format!("model.layers.{i}.mlp.up_proj.weight"),
                b.ffn1.weight_id,
            ));
            out.push((
                format!("model.layers.{i}.mlp.down_proj.weight"),
                b.ffn2.weight_id,
            ));
        }
        out.push(("model.norm.weight".to_string(), self.norm_f.weight_id));
        out.push(("lm_head.weight".to_string(), self.lm_head.weight_id));
        out
    }

}

impl Module for TransformerLM {
    fn forward(&self, g: &mut Graph, x_id: usize) -> usize {
        let mut h = self.token_emb.forward(g, x_id);
        for block in &self.blocks {
            h = block.forward_with_mask(g, h, true);
        }
        h = self.norm_f.forward(g, h);
        self.lm_head.forward(g, h)
    }
    fn params(&self) -> Vec<usize> {
        let mut p = self.token_emb.params();
        for b in &self.blocks { p.extend(b.params()); }
        p.extend(self.norm_f.params());
        p.extend(self.lm_head.params());
        p
    }
}

