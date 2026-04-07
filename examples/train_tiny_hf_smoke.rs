use eisen::graph::Graph;
use eisen::nn::embedding::Embedding;
use eisen::nn::linear::Linear;
use eisen::nn::optim::AdamW;
use eisen::nn::rmsnorm::RMSNorm;
use eisen::nn::transformer::TransformerBlock;
use eisen::nn::Module;
use eisen::tensor::Device;
use eisen::tools::huggingface::{write_llama_config, write_safetensors, LlamaConfig};
use std::fs;

/// Tiny CPU-only LM for a quick Phase 7 smoke test.
struct TinyTransformerLM {
    token_emb: Embedding,
    blocks: Vec<TransformerBlock>,
    norm_f: RMSNorm,
    lm_head: Linear,
}

impl TinyTransformerLM {
    fn new(
        g: &mut Graph,
        vocab_size: usize,
        hidden_dim: usize,
        num_heads: usize,
        ffn_dim: usize,
        num_layers: usize,
    ) -> Self {
        let token_emb = Embedding::new(g, vocab_size, hidden_dim);
        let mut blocks = Vec::new();
        for _ in 0..num_layers {
            blocks.push(TransformerBlock::new(g, hidden_dim, num_heads, ffn_dim));
        }
        let norm_f = RMSNorm::new(g, hidden_dim, 1e-5);
        let lm_head = Linear::new(g, hidden_dim, vocab_size, false);
        Self {
            token_emb,
            blocks,
            norm_f,
            lm_head,
        }
    }

    fn named_params(&self) -> Vec<(String, usize)> {
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

impl Module for TinyTransformerLM {
    fn forward(&self, g: &mut Graph, x_id: usize) -> usize {
        let mut h_id = self.token_emb.forward(g, x_id);
        for block in &self.blocks {
            h_id = block.forward_with_mask(g, h_id, true);
        }
        h_id = self.norm_f.forward(g, h_id);
        self.lm_head.forward(g, h_id)
    }

    fn params(&self) -> Vec<usize> {
        let mut p = Vec::new();
        p.extend(self.token_emb.params());
        for block in &self.blocks {
            p.extend(block.params());
        }
        p.extend(self.norm_f.params());
        p.extend(self.lm_head.params());
        p
    }
}

fn main() {
    println!("=== Phase 7 Tiny HF Smoke Run (CPU) ===");

    let mut g = Graph::new(Device::Cpu);

    // Tiny dimensions so this runs fast on CPU.
    let vocab_size = 32usize;
    let seq_len = 8usize;
    let hidden_dim = 32usize;
    let num_heads = 4usize;
    let ffn_dim = 64usize;
    let num_layers = 1usize;
    let batch_size = 4usize;

    let model = TinyTransformerLM::new(
        &mut g, vocab_size, hidden_dim, num_heads, ffn_dim, num_layers,
    );
    let mut optim = AdamW::new(model.params(), 1e-3);

    // Deterministic synthetic token stream: 0..31 repeating.
    let stream: Vec<usize> = (0..4096).map(|i| i % vocab_size).collect();
    let steps = 40usize;

    for step in 0..steps {
        let start = (step * batch_size * seq_len) % (stream.len() - batch_size * seq_len - 1);
        let mut x_batch = Vec::with_capacity(batch_size * seq_len);
        let mut y_batch = Vec::with_capacity(batch_size * seq_len);

        for b in 0..batch_size {
            let base = start + b * seq_len;
            for t in 0..seq_len {
                x_batch.push(stream[base + t] as f32);
                y_batch.push(stream[base + t + 1]);
            }
        }

        let x_id = g.alloc(vec![batch_size, seq_len], x_batch);
        let logits_id = model.forward(&mut g, x_id);
        let flat_logits = g.reshape(logits_id, vec![batch_size * seq_len, vocab_size]);
        let loss_id = g.cross_entropy(flat_logits, &y_batch);
        let loss = g.tensors[loss_id].sync_to_cpu()[0];

        optim.zero_grad(&mut g);
        g.backward(loss_id);
        optim.step(&mut g);
        g.clear_activations();

        if step % 10 == 0 {
            println!("step {step:03} | loss {loss:.4}");
        }
    }

    let out_dir = "data/hf_export_tiny_smoke";
    fs::create_dir_all(out_dir).expect("Failed to create output dir");
    write_safetensors(
        &g,
        &model.named_params(),
        &format!("{out_dir}/model.safetensors"),
    )
    .expect("Failed to write safetensors");
    write_llama_config(
        &format!("{out_dir}/config.json"),
        &LlamaConfig {
            vocab_size,
            hidden_size: hidden_dim,
            intermediate_size: ffn_dim,
            num_hidden_layers: num_layers,
            num_attention_heads: num_heads,
            max_position_embeddings: seq_len,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
        },
    )
    .expect("Failed to write config");

    println!("Tiny smoke export written to: {out_dir}");
    println!(
        "Now run: python scripts/validate_hf_export.py --export-dir {}",
        out_dir
    );
}
