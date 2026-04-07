use eisen::graph::Graph;
use eisen::nn::embedding::Embedding;
use eisen::nn::linear::Linear;
use eisen::nn::optim::AdamW;
use eisen::nn::rmsnorm::RMSNorm;
use eisen::nn::transformer::{TransformerBlock, TransformerLM};
use eisen::nn::Module;
use eisen::tensor::Device;
use eisen::tools::huggingface::{write_llama_config, write_safetensors, LlamaConfig};
use std::fs;

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

    let model = TransformerLM::new(
        &mut g, vocab_size, hidden_dim, num_heads, ffn_dim, num_layers,
    );
    g.mark_params();
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
