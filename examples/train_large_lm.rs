use eisen::graph::Graph;
use eisen::nn::embedding::Embedding;
use eisen::nn::linear::Linear;
use eisen::nn::optim::AdamW;
use eisen::nn::rmsnorm::RMSNorm;
use eisen::nn::transformer::TransformerBlock;
use eisen::nn::Module;
use eisen::data::tokenizer::BPETokenizer;
use eisen::data::dataloader::BinaryDataLoader;
use eisen::tensor::Device;
use cudarc::driver::CudaContext;
use std::sync::Arc;
use std::time::Instant;
use std::io::{BufWriter, Write};
use std::fs::File;

fn setup_gpu() -> Option<Device> {
    match CudaContext::new(0) {
        Ok(ctx) => {
            let stream = ctx.default_stream();
            Some(Device::Gpu(ctx, stream))
        }
        Err(_) => None,
    }
}


/// True GPT-Style Causal Language Model
struct TransformerLM {
    token_emb: Embedding,
    blocks: Vec<TransformerBlock>,
    norm_f: RMSNorm,
    lm_head: Linear,
}

impl TransformerLM {
    fn new(g: &mut Graph, vocab_size: usize, hidden_dim: usize, num_heads: usize, ffn_dim: usize, num_layers: usize) -> Self {
        let token_emb = Embedding::new(g, vocab_size, hidden_dim);
        let mut blocks = Vec::new();
        for _ in 0..num_layers { 
            blocks.push(TransformerBlock::new(g, hidden_dim, num_heads, ffn_dim)); 
        }
        let norm_f = RMSNorm::new(g, hidden_dim, 1e-5);
        let lm_head = Linear::new(g, hidden_dim, vocab_size, false);

        Self { token_emb, blocks, norm_f, lm_head }
    }
}

impl Module for TransformerLM {
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
        for block in &self.blocks { p.extend(block.params()); }
        p.extend(self.norm_f.params());
        p.extend(self.lm_head.params());
        p
    }
}

/// Helper to dump all persistent parameters from VRAM to disk
fn save_weights(g: &Graph, path: &str) {
    println!("Saving model weights to {}...", path);
    let file = File::create(path).expect("Failed to create weights file");
    let mut writer = BufWriter::new(file);
    
    // We only iterate up to num_params, ignoring temporary activations in the VRAM pool!
    for i in 0..g.num_params {
        let data = g.tensors[i].sync_to_cpu();
        for &val in &data {
            writer.write_all(&val.to_le_bytes()).unwrap();
        }
    }
    writer.flush().unwrap();
    println!("Weights successfully saved.");
}

fn main() {
    println!("==================================================");
    println!("=== Eisen Engine: Large Scale Pre-Training Run ===");
    println!("==================================================");
    
    let device = setup_gpu().expect("CUDA GPU Required!");
    println!("Device initialized: {:?}", device);

    let tokenizer_path = "data/tokenizer.model";
    let bin_path = "data/german_large_corpus.bin";
    let output_model_path = "data/eisen_model.bin";

    // 1. Load Tokenizer from Disk
    println!("Loading tokenizer from {}...", tokenizer_path);
    let tokenizer = Arc::new(
        BPETokenizer::load(tokenizer_path)
            .expect("Could not load tokenizer. Did you run tools/train_tokenizer.rs?")
    );
    let vocab_size = tokenizer.vocab.len();
    println!("Loaded Vocab Size: {}", vocab_size);

    // 2. Hyperparameters (Scaled up for 6-8GB VRAM)
    let seq_len = 256;       // 8x longer context than our test run
    let hidden_dim = 384;    // Wider embeddings
    let num_heads = 6;       // 6 heads (64 dim per head)
    let ffn_dim = 1536;      // Standard 4x expansion
    let num_layers = 6;      // 6 full Transformer blocks
    let batch_size = 16;     // Kept moderate to survive FP32 memory limits
    let lr = 3e-4;           // Standard pre-training LR
    
    let tokens_per_step = batch_size * seq_len;
    
    println!("\nModel Architecture:");
    println!("- Layers: {}", num_layers);
    println!("- Hidden Dim: {} ({} heads)", hidden_dim, num_heads);
    println!("- Context Len: {}", seq_len);
    println!("- Tokens per batch: {}", tokens_per_step);

    let mut g = Graph::new(device);
    let model = TransformerLM::new(&mut g, vocab_size, hidden_dim, num_heads, ffn_dim, num_layers);
    let mut optim = AdamW::new(model.params(), lr);

    // Lock parameters to activate the Zero-Leak VRAM pool
    g.mark_params();
    println!("Engine VRAM Pool activated. Parameters locked: {}", g.num_params);

    // 3. Training Loop with Telemetry
    println!("\nStarting high-speed training loop...");
    let mut dataloader = BinaryDataLoader::new(bin_path, seq_len, batch_size);
    
    let mut step = 0;
    let mut running_loss = 0.0;
    let log_interval = 100;
    let save_interval = 2500; // Save checkpoint every ~2,500 steps
    let mut last_log_time = Instant::now();

    // Infinite loop over the dataloader stream
    while let Some((x_batch, y_batch)) = dataloader.next_batch() {
        // Forward
        let x_id = g.alloc(vec![batch_size, seq_len], x_batch);
        let logits_id = model.forward(&mut g, x_id);
        
        // Loss
        let flat_logits_id = g.reshape(logits_id, vec![tokens_per_step, vocab_size]);
        let loss_id = g.cross_entropy(flat_logits_id, &y_batch);
        
        // Backward
        optim.zero_grad(&mut g);
        g.backward(loss_id);
        optim.step(&mut g);
        
        // Telemetry
        let loss = g.tensors[loss_id].sync_to_cpu()[0];
        running_loss += loss;
        
        if step % log_interval == 0 && step > 0 {
            let elapsed = last_log_time.elapsed().as_secs_f32();
            let tokens_processed = (log_interval * tokens_per_step) as f32;
            let throughput = tokens_processed / elapsed;
            let avg_loss = running_loss / log_interval as f32;
            
            println!(
                "Step {:06} | Avg Loss: {:.4} | Throughput: {:.0} tok/s | Batch Time: {:.3}s", 
                step, avg_loss, throughput, elapsed / log_interval as f32
            );
            
            running_loss = 0.0;
            last_log_time = Instant::now();
        }

        // --- CHECKPOINT SAVER ---
        if step % save_interval == 0 && step > 0 {
            save_weights(&g, output_model_path);
        }
        
        g.clear_activations();
        step += 1;
        
        // Stop condition (if you want to limit the run to, say, 50,000 steps)
        // if step > 50_000 { break; }
    }
    
    println!("\n=== End of Dataset Reached ===");
    
    // Final save before exiting!
    save_weights(&g, output_model_path);
    
    // 4. Verification Generation
    println!("\nGenerating from prompt to verify learned syntax...");
    let prompt = "Die künstliche Intelligenz";
    let mut input_tokens = tokenizer.encode(prompt);
    print!("{}", prompt);

    for _ in 0..100 {
        let start = input_tokens.len().saturating_sub(seq_len);
        let context = &input_tokens[start..];
        let current_seq_len = context.len();

        let x_id = g.alloc(vec![1, current_seq_len], context.iter().map(|&t| t as f32).collect());
        let logits_id = model.forward(&mut g, x_id);
        
        let logits_data = g.tensors[logits_id].sync_to_cpu();
        let last_token_logits_start = (current_seq_len - 1) * vocab_size;
        let last_token_logits = &logits_data[last_token_logits_start..last_token_logits_start + vocab_size];
        
        let predicted_id = last_token_logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap();
        
        print!("{}", tokenizer.decode(&[predicted_id]));
        input_tokens.push(predicted_id);
        g.clear_activations();
    }
    println!("\n\nExperiment Complete! Your weights are safely stored.");
}
