use eisen::graph::Graph;
use eisen::nn::embedding::Embedding;
use eisen::nn::linear::Linear;
use eisen::nn::optim::AdamW;
use eisen::nn::rmsnorm::RMSNorm;
use eisen::nn::transformer::TransformerBlock;
use eisen::nn::Module;
use eisen::data::tokenizer::BPETokenizer;
use eisen::data::dataloader::StreamingDataLoader;
use eisen::tensor::Device;
use cudarc::driver::CudaContext;
use std::fs;
use std::sync::Arc;

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
        for _ in 0..num_layers { blocks.push(TransformerBlock::new(g, hidden_dim, num_heads, ffn_dim)); }
        let norm_f = RMSNorm::new(g, hidden_dim, 1e-5);
        let lm_head = Linear::new(g, hidden_dim, vocab_size, false);

        Self { token_emb, blocks, norm_f, lm_head }
    }
}

impl Module for TransformerLM {
    fn forward(&self, g: &mut Graph, x_id: usize) -> usize {
        let mut h_id = self.token_emb.forward(g, x_id);
        for block in &self.blocks { h_id = block.forward_with_mask(g, h_id, true); }
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

fn main() {
    println!("=== Eisen Phase 5: High-Throughput Streaming Dataloader ===");
    let device = setup_gpu().expect("CUDA GPU Required!");

    // 1. Setup Data Pipeline
    let file_path = "data/german_corpus.txt";
    
    println!("Training BPE Tokenizer...");
    let sample_text = fs::read_to_string(file_path).unwrap();
    let vocab_size = 1024;
    let tokenizer = Arc::new(BPETokenizer::train(&sample_text, vocab_size));
    println!("BPE Training Complete. Vocab Size: {}", tokenizer.vocab.len());

    // 2. Hyperparameters
    let seq_len = 16;
    let hidden_dim = 64;
    let num_heads = 4;
    let ffn_dim = 128;
    let num_layers = 2;
    let batch_size = 32;
    let epochs = 30;

    // --- High-Throughput Dataloader Fix ---
    // Bumping from 5 to 1000 batches. 1000 batches of (32x16) size is only ~4MB of system RAM.
    // This creates a massive runway for the background tokenization thread to stay ahead of the GPU!
    let prefetch_batches = 1000; 

    let mut g = Graph::new(device);
    let model = TransformerLM::new(&mut g, vocab_size, hidden_dim, num_heads, ffn_dim, num_layers);
    let mut optim = AdamW::new(model.params(), 0.003);

    g.mark_params(); // Activate Zero-Leak VRAM Pooling

    // 3. Training Loop
    println!("Starting Training...");
    for epoch in 1..=epochs {
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        let dataloader = StreamingDataLoader::new(
            file_path.to_string(), 
            tokenizer.clone(), 
            seq_len, 
            batch_size, 
            prefetch_batches // <--- Unleash the background thread!
        );

        while let Some((x_batch, y_batch)) = dataloader.next_batch() {
            let x_id = g.alloc(vec![batch_size, seq_len], x_batch);
            
            let logits_id = model.forward(&mut g, x_id);
            let flat_logits_id = g.reshape(logits_id, vec![batch_size * seq_len, vocab_size]);
            let loss_id = g.cross_entropy(flat_logits_id, &y_batch);
            
            total_loss += g.tensors[loss_id].sync_to_cpu()[0];
            num_batches += 1;

            optim.zero_grad(&mut g);
            g.backward(loss_id);
            optim.step(&mut g);
            
            g.clear_activations(); 
        }

        if epoch % 5 == 0 {
            println!("Epoch {:03} | Avg Loss: {:.6}", epoch, total_loss / num_batches as f32);
        }
    }

    // 4. Generation
    println!("\nGenerating from prompt...");
    let prompt = "Und der Haifisch";
    let mut input_tokens = tokenizer.encode(prompt);
    print!("{}", prompt);

    for _ in 0..30 {
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
    println!("\n\nExperiment Complete!");
}
