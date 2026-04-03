use eisen::graph::Graph;
use eisen::nn::embedding::Embedding;
use eisen::nn::linear::Linear;
use eisen::nn::optim::AdamW;
use eisen::nn::rmsnorm::RMSNorm;
use eisen::nn::transformer::TransformerBlock;
use eisen::nn::Module;
use eisen::data::tokenizer::BPETokenizer;
use eisen::data::dataloader::BinaryDataLoader;
use eisen::tools::pre_tokenize;
use eisen::tensor::Device;
use cudarc::driver::CudaContext;
use std::fs;
use std::sync::Arc;
use std::path::Path;

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

fn main() {
    println!("=== Eisen Phase 5: Adjusted Hyperparameters for Small Corpus ===");
    let device = setup_gpu().expect("CUDA GPU Required!");

    let txt_path = "data/german_large_corpus.txt";
    let bin_path = "data/german_large_corpus.bin";
    let tok_path = "data/tokenizer.model";
    
    // 1. Tokenizer & Pre-processing
    let sample_text = fs::read_to_string(txt_path).unwrap();
    let tokenizer = BPETokenizer::load(&tok_path)
        .expect("Tokenizer not found. Please train it first.");
    
    let vocab_size = tokenizer.vocab.len();
    println!("Vocab Size determined by Tokenizer: {}", vocab_size);

    if !Path::new(bin_path).exists() {
        println!("Binary corpus not found. Pre-tokenization required.");
        // In a real scenario, you'd call pre_tokenize(txt_path, bin_path, tokenizer.clone());
        return;
    }

    // 2. Hyperparameters (Re-tuned for ~6.5k tokens)
    let seq_len = 32;       
    let hidden_dim = 64;    // Slightly narrower
    let num_layers = 2;     // 4 layers was too deep for this little data
    let batch_size = 16;    // Smaller batch = more weight updates per epoch
    let epochs = 50;        // We need many passes for a tiny file
    let lr = 0.0007;        // Slightly more conservative learning rate

    let mut g = Graph::new(device);
    let model = TransformerLM::new(&mut g, vocab_size, hidden_dim, 4, 128, num_layers);
    let mut optim = AdamW::new(model.params(), lr);

    g.mark_params();

    // 3. Training Loop
    println!("Starting Training...");
    let mut dataloader = BinaryDataLoader::new(bin_path, seq_len, batch_size);

    for epoch in 1..=epochs {
        let mut step = 0;
        let mut epoch_loss = 0.0;
        dataloader.reset();

        while let Some((x_batch, y_batch)) = dataloader.next_batch() {
            let x_id = g.alloc(vec![batch_size, seq_len], x_batch);
            let logits_id = model.forward(&mut g, x_id);
            
            let flat_logits_id = g.reshape(logits_id, vec![batch_size * seq_len, vocab_size]);
            let loss_id = g.cross_entropy(flat_logits_id, &y_batch);
            
            optim.zero_grad(&mut g);
            g.backward(loss_id);
            optim.step(&mut g);
            
            epoch_loss += g.tensors[loss_id].sync_to_cpu()[0];
            g.clear_activations();
            step += 1;
        }
        
        if epoch % 5 == 0 || epoch == 1 {
            println!("Epoch {:02} | Avg Loss: {:.6}", epoch, epoch_loss / step as f32);
        }
    }
    
    // 4. Generation Logic
    println!("\nGenerating from prompt...");
    let prompt = "Und der Haifisch";
    let mut input_tokens = tokenizer.encode(prompt);
    print!("{}", prompt);

    for _ in 0..60 {
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
    println!("\n\nTraining Run Complete!");
}
