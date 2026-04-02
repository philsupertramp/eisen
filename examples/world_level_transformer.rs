use eisen::graph::Graph;
use eisen::nn::embedding::Embedding;
use eisen::nn::linear::Linear;
use eisen::nn::optim::AdamW;
use eisen::nn::rmsnorm::RMSNorm;
use eisen::nn::transformer::TransformerBlock;
use eisen::nn::Module;
use eisen::tensor::Device;
use cudarc::driver::CudaContext;
use std::collections::HashMap;
use std::fs;

fn setup_gpu() -> Option<Device> {
    match CudaContext::new(0) {
        Ok(ctx) => {
            let stream = ctx.default_stream();
            Some(Device::Gpu(ctx, stream))
        }
        Err(_) => None,
    }
}

/// Simple Word-Level Tokenizer
struct Tokenizer {
    word_to_id: HashMap<String, usize>,
    id_to_word: Vec<String>,
}

impl Tokenizer {
    fn from_text(text: &str) -> Self {
        let mut word_to_id = HashMap::new();
        let mut id_to_word = Vec::new();
        
        let cleaned = text.to_lowercase()
            .replace(".", " . ")
            .replace(",", " , ")
            .replace("!", " ! ")
            .replace("?", " ? ")
            .replace("\"", "");

        for word in cleaned.split_whitespace() {
            if !word_to_id.contains_key(word) {
                word_to_id.insert(word.to_string(), id_to_word.len());
                id_to_word.push(word.to_string());
            }
        }
        Self { word_to_id, id_to_word }
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        text.to_lowercase()
            .replace(".", " . ")
            .replace(",", " , ")
            .split_whitespace()
            .filter_map(|w| self.word_to_id.get(w).cloned())
            .collect()
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
        let lm_head = Linear::new(g, hidden_dim, vocab_size, false); // No bias in modern LM heads

        Self { token_emb, blocks, norm_f, lm_head }
    }
}

impl Module for TransformerLM {
    fn forward(&self, g: &mut Graph, x_id: usize) -> usize {
        // 1. Embed Tokens: [Batch, Seq] -> [Batch, Seq, HiddenDim]
        let mut h_id = self.token_emb.forward(g, x_id);
        
        // 2. Pass through Transformer Blocks (with RoPE and Causal Masking applied internally!)
        for block in &self.blocks {
            h_id = block.forward_with_mask(g, h_id, true);
        }
        
        // 3. Final Pre-Norm
        h_id = self.norm_f.forward(g, h_id);
        
        // 4. LM Head Projection: [Batch, Seq, HiddenDim] -> [Batch, Seq, VocabSize]
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
    println!("=== Eisen Phase 4: Full Transformer LLM on GPU ===");
    
    let device = setup_gpu().expect("This Transformer example requires a CUDA GPU!");
    println!("Device: {:?}", device);

    let file_path = "data/german_corpus.txt";
    let raw_text = fs::read_to_string(file_path)
        .expect("Please ensure data/german_corpus.txt contains the Dreigroschenoper text.");
    
    println!("Loading corpus...");
    let tokenizer = Tokenizer::from_text(&raw_text);
    let tokens = tokenizer.encode(&raw_text);
    let vocab_size = tokenizer.id_to_word.len();
    println!("Vocab Size: {} | Total Tokens: {}", vocab_size, tokens.len());

    // --- Hyperparameters ---
    let seq_len = 16;       // Context window
    let hidden_dim = 64;    // Embedding dimension
    let num_heads = 4;      // 4 Attention Heads (16 dim per head)
    let ffn_dim = 128;      // Feed-Forward expansion
    let num_layers = 2;     // 2 full Transformer blocks
    let batch_size = 32;
    let epochs = 30;
    let lr = 0.003;

    let mut g = Graph::new(device);
    let model = TransformerLM::new(&mut g, vocab_size, hidden_dim, num_heads, ffn_dim, num_layers);
    let mut optim = AdamW::new(model.params(), lr);

    // --- VRAM POOLING ACTIVATION ---
    println!("Locking parameters and activating VRAM zero-allocation pool...");
    g.mark_params();

    println!("Starting Training...");
    for epoch in 1..=epochs {
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        for i in (0..tokens.len() - seq_len - 1).step_by(batch_size * seq_len) {
            let end = (i + batch_size * seq_len).min(tokens.len() - seq_len - 1);
            let current_batch_size = (end - i) / seq_len;
            if current_batch_size == 0 { continue; }

            let mut x_batch = Vec::with_capacity(current_batch_size * seq_len);
            let mut y_batch = Vec::with_capacity(current_batch_size * seq_len);

            // Create causal language modeling pairs (predict next token for EVERY position)
            for b in 0..current_batch_size {
                let start_idx = i + b * seq_len;
                for s in 0..seq_len {
                    x_batch.push(tokens[start_idx + s] as f32);
                    y_batch.push(tokens[start_idx + s + 1]); // The target is shifted by 1
                }
            }

            // 1. Forward Pass
            let x_id = g.alloc(vec![current_batch_size, seq_len], x_batch);
            let logits_id = model.forward(&mut g, x_id);
            
            // 2. Reshape for Loss: [Batch, Seq, Vocab] -> [Batch * Seq, Vocab]
            let flat_logits_id = g.reshape(logits_id, vec![current_batch_size * seq_len, vocab_size]);
            
            // 3. Compute Loss
            let loss_id = g.cross_entropy(flat_logits_id, &y_batch);
            
            // Pull the scalar loss to the CPU to log it
            let loss_val = g.tensors[loss_id].sync_to_cpu()[0];
            total_loss += loss_val;
            num_batches += 1;

            // 4. Backward Pass & Step
            optim.zero_grad(&mut g);
            g.backward(loss_id);
            optim.step(&mut g);
            
            // 5. Checkpoint & Reclaim VRAM
            // This drops all intermediate activations and gradients from this step 
            // back into the `vram_pool`, ensuring zero allocations on the next step!
            g.clear_activations();
        }

        if epoch % 5 == 0 {
            println!("Epoch {:03} | Avg Loss: {:.6}", epoch, total_loss / num_batches as f32);
        }
    }

    // --- TEXT GENERATION ---
    println!("\nGenerating from prompt...");
    let prompt = "und der haifisch";
    let mut input_tokens = tokenizer.encode(prompt);
    print!("Prompt: {} ", prompt);

    for _ in 0..20 {
        // Truncate input to maximum context length
        let start = input_tokens.len().saturating_sub(seq_len);
        let context = &input_tokens[start..];
        let current_seq_len = context.len();

        let x_id = g.alloc(vec![1, current_seq_len], context.iter().map(|&t| t as f32).collect());
        let logits_id = model.forward(&mut g, x_id);
        
        // Sync the [1, Seq, Vocab] logits back to CPU
        let logits_data = g.tensors[logits_id].sync_to_cpu();
        
        // We only care about the predictions for the VERY LAST token in the sequence
        let last_token_logits_start = (current_seq_len - 1) * vocab_size;
        let last_token_logits = &logits_data[last_token_logits_start..last_token_logits_start + vocab_size];
        
        // Greedy decoding (Argmax)
        let predicted_id = last_token_logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap();
        
        print!("{} ", tokenizer.id_to_word[predicted_id]);
        input_tokens.push(predicted_id);
        
        // Don't leak VRAM during inference!
        g.clear_activations();
    }
    println!("\n\nExperiment Complete: Hardware-accelerated Transformer generated text successfully!");
}
