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

/// Custom Bare-Metal Byte-Pair Encoding (BPE) Tokenizer
struct BPETokenizer {
    vocab: Vec<String>,
    merges: HashMap<(usize, usize), usize>,
}

impl BPETokenizer {
    fn train(text: &str, target_vocab_size: usize) -> Self {
        let mut vocab = Vec::new();
        let mut char_to_id = HashMap::new();
        let mut sequence = Vec::new();

        // 1. Initialize vocabulary with all unique characters
        for ch in text.chars() {
            let ch_str = ch.to_string();
            let id = *char_to_id.entry(ch_str.clone()).or_insert_with(|| {
                let new_id = vocab.len();
                vocab.push(ch_str);
                new_id
            });
            sequence.push(id);
        }

        let mut merges = HashMap::new();
        let num_merges = target_vocab_size.saturating_sub(vocab.len());

        println!("BPE Training: Initial character vocab size: {}", vocab.len());

        // 2. Iteratively merge the most frequent adjacent token pairs
        for _ in 0..num_merges {
            let mut pair_counts = HashMap::new();
            for window in sequence.windows(2) {
                *pair_counts.entry((window[0], window[1])).or_insert(0) += 1;
            }

            if let Some((&best_pair, &count)) = pair_counts.iter().max_by_key(|&(_, count)| count) {
                if count < 2 { break; } // No more repeating patterns to merge

                let new_id = vocab.len();
                let new_token = format!("{}{}", vocab[best_pair.0], vocab[best_pair.1]);
                vocab.push(new_token);
                merges.insert(best_pair, new_id);

                // Apply the learned merge to the training sequence
                let mut new_sequence = Vec::with_capacity(sequence.len());
                let mut i = 0;
                while i < sequence.len() {
                    if i < sequence.len() - 1 && (sequence[i], sequence[i+1]) == best_pair {
                        new_sequence.push(new_id);
                        i += 2;
                    } else {
                        new_sequence.push(sequence[i]);
                        i += 1;
                    }
                }
                sequence = new_sequence;
            } else {
                break;
            }
        }

        Self { vocab, merges }
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        let mut sequence = Vec::new();
        for ch in text.chars() {
            let ch_str = ch.to_string();
            // Simple fallback: if character wasn't in training text, we skip it
            if let Some(pos) = self.vocab.iter().position(|v| v == &ch_str) {
                sequence.push(pos);
            }
        }

        if sequence.is_empty() { return sequence; }

        // Recursively apply merges in order of priority (lowest ID = earliest learned)
        loop {
            let mut best_pair = None;
            let mut min_rank = usize::MAX;

            for i in 0..sequence.len() - 1 {
                let pair = (sequence[i], sequence[i+1]);
                if let Some(&new_id) = self.merges.get(&pair) {
                    if new_id < min_rank {
                        min_rank = new_id;
                        best_pair = Some(pair);
                    }
                }
            }

            if let Some(pair) = best_pair {
                let new_id = self.merges[&pair];
                let mut new_sequence = Vec::with_capacity(sequence.len());
                let mut i = 0;
                while i < sequence.len() {
                    if i < sequence.len() - 1 && (sequence[i], sequence[i+1]) == pair {
                        new_sequence.push(new_id);
                        i += 2;
                    } else {
                        new_sequence.push(sequence[i]);
                        i += 1;
                    }
                }
                sequence = new_sequence;
            } else {
                break;
            }
        }
        sequence
    }

    fn decode(&self, ids: &[usize]) -> String {
        ids.iter().map(|&id| self.vocab.get(id).cloned().unwrap_or_default()).collect()
    }

    /// Exports the tokenizer to a Hugging Face compatible tokenizer.json file
    fn export_huggingface(&self, file_path: &str) {
        // 1. Build the vocabulary mapping
        let mut hf_vocab = std::collections::HashMap::new();
        for (id, token) in self.vocab.iter().enumerate() {
            hf_vocab.insert(token.clone(), id);
        }

        // 2. Recover merge order by sorting by the newly created token ID
        let mut ordered_merges: Vec<(&(usize, usize), &usize)> = self.merges.iter().collect();
        ordered_merges.sort_by_key(|&(_, &id)| id);

        // 3. Format merges as space-separated strings
        let mut hf_merges = Vec::new();
        for (&(left_id, right_id), _) in ordered_merges {
            let left_token = &self.vocab[left_id];
            let right_token = &self.vocab[right_id];
            hf_merges.push(format!("{} {}", left_token, right_token));
        }

        // 4. Construct the standard Hugging Face tokenizer JSON schema
        let hf_tokenizer = serde_json::json!({
            "version": "1.0",
            "model": {
                "type": "BPE",
                "vocab": hf_vocab,
                "merges": hf_merges
            }
        });

        // 5. Write to disk
        std::fs::write(
            file_path,
            serde_json::to_string_pretty(&hf_tokenizer).expect("Failed to serialize JSON"),
        ).expect("Failed to write tokenizer.json to disk");
        
        println!("Exported Hugging Face tokenizer to {}", file_path);
    }
}

fn main() {
    use crate::nn::transformer::TransformerLM;
    println!("=== Eisen Phase 5: BPE Tokenization + Transformer LLM on GPU ===");
    
    let device = setup_gpu().expect("This Transformer example requires a CUDA GPU!");
    println!("Device: {:?}", device);

    let file_path = "data/german_corpus.txt";
    let raw_text = fs::read_to_string(file_path)
        .expect("Please ensure data/german_corpus.txt contains the Dreigroschenoper text.");
    
    println!("Loading and tokenizing corpus...");
    
    // Train a BPE Tokenizer with a target vocab of 1024 subwords!
    let vocab_target = 1024;
    let tokenizer = BPETokenizer::train(&raw_text, vocab_target);
    let tokens = tokenizer.encode(&raw_text);
    let vocab_size = tokenizer.vocab.len();
    
    println!("Final BPE Vocab Size: {} | Total Sub-Word Tokens: {}", vocab_size, tokens.len());

    // --- Hyperparameters ---
    let seq_len = 16;       // Context window
    let hidden_dim = 64;    // Embedding dimension
    let num_heads = 4;      // 4 Attention Heads (16 dim per head)
    let ffn_dim = 128;      // Feed-Forward expansion
    let num_layers = 2;     // 2 full Transformer blocks
    let batch_size = 32;
    let epochs = 50;
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

            for b in 0..current_batch_size {
                let start_idx = i + b * seq_len;
                for s in 0..seq_len {
                    x_batch.push(tokens[start_idx + s] as f32);
                    y_batch.push(tokens[start_idx + s + 1]); 
                }
            }

            // 1. Forward Pass
            let x_id = g.alloc(vec![current_batch_size, seq_len], x_batch);
            let logits_id = model.forward(&mut g, x_id);
            
            // 2. Reshape for Loss
            let flat_logits_id = g.reshape(logits_id, vec![current_batch_size * seq_len, vocab_size]);
            
            // 3. Compute Loss
            let loss_id = g.cross_entropy(flat_logits_id, &y_batch);
            
            let loss_val = g.tensors[loss_id].sync_to_cpu()[0];
            total_loss += loss_val;
            num_batches += 1;

            // 4. Backward Pass & Step
            optim.zero_grad(&mut g);
            g.backward(loss_id);
            optim.step(&mut g);
            
            // 5. Clean VRAM Pool
            g.clear_activations();
        }

        if epoch % 5 == 0 {
            println!("Epoch {:03} | Avg Loss: {:.6}", epoch, total_loss / num_batches as f32);
        }
    }

    // --- TEXT GENERATION ---
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
        // Zero VRAM leak hacks needed here either!
    }
    println!("\n\nExperiment Complete: BPE + Transformer generated text successfully!");
}
