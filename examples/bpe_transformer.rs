use eisen::graph::Graph;
use eisen::nn::optim::AdamW;
use eisen::nn::transformer::TransformerLM;
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
