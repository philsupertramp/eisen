use eisen::graph::Graph;
use eisen::nn::linear::Linear;
use eisen::nn::embedding::Embedding;
use eisen::nn::optim::AdamW;
use eisen::nn::Module;
use std::collections::HashMap;
use std::fs;

/// Simple Word-Level Tokenizer for German text
struct Tokenizer {
    word_to_id: HashMap<String, usize>,
    id_to_word: Vec<String>,
}

impl Tokenizer {
    fn from_text(text: &str) -> Self {
        let mut word_to_id = HashMap::new();
        let mut id_to_word = Vec::new();
        
        // Basic punctuation handling for German
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

/// The Bengio LM Architecture
struct BengioLM {
    window_size: usize,
    hidden_dim: usize,
    embedding: Embedding,
    hidden: Linear,
    head: Linear,
}

impl BengioLM {
    fn new(g: &mut Graph, vocab_size: usize, window_size: usize, hidden_dim: usize) -> Self {
        let embedding = Embedding::new(g, vocab_size, hidden_dim);
        let hidden = Linear::new(g, window_size * hidden_dim, hidden_dim, true);
        let head = Linear::new(g, hidden_dim, vocab_size, true);
        Self { window_size, hidden_dim, embedding, hidden, head }
    }
}

impl Module for BengioLM {
    fn forward(&self, g: &mut Graph, x_id: usize) -> usize {
        let batch_size = g.tensors[x_id].shape[0];
        let x = self.embedding.forward(g, x_id);
        let flat_id = g.reshape(x, vec![batch_size, self.window_size * self.hidden_dim]);
        let h = self.hidden.forward(g, flat_id);
        let act = g.silu(h);
        self.head.forward(g, act)
    }

    fn params(&self) -> Vec<usize> {
        let mut p = Vec::new();
        p.extend(self.embedding.params());
        p.extend(self.hidden.params());
        p.extend(self.head.params());
        p
    }
}

fn main() {
    // 1. Load the text file
    // Note: Create a 'data/german_corpus.txt' file with some German text
    let file_path = "data/german_corpus.txt";
    let raw_text = fs::read_to_string(file_path)
        .expect("Please create data/german_corpus.txt with some German sentences.");
    
    println!("Loading corpus from {}...", file_path);
    let tokenizer = Tokenizer::from_text(&raw_text);
    let tokens = tokenizer.encode(&raw_text);
    println!("Vocab Size: {} | Total Tokens: {}", tokenizer.id_to_word.len(), tokens.len());

    // 2. Setup Hyperparameters
    let window_size = 3;
    let hidden_dim = 64;
    let batch_size = 16;
    let epochs = 50;
    let lr = 0.005;

    let mut g = Graph::default();
    let model = BengioLM::new(&mut g, tokenizer.id_to_word.len(), window_size, hidden_dim);
    let mut optim = AdamW::new(model.params(), lr);

    // 3. Training Loop with Mini-Batching
    println!("Starting training (CPU, zero-dependency)...");
    for epoch in 1..=epochs {
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        // Slide over the tokens to create batches
        for i in (0..tokens.len() - window_size - 1).step_by(batch_size) {
            let end = (i + batch_size).min(tokens.len() - window_size - 1);
            let current_batch_size = end - i;
            
            let mut x_batch = Vec::new();
            let mut y_batch = Vec::new();

            for j in i..end {
                for k in 0..window_size {
                    x_batch.push(tokens[j + k] as f32);
                }
                y_batch.push(tokens[j + window_size]);
            }

            // Forward
            let x_id = g.alloc(vec![current_batch_size, window_size], x_batch);
            let logits_id = model.forward(&mut g, x_id);
            let loss_id = g.cross_entropy(logits_id, &y_batch);
            
            total_loss += g.tensors[loss_id].data.as_cpu()[0];
            num_batches += 1;

            // Backward & Optimize
            optim.zero_grad(&mut g);
            g.backward(loss_id);
            optim.step(&mut g);
            
            // Clear memory
            g.tape.nodes.clear();
            // Optional: for real memory pressure, you'd prune g.tensors here too
        }

        if epoch % 5 == 0 {
            println!("Epoch {:03} | Avg Loss: {:.6}", epoch, total_loss / num_batches as f32);
        }
    }

    // 4. Test Generation
    println!("\nGenerating from prompt...");
    let prompt = "und der haifisch,"; 
    let mut input_tokens = tokenizer.encode(prompt);
    print!("Prompt: {} ", prompt);

    for _ in 0..20 {
        let context = &input_tokens[input_tokens.len() - window_size..];
        let x_id = g.alloc(vec![1, window_size], context.iter().map(|&t| t as f32).collect());
        let logits_id = model.forward(&mut g, x_id);
        
        let predicted_id = g.tensors[logits_id].data.as_cpu().iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap();
        
        print!("{} ", tokenizer.id_to_word[predicted_id]);
        input_tokens.push(predicted_id);
        g.tape.nodes.clear();
    }
    println!();
}
