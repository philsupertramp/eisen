use eisen::graph::Graph;
use eisen::nn::embedding::Embedding;
use eisen::nn::linear::Linear;
use eisen::nn::rmsnorm::RMSNorm;
use eisen::nn::transformer::TransformerBlock;
use eisen::nn::Module;
use eisen::data::tokenizer::BPETokenizer;
use eisen::tensor::Device;
use cudarc::driver::CudaContext;

use std::fs::File;
use std::io::{self, Read, Write};
use std::sync::Arc;

fn setup_gpu() -> Option<Device> {
    match CudaContext::new(0) {
        Ok(ctx) => Some(Device::Gpu(ctx.clone(), ctx.default_stream())),
        Err(_) => None,
    }
}

// Stelle sicher, dass dies exakt der 13.8M Parameter Architektur entspricht
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

/// Lädt flache f32 Gewichte sicher von der Festplatte
fn load_weights(g: &mut Graph, params: &[usize], path: &str) {
    let mut file = File::open(path).expect("Konnte Modellgewichte nicht finden!");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap();

    let raw_floats: Vec<f32> = buffer
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let mut offset = 0;
    for &p_id in params {
        let size = g.tensors[p_id].size();
        if offset + size > raw_floats.len() {
            panic!("Gewichtsdatei ist abgeschnitten! Erwartete mehr als {} Floats.", raw_floats.len());
        }
        let chunk = &raw_floats[offset..offset + size];
        g.load_tensor_data(p_id, chunk); 
        offset += size;
    }
    println!("Erfolgreich {} Parameter geladen.", offset);
}

fn main() {
    println!("=== Eisen Interactive CLI ===");
    let device = setup_gpu().expect("CUDA wird benötigt!");

    let tokenizer = BPETokenizer::load("data/tokenizer.model").unwrap();
    let vocab_size = tokenizer.vocab.len();

    // 13.8M Param Architektur
    let seq_len = 256;
    let hidden_dim = 384;
    let num_heads = 6;
    let ffn_dim = 1536;
    let num_layers = 6;

    let mut g = Graph::new(device);
    let model = TransformerLM::new(&mut g, vocab_size, hidden_dim, num_heads, ffn_dim, num_layers);
    
    // KRITISCHER FIX: Parameter sperren, damit clear_activations sie nicht löscht!
    g.mark_params(); 
    
    // Autograd-Tracking für Inferenz deaktivieren, um Speicher zu sparen
    g.no_grad = true;
    
    println!("Lade Gewichte vom Checkpoint...");
    load_weights(&mut g, &model.params(), "data/eisen_model.bin");

    println!("\nModell bereit! Gib deinen Prompt unten ein. Tippe 'exit' zum Beenden.");
    println!("---------------------------------------------------------");

    let mut input = String::new();
    loop {
        print!("> ");
        io::stdout().flush().unwrap();
        input.clear();
        io::stdin().read_line(&mut input).unwrap();
        
        let prompt = input.trim();
        if prompt == "exit" { break; }
        if prompt.is_empty() { continue; }

        let mut tokens = tokenizer.encode(prompt);
        print!("Eisen: {}", prompt);
        io::stdout().flush().unwrap();

        for _ in 0..100 {
            let start = tokens.len().saturating_sub(seq_len);
            let context = &tokens[start..];
            let current_seq_len = context.len();

            let x_id = g.alloc(vec![1, current_seq_len], context.iter().map(|&t| t as f32).collect());
            let logits_id = model.forward(&mut g, x_id);
            
            let logits_data = g.tensors[logits_id].sync_to_cpu();
            let last_token_start = (current_seq_len - 1) * vocab_size;
            let last_logits = &logits_data[last_token_start..last_token_start + vocab_size];
            
            let predicted_id = last_logits.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i).unwrap();
            
            print!("{}", tokenizer.decode(&[predicted_id]));
            io::stdout().flush().unwrap();

            tokens.push(predicted_id);
            
            // Tape leeren
            g.clear_activations(); 
            // KRITISCHER OOM FIX: Den VRAM Pool hart leeren, da sich die Größen (seq_len) dynamisch ändern!
            g.vram_pool.clear(); 
        }
        println!("\n");
    }
}
