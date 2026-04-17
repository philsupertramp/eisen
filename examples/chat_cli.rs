use eisen::graph::Graph;
use eisen::nn::embedding::Embedding;
use eisen::nn::linear::Linear;
use eisen::nn::rmsnorm::RMSNorm;
use eisen::nn::transformer::{TransformerBlock, TransformerLM};
use eisen::nn::Module;
use eisen::data::tokenizer::BPETokenizer;
use eisen::tensor::Device;
use cudarc::driver::CudaContext;

use serde::Deserialize;
use std::env;
use std::fs::File;
use std::io::{self, Read, Write};
use std::sync::Arc;
use rand::Rng; // Requires `rand` crate in Cargo.toml

// --- Configuration Structs ---

#[derive(Deserialize, Debug)]
struct RunManifest {
    hyperparams: Hyperparams,
}

#[derive(Deserialize, Debug)]
struct Hyperparams {
    hidden_dim: usize,
    num_heads: usize,
    ffn_dim: usize,
    num_layers: usize,
    seq_len: usize,
}

// --- Advanced Sampling Configuration & Logic ---

#[derive(Clone, Debug)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.1,
        }
    }
}

pub fn sample_logits(logits: &mut [f32], context: &[usize], config: &SamplerConfig) -> usize {
    // 1. Repetition Penalty
    if config.repetition_penalty != 1.0 {
        for &token_id in context {
            let score = logits[token_id];
            // If score is negative, multiply to penalize (make more negative). 
            // If positive, divide to penalize (make closer to 0).
            logits[token_id] = if score < 0.0 {
                score * config.repetition_penalty
            } else {
                score / config.repetition_penalty
            };
        }
    }

    // 2. Greedy fallback if temperature is effectively 0
    if config.temperature < 1e-4 {
        return logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap();
    }

    // 3. Temperature Scaling
    for logit in logits.iter_mut() {
        *logit /= config.temperature;
    }

    // Create a vector of (index, logit) so we can sort and truncate
    let mut pairs: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();

    // Sort descending by logit
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // 4. Top-K Sampling
    if config.top_k > 0 && config.top_k < pairs.len() {
        pairs.truncate(config.top_k);
    }

    // Softmax over the remaining Top-K logits
    let max_logit = pairs.first().unwrap().1;
    let mut probs: Vec<f32> = Vec::with_capacity(pairs.len());
    let mut exp_sum = 0.0;
    
    for &(_, logit) in &pairs {
        let exp = (logit - max_logit).exp();
        probs.push(exp);
        exp_sum += exp;
    }
    
    for prob in probs.iter_mut() {
        *prob /= exp_sum;
    }

    // 5. Top-P (Nucleus) Sampling
    if config.top_p < 1.0 {
        let mut cumsum = 0.0;
        let mut cutoff_idx = probs.len();
        
        for (i, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if cumsum > config.top_p {
                cutoff_idx = i + 1;
                break;
            }
        }
        
        pairs.truncate(cutoff_idx);
        probs.truncate(cutoff_idx);

        // Re-normalize probabilities after truncation
        let p_sum: f32 = probs.iter().sum();
        for prob in probs.iter_mut() {
            *prob /= p_sum;
        }
    }

    // 6. Sample from the resulting distribution
    let mut rng = rand::thread_rng();
    let r: f32 = rng.r#gen();
    let mut cumsum = 0.0;
    
    for (i, &prob) in probs.iter().enumerate() {
        cumsum += prob;
        if r <= cumsum {
            return pairs[i].0;
        }
    }

    // Fallback to the last valid token just in case of precision issues
    pairs.last().unwrap().0
}


// --- Framework Helpers ---

fn setup_gpu() -> Option<Device> {
    match CudaContext::new(0) {
        Ok(ctx) => { let stream = ctx.default_stream(); Some(Device::Gpu(ctx, stream)) }
        Err(_) => None,
    }
}

/// Loads flat f32 weights safely from disk
fn load_weights(g: &mut Graph, params: &[usize], path: &str) {
    let mut file = File::open(path).expect("Could not find model weights!");
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
            panic!("Weight file is truncated! Expected more than {} floats.", raw_floats.len());
        }
        let chunk = &raw_floats[offset..offset + size];
        g.load_tensor_data(p_id, chunk); 
        offset += size;
    }
    println!("Successfully loaded {} parameters.", offset);
}

fn main() {
    println!("=== Eisen Interactive CLI ===");
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <cpu|gpu>", args[0]);
        std::process::exit(1);
    }
    let device_type = &args[1];

    let mut tokenizer_file = "./data/tokenizer.model";
    let mut model_file = "data/eisen_model.bin";

    if args.len() > 2 {
        if args.len() < 4 {
            eprintln!("Usage: {} <cpu|gpu> <tokenizer_file> <model_file>", args[0]);
            std::process::exit(1);
        }
        tokenizer_file = &args[2];
        model_file = &args[3];
    }

    let device: Device = if device_type == "gpu" {
        println!("Using GPU");
        setup_gpu().expect("CUDA is required!")
    } else {
        println!("Using CPU");
        Device::Cpu
    };

    println!("Loading tokenizer...");
    let tokenizer = BPETokenizer::load(tokenizer_file).unwrap();
    let vocab_size = tokenizer.vocab.len();

    // Dynamically load architecture parameters from the run manifest
    println!("Reading model configuration from run manifest...");
    let manifest_path = "data/run_manifest.json";
    let manifest_file = File::open(manifest_path).expect("Could not find data/run_manifest.json! Please ensure training has completed.");
    let manifest: RunManifest = serde_json::from_reader(manifest_file).expect("Failed to parse run_manifest.json");

    let seq_len = manifest.hyperparams.seq_len;
    let hidden_dim = manifest.hyperparams.hidden_dim;
    let num_heads = manifest.hyperparams.num_heads;
    let num_layers = manifest.hyperparams.num_layers;
    let ffn_dim = manifest.hyperparams.ffn_dim;

    println!("Loaded Architecture: Layers={}, Hidden={}, Heads={}, FFN={}, SeqLen={}", 
             num_layers, hidden_dim, num_heads, ffn_dim, seq_len);

    let mut g = Graph::new(device);
    let model = TransformerLM::new(&mut g, vocab_size, hidden_dim, num_heads, ffn_dim, num_layers);
    
    // CRITICAL FIX: Lock parameters so clear_activations doesn't delete them!
    g.mark_params(); 
    
    // Disable autograd tracking for inference to save memory
    g.no_grad = true;
    
    println!("Loading weights from checkpoint...");
    load_weights(&mut g, &model.params(), model_file);

    // Initialize our advanced sampler config
    let sampler_config = SamplerConfig {
        temperature: 0.1,
        top_k: 40,
        top_p: 0.99,
        repetition_penalty: 1.10,
    };

    println!("\nModel ready! Type your prompt below. Type 'exit' to quit.");
    println!("Sampler Settings: Temp={:.2}, TopK={}, TopP={:.2}, RepPen={:.2}", 
             sampler_config.temperature, sampler_config.top_k, sampler_config.top_p, sampler_config.repetition_penalty);
    println!("---------------------------------------------------------");

    let mut input = String::new();
    loop {
        print!("> ");
        io::stdout().flush().unwrap();
        input.clear();
        io::stdin().read_line(&mut input).unwrap();
        
        let mut prompt: String = "<story>\n".to_string();
        prompt.push_str(input.trim());
        if prompt == "exit" { break; }
        if prompt.is_empty() { continue; }

        prompt.push_str(" ");

        let mut tokens = tokenizer.encode(&prompt);
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
            
            // We clone the logits locally so we can mutate them for temperature & penalties
            let mut last_logits = logits_data[last_token_start..last_token_start + vocab_size].to_vec();

            // Use the advanced sampler rather than basic argmax
            let predicted_id = sample_logits(&mut last_logits, context, &sampler_config);
            
            print!("{}", tokenizer.decode(&[predicted_id]));
            io::stdout().flush().unwrap();

            tokens.push(predicted_id);
            
            // Clear the tape
            g.clear_activations(); 
            // CRITICAL OOM FIX: Hard clear the VRAM pool since sizes (seq_len) change dynamically!
            g.vram_pool.clear(); 
        }
        println!("\n");
    }
}
