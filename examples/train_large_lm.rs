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

use std::fs::File;
use std::fs;
use std::sync::{Arc, RwLock};
use std::path::Path;
use std::time::Instant;
use std::net::TcpListener;
use std::io::{Read, Write, BufWriter};
use std::thread;

fn setup_gpu() -> Option<Device> {
    match CudaContext::new(0) {
        Ok(ctx) => {
            let stream = ctx.default_stream();
            Some(Device::Gpu(ctx, stream))
        }
        Err(_) => None,
    }
}

// =========================================================================
// === EISENBOARD: ZERO-DEPENDENCY RAW TCP DASHBOARD ===
// =========================================================================

#[derive(Clone, Default)]
struct TrainStats {
    epoch: usize,
    step: usize,
    loss: f32,
    tps: f32, // Tokens per second
}

const DASHBOARD_HTML: &str = r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>EisenBoard | Live Training</title>
    <style>
        body { background-color: #0d1117; color: #c9d1d9; font-family: monospace; padding: 2rem; margin: 0; }
        h1 { color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 10px; }
        .stats-grid { display: flex; gap: 20px; margin-bottom: 20px; }
        .stat-box { background: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 6px; min-width: 150px; }
        .stat-value { font-size: 24px; color: #7ee787; font-weight: bold; margin-top: 5px; }
        canvas { background: #161b22; border: 1px solid #30363d; border-radius: 6px; width: 100%; max-width: 1000px; height: 400px; }
    </style>
</head>
<body>
    <h1>EisenBoard 🚀</h1>
    <div class="stats-grid">
        <div class="stat-box"><div>EPOCH</div><div id="epoch" class="stat-value">0</div></div>
        <div class="stat-box"><div>STEP</div><div id="step" class="stat-value">0</div></div>
        <div class="stat-box"><div>LOSS</div><div id="loss" class="stat-value">0.0000</div></div>
        <div class="stat-box"><div>TOKENS / SEC</div><div id="tps" class="stat-value">0</div></div>
    </div>
    <canvas id="lossChart" width="1000" height="400"></canvas>

    <script>
        const ctx = document.getElementById('lossChart').getContext('2d');
        let history = [];

        function drawChart() {
            ctx.clearRect(0, 0, 1000, 400);
            if (history.length < 2) return;

            // Draw grid
            ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
            for(let i=0; i<10; i++) { ctx.beginPath(); ctx.moveTo(0, i*40); ctx.lineTo(1000, i*40); ctx.stroke(); }

            // Find min/max for scaling
            let minLoss = Math.min(...history.map(d => d.loss)) * 0.95;
            let maxLoss = Math.max(...history.map(d => d.loss)) * 1.05;
            let range = maxLoss - minLoss;
            if (range === 0) range = 1;

            // Draw line
            ctx.strokeStyle = '#ff7b72'; ctx.lineWidth = 2; ctx.beginPath();
            history.forEach((point, i) => {
                let x = (i / (Math.max(history.length - 1, 1))) * 1000;
                let y = 400 - (((point.loss - minLoss) / range) * 400);
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            });
            ctx.stroke();
        }

        setInterval(async () => {
            try {
                const res = await fetch('/api/stats');
                const data = await res.json();
                
                document.getElementById('epoch').innerText = data.epoch;
                document.getElementById('step').innerText = data.step;
                document.getElementById('loss').innerText = data.loss.toFixed(4);
                document.getElementById('tps').innerText = Math.round(data.tps).toLocaleString();

                // Only add new steps to history
                if (history.length === 0 || history[history.length - 1].step !== data.step) {
                    history.push(data);
                    if (history.length > 100) history.shift(); // Keep last 100 points
                    drawChart();
                }
            } catch (e) { console.log("Waiting for engine..."); }
        }, 1000);
    </script>
</body>
</html>
"#;

fn spawn_eisenboard(stats: Arc<RwLock<TrainStats>>) {
    thread::spawn(move || {
        let listener = TcpListener::bind("0.0.0.0:8080").expect("Failed to bind EisenBoard to port 8080");
        println!("\n🌐 EisenBoard Live! Open http://localhost:8080 in your browser\n");

        for stream in listener.incoming() {
            if let Ok(mut stream) = stream {
                // Prevent hanging on broken connections
                let _ = stream.set_read_timeout(Some(std::time::Duration::from_millis(500)));
                let mut buffer = [0; 1024];
                
                if stream.read(&mut buffer).unwrap_or(0) > 0 {
                    let request = String::from_utf8_lossy(&buffer[..]);
                    
                    if request.starts_with("GET /api/stats") {
                        // Serve JSON API
                        let s = stats.read().unwrap();
                        let json = format!(
                            r#"{{"epoch":{},"step":{},"loss":{:.6},"tps":{:.2}}}"#,
                            s.epoch, s.step, s.loss, s.tps
                        );
                        let response = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
                            json
                        );
                        let _ = stream.write_all(response.as_bytes());
                    } else if request.starts_with("GET / ") {
                        // Serve HTML Dashboard
                        let response = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n{}",
                            DASHBOARD_HTML
                        );
                        let _ = stream.write_all(response.as_bytes());
                    }
                }
            }
        }
    });
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

    // Set up shared stats for the dashboard and spawn the server thread!
    let shared_stats = Arc::new(RwLock::new(TrainStats::default()));
    spawn_eisenboard(shared_stats.clone());

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
    let mut timer = Instant::now();

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

        // --- UPDATE EISENBOARD EVERY 10 STEPS ---
        if step % 10 == 0 {
            let elapsed = timer.elapsed().as_secs_f32();
            let tokens_processed = (batch_size * seq_len * 10) as f32;
            let current_tps = if elapsed > 0.0 { tokens_processed / elapsed } else { 0.0 };
            
            // Acquire write lock, update, and drop instantly
            if let Ok(mut s) = shared_stats.write() {
                s.epoch = 1;
                s.step = step;
                s.loss = loss;
                s.tps = current_tps;
            }
            
            timer = Instant::now(); // Reset timer
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
