use eisen::graph::Graph;
use eisen::nn::embedding::Embedding;
use eisen::nn::linear::Linear;
use eisen::nn::optim::AdamW;
use eisen::nn::rmsnorm::RMSNorm;
use eisen::nn::scheduler::CosineScheduler;
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
        Ok(ctx) => { let stream = ctx.default_stream(); Some(Device::Gpu(ctx, stream)) }
        Err(_) => None,
    }
}

// =========================================================================
// === EISENBOARD: ZERO-DEPENDENCY RAW TCP DASHBOARD ===
// =========================================================================

#[derive(Clone, Default)]
struct TrainStats {
    step: usize,
    loss: f32,
    lr: f32,
    tps: f32,
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
        .stats-grid { display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap; }
        .stat-box { background: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 6px; min-width: 150px; }
        .stat-value { font-size: 24px; color: #7ee787; font-weight: bold; margin-top: 5px; }
        .stat-value.lr { color: #f78166; }
        canvas { background: #161b22; border: 1px solid #30363d; border-radius: 6px; width: 100%; max-width: 1000px; height: 400px; }
    </style>
</head>
<body>
    <h1>EisenBoard 🚀</h1>
    <div class="stats-grid">
        <div class="stat-box"><div>STEP</div><div id="step" class="stat-value">0</div></div>
        <div class="stat-box"><div>LOSS</div><div id="loss" class="stat-value">0.0000</div></div>
        <div class="stat-box"><div>LR</div><div id="lr" class="stat-value lr">0.0000</div></div>
        <div class="stat-box"><div>TOKENS / SEC</div><div id="tps" class="stat-value">0</div></div>
    </div>
    <canvas id="lossChart" width="1000" height="400"></canvas>
    <script>
        const ctx = document.getElementById('lossChart').getContext('2d');
        let history = [];
        function drawChart() {
            ctx.clearRect(0, 0, 1000, 400);
            if (history.length < 2) return;
            ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
            for(let i=0; i<10; i++) { ctx.beginPath(); ctx.moveTo(0, i*40); ctx.lineTo(1000, i*40); ctx.stroke(); }
            let minLoss = Math.min(...history.map(d => d.loss)) * 0.95;
            let maxLoss = Math.max(...history.map(d => d.loss)) * 1.05;
            let range = maxLoss - minLoss || 1;
            ctx.strokeStyle = '#ff7b72'; ctx.lineWidth = 2; ctx.beginPath();
            history.forEach((point, i) => {
                let x = (i / Math.max(history.length - 1, 1)) * 1000;
                let y = 400 - (((point.loss - minLoss) / range) * 400);
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            });
            ctx.stroke();
        }
        setInterval(async () => {
            try {
                const res = await fetch('/api/stats');
                const data = await res.json();
                document.getElementById('step').innerText = data.step;
                document.getElementById('loss').innerText = data.loss.toFixed(4);
                document.getElementById('lr').innerText = data.lr.toExponential(2);
                document.getElementById('tps').innerText = Math.round(data.tps).toLocaleString();
                if (history.length === 0 || history[history.length-1].step !== data.step) {
                    history.push(data);
                    if (history.length > 200) history.shift();
                    drawChart();
                }
            } catch (e) {}
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
                let _ = stream.set_read_timeout(Some(std::time::Duration::from_millis(500)));
                let mut buffer = [0; 1024];
                if stream.read(&mut buffer).unwrap_or(0) > 0 {
                    let request = String::from_utf8_lossy(&buffer[..]);
                    if request.starts_with("GET /api/stats") {
                        let s = stats.read().unwrap();
                        let json = format!(
                            r#"{{"step":{},"loss":{:.6},"lr":{:.8},"tps":{:.2}}}"#,
                            s.step, s.loss, s.lr, s.tps
                        );
                        let response = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
                            json
                        );
                        let _ = stream.write_all(response.as_bytes());
                    } else if request.starts_with("GET / ") {
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

fn save_weights(g: &Graph, path: &str) {
    println!("Saving model weights to {}...", path);
    let file = File::create(path).expect("Failed to create weights file");
    let mut writer = BufWriter::new(file);
    for i in 0..g.num_params {
        let data = g.tensors[i].sync_to_cpu();
        for &val in &data { writer.write_all(&val.to_le_bytes()).unwrap(); }
    }
    writer.flush().unwrap();
    println!("Weights saved.");
}

fn main() {
    println!("==================================================================");
    println!("=== Eisen Engine: Phase 6 — Hyper-Optimized Pre-Training Run  ===");
    println!("==================================================================");

    let device = setup_gpu().expect("CUDA GPU Required!");
    println!("Device initialized: {:?}", device);

    let shared_stats = Arc::new(RwLock::new(TrainStats::default()));
    spawn_eisenboard(shared_stats.clone());

    let tokenizer_path = "data/tokenizer.model";
    let bin_path       = "data/german_large_corpus.bin";
    let output_path    = "data/eisen_model.bin";

    println!("Loading tokenizer from {}...", tokenizer_path);
    let tokenizer = Arc::new(
        BPETokenizer::load(tokenizer_path)
            .expect("Could not load tokenizer. Did you run tools/train_tokenizer.rs?")
    );
    let vocab_size = tokenizer.vocab.len();
    println!("Vocab size: {}", vocab_size);

    // ---------------------------------------------------------------
    // Hyperparameters
    // ---------------------------------------------------------------
    let seq_len    = 256;
    let hidden_dim = 384;
    let num_heads  = 6;
    let ffn_dim    = 1536;
    let num_layers = 6;
    let batch_size = 16;    // Micro-batch size (fits in VRAM)

    // ---- Gradient Accumulation ------------------------------------
    // We accumulate gradients over `accum_steps` micro-batches before
    // calling optim.step(). Effective batch = batch_size * accum_steps.
    // This simulates a much larger batch without increasing VRAM usage,
    // which dramatically reduces gradient noise during training.
    let accum_steps: usize = 4;
    let effective_batch = batch_size * accum_steps;

    // ---- Cosine LR with Warmup ------------------------------------
    // Standard recipe: warm up for ~1% of total steps, then decay.
    let total_steps   = 50_000_usize; // One full run cap
    let warmup_steps  = 500_usize;
    let lr_max        = 3e-4_f32;
    let lr_min        = 3e-5_f32;
    let scheduler     = CosineScheduler::new(lr_max, lr_min, warmup_steps, total_steps);

    let tokens_per_micro  = batch_size * seq_len;
    let tokens_per_step   = tokens_per_micro * accum_steps;

    println!("\nPhase 6 Architecture:");
    println!("  Layers: {} | Hidden: {} | Heads: {} | FFN: {}", num_layers, hidden_dim, num_heads, ffn_dim);
    println!("  Micro-batch: {} | Accum steps: {} | Effective batch: {}", batch_size, accum_steps, effective_batch);
    println!("  Tokens/optimizer step: {}", tokens_per_step);
    println!("  Warmup: {} steps → cosine decay over {} steps", warmup_steps, total_steps);
    println!("  LR: {:.2e} → {:.2e}", lr_max, lr_min);

    let mut g = Graph::new(device);
    let model  = TransformerLM::new(&mut g, vocab_size, hidden_dim, num_heads, ffn_dim, num_layers);

    // Start with the scheduler's initial lr (warmup step 0)
    let mut optim = AdamW::new(model.params(), scheduler.get_lr(0));
    g.mark_params();
    println!("\nVRAM pool active. Parameter tensors locked: {}", g.num_params);

    // ---------------------------------------------------------------
    // Training Loop
    // ---------------------------------------------------------------
    println!("\nStarting training...");
    let mut dataloader  = BinaryDataLoader::new(bin_path, seq_len, batch_size);
    let mut step        = 0_usize;
    let mut running_loss = 0.0_f32;
    let log_interval    = 50;   // Log every N optimizer steps
    let save_interval   = 2500;

    let mut timer        = Instant::now();
    let mut last_log     = Instant::now();

    'training: loop {
        if step >= total_steps { break; }

        // -----------------------------------------------------------
        // Gradient Accumulation Inner Loop
        //
        // We call zero_grad once per optimizer step (not per micro-batch)
        // so that gradients accumulate additively across micro-batches.
        // Each micro-batch's backward pass calls atomicAdd on the same
        // param grad buffers, giving us the gradient sum over all
        // accum_steps * batch_size examples.
        //
        // Because cross_entropy already averages over the batch dimension,
        // summing N averaged-micro-batch gradients is mathematically
        // equivalent to the gradient of the averaged loss over the full
        // effective batch. No manual rescaling needed.
        // -----------------------------------------------------------
        let current_lr = scheduler.get_lr(step);
        optim.lr = current_lr;
        optim.zero_grad(&mut g);

        let mut step_loss = 0.0_f32;
        let mut micro_count = 0_usize;

        for _ in 0..accum_steps {
            match dataloader.next_batch() {
                Some((x_batch, y_batch)) => {
                    let x_id        = g.alloc(vec![batch_size, seq_len], x_batch);
                    let logits_id   = model.forward(&mut g, x_id);
                    let flat_logits = g.reshape(logits_id, vec![tokens_per_micro, vocab_size]);
                    let loss_id     = g.cross_entropy(flat_logits, &y_batch);

                    // Sync loss scalar to CPU for logging (this is a small scalar — negligible overhead)
                    step_loss += g.tensors[loss_id].sync_to_cpu()[0];
                    micro_count += 1;

                    // Backward accumulates into param .grad buffers.
                    // clear_activations() recycles activation VRAM back to the pool
                    // but does NOT touch param grads (indices 0..num_params).
                    g.backward(loss_id);
                    g.clear_activations();
                }
                None => {
                    // Dataset exhausted mid-accumulation: reset and break the outer loop.
                    println!("\n=== End of Dataset Reached at step {} ===", step);
                    break 'training;
                }
            }
        }

        // One optimizer step covers all accumulated micro-batch gradients.
        optim.step(&mut g);
        step += 1;

        let avg_loss = step_loss / micro_count as f32;
        running_loss += avg_loss;

        // -----------------------------------------------------------
        // Logging
        // -----------------------------------------------------------
        if step % log_interval == 0 {
            let elapsed = last_log.elapsed().as_secs_f32();
            let tokens_processed = (log_interval * tokens_per_step) as f32;
            let throughput = tokens_processed / elapsed.max(1e-6);
            let avg = running_loss / log_interval as f32;

            println!(
                "Step {:06} | Loss {:.4} | LR {:.2e} | {:.0} tok/s",
                step, avg, current_lr, throughput
            );

            running_loss = 0.0;
            last_log = Instant::now();
        }

        // -----------------------------------------------------------
        // EisenBoard Telemetry
        // -----------------------------------------------------------
        if step % 10 == 0 {
            let elapsed = timer.elapsed().as_secs_f32();
            let tps = (10 * tokens_per_step) as f32 / elapsed.max(1e-6);
            if let Ok(mut s) = shared_stats.write() {
                s.step = step;
                s.loss = avg_loss;
                s.lr   = current_lr;
                s.tps  = tps;
            }
            timer = Instant::now();
        }

        // -----------------------------------------------------------
        // Checkpoint
        // -----------------------------------------------------------
        if step % save_interval == 0 && step > 0 {
            save_weights(&g, output_path);
        }
    }

    save_weights(&g, output_path);

    // ---------------------------------------------------------------
    // Verification Generation
    // ---------------------------------------------------------------
    println!("\nGenerating from prompt to verify learned syntax...");
    let prompt = "Die künstliche Intelligenz";
    let mut input_tokens = tokenizer.encode(prompt);
    print!("{}", prompt);

    for _ in 0..100 {
        let start = input_tokens.len().saturating_sub(seq_len);
        let context = &input_tokens[start..];
        let current_seq_len = context.len();

        let x_id      = g.alloc(vec![1, current_seq_len], context.iter().map(|&t| t as f32).collect());
        let logits_id = model.forward(&mut g, x_id);

        let logits_data = g.tensors[logits_id].sync_to_cpu();
        let last_start  = (current_seq_len - 1) * vocab_size;
        let last_logits = &logits_data[last_start..last_start + vocab_size];

        let predicted = last_logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap();

        print!("{}", tokenizer.decode(&[predicted]));
        input_tokens.push(predicted);
        g.clear_activations();
    }

    println!("\n\nPhase 6 Run Complete! Weights stored at {}", output_path);
}
