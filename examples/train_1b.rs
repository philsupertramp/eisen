// examples/train_1b.rs
//
// Eisen Engine — 1.07B Parameter Pre-Training Run
//
// Architecture:
//   hidden=1536 | heads=12 | head_dim=128 | ffn=4096 | layers=48 | vocab=4096
//
// Memory strategy:
//   - Embeddings, final norm, lm_head: always VRAM (frequently accessed, small)
//   - Transformer block matrix weights are demoted to CPU immediately during
//     init to avoid OOM before streaming layout is planned
//   - Tiny RMSNorm scale vectors stay in VRAM (RMSNorm kernels are GPU-only)
//   - plan_streaming() then decides final residency and reports peak temp usage
//   - CPU RAM holds streamed weights (+ grad + Adam moments)
//
// Run:
//   cargo run --release --example train_1b [--features bf16]

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
use std::io::{BufWriter, Write, Read};
use std::sync::{Arc, RwLock};
use std::time::Instant;
use std::net::TcpListener;
use std::thread;
use std::env;

// ─── GPU setup ────────────────────────────────────────────────────────────────

fn setup_gpu() -> Device {
    let ctx    = CudaContext::new(0).expect("CUDA GPU required for 1B training");
    let stream = ctx.default_stream();
    Device::Gpu(ctx, stream)
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

// ─── EisenBoard (carried over from train_large_lm.rs unchanged) ───────────────

#[derive(Clone, Default)]
struct TrainStats {
    step:           usize,
    loss:           f32,
    lr:             f32,
    tps:            f32,
    batch_time_ms:  f32,
    total_tokens:   usize,
    history:        Vec<(usize, f32)>,
}

const DASHBOARD_HTML: &str = r#"<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>EisenBoard | 1B</title>
<style>
body{background:#0d1117;color:#c9d1d9;font-family:monospace;padding:2rem;margin:0}
h1{color:#58a6ff;border-bottom:1px solid #30363d;padding-bottom:10px}
.grid{display:flex;gap:15px;margin-bottom:20px;flex-wrap:wrap;max-width:1100px}
.box{background:#161b22;border:1px solid #30363d;padding:15px;border-radius:6px;flex:1 1 150px}
.lbl{font-size:12px;color:#8b949e;text-transform:uppercase;letter-spacing:1px}
.val{font-size:24px;color:#7ee787;font-weight:bold;margin-top:5px}
.val.lr{color:#f78166}
canvas{background:#161b22;border:1px solid #30363d;border-radius:6px;width:100%;max-width:1100px;height:400px}
</style></head><body>
<h1>EisenBoard 🚀  <small style="font-size:14px;color:#8b949e">1.07B — Eisen Engine</small></h1>
<div class="grid">
  <div class="box"><div class="lbl">STEP</div><div id="s" class="val">0</div></div>
  <div class="box"><div class="lbl">LOSS</div><div id="l" class="val">0.0000</div></div>
  <div class="box"><div class="lbl">LR</div><div id="r" class="val lr">0</div></div>
  <div class="box"><div class="lbl">TOK/SEC</div><div id="t" class="val">0</div></div>
  <div class="box"><div class="lbl">BATCH ms</div><div id="b" class="val">0</div></div>
  <div class="box"><div class="lbl">TOTAL TOKENS</div><div id="k" class="val">0</div></div>
</div>
<canvas id="c" width="1100" height="400"></canvas>
<script>
const ctx=document.getElementById('c').getContext('2d');
function draw(h){
  ctx.clearRect(0,0,1100,400);if(!h||h.length<2)return;
  ctx.strokeStyle='#30363d';ctx.lineWidth=1;
  for(let i=0;i<=10;i++){ctx.beginPath();ctx.moveTo(0,i*40);ctx.lineTo(1100,i*40);ctx.stroke();}
  let mn=Math.min(...h.map(d=>d[1]))*0.98,mx=Math.max(...h.map(d=>d[1]))*1.02,rng=mx-mn||1;
  ctx.strokeStyle='#ff7b72';ctx.lineWidth=2;ctx.beginPath();
  h.forEach((p,i)=>{let x=i/(h.length-1)*1100,y=400-((p[1]-mn)/rng*400);i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);});
  ctx.stroke();
}
setInterval(async()=>{
  try{const d=await(await fetch('/api/stats')).json();
  document.getElementById('s').innerText=d.step.toLocaleString();
  document.getElementById('l').innerText=d.loss.toFixed(4);
  document.getElementById('r').innerText=d.lr.toExponential(2);
  document.getElementById('t').innerText=Math.round(d.tps).toLocaleString();
  document.getElementById('b').innerText=Math.round(d.batch_ms)+' ms';
  let tk=d.total_tokens;
  document.getElementById('k').innerText=tk>1e9?(tk/1e9).toFixed(2)+'B':tk>1e6?(tk/1e6).toFixed(2)+'M':tk.toLocaleString();
  draw(d.history);}catch(e){}
},1000);
</script></body></html>"#;

fn spawn_eisenboard(stats: Arc<RwLock<TrainStats>>) {
    thread::spawn(move || {
        let listener = TcpListener::bind("0.0.0.0:8080")
            .expect("EisenBoard: failed to bind port 8080");
        println!("🌐  EisenBoard: http://localhost:8080\n");
        for stream in listener.incoming() {
            if let Ok(mut s) = stream {
                let _ = s.set_read_timeout(Some(std::time::Duration::from_millis(500)));
                let mut buf = [0u8; 1024];
                if s.read(&mut buf).unwrap_or(0) > 0 {
                    let req = String::from_utf8_lossy(&buf);
                    if req.starts_with("GET /api/stats") {
                        let st = stats.read().unwrap();
                        let hist: Vec<String> = st.history.iter()
                            .map(|(step, loss)| format!("[{},{}]", step, loss))
                            .collect();
                        let json = format!(
                            r#"{{"step":{},"loss":{:.6},"lr":{:.8},"tps":{:.1},"batch_ms":{:.1},"total_tokens":{},"history":[{}]}}"#,
                            st.step, st.loss, st.lr, st.tps, st.batch_time_ms,
                            st.total_tokens, hist.join(",")
                        );
                        let _ = s.write_all(
                            format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}", json)
                            .as_bytes()
                        );
                    } else if req.starts_with("GET / ") {
                        let _ = s.write_all(
                            format!("HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n{}", DASHBOARD_HTML)
                            .as_bytes()
                        );
                    }
                }
            }
        }
    });
}

// ─── Model ────────────────────────────────────────────────────────────────────

struct TransformerLM {
    token_emb: Embedding,
    blocks:    Vec<TransformerBlock>,
    norm_f:    RMSNorm,
    lm_head:   Linear,
}

impl TransformerLM {
    fn new(
        g:          &mut Graph,
        vocab_size: usize,
        hidden_dim: usize,
        num_heads:  usize,
        ffn_dim:    usize,
        num_layers: usize,
    ) -> Self {
        let token_emb = Embedding::new(g, vocab_size, hidden_dim);

        // Build blocks one-by-one and immediately demote their parameters to
        // CPU so model init does not require full-model VRAM residency.
        let mut blocks = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let block = TransformerBlock::new(g, hidden_dim, num_heads, ffn_dim);
            for pid in block.params() {
                // Keep tiny 1D norm scales resident; stream only matrix weights.
                // This avoids hitting GPU-only RMSNorm kernels with CPU weights.
                if g.tensors[pid].shape.len() == 2 {
                    g.demote_tensor_to_cpu(pid);
                }
            }
            blocks.push(block);
        }

        let norm_f  = RMSNorm::new(g, hidden_dim, 1e-5);
        let lm_head = Linear::new(g, hidden_dim, vocab_size, false);
        Self { token_emb, blocks, norm_f, lm_head }
    }
}

impl Module for TransformerLM {
    fn forward(&self, g: &mut Graph, x_id: usize) -> usize {
        let mut h = self.token_emb.forward(g, x_id);
        for block in &self.blocks {
            h = block.forward_with_mask(g, h, true);
        }
        h = self.norm_f.forward(g, h);
        self.lm_head.forward(g, h)
    }
    fn params(&self) -> Vec<usize> {
        let mut p = self.token_emb.params();
        for b in &self.blocks { p.extend(b.params()); }
        p.extend(self.norm_f.params());
        p.extend(self.lm_head.params());
        p
    }
}

// ─── Checkpoint I/O ───────────────────────────────────────────────────────────

fn save_weights(g: &Graph, path: &str) {
    println!("Saving checkpoint to {}…", path);
    let f   = File::create(path).expect("Cannot create checkpoint file");
    let mut w = BufWriter::new(f);
    for i in 0..g.num_params {
        let data = g.tensors[i].sync_to_cpu();
        for &v in &data {
            w.write_all(&v.to_le_bytes()).unwrap();
        }
    }
    w.flush().unwrap();
    println!("Checkpoint saved ({} param tensors).", g.num_params);
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Eisen Engine — 1.07B Parameter Pre-Training            ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    let device       = setup_gpu();
    let shared_stats = Arc::new(RwLock::new(TrainStats::default()));
    spawn_eisenboard(shared_stats.clone());

    // ── Paths ────────────────────────────────────────────────────────────────
    let tokenizer_path = "data/tokenizer.model";
    let bin_path       = "data/german_large_corpus.bin";
    let ckpt_path      = "data/eisen_1b.bin";

    // ── Tokenizer ────────────────────────────────────────────────────────────
    println!("\nLoading tokenizer…");
    let tokenizer = Arc::new(
        BPETokenizer::load(tokenizer_path)
            .expect("Tokenizer not found — run examples/train_tokenizer.rs first"),
    );
    let vocab_size = tokenizer.vocab.len();
    println!("Vocab size: {}", vocab_size);

    // ── Architecture ─────────────────────────────────────────────────────────
    //
    // 1.07B parameters:
    //   Transformer blocks : 48 × (4×1536² + 2×1536×4096 + 2×1536) = 1.057B
    //   Embedding + head   : 2 × 4096 × 1536                        = 12.6M
    //   Final norm         : 1536                                    = ~0
    //   ──────────────────────────────────────────────────────────────────────
    //   Total              : ~1.07B
    //
    let hidden_dim = 1536;
    let num_heads  = 12;     // head_dim = 128
    let ffn_dim    = 4096;
    let num_layers = 48;

    // ── Training hyperparameters ─────────────────────────────────────────────
    //
    // Effective batch = micro_batch × accum_steps = 2 × 8 = 16 sequences
    // Effective tokens per step = 16 × 128 = 2,048
    //
    // LR recipe (Chinchilla-style):
    //   Warmup 1,000 steps → cosine decay to lr_min over 100,000 steps
    //   Peak lr 3e-4 is standard for models in this range with AdamW
    //
    let seq_len          = env_usize("EISEN_SEQ_LEN", 128);
    // Keep this conservative by default: attention transpose/bmm activations
    // can still OOM at 4 on smaller consumer GPUs even with streaming.
    let micro_batch_size = env_usize("EISEN_MICRO_BATCH", 2);
    let accum_steps      = env_usize("EISEN_ACCUM_STEPS", 8);
    let effective_batch  = micro_batch_size * accum_steps;
    let tokens_per_step  = effective_batch * seq_len;

    let total_steps  = 100_000_usize;
    let warmup_steps = 1_000_usize;
    let lr_max       = 3e-4_f32;
    let lr_min       = 3e-5_f32;
    let scheduler    = CosineScheduler::new(lr_max, lr_min, warmup_steps, total_steps);

    let save_interval = 2_500_usize;
    let log_interval  = 50_usize;

    // ── Model + streaming layout ──────────────────────────────────────────────
    println!("\nBuilding model…");
    let mut g = Graph::new(device);
    let model = TransformerLM::new(
        &mut g, vocab_size, hidden_dim, num_heads, ffn_dim, num_layers,
    );

    // Lock the parameter watermark before plan_streaming inspects it
    g.mark_params();

    println!("Pinning model...");
    // Pinned params: always stay in VRAM.
    // - token_emb: looked up every forward pass, very cheap to keep resident
    // - norm_f:    tiny (1536 floats), used in final layer
    // - lm_head:   large but accessed every step; streaming it would add
    //              vocab×hidden×3 PCIe traffic every micro-batch
    let pinned: Vec<usize> = model.token_emb.params()
        .into_iter()
        .chain(model.norm_f.params())
        .chain(model.lm_head.params())
        .collect();

    println!("Planing memory streaming...");
    let vram_budget_mb   = env_usize("EISEN_VRAM_BUDGET_MB", 5120);
    let reserve_mb       = env_usize("EISEN_ACTIVATION_RESERVE_MB", 500);
    let report = g.plan_streaming(
        vram_budget_mb * 1024 * 1024,
        reserve_mb * 1024 * 1024,
        &pinned,
    );
    println!("{}", report);

    // ── Optimizer (AFTER plan_streaming) ─────────────────────────────────────
    // Moment buffers are lazily initialised on first step.
    // plan_streaming already converted overflow params to Storage::Cpu,
    // so AdamW will correctly allocate CPU moments for those params.
    let mut optim = AdamW::new(model.params(), scheduler.get_lr(0));

    println!("\nArchitecture summary:");
    println!("  Layers:  {} | Hidden: {} | Heads: {} | FFN: {}", num_layers, hidden_dim, num_heads, ffn_dim);
    println!("  Micro-batch: {} | Accum: {} | Effective batch: {}", micro_batch_size, accum_steps, effective_batch);
    println!("  Tokens/step: {} | Total steps: {}", tokens_per_step, total_steps);
    println!("  LR: {:.1e} → {:.1e} over {} steps ({} warmup)",
        lr_max, lr_min, total_steps, warmup_steps);

    #[cfg(feature = "bf16")]
    println!("  Precision: BF16 forward + FP32 accumulation (streaming-aware)");
    #[cfg(not(feature = "bf16"))]
    println!("  Precision: FP32");

    // ── Training loop ─────────────────────────────────────────────────────────
    println!("\nStarting training…");
    let mut dataloader   = BinaryDataLoader::new(bin_path, seq_len, micro_batch_size);
    let mut step         = 0_usize;
    let mut running_loss = 0.0_f32;
    let mut total_tokens = 0_usize;
    let mut last_log     = Instant::now();
    let mut board_timer  = Instant::now();

    'training: loop {
        if step >= total_steps { break; }

        let current_lr = scheduler.get_lr(step);
        optim.lr = current_lr;
        optim.zero_grad(&mut g);

        let mut step_loss   = 0.0_f32;
        let mut micro_count = 0_usize;

        // ── Gradient accumulation inner loop ──────────────────────────────────
        //
        // zero_grad is called ONCE per optimizer step, before this loop.
        // Each micro-batch backward accumulates into the same grad buffers:
        //   - GPU-resident params: atomicAdd in VRAM
        //   - CPU-streamed params: Vec += in matmul_streamed backward closure
        //
        // With checkpointing: no_grad forward → restore_save_point → recompute
        // is NOT used here because the streaming temp itself is sync-freed
        // immediately, so the VRAM overhead without checkpointing is acceptable
        // for this micro_batch_size. Add gradient checkpointing if you scale
        // to micro_batch_size > 4 or seq_len > 512.
        for _ in 0..accum_steps {
            let batch = match dataloader.next_batch() {
                Some(b) => b,
                None => {
                    println!("\nDataset exhausted at step {}.", step);
                    break 'training;
                }
            };
            let (x_batch, y_batch) = batch;
            let tokens_this_micro  = micro_batch_size * seq_len;

            let x_id        = g.alloc(vec![micro_batch_size, seq_len], x_batch);
            let logits_id   = model.forward(&mut g, x_id);
            let flat_logits = g.reshape(logits_id, vec![tokens_this_micro, vocab_size]);
            let loss_id     = g.cross_entropy(flat_logits, &y_batch);

            step_loss += g.tensors[loss_id].sync_to_cpu()[0];
            micro_count += 1;

            g.backward(loss_id);
            g.clear_activations();
        }

        optim.step(&mut g);
        step += 1;
        total_tokens += tokens_per_step;

        let avg_loss = step_loss / micro_count as f32;
        running_loss += avg_loss;

        // ── Logging ───────────────────────────────────────────────────────────
        if step % log_interval == 0 {
            let elapsed     = last_log.elapsed().as_secs_f32();
            let throughput  = (log_interval * tokens_per_step) as f32 / elapsed.max(1e-6);
            let avg         = running_loss / log_interval as f32;

            println!(
                "Step {:06} | Loss {:.4} | LR {:.2e} | {:.0} tok/s",
                step, avg, current_lr, throughput
            );

            running_loss = 0.0;
            last_log     = Instant::now();
        }

        // ── EisenBoard telemetry ──────────────────────────────────────────────
        if step % 10 == 0 {
            let elapsed       = board_timer.elapsed().as_secs_f32();
            let tps           = (10 * tokens_per_step) as f32 / elapsed.max(1e-6);
            let batch_time_ms = elapsed * 100.0;   // ms per step

            if let Ok(mut s) = shared_stats.write() {
                s.step          = step;
                s.loss          = avg_loss;
                s.lr            = current_lr;
                s.tps           = tps;
                s.batch_time_ms = batch_time_ms;
                s.total_tokens  = total_tokens;
                if s.history.len() >= 300 { s.history.remove(0); }
                s.history.push((step, avg_loss));
            }
            board_timer = Instant::now();
        }

        // ── Checkpoint ────────────────────────────────────────────────────────
        if step % save_interval == 0 {
            save_weights(&g, ckpt_path);
        }
    }

    // Final save
    save_weights(&g, ckpt_path);

    // ── Generation smoke test ─────────────────────────────────────────────────
    println!("\nSanity-check generation…");
    let prompt = "Die künstliche Intelligenz";
    let mut tokens = tokenizer.encode(prompt);
    print!("{}", prompt);

    g.no_grad = true;
    for _ in 0..80 {
        let start   = tokens.len().saturating_sub(seq_len);
        let context = &tokens[start..];
        let csl     = context.len();

        let x_id      = g.alloc(vec![1, csl], context.iter().map(|&t| t as f32).collect());
        let logits_id = model.forward(&mut g, x_id);
        let logits    = g.tensors[logits_id].sync_to_cpu();

        let predicted = logits[(csl - 1) * vocab_size..]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        print!("{}", tokenizer.decode(&[predicted]));
        tokens.push(predicted);
        g.clear_activations();
    }

    println!("\n\nRun complete. Weights at {}", ckpt_path);
}
