// examples/train_llm.rs
//
// Eisen Engine - default settings 1.07B Parameter Pre-Training Run
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
//   cargo run --release --example train_llm [--features bf16]

use cudarc::driver::CudaContext;
use eisen::data::dataloader::BinaryDataLoader;
use eisen::data::tokenizer::BPETokenizer;
use eisen::graph::Graph;
use eisen::nn::Module;
use eisen::nn::optim::AdamW;
use eisen::nn::scheduler::CosineScheduler;
use eisen::nn::transformer::TransformerLM;
use eisen::tensor::Device;
use eisen::tools::huggingface::{LlamaConfig, write_llama_config, write_safetensors};
 
use eisen::data::fim::{FimConfig, FimTokens};
use eisen::data::dataloader::BatchResult;

use std::env;
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::{Arc, RwLock};
use std::time::{Instant, UNIX_EPOCH, SystemTime};

use eisenboard::{StepRecord, TrainStats, spawn_eisenboard};

// ─── GPU setup ────────────────────────────────────────────────────────────────

fn setup_gpu() -> Device {
    let ctx = CudaContext::new(0).expect("CUDA GPU required for 1B training");
    let stream = ctx.default_stream();
    Device::Gpu(ctx, stream)
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    env::var(name)
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(default)
}

fn env_bool(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(default)
}

fn apply_reproducibility_controls(deterministic: bool, seed: u64) {
    unsafe { env::set_var("EISEN_SEED", seed.to_string()) };
    if deterministic {
        unsafe { env::set_var("CUBLAS_WORKSPACE_CONFIG", ":4096:8") };
        unsafe { env::set_var("CUDA_LAUNCH_BLOCKING", "1") };
    }
}

fn write_run_manifest(path: &str, content: &str) {
    fs::write(path, content).expect("Failed to write run manifest");
    println!("Run manifest written to {}", path);
}

// ─── Checkpoint I/O ───────────────────────────────────────────────────────────

fn save_weights(g: &Graph, path: &str) {
    println!("Saving checkpoint to {}…", path);
    let f = File::create(path).expect("Cannot create checkpoint file");
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

fn save_hf_bundle(
    g: &Graph,
    model: &TransformerLM,
    vocab_size: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    num_layers: usize,
    num_heads: usize,
    seq_len: usize,
    dir: &str,
) {
    fs::create_dir_all(dir).expect("Failed to create HF output directory");
    let safe_path = format!("{dir}/model.safetensors");
    let cfg_path = format!("{dir}/config.json");
    write_safetensors(g, &model.named_params(), &safe_path).expect("Failed to export safetensors");
    write_llama_config(
        &cfg_path,
        &LlamaConfig {
            vocab_size,
            hidden_size: hidden_dim,
            intermediate_size: ffn_dim,
            num_hidden_layers: num_layers,
            num_attention_heads: num_heads,
            max_position_embeddings: seq_len,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
        },
    )
    .expect("Failed to write HF config");
    println!("HF bundle exported: {}", dir);
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Eisen Engine — 1.07B Parameter Pre-Training            ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    let device = setup_gpu();
    let shared_stats = Arc::new(RwLock::new(TrainStats::default()));
    let board_bind = env::var("EISEN_BOARD_BIND").unwrap_or_else(|_| "0.0.0.0:8080".to_string());
    spawn_eisenboard(shared_stats.clone(), &board_bind);

    // ── Paths ────────────────────────────────────────────────────────────────
    let tokenizer_path = "data/tokenizer.model";
    let bin_path = "data/german_large_corpus.bin";
    let output_path = "data/eisen_model.bin";
    let hf_out_dir = "data/hf_export";

    let grad_clip_norm = env_f32("EISEN_GRAD_CLIP_NORM", 1.0);
    let seed = env_u64("EISEN_SEED", 1337);
    let deterministic = env_bool("EISEN_DETERMINISTIC", true);
    apply_reproducibility_controls(deterministic, seed);

    // ── Tokenizer ────────────────────────────────────────────────────────────
    println!("\nLoading tokenizer…");
    let tokenizer = Arc::new(
        BPETokenizer::load(tokenizer_path)
            .expect("Tokenizer not found — run examples/train_tokenizer.rs first"),
    );
    let base_vocab_size = tokenizer.vocab.len();
    // FIM adds 4 special tokens; the model vocab must be trained-aware of them.
    let fim_overhead = FimTokens::vocab_overhead(); // = 4
    let vocab_size = base_vocab_size + fim_overhead;
    println!("Base vocab: {}  FIM overhead: {}  Total vocab: {}",
             base_vocab_size, fim_overhead, vocab_size);

    // ----- FIM config ----
    let fim_rate    = env_f32("EISEN_FIM_RATE",     0.5);
    let fim_spm     = env_f32("EISEN_FIM_SPM_RATE", 0.5);
    let use_fim     = fim_rate > 0.0;
    let fim_config  = FimConfig::new(base_vocab_size)
        .with_rate(fim_rate)
        .with_spm_rate(fim_spm);
 
    if use_fim {
        println!(
            "FIM enabled — rate={:.0}%, SPM={:.0}%, special tokens: prefix={} suffix={} middle={} pad={}",
            fim_rate * 100.0, fim_spm * 100.0,
            fim_config.tokens.prefix, fim_config.tokens.suffix,
            fim_config.tokens.middle, fim_config.tokens.pad,
        );
    } 
    // ── Architecture ─────────────────────────────────────────────────────────
    let hidden_dim = env_usize("EISEN_HIDDEN_DIM", 32);
    let num_heads = env_usize("EISEN_NUM_HEADS", 4);
    let ffn_dim = env_usize("EISEN_FFN_DIM", 336);
    let num_layers = env_usize("EISEN_NUM_LAYERS", 128);

    // ── Training hyperparameters ─────────────────────────────────────────────
    let seq_len = env_usize("EISEN_SEQ_LEN", 512);
    let micro_batch_size = env_usize("EISEN_MICRO_BATCH", 2);
    let accum_steps = env_usize("EISEN_ACCUM_STEPS", 32);
    let effective_batch = micro_batch_size * accum_steps;
    let tokens_per_step = effective_batch * seq_len;
    let tokens_per_micro = micro_batch_size * seq_len;

    let epochs = env_usize("EISEN_EPOCHS", 1);
    let mut current_epoch = 1;

    let mut dataloader = {
        let dl = BinaryDataLoader::new(bin_path, seq_len, micro_batch_size)
            .with_seed(seed);
        if use_fim {
            dl.with_fim(fim_config.clone())
        } else {
            dl
        }
    };

    // Scale the total steps based on the number of epochs requested
    let total_steps = dataloader.total_batches() * epochs;
    let warmup_steps = 2usize;
    let lr_max = 1e-3_f32;
    let lr_min = 1e-4_f32;
    let scheduler = CosineScheduler::new(lr_max, lr_min, warmup_steps, total_steps);

    let save_interval = 2_500_usize;
    let log_interval = 50_usize;

    // ── Model + streaming layout ──────────────────────────────────────────────
    println!("\nBuilding model…");
    let mut g = Graph::new(device);
    let model = TransformerLM::new(
        &mut g, vocab_size, hidden_dim, num_heads, ffn_dim, num_layers
    );

    g.mark_params();

    println!("Pinning model...");
    let pinned: Vec<usize> = model
        .token_emb
        .params()
        .into_iter()
        .chain(model.norm_f.params())
        .chain(model.lm_head.params())
        .collect();

    println!("Planing memory streaming...");
    let vram_budget_mb = env_usize("EISEN_VRAM_BUDGET_MB", 6144);
    let reserve_mb = env_usize("EISEN_ACTIVATION_RESERVE_MB", 1024);
    let report = g.plan_streaming(
        vram_budget_mb * 1024 * 1024,
        reserve_mb * 1024 * 1024,
        &pinned,
    );
    println!("{}", report);

    g.print_vram_state("Model init.");

    // ── Optimizer ────────────────────────────────────────────────────────────
    let mut optim = AdamW::new(model.params(), scheduler.get_lr(0));
    optim.weight_decay = 0.1;
    optim.beta1 = 0.9;
    optim.beta2 = 0.95;
    optim.eps = 1e-8;
    optim.set_grad_clip_norm(grad_clip_norm);

    println!("Pre-allocating optimizer moment buffers...");
    optim.init_moments(&mut g);

    let total_params: usize = model
        .params()
        .iter()
        .map(|&id| g.tensors[id].shape.iter().product::<usize>())
        .sum();
        
    println!("\nArchitecture summary:");
    println!("Trainable parameters: {}", total_params);
    println!(
        "  Layers:  {} | Hidden: {} | Heads: {} | FFN: {}",
        num_layers, hidden_dim, num_heads, ffn_dim
    );
    println!(
        "  Micro-batch: {} | Accum: {} | Effective batch: {}",
        micro_batch_size, accum_steps, effective_batch
    );
    println!(
        "  Tokens/step: {} | Total steps: {} ({} Epochs)",
        tokens_per_step, total_steps, epochs
    );
    println!(
        "  LR: {:.1e} → {:.1e} over {} steps ({} warmup)",
        lr_max, lr_min, total_steps, warmup_steps
    );
    println!("  Grad clip norm: {:.3}", grad_clip_norm);
    println!(
        "  Reproducibility: deterministic={} | seed={}",
        deterministic, seed
    );

    #[cfg(feature = "bf16")]
    if g.uses_bf16_mixed_precision() {
        println!("  Precision: BF16 forward + FP32 accumulation (streaming-aware)");
    } else {
        println!("  Precision: FP32");
    }
    if let Ok(mut s) = shared_stats.write() {
        s.vocab_size = vocab_size;
        s.hidden_dim = hidden_dim;
        s.num_heads = num_heads;
        s.ffn_dim = ffn_dim;
        s.num_layers = num_layers;
        s.total_params = total_params;
        s.seq_len = seq_len;
        s.micro_batch_size = micro_batch_size;
        s.accum_steps = accum_steps;
        s.effective_batch = effective_batch;
    }

    // ── Stability + reproducibility controls ────────────────────────────────
    let grad_clip_max_norm = env_f32("EISEN_GRAD_CLIP_NORM", 1.0);
    let run_seed = env_u64("EISEN_SEED", 1337);
    let deterministic = env_bool("EISEN_DETERMINISTIC", true);
    let manifest_path =
        env::var("EISEN_RUN_MANIFEST").unwrap_or_else(|_| "data/run_manifest.json".to_string());
    let started_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    if deterministic {
        unsafe {
            env::set_var("CUBLAS_WORKSPACE_CONFIG", ":4096:8");
        }
        println!(
            "Determinism: enabled (seed={}, cublas workspace pinned)",
            run_seed
        );
    } else {
        println!("Determinism: disabled (seed={} logged only)", run_seed);
    }

    let run_manifest = format!(
        "{{\n  \"phase\": 8,\n  \"started_unix\": {},\n  \"seed\": {},\n  \"deterministic\": {},\n  \"grad_clip_max_norm\": {:.6},\n  \"hyperparams\": {{\n    \"epochs\": {},\n    \"hidden_dim\": {},\n    \"num_heads\": {},\n    \"ffn_dim\": {},\n    \"num_layers\": {},\n    \"seq_len\": {},\n    \"micro_batch_size\": {},\n    \"accum_steps\": {},\n    \"effective_batch\": {},\n    \"lr_max\": {:.8},\n    \"lr_min\": {:.8},\n    \"warmup_steps\": {},\n    \"total_steps\": {}\n  }}\n}}",
        started_unix,
        run_seed,
        deterministic,
        grad_clip_max_norm,
        epochs,
        hidden_dim,
        num_heads,
        ffn_dim,
        num_layers,
        seq_len,
        micro_batch_size,
        accum_steps,
        effective_batch,
        lr_max,
        lr_min,
        warmup_steps,
        total_steps,
    );
    write_run_manifest(&manifest_path, &run_manifest);

    // ── Training loop ─────────────────────────────────────────────────────────
    println!("\nStarting training (Epoch 1/{})...", epochs);
    let mut step = 0_usize;
    let mut running_loss = 0.0_f32;
    let mut cumulative_tokens = 0_usize;

    let mut last_log = Instant::now();
    let mut board_timer = Instant::now();

    let mut last_grad_norm = 0.0_f32;
    let mut last_clip_coef = 1.0_f32;
    let mut best_loss = f32::MAX;

    g.print_vram_state("Optimizer init");

    'training: loop {
        if step >= total_steps {
            break;
        }

        let current_lr = scheduler.get_lr(step);
        optim.lr = current_lr;
        optim.zero_grad(&mut g);

        let mut step_loss = 0.0_f32;
        let mut micro_count = 0_usize;

        for _ in 0..accum_steps {
            let batch: BatchResult = match dataloader.next_batch() {
                Some(b) => b,
                None => {
                    if current_epoch < epochs {
                        current_epoch += 1;
                        println!("\nEpoch {}/{} starting at step {}…",
                                 current_epoch, epochs, step);
                        dataloader = {
                            let dl = BinaryDataLoader::new(bin_path, seq_len, micro_batch_size)
                                .with_seed(seed.wrapping_add(current_epoch as u64));
                            if use_fim { dl.with_fim(fim_config.clone()) } else { dl }
                        };
                        match dataloader.next_batch() {
                            Some(b) => b,
                            None => { println!("Dataset empty."); break 'training; }
                        }
                    } else {
                        println!("\nDataset exhausted at step {}. Done.", step);
                        break 'training;
                    }
                }
            };
 
            let tokens_this_micro = micro_batch_size * seq_len;
 
            let x_id = g.alloc_pooled(vec![micro_batch_size, seq_len]);
            g.load_tensor_data(x_id, &batch.x);
 
            let logits_id   = model.forward(&mut g, x_id);
            let flat_logits = g.reshape(logits_id, vec![tokens_this_micro, vocab_size]);
 
            // Use masked CE when FIM has produced ignore-index positions;
            // fall back to standard CE otherwise (no overhead).
            let loss_id = if batch.has_masked {
                g.cross_entropy_masked(flat_logits, &batch.targets)
            } else {
                g.cross_entropy(flat_logits, &batch.targets)
            };
 
            step_loss += g.tensors[loss_id].sync_to_cpu()[0];
            micro_count += 1;
 
            g.backward(loss_id);
            g.clear_activations();
        }

        if micro_count == 0 {
            break;
        }

        let (grad_norm, grad_clip_coef) = optim.clip_grad_norm(&mut g, grad_clip_max_norm);
        last_grad_norm = grad_norm;
        last_clip_coef = grad_clip_coef;

        g.print_vram_state("PRE step");
        optim.step(&mut g);
        g.print_vram_state("POST step");
        step += 1;
        cumulative_tokens += micro_count * tokens_per_micro;

        let avg_loss = step_loss / micro_count as f32;
        running_loss += avg_loss;

        // ── Logging ───────────────────────────────────────────────────────────
        if step % log_interval == 0 {
            let elapsed = last_log.elapsed().as_secs_f32();
            let throughput = (log_interval * tokens_per_step) as f32 / elapsed.max(1e-6);
            let avg = running_loss / log_interval as f32;

            println!(
                "Epoch {} | Step {:06} | Loss {:.4} | LR {:.2e} | {:.0} tok/s",
                current_epoch, step, avg, current_lr, throughput
            );

            running_loss = 0.0;
            last_log = Instant::now();
        }

        // ── EisenBoard telemetry ──────────────────────────────────────────────
        if step % 1 == 0 {
            let elapsed = board_timer.elapsed().as_secs_f32();
            let tps = (10 * tokens_per_step) as f32 / elapsed.max(1e-6);
            let batch_time_ms = (elapsed * 1000.0) / 10.0;

            if let Ok(mut s) = shared_stats.write() {
                s.step = step;
                s.loss = avg_loss;
                s.lr = current_lr;
                s.tps = tps;
                s.batch_time_ms = batch_time_ms;
                s.total_tokens = cumulative_tokens;
                s.grad_norm = last_grad_norm;
                s.grad_clip_coef = last_clip_coef;
                s.accum_steps = accum_steps;
                s.seq_len = seq_len;
                s.micro_batch_size = micro_batch_size;
                s.effective_batch = effective_batch;

                if s.history.len() >= 200 {
                    s.history.remove(0);
                }
                s.history.push(StepRecord {
                    step,
                    loss: avg_loss,
                    grad_norm: optim.last_grad_norm(),
                });
            }
            board_timer = Instant::now();
        }

        // ── Checkpoint ────────────────────────────────────────────────────────
        if avg_loss < best_loss && !avg_loss.is_nan() {
            println!("🚀 NEW BEST LOSS: {:.4} (previous was {:.4})", avg_loss, best_loss);
            best_loss = avg_loss;
            save_weights(&g, output_path);
            save_hf_bundle(
                &g, &model, vocab_size, hidden_dim, ffn_dim, num_layers, num_heads, seq_len,
                hf_out_dir,
            );
        }
    }

    // Final save
    save_weights(&g, output_path);
    save_hf_bundle(
        &g, &model, vocab_size, hidden_dim, ffn_dim, num_layers, num_heads, seq_len, hf_out_dir,
    );

    // ── Generation smoke test ─────────────────────────────────────────────────
    println!("\nSanity-check generation…");
    let prompt = "Die künstliche Intelligenz";
    let mut tokens = tokenizer.encode(prompt);
    print!("{}", prompt);

    g.no_grad = true;
    for _ in 0..80 {
        let start = tokens.len().saturating_sub(seq_len);
        let context = &tokens[start..];
        let csl = context.len();

        let context_f32: Vec<f32> = context.iter().map(|&t| t as f32).collect();
        let x_id = g.alloc_pooled(vec![1, csl]);
        g.load_tensor_data(x_id, &context_f32);
        
        let logits_id = model.forward(&mut g, x_id);
        let logits = g.tensors[logits_id].sync_to_cpu();

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

    let finished_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let completed_manifest = format!(
        "{{\n  \"phase\": 8,\n  \"started_unix\": {},\n  \"seed\": {},\n  \"deterministic\": {},\n  \"grad_clip_max_norm\": {:.6},\n  \"hyperparams\": {{\n    \"epochs\": {},\n    \"hidden_dim\": {},\n    \"num_heads\": {},\n    \"ffn_dim\": {},\n    \"num_layers\": {},\n    \"seq_len\": {},\n    \"micro_batch_size\": {},\n    \"accum_steps\": {},\n    \"effective_batch\": {},\n    \"lr_max\": {:.8},\n    \"lr_min\": {:.8},\n    \"warmup_steps\": {},\n    \"total_steps\": {}\n  }}\n}}",
        started_unix,
        run_seed,
        deterministic,
        grad_clip_max_norm,
        epochs,
        hidden_dim,
        num_heads,
        ffn_dim,
        num_layers,
        seq_len,
        micro_batch_size,
        accum_steps,
        effective_batch,
        lr_max,
        lr_min,
        warmup_steps,
        total_steps,
    );
    write_run_manifest(&manifest_path, &completed_manifest);


    println!("\n\nRun complete. Weights at {}", output_path);
}
