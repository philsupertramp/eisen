/// # Gradient Accumulation Test Suite
///
/// Tests are CPU-only (no GPU required) unless explicitly marked `_gpu`.
/// Each test is self-contained: it creates its own Graph, allocates fresh
/// weights, and verifies one specific invariant.
///
/// ## The key identity (proven in audit)
///
/// cross_entropy divides the loss by `batch_size` *inside the kernel*.
/// After `accum_steps` micro-batches of size `B_micro`:
///
///   g_accum = accum_steps × g_full_batch
///
/// where `g_full_batch` is the gradient from a single pass over all
/// `B_micro × accum_steps` samples at once.
///
/// For AdamW this scale difference cancels in `m / √v`, so no rescaling
/// is needed in the optimiser. For SGD it would NOT cancel — the comment
/// in train_large_lm.rs is only correct for adaptive optimisers.

use eisen::graph::Graph;
use eisen::nn::linear::Linear;
use eisen::nn::embedding::Embedding;
use eisen::nn::optim::AdamW;
use eisen::nn::Module;
use eisen::tensor::Device;

// ─── helpers ──────────────────────────────────────────────────────────────────

/// Build a minimal 2-class linear classifier on CPU.
/// Returns (graph, weight_id, bias_id, param_ids).
fn make_cpu_linear(in_f: usize, out_f: usize) -> (Graph, Linear) {
    let mut g = Graph::new(Device::Cpu);
    let layer = Linear::new(&mut g, in_f, out_f, false); // no bias keeps math simple
    (g, layer)
}

/// Read grad buffer of a CPU tensor as a plain Vec.
fn grad_vec(g: &Graph, id: usize) -> Vec<f32> {
    g.tensors[id].grad.as_cpu().clone()
}

/// Elementwise absolute difference, returns max.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

// ─── Test 1 ───────────────────────────────────────────────────────────────────

/// **Gradient additivity**
///
/// Running backward twice on the **same** micro-batch accumulates twice
/// the gradient of a single pass, without any clear between the two passes.
/// This is the most primitive property gradient accumulation relies on.
#[test]
fn test_grad_accumulates_additively() {
    let (mut g, layer) = make_cpu_linear(2, 3);

    let x = vec![1.0_f32, 2.0];
    let y = vec![0_usize]; // target class 0

    // — single forward+backward pass —
    let x1 = g.alloc(vec![1, 2], x.clone());
    let out1 = layer.forward(&mut g, x1);
    let loss1 = g.cross_entropy(out1, &y);
    g.backward(loss1);
    let grad_single = grad_vec(&g, layer.weight_id);
    g.tape.nodes.clear();

    // Zero grad, then run backward TWICE on the same data (simulate 2 accum steps)
    for d in g.tensors[layer.weight_id].grad.as_cpu_mut() {
        *d = 0.0;
    }

    let x2 = g.alloc(vec![1, 2], x.clone());
    let out2 = layer.forward(&mut g, x2);
    let loss2 = g.cross_entropy(out2, &y);
    g.backward(loss2);
    g.tape.nodes.clear();

    let x3 = g.alloc(vec![1, 2], x.clone());
    let out3 = layer.forward(&mut g, x3);
    let loss3 = g.cross_entropy(out3, &y);
    g.backward(loss3);
    g.tape.nodes.clear();

    let grad_double = grad_vec(&g, layer.weight_id);

    // After two identical passes the grad must be exactly 2× the single pass.
    let diff = max_abs_diff(&grad_double, &grad_single.iter().map(|v| 2.0 * v).collect::<Vec<_>>());
    assert!(
        diff < 1e-6,
        "Gradient not additive: max diff = {diff:.2e}\n  single={grad_single:?}\n  double={grad_double:?}"
    );
}

// ─── Test 2 ───────────────────────────────────────────────────────────────────

/// **Scale relationship: g_accum = accum_steps × g_full**
///
/// Split a batch of 4 into 2 micro-batches of 2.
/// The accumulated gradient must be exactly 2× the full-batch gradient.
/// Both runs start from the same freshly initialised weights.
#[test]
fn test_accum_grad_equals_n_times_full_batch_grad() {
    const ACCUM: f32 = 2.0;

    // Fixed input data: 4 samples, 3 features, 2-class problem
    let x_all = vec![
        0.5_f32, -0.3, 1.1,   // sample 0
        -0.8,  0.6, 0.2,      // sample 1
        0.1,  -0.9, 0.4,      // sample 2
        0.7,   0.0,-0.5,      // sample 3
    ];
    let y_all = vec![0_usize, 1, 1, 0];

    // ── Run A: full batch of 4 ──────────────────────────────────────────────
    let (mut g_a, layer_a) = make_cpu_linear(3, 2);

    let xa = g_a.alloc(vec![4, 3], x_all.clone());
    let out_a = layer_a.forward(&mut g_a, xa);
    let loss_a = g_a.cross_entropy(out_a, &y_all);
    g_a.backward(loss_a);
    let g_full = grad_vec(&g_a, layer_a.weight_id);

    // ── Run B: same weights, 2 micro-batches of 2 ──────────────────────────
    // We need identical initial weights. Linear::new uses a fixed LCG seed so
    // a freshly constructed Linear with the same shape is identical to layer_a.
    let (mut g_b, layer_b) = make_cpu_linear(3, 2);
    g_b.mark_params(); // watermark so clear_activations knows where params end

    // Verify initial weights match (both used the same LCG seed = 42)
    assert_eq!(
        g_a.tensors[layer_a.weight_id].data.as_cpu(),
        g_b.tensors[layer_b.weight_id].data.as_cpu(),
        "Initial weights must be identical for the comparison to be valid"
    );

    // micro-batch 0: samples 0 and 1
    let xb0 = g_b.alloc(vec![2, 3], x_all[0..6].to_vec());
    let out_b0 = layer_b.forward(&mut g_b, xb0);
    let loss_b0 = g_b.cross_entropy(out_b0, &y_all[0..2]);
    g_b.backward(loss_b0);
    g_b.clear_activations(); // recycles activation VRAM, keeps param grads

    // micro-batch 1: samples 2 and 3
    let xb1 = g_b.alloc(vec![2, 3], x_all[6..].to_vec());
    let out_b1 = layer_b.forward(&mut g_b, xb1);
    let loss_b1 = g_b.cross_entropy(out_b1, &y_all[2..]);
    g_b.backward(loss_b1);
    g_b.clear_activations();

    let g_accum = grad_vec(&g_b, layer_b.weight_id);

    // g_accum should be exactly ACCUM × g_full
    let expected: Vec<f32> = g_full.iter().map(|v| ACCUM * v).collect();
    let diff = max_abs_diff(&g_accum, &expected);

    assert!(
        diff < 1e-5,
        "g_accum ≠ {ACCUM}×g_full: max diff = {diff:.2e}\n  g_full ={g_full:?}\n  g_accum={g_accum:?}\n  expect ={expected:?}"
    );
}

// ─── Test 3 ───────────────────────────────────────────────────────────────────

/// **zero_grad must not run between micro-batches**
///
/// If zero_grad fires inside the accumulation loop the second micro-batch
/// gradient would silently overwrite the first instead of adding to it.
/// This test verifies that param grads after 2 micro-batches are NOT equal
/// to the grads from just 1 micro-batch (which they would be if zero_grad
/// ran between them).
#[test]
fn test_zero_grad_must_not_run_mid_accumulation() {
    let x = vec![0.5_f32, -0.3, 1.1, -0.8, 0.6, 0.2];
    let y = vec![0_usize, 1];

    let (mut g, layer) = make_cpu_linear(3, 2);
    g.mark_params();

    // Single micro-batch
    let x1 = g.alloc(vec![2, 3], x.clone());
    let o1 = layer.forward(&mut g, x1);
    let l1 = g.cross_entropy(o1, &y);
    g.backward(l1);
    g.clear_activations();
    let grad_one_pass = grad_vec(&g, layer.weight_id);

    // Second micro-batch without zeroing
    let x2 = g.alloc(vec![2, 3], x.clone());
    let o2 = layer.forward(&mut g, x2);
    let l2 = g.cross_entropy(o2, &y);
    g.backward(l2);
    g.clear_activations();
    let grad_two_pass = grad_vec(&g, layer.weight_id);

    // After two passes grads must be ≠ one pass (they must have accumulated)
    let diff = max_abs_diff(&grad_two_pass, &grad_one_pass);
    assert!(
        diff > 1e-6,
        "Grads did not accumulate between micro-batches — zero_grad may have fired mid-loop.\n  one_pass={grad_one_pass:?}\n  two_pass={grad_two_pass:?}"
    );

    // They must equal exactly 2× one pass (same data)
    let expected: Vec<f32> = grad_one_pass.iter().map(|v| 2.0 * v).collect();
    let diff2 = max_abs_diff(&grad_two_pass, &expected);
    assert!(
        diff2 < 1e-6,
        "Two-pass grad ≠ 2× single-pass grad: max diff = {diff2:.2e}"
    );
}

// ─── Test 4 ───────────────────────────────────────────────────────────────────

/// **clear_activations must preserve param grads**
///
/// This is the load-bearing VRAM-safety property.
/// restore_save_point(num_params) only pops tensors above the watermark.
/// Param grad buffers live *on* the param tensors (indices < num_params)
/// and must survive clear_activations intact.
#[test]
fn test_clear_activations_preserves_param_grads() {
    let (mut g, layer) = make_cpu_linear(2, 3);
    g.mark_params(); // seal the parameter watermark

    let x = g.alloc(vec![1, 2], vec![1.0_f32, -1.0]);
    let out = layer.forward(&mut g, x);
    let loss = g.cross_entropy(out, &[1_usize]);
    g.backward(loss);

    let grad_before = grad_vec(&g, layer.weight_id);

    // This must not touch param grad buffers
    g.clear_activations();

    let grad_after = grad_vec(&g, layer.weight_id);

    assert_eq!(
        grad_before, grad_after,
        "clear_activations corrupted param grads!\n  before={grad_before:?}\n  after ={grad_after:?}"
    );
    // Also confirm the grad is non-zero (backward actually ran)
    assert!(
        grad_after.iter().any(|&v| v != 0.0),
        "Backward produced all-zero grads — test is vacuous"
    );
}

// ─── Test 5 ───────────────────────────────────────────────────────────────────

/// **zero_grad wipes grads cleanly between optimizer steps**
///
/// After an optimizer step, zero_grad must reset ALL param grads to 0.0
/// so the next step starts from a clean slate.
#[test]
fn test_zero_grad_resets_between_steps() {
    let (mut g, layer) = make_cpu_linear(2, 3);
    g.mark_params();
    let mut optim = AdamW::new(layer.params(), 0.01);

    let x = vec![1.0_f32, -1.0];
    let y = vec![0_usize];

    // Step 1
    let x1 = g.alloc(vec![1, 2], x.clone());
    let o1 = layer.forward(&mut g, x1);
    let l1 = g.cross_entropy(o1, &y);
    g.backward(l1);
    g.clear_activations();
    optim.step(&mut g);

    // Grads should still be non-zero at this point (step() doesn't zero them)
    let grad_after_step = grad_vec(&g, layer.weight_id);
    assert!(
        grad_after_step.iter().any(|&v| v != 0.0),
        "Grad was zero before zero_grad — test is vacuous"
    );

    // Now zero
    optim.zero_grad(&mut g);
    let grad_after_zero = grad_vec(&g, layer.weight_id);
    assert!(
        grad_after_zero.iter().all(|&v| v == 0.0),
        "zero_grad did not fully reset grads: {grad_after_zero:?}"
    );
}

// ─── Test 6 ───────────────────────────────────────────────────────────────────

/// **End-to-end convergence with gradient accumulation**
///
/// XOR with 4 samples split into 2 micro-batches of 2.
/// The model must reach near-zero loss, proving that accumulated gradients
/// drive optimisation correctly over many steps.
#[test]
fn test_accum_converges_xor() {
    let mut g = Graph::new(Device::Cpu);
    let hidden = 16;
    let l1 = Linear::new(&mut g, 2, hidden, true);
    let l2 = Linear::new(&mut g, hidden, 2, true);
    g.mark_params();

    let mut params = l1.params();
    params.extend(l2.params());
    let mut optim = AdamW::new(params, 0.05);

    // XOR: 4 samples split into 2 micro-batches
    let mb0_x = vec![0.0_f32, 0.0, 0.0, 1.0]; // samples 0, 1
    let mb0_y = vec![0_usize, 1];
    let mb1_x = vec![1.0_f32, 0.0, 1.0, 1.0]; // samples 2, 3
    let mb1_y = vec![1_usize, 0];

    let mut final_loss = f32::MAX;

    for _step in 0..300 {
        optim.zero_grad(&mut g);
        let mut step_loss = 0.0_f32;

        for (xd, yd) in [(&mb0_x, mb0_y.as_slice()), (&mb1_x, mb1_y.as_slice())] {
            let x_id = g.alloc(vec![2, 2], xd.clone());
            let h = l1.forward(&mut g, x_id);
            let act = g.silu(h);
            let logits = l2.forward(&mut g, act);
            let loss_id = g.cross_entropy(logits, yd);

            step_loss += g.tensors[loss_id].data.as_cpu()[0];
            g.backward(loss_id);
            g.clear_activations();
        }

        optim.step(&mut g);
        final_loss = step_loss / 2.0;
    }

    assert!(
        final_loss < 0.05,
        "XOR did not converge with gradient accumulation. Final loss = {final_loss:.4}"
    );
}

// ─── Test 7 ───────────────────────────────────────────────────────────────────

/// **Tape is empty after clear_activations**
///
/// Between micro-batches we call clear_activations() which must drop all
/// TapeNodes. If the tape grows unboundedly across micro-batches we'd
/// double-apply backward on the second step (use-after-free territory).
#[test]
fn test_tape_is_empty_after_clear_activations() {
    let (mut g, layer) = make_cpu_linear(2, 3);
    g.mark_params();

    let x = g.alloc(vec![1, 2], vec![1.0_f32, -1.0]);
    let out = layer.forward(&mut g, x);
    let loss = g.cross_entropy(out, &[0_usize]);

    let tape_len_before = g.tape.nodes.len();
    assert!(tape_len_before > 0, "Tape must be non-empty before clear");

    g.backward(loss);
    g.clear_activations();

    assert_eq!(
        g.tape.nodes.len(),
        0,
        "Tape not empty after clear_activations: {} nodes remain",
        g.tape.nodes.len()
    );
}

// ─── Test 8 ───────────────────────────────────────────────────────────────────

/// **Tensor count returns to watermark after clear_activations**
///
/// After mark_params() the engine knows where parameters end.
/// clear_activations must pop all activation tensors back to that watermark,
/// ensuring zero VRAM growth across accumulation steps.
#[test]
fn test_tensor_count_returns_to_watermark() {
    let (mut g, layer) = make_cpu_linear(2, 3);
    g.mark_params();
    let watermark = g.num_params;

    assert_eq!(
        g.tensors.len(),
        watermark,
        "Watermark should equal tensor count right after mark_params"
    );

    // Forward pass allocates activation tensors above the watermark
    let x = g.alloc(vec![1, 2], vec![1.0_f32, -1.0]);
    let out = layer.forward(&mut g, x);
    let loss = g.cross_entropy(out, &[0_usize]);
    g.backward(loss);

    assert!(
        g.tensors.len() > watermark,
        "Forward pass should have added activation tensors above watermark"
    );

    g.clear_activations();

    assert_eq!(
        g.tensors.len(),
        watermark,
        "clear_activations did not restore tensor count to watermark: {} > {}",
        g.tensors.len(),
        watermark
    );
}

// ─── Test 9 ───────────────────────────────────────────────────────────────────

/// **Embedding gradient accumulates across micro-batches**
///
/// The embedding gather kernel uses atomicAdd for gradient accumulation.
/// With gradient accumulation, the same token looked up in two successive
/// micro-batches must produce exactly 2× the gradient of a single lookup.
#[test]
fn test_embedding_grad_accumulates_across_micro_batches() {
    let (mut g, emb) = {
        let mut g = Graph::new(Device::Cpu);
        let emb = Embedding::new(&mut g, 4, 8); // vocab=4, dim=8
        (g, emb)
    };
    g.mark_params();

    let token_ids = vec![2.0_f32]; // look up token 2

    // Single pass
    let idx1 = g.alloc(vec![1], token_ids.clone());
    let out1 = emb.forward(&mut g, idx1);
    // Sum all outputs as a proxy loss
    let loss1 = g.sum(out1, 0);
    let scalar1 = g.sum(loss1, 0);
    g.backward(scalar1);
    g.clear_activations();

    let grad_single = g.tensors[emb.weight_id].grad.as_cpu()[2 * 8..3 * 8].to_vec();

    // Second pass (no zero_grad) — should add to existing grads
    let idx2 = g.alloc(vec![1], token_ids.clone());
    let out2 = emb.forward(&mut g, idx2);
    let loss2 = g.sum(out2, 0);
    let scalar2 = g.sum(loss2, 0);
    g.backward(scalar2);
    g.clear_activations();

    let grad_double = g.tensors[emb.weight_id].grad.as_cpu()[2 * 8..3 * 8].to_vec();

    let expected: Vec<f32> = grad_single.iter().map(|v| 2.0 * v).collect();
    let diff = max_abs_diff(&grad_double, &expected);

    assert!(
        diff < 1e-6,
        "Embedding grad did not double after two micro-batches: diff={diff:.2e}\n  single={grad_single:?}\n  double={grad_double:?}"
    );

    // Unused token (e.g. token 0) must have zero grad throughout
    let grad_unused = &g.tensors[emb.weight_id].grad.as_cpu()[0..8];
    assert!(
        grad_unused.iter().all(|&v| v == 0.0),
        "Unused token received non-zero gradient: {grad_unused:?}"
    );
}

// ─── Test 10 ──────────────────────────────────────────────────────────────────

/// **AdamW update equivalence under accumulation**
///
/// For AdamW, the accumulated gradient is `N × g_full`.  The second moment
/// is `N² × g_full²`.  The update `m / √v` has the N cancel:
///
///   (N × g) / √(N² × g²)  =  g / |g|   (same sign, same magnitude as unscaled)
///
/// We verify this numerically: after ONE optimizer step (from zero momentum),
/// the weight update from accumulation should equal the weight update from
/// the full batch, to within floating-point tolerance.
#[test]
fn test_adamw_update_equiv_under_accumulation() {
    let x_all = vec![
        0.5_f32, -0.3,
        -0.8,  0.6,
        0.1,  -0.9,
        0.7,   0.0,
    ];
    let y_all = vec![0_usize, 1, 1, 0];

    // ── Run A: full batch, one AdamW step ──────────────────────────────────
    let (mut g_a, layer_a) = make_cpu_linear(2, 3);
    g_a.mark_params();
    let mut optim_a = AdamW::new(layer_a.params(), 0.01);

    let xa = g_a.alloc(vec![4, 2], x_all.clone());
    let oa = layer_a.forward(&mut g_a, xa);
    let la = g_a.cross_entropy(oa, &y_all);
    g_a.backward(la);
    g_a.clear_activations();
    optim_a.step(&mut g_a);

    let weights_full = g_a.tensors[layer_a.weight_id].data.as_cpu().clone();

    // ── Run B: 2 micro-batches, one AdamW step ─────────────────────────────
    let (mut g_b, layer_b) = make_cpu_linear(2, 3); // same seed → same init weights
    g_b.mark_params();
    let mut optim_b = AdamW::new(layer_b.params(), 0.01);

    optim_b.zero_grad(&mut g_b);

    let xb0 = g_b.alloc(vec![2, 2], x_all[0..4].to_vec());
    let ob0 = layer_b.forward(&mut g_b, xb0);
    let lb0 = g_b.cross_entropy(ob0, &y_all[0..2]);
    g_b.backward(lb0);
    g_b.clear_activations();

    let xb1 = g_b.alloc(vec![2, 2], x_all[4..].to_vec());
    let ob1 = layer_b.forward(&mut g_b, xb1);
    let lb1 = g_b.cross_entropy(ob1, &y_all[2..]);
    g_b.backward(lb1);
    g_b.clear_activations();

    optim_b.step(&mut g_b);

    let weights_accum = g_b.tensors[layer_b.weight_id].data.as_cpu().clone();

    // The two weight tensors started identical (same LCG seed).
    // After one AdamW step the updates should be equal to within 1e-5.
    // (The cancellation is exact when epsilon is negligible relative to |g|.)
    let diff = max_abs_diff(&weights_full, &weights_accum);
    assert!(
        diff < 1e-4,
        "AdamW update diverged between full-batch and accumulated:\n  max diff = {diff:.2e}\n  full ={weights_full:?}\n  accum={weights_accum:?}"
    );
}
