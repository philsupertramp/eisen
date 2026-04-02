use eisen::graph::Graph;
use eisen::tensor::Device;
use cudarc::driver::CudaContext;
use eisen::nn::Module;

fn setup_gpu() -> Option<Device> {
    match CudaContext::new(0) {
        Ok(ctx) => {
            let stream = ctx.default_stream();
            Some(Device::Gpu(ctx, stream))
        }
        Err(_) => None,
    }
}

#[test]
fn test_gpu_add_forward_backward() {
    let device = match setup_gpu() {
        Some(d) => d,
        None => { eprintln!("No GPU found, skipping test."); return; }
    };
    let mut g = Graph::new(device);

    let a_id = g.alloc(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b_id = g.alloc(vec![2, 2], vec![0.5, 1.5, 2.5, 3.5]);

    let c_id = g.add(a_id, b_id);
    let c_data = g.tensors[c_id].sync_to_cpu();
    
    assert_eq!(c_data, vec![1.5, 3.5, 5.5, 7.5]);

    g.backward(c_id);

    let a_grad = g.sync_grad_to_cpu(a_id);
    let b_grad = g.sync_grad_to_cpu(b_id);
    
    assert_eq!(a_grad, vec![1.0, 1.0, 1.0, 1.0]);
    assert_eq!(b_grad, vec![1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_gpu_mul_forward_backward() {
    let device = match setup_gpu() {
        Some(d) => d,
        None => return,
    };
    let mut g = Graph::new(device);

    let a_id = g.alloc(vec![3], vec![2.0, -3.0, 4.0]);
    let b_id = g.alloc(vec![3], vec![5.0, 2.0, -1.0]);

    let c_id = g.mul(a_id, b_id);
    let c_data = g.tensors[c_id].sync_to_cpu();
    
    assert_eq!(c_data, vec![10.0, -6.0, -4.0]);

    g.backward(c_id);

    let a_grad = g.sync_grad_to_cpu(a_id);
    let b_grad = g.sync_grad_to_cpu(b_id);
    
    // For y = a * b, dy/da = b (5, 2, -1) and dy/db = a (2, -3, 4)
    assert_eq!(a_grad, vec![5.0, 2.0, -1.0]);
    assert_eq!(b_grad, vec![2.0, -3.0, 4.0]);
}

#[test]
fn test_gpu_matmul_forward_backward() {
    let device = match setup_gpu() {
        Some(d) => d,
        None => return,
    };
    let mut g = Graph::new(device);

    // Matrix A: 2x3
    let a_id = g.alloc(vec![2, 3], vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    ]);
    // Matrix B: 3x2
    let b_id = g.alloc(vec![3, 2], vec![
        7.0, 8.0,
        9.0, 1.0,
        2.0, 3.0
    ]);

    let c_id = g.matmul(a_id, b_id);
    let c_data = g.tensors[c_id].sync_to_cpu();
    
    // Check Forward Pass
    let expected_fwd = vec![
        31.0, 19.0,
        85.0, 55.0
    ];
    assert_eq!(c_data, expected_fwd);

    g.backward(c_id);

    // Check Backward Pass
    let a_grad = g.sync_grad_to_cpu(a_id);
    let b_grad = g.sync_grad_to_cpu(b_id);
    
    // Expected Gradients assuming seed grad is [1, 1, 1, 1]
    // dA = dC @ B^T
    let expected_a_grad = vec![
        15.0, 10.0, 5.0,
        15.0, 10.0, 5.0
    ];
    // dB = A^T @ dC
    let expected_b_grad = vec![
        5.0,  5.0,
        7.0,  7.0,
        9.0,  9.0
    ];
    
    assert_eq!(a_grad, expected_a_grad);
    assert_eq!(b_grad, expected_b_grad);
}

#[test]
fn test_gpu_silu_forward_backward() {
    let device = match setup_gpu() {
        Some(d) => d,
        None => return,
    };
    let mut g = Graph::new(device);

    let x_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let a_id = g.alloc(vec![5], x_data.clone());

    let c_id = g.silu(a_id);
    let c_data = g.tensors[c_id].sync_to_cpu();
    
    // Calculate expected CPU values to match against GPU
    let mut expected_fwd = vec![0.0; 5];
    let mut expected_grad = vec![0.0; 5];
    for i in 0..5 {
        let x = x_data[i];
        let sig = 1.0 / (1.0 + (-x).exp());
        let silu = x * sig;
        expected_fwd[i] = silu;
        expected_grad[i] = silu + sig * (1.0 - silu);
    }

    // Verify Forward Pass
    for (found, expected) in c_data.iter().zip(expected_fwd.iter()) {
        assert!((found - expected).abs() < 1e-5, "Forward SiLU mismatch: found {}, expected {}", found, expected);
    }

    g.backward(c_id);

    // Verify Backward Pass
    let a_grad = g.sync_grad_to_cpu(a_id);
    for (found, expected) in a_grad.iter().zip(expected_grad.iter()) {
        assert!((found - expected).abs() < 1e-5, "Backward SiLU mismatch: found {}, expected {}", found, expected);
    }
}

#[test]
fn test_gpu_gather_forward_backward() {
    let device = match setup_gpu() {
        Some(d) => d,
        None => return,
    };
    let mut g = Graph::new(device);

    // Weights matrix: 3 vocab items, embedding dimension 2
    let w_data = vec![
        0.1, 0.2, // Row 0
        0.3, 0.4, // Row 1
        0.5, 0.6  // Row 2
    ];
    let w_id = g.alloc(vec![3, 2], w_data);

    // Indices (Notice '2' is repeated, creating a gradient collision during backprop!)
    let idx_data = vec![2.0, 0.0, 2.0, 1.0];
    let idx_id = g.alloc(vec![4], idx_data);

    let c_id = g.gather(w_id, idx_id);
    let c_data = g.tensors[c_id].sync_to_cpu();
    
    // Check Forward Pass
    let expected_fwd = vec![
        0.5, 0.6, // from Row 2
        0.1, 0.2, // from Row 0
        0.5, 0.6, // from Row 2
        0.3, 0.4  // from Row 1
    ];
    for (found, expected) in c_data.iter().zip(expected_fwd.iter()) {
        assert!((found - expected).abs() < 1e-5, "Forward Gather mismatch");
    }

    g.backward(c_id);

    let w_grad = g.sync_grad_to_cpu(w_id);
    
    // We expect the seed gradient to be 1.0 everywhere.
    // Row 2 was accessed twice, so its gradient should accumulate to 2.0!
    let expected_w_grad = vec![
        1.0, 1.0, // Row 0 accessed once
        1.0, 1.0, // Row 1 accessed once
        2.0, 2.0  // Row 2 accessed twice! (atomicAdd test)
    ];
    
    for (found, expected) in w_grad.iter().zip(expected_w_grad.iter()) {
        assert!((found - expected).abs() < 1e-5, "Backward Gather mismatch (AtomicAdd failed?)");
    }
}

#[test]
fn test_gpu_rmsnorm_forward_backward() {
    let device = match setup_gpu() {
        Some(d) => d,
        None => return,
    };
    let mut g = Graph::new(device);

    let dim = 4;
    let eps = 1e-5;
    
    let x_data = vec![
        1.0, -2.0, 3.0, -4.0, // Vector 1
        0.1, 0.2, -0.1, 0.0,  // Vector 2
    ];
    let w_data = vec![1.0, 1.0, 1.0, 1.0];
    
    let x_id = g.alloc(vec![2, 4], x_data.clone());
    let w_id = g.alloc(vec![4], w_data.clone());

    let out_id = g.rms_norm(x_id, w_id, eps);
    
    let out_data = g.tensors[out_id].sync_to_cpu();
    
    // Quick sanity check against NaN
    assert!(out_data.iter().all(|v| !v.is_nan()), "RMSNorm produced NaNs");
    
    g.backward(out_id);
    
    let w_grad = g.sync_grad_to_cpu(w_id);
    let x_grad = g.sync_grad_to_cpu(x_id);
    
    assert!(w_grad.iter().any(|&v| v != 0.0), "RMSNorm weight received no gradient");
    assert!(x_grad.iter().any(|&v| v != 0.0), "RMSNorm input received no gradient");
}


#[test]
fn test_gpu_cross_entropy_forward_backward() {
    let device = match setup_gpu() {
        Some(d) => d,
        None => return,
    };
    let mut g = Graph::new(device);

    let logits_data = vec![
        2.0, 1.0, 0.1, // batch 0
        0.0, 3.0, 0.5  // batch 1
    ];
    let logits_id = g.alloc(vec![2, 3], logits_data);
    let targets = vec![0, 1];

    let loss_id = g.cross_entropy(logits_id, &targets);
    let loss_data = g.tensors[loss_id].sync_to_cpu();
    
    // Expected loss is ~0.27 based on probabilities
    assert!((loss_data[0] - 0.27).abs() < 0.05, "Loss mismatch: {}", loss_data[0]);

    g.backward(loss_id);
    let grad_data = g.sync_grad_to_cpu(logits_id);
    
    // Check that target gradients are negative and others are positive
    assert!(grad_data[0] < 0.0); // target for batch 0
    assert!(grad_data[1] > 0.0);
    assert!(grad_data[2] > 0.0);

    assert!(grad_data[3] > 0.0);
    assert!(grad_data[4] < 0.0); // target for batch 1
    assert!(grad_data[5] > 0.0);
}


#[test]
fn test_gpu_sum_max_forward_backward() {
    let device = match setup_gpu() {
        Some(d) => d,
        None => return,
    };
    let mut g = Graph::new(device);

    // Matrix: 2x3
    let a_data = vec![
        1.0, 2.0, 3.0,
        4.0, 9.0, 6.0
    ];
    let a_id = g.alloc(vec![2, 3], a_data);

    // --- TEST SUM ---
    let sum_id = g.sum(a_id, 1); // Sum across columns
    let sum_data = g.tensors[sum_id].sync_to_cpu();
    
    assert_eq!(sum_data, vec![6.0, 19.0]);

    g.backward(sum_id);
    let a_grad_sum = g.sync_grad_to_cpu(a_id);
    // Gradient should be broadcasted to all elements
    assert_eq!(a_grad_sum, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

    g.tape.nodes.clear(); // Reset tape
    
    // Reset gradients to 0 for next test
    g.tensors[a_id].grad = eisen::tensor::Storage::Gpu(
        setup_gpu().unwrap().into_gpu().1.alloc_zeros(6).unwrap()
    );

    // --- TEST MAX ---
    let max_id = g.max(a_id, 1); // Max across columns
    let max_data = g.tensors[max_id].sync_to_cpu();
    
    assert_eq!(max_data, vec![3.0, 9.0]);

    g.backward(max_id);
    let a_grad_max = g.sync_grad_to_cpu(a_id);
    
    // Gradients should ONLY route to the winning indices: index 2 (val: 3.0) and index 4 (val: 9.0)
    assert_eq!(a_grad_max, vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
}

#[test]
fn test_gpu_bmm_and_softmax_attention_primitives() {
    let device = match setup_gpu() {
        Some(d) => d,
        None => return,
    };
    let mut g = Graph::new(device);

    // Mock an Attention scenario: Batch=2, SeqLen=2, HeadDim=3
    // Q: [2, 2, 3]
    let q_data = vec![
        1.0, 0.0, 1.0,  0.0, 1.0, 0.0, // Batch 1
        1.0, 1.0, 1.0,  0.0, 0.0, 1.0  // Batch 2
    ];
    // K: [2, 2, 3] 
    let k_data = vec![
        1.0, 0.0, 1.0,  0.0, 1.0, 0.0, // Batch 1 (Same as Q to test self-attention similarity)
        1.0, 0.0, 0.0,  0.0, 0.0, 1.0  // Batch 2
    ];
    
    let q_id = g.alloc(vec![2, 2, 3], q_data);
    let k_id = g.alloc(vec![2, 2, 3], k_data);

    // 1. Compute Scores = Q @ K^T 
    // BMM with trans_b = true avoids the memory allocation of a Transpose node!
    let scores_id = g.bmm(q_id, k_id, true);
    
    let scores = g.tensors[scores_id].sync_to_cpu();
    
    // Batch 1: 
    // Q0 @ K0 = (1*1 + 0*0 + 1*1) = 2.0
    // Q0 @ K1 = (1*0 + 0*1 + 1*0) = 0.0
    // Q1 @ K0 = 0.0
    // Q1 @ K1 = 1.0
    assert_eq!(scores[0..4], vec![2.0, 0.0, 0.0, 1.0]);

    // 2. Compute Probabilities = Softmax(Scores)
    let probs_id = g.softmax(scores_id);
    let probs = g.tensors[probs_id].sync_to_cpu();
    
    // Batch 1, Row 0 softmax([2.0, 0.0]) => ~[0.88, 0.12]
    assert!((probs[0] - 0.88079).abs() < 1e-4);
    assert!((probs[1] - 0.11920).abs() < 1e-4);

    // Run a backward pass through Softmax and BMM to ensure autograd hooks are wired
    g.backward(probs_id);
    
    let dq = g.sync_grad_to_cpu(q_id);
    let dk = g.sync_grad_to_cpu(k_id);
    
    assert!(dq.iter().any(|&v| v != 0.0), "Q gradients failed to accumulate");
    assert!(dk.iter().any(|&v| v != 0.0), "K gradients failed to accumulate");
}

#[test]
fn test_gpu_mha_layer_forward_backward() {
    let device = match setup_gpu() {
        Some(d) => d,
        None => return,
    };
    let mut g = Graph::new(device);
    
    let hidden_dim = 4;
    let num_heads = 2; // Test MHA!
    
    // Instantiate our new Multi-Head Attention layer
    let mha = eisen::nn::attention::MultiHeadAttention::new(&mut g, hidden_dim, num_heads);
    
    // Mock Input: Batch=1, SeqLen=3, HiddenDim=4
    let x_data = vec![
        1.0, 0.0, 1.0, 0.0, // token 1
        0.0, 1.0, 0.0, 1.0, // token 2
        1.0, 1.0, 1.0, 1.0, // token 3
    ];
    let x_id = g.alloc(vec![1, 3, 4], x_data);
    
    // Forward Pass (Causal Mask = True)
    let out_id = mha.forward(&mut g, x_id);
    let out_data = g.tensors[out_id].sync_to_cpu();
    
    // Verify structural integrity
    assert_eq!(out_data.len(), 12);
    assert_eq!(g.tensors[out_id].shape, vec![1, 3, 4]);
    assert!(out_data.iter().all(|v| !v.is_nan()), "MHA produced NaNs");
    
    // Backward Pass
    g.backward(out_id);
    let x_grad = g.sync_grad_to_cpu(x_id);
    
    // Verify gradients successfully flowed through all the projections, MHA transposes, and BMMs!
    assert!(x_grad.iter().any(|&v| v != 0.0), "MHA input received no gradient");
}

#[test]
fn test_gpu_rope_forward_backward() {
    let device = match setup_gpu() {
        Some(d) => d,
        None => return,
    };
    let mut g = Graph::new(device);
    
    // Batch=1, Seq=2, HiddenDim=4 (HeadDim=4)
    // Sequence pos 0: [1.0, 1.0, 1.0, 1.0]
    // Sequence pos 1: [1.0, 1.0, 1.0, 1.0]
    let q_data = vec![
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
    ];
    let q_id = g.alloc(vec![1, 2, 4], q_data);
    
    let rope_id = g.rope(q_id, 4);
    let rope_data = g.tensors[rope_id].sync_to_cpu();
    
    // Position 0 should have no rotation applied (angle = 0)
    assert_eq!(rope_data[0..4], vec![1.0, 1.0, 1.0, 1.0]);
    
    // Position 1 should be rotated. 
    // Frequency for pair 0: 10000^(-0/4) = 1.0. Angle = 1.0 * 1 = 1.0 rad.
    // cos(1) ~ 0.5403, sin(1) ~ 0.8414
    // [1*cos - 1*sin, 1*cos + 1*sin] -> [0.5403 - 0.8414, 0.5403 + 0.8414]
    assert!((rope_data[4] - (0.5403 - 0.8414)).abs() < 1e-3);
    assert!((rope_data[5] - (0.5403 + 0.8414)).abs() < 1e-3);

    // Test backward pass inverse transformation
    g.backward(rope_id);
    let q_grad = g.sync_grad_to_cpu(q_id);
    
    // Seed grad is 1.0 everywhere.
    // Pos 0 grads should be [1.0, 1.0] because cos(0)=1, sin(0)=0.
    assert_eq!(q_grad[0..4], vec![1.0, 1.0, 1.0, 1.0]);
    
    // Pos 1 grads should be inverse rotated:
    // grad_x1 = 1*cos(1) + 1*sin(1)
    // grad_x2 = 1*cos(1) - 1*sin(1)
    assert!((q_grad[4] - (0.5403 + 0.8414)).abs() < 1e-3);
    assert!((q_grad[5] - (0.5403 - 0.8414)).abs() < 1e-3);
}

#[test]
fn test_gpu_transformer_block_forward_backward() {
    let device = match setup_gpu() {
        Some(d) => d,
        None => return,
    };
    let mut g = Graph::new(device);
    
    let hidden_dim = 16;
    let num_heads = 4;
    let ffn_dim = 64;
    
    // Instantiate the complete Transformer Block
    let block = eisen::nn::transformer::TransformerBlock::new(&mut g, hidden_dim, num_heads, ffn_dim);
    
    // Mock Input: Batch=2, SeqLen=4, HiddenDim=16
    let x_data = vec![0.5; 2 * 4 * 16];
    let x_id = g.alloc(vec![2, 4, 16], x_data);
    
    // Forward Pass
    let out_id = block.forward(&mut g, x_id);
    let out_data = g.tensors[out_id].sync_to_cpu();
    
    // Verify structural integrity
    assert_eq!(out_data.len(), 128);
    assert_eq!(g.tensors[out_id].shape, vec![2, 4, 16]);
    assert!(out_data.iter().all(|v| !v.is_nan()), "Transformer block produced NaNs");
    
    // Backward Pass
    g.backward(out_id);
    let x_grad = g.sync_grad_to_cpu(x_id);
    
    // Verify gradients successfully flowed through the entire block!
    assert!(x_grad.iter().any(|&v| v != 0.0), "Transformer block input received no gradient");
}

#[test]
fn test_gpu_gradient_checkpointing() {
    let device = match setup_gpu() {
        Some(d) => d,
        None => return,
    };
    let mut g = Graph::new(device);

    let hidden_dim = 16;
    let block = eisen::nn::transformer::TransformerBlock::new(&mut g, hidden_dim, 4, 64);
    g.mark_params();
    
    // 1. Allocate block input
    let x_data = vec![0.5; 2 * 4 * 16];
    let x_id = g.alloc(vec![2, 4, 16], x_data);
    
    // ** SAVE POINT: Captures the graph exactly at `x_id`. **
    let save_point = g.mark_save_point();
    
    // 2. FORWARD PASS WITH CHECKPOINTING (Saves VRAM)
    g.no_grad = true;
    let checkpointed_out_id = block.forward(&mut g, x_id);
    let checkpointed_out_data = g.tensors[checkpointed_out_id].sync_to_cpu();
    g.no_grad = false; // Re-enable tracking
    
    // Checkpointing Engine Action: We immediately drop the output and all 
    // internal `TapeNode` activations, returning them to the VRAM pool!
    g.restore_save_point(save_point);
    
    // 3. RECOMPUTATION & BACKWARD PASS
    // We re-run the exact same block with `no_grad = false`. 
    // The engine automatically re-allocates from the VRAM pool we just filled!
    let recomputed_out_id = block.forward(&mut g, x_id);
    
    // Verify mathematical integrity: the checkpointed pass and recomputed pass must be identical!
    let recomputed_out_data = g.tensors[recomputed_out_id].sync_to_cpu();
    assert_eq!(checkpointed_out_data, recomputed_out_data, "Gradient Checkpointing corrupted forward pass data");
    
    // Backpropagate to prove the Wengert List was correctly re-assembled during recomputation.
    g.backward(recomputed_out_id);
    let x_grad = g.sync_grad_to_cpu(x_id);
    assert!(x_grad.iter().any(|&v| v != 0.0), "Gradient Checkpointing failed to backpropagate");
    
    println!("Milestone 4 Passed: Gradient Checkpointing is mechanically sound.");
}

// Helper trait extension to easily extract stream for test resets
trait IntoGpu { fn into_gpu(self) -> (std::sync::Arc<CudaContext>, std::sync::Arc<cudarc::driver::CudaStream>); }
impl IntoGpu for Device { fn into_gpu(self) -> (std::sync::Arc<CudaContext>, std::sync::Arc<cudarc::driver::CudaStream>) { match self { Device::Gpu(c, s) => (c, s), _ => unreachable!() } } }
