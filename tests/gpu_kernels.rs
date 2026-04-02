use eisen::graph::Graph;
use eisen::tensor::Device;
use cudarc::driver::CudaContext;
use std::sync::Arc;

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
// Helper trait extension to easily extract stream for test resets
trait IntoGpu { fn into_gpu(self) -> (std::sync::Arc<CudaContext>, std::sync::Arc<cudarc::driver::CudaStream>); }
impl IntoGpu for Device { fn into_gpu(self) -> (std::sync::Arc<CudaContext>, std::sync::Arc<cudarc::driver::CudaStream>) { match self { Device::Gpu(c, s) => (c, s), _ => unreachable!() } } }
