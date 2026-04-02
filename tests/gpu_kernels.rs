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
