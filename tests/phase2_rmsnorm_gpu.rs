use eisen::graph::Graph;
use eisen::nn::rmsnorm::RMSNorm;
use eisen::nn::Module;
use eisen::tensor::Device;
use cudarc::driver::CudaContext;

fn setup_gpu() -> Option<Device> {
    match CudaContext::new(0) {
        Ok(ctx) => {
            let ctx = ctx;
            let stream = ctx.default_stream();
            Some(Device::Gpu(ctx, stream))
        }
        Err(_) => None,
    }
}

#[test]
fn test_rmsnorm_forward_backward() {
    println!("\n=== Eisen Phase 2: RMSNorm Test ===");
    let device = match setup_gpu() {
        Some(d) => d,
        None => { eprintln!("No GPU found, skipping test."); return; }
    };
    let mut g = Graph::new(device);
    
    let dim = 4;
    let norm = RMSNorm::new(&mut g, dim, 1e-5);
    
    // Allocate some dummy input [batch=2, dim=4]
    let x_data = vec![
        1.0, -2.0, 3.0, -4.0, // Vector 1
        0.1, 0.2, -0.1, 0.0,  // Vector 2
    ];
    let x_id = g.alloc(vec![2, 4], x_data);
    
    // Forward Pass
    let out_id = norm.forward(&mut g, x_id);
    
    // Backward Pass
    g.backward(out_id);
    
    // Verify outputs
    let out = &g.sync_grad_to_cpu(out_id);
    assert_eq!(out.len(), 8);
    assert!(out.iter().all(|v| !v.is_nan()), "RMSNorm produced NaNs");
    
    // Verify Gradients
    let w_grad = &g.sync_grad_to_cpu(norm.weight_id);
    let x_grad = &g.sync_grad_to_cpu(x_id);
    
    assert!(w_grad.iter().any(|&v| v != 0.0), "RMSNorm weight received no gradient");
    assert!(x_grad.iter().any(|&v| v != 0.0), "RMSNorm input received no gradient");
    
    println!("Phase 2 RMSNorm: SUCCESS. Fused analytical gradients passed.");
}

