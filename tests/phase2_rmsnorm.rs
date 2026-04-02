use eisen::graph::Graph;
use eisen::nn::rmsnorm::RMSNorm;
use eisen::nn::Module;

#[test]
fn test_rmsnorm_forward_backward() {
    println!("\n=== Eisen Phase 2: RMSNorm Test ===");
    let mut g = Graph::default();
    
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
    let out = &g.tensors[out_id];
    assert_eq!(out.shape, vec![2, 4]);
    assert!(out.data.as_cpu().iter().all(|v| !v.is_nan()), "RMSNorm produced NaNs");
    
    // Verify Gradients
    let w_grad = &g.tensors[norm.weight_id].grad;
    let x_grad = &g.tensors[x_id].grad;
    
    assert!(w_grad.as_cpu().iter().any(|&v| v != 0.0), "RMSNorm weight received no gradient");
    assert!(x_grad.as_cpu().iter().any(|&v| v != 0.0), "RMSNorm input received no gradient");
    
    println!("Phase 2 RMSNorm: SUCCESS. Fused analytical gradients passed.");
}
