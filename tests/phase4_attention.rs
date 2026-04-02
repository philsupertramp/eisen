use eisen::graph::Graph;
use eisen::tensor::Device;
use cudarc::driver::CudaContext;
use std::sync::Arc;
use eisen::nn::attention::Attention;
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
fn test_gpu_attention_layer_forward_backward() {
    let device = match setup_gpu() {
        Some(d) => d,
        None => return,
    };
    let mut g = Graph::new(device);
    
    let hidden_dim = 4;
    // Instantiate our new Attention layer
    let attention = Attention::new(&mut g, hidden_dim);
    
    // Mock Input: Batch=1, SeqLen=3, HiddenDim=4
    let x_data = vec![
        1.0, 0.0, 1.0, 0.0, // token 1
        0.0, 1.0, 0.0, 1.0, // token 2
        1.0, 1.0, 1.0, 1.0, // token 3
    ];
    let x_id = g.alloc(vec![1, 3, 4], x_data);
    
    // Forward Pass (Causal Mask = True)
    let out_id = attention.forward(&mut g, x_id);
    let out_data = g.tensors[out_id].sync_to_cpu();
    
    // Verify structural integrity and mathematical stability
    assert_eq!(out_data.len(), 12);
    assert!(out_data.iter().all(|v| !v.is_nan()), "Attention produced NaNs");
    
    // Backward Pass
    g.backward(out_id);
    let x_grad = g.sync_grad_to_cpu(x_id);
    
    // Verify gradients successfully flowed through all the projections and BMMs!
    assert!(x_grad.iter().any(|&v| v != 0.0), "Attention input received no gradient");
}

