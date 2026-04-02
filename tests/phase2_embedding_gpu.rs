use eisen::graph::Graph;
use eisen::nn::embedding::Embedding;
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
fn test_embedding_forward_backward() {
    println!("\n=== Eisen Phase 2: Embedding Layer Test ===");
    let device = match setup_gpu() {
        Some(d) => d,
        None => { eprintln!("No GPU found, skipping test."); return; }
    };
    let mut g = Graph::new(device);
    
    let vocab_size = 5;
    let hidden_dim = 3;
    let emb = Embedding::new(&mut g, vocab_size, hidden_dim);
    
    // We want to lookup tokens: [1, 3, 1]
    // We use f32 because our engine currently only supports f32 tensors!
    let token_ids = vec![1.0, 3.0, 1.0];
    let x_id = g.alloc(vec![3], token_ids);
    
    let out_id = emb.forward(&mut g, x_id);
    
    let out_shape = &g.tensors[out_id].shape;
    assert_eq!(out_shape, &vec![3, 3], "Output shape should be [seq_len, hidden_dim]");
    
    // Forward pass assertion: Since sequence index 0 and 2 are both token '1', 
    // their output vectors must be exactly identical.
    let out_data = &g.tensors[out_id].sync_to_cpu();
    let vec_0 = &out_data[0..3];
    let vec_2 = &out_data[6..9];
    assert_eq!(vec_0, vec_2, "Embeddings for the same token ID must match");
    
    // --- BACKWARD PASS ---
    g.backward(out_id);
    
    let weight_grads = &g.sync_grad_to_cpu(emb.weight_id);
    
    // Token '1' was used twice, so its row in the embedding matrix should have accumulated a gradient of 2.0
    assert_eq!(&weight_grads[3..6], &[2.0, 2.0, 2.0], "Token 1 should have gradient 2.0");
    
    // Token '3' was used once, gradient should be 1.0
    assert_eq!(&weight_grads[9..12], &[1.0, 1.0, 1.0], "Token 3 should have gradient 1.0");
    
    // Token '0', '2', '4' were unused, gradients should be 0.0
    assert_eq!(&weight_grads[0..3], &[0.0, 0.0, 0.0], "Unused tokens should have 0.0 gradient");
    
    println!("Phase 2 Embedding Layer: SUCCESS. Gradients correctly accumulated for repeated tokens.");
}

