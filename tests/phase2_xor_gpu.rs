use eisen::graph::Graph;
use eisen::nn::linear::Linear;
use eisen::nn::optim::AdamW;
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
fn test_xor_mlp_gpu() {
    println!("\n=== Eisen Phase 3: GPU XOR MLP (No-SiLU Bypass) ===");
    let device = match setup_gpu() {
        Some(d) => d,
        None => { eprintln!("No GPU found, skipping test."); return; }
    };
    let mut g = Graph::new(device);
    
    // Architecture: 2 Inputs -> 8 Hidden -> 1 Output
    let hidden_dim = 8;
    let l1 = Linear::new(&mut g, 2, hidden_dim, true);
    let l2 = Linear::new(&mut g, hidden_dim, 1, true);
    
    let layers: Vec<&dyn Module> = vec![&l1, &l2];
    let mut params = Vec::new();
    for layer in layers { params.extend(layer.params()); }
    
    let mut optim = AdamW::new(params, 0.05);
    
    let x_data = vec![
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    ];
    // XOR Truth Table. We use negative targets so we can do (Pred - Target) directly
    // since we haven't written a subtraction op!
    let y_targets = vec![0.0, -1.0, -1.0, 0.0]; 

    let mut final_loss = f32::MAX;

    for epoch in 1..=500 {
        let x_id = g.alloc(vec![4, 2], x_data.clone());
        let y_id = g.alloc(vec![4, 1], y_targets.clone());
        
        let h_id = l1.forward(&mut g, x_id);
        
        // BYPASS: Use x^2 as our non-linearity since we don't have SiLU yet!
        let act_id = g.mul(h_id, h_id); 
        
        let logits_id = l2.forward(&mut g, act_id);
        
        // BYPASS: Calculate Mean Squared Error manually since we don't have CrossEntropy!
        // Diff = Logits + (-Targets)
        let diff_id = g.add(logits_id, y_id);
        // Sq = Diff * Diff
        let loss_id = g.mul(diff_id, diff_id);

        if epoch % 100 == 0 || epoch == 500 {
            // Pull the un-summed vector back to calculate the metric
            let current_loss_vec = g.tensors[loss_id].sync_to_cpu();
            final_loss = current_loss_vec.iter().sum::<f32>() / 4.0;
            println!("Epoch {:03} | MSE Loss: {:.6}", epoch, final_loss);
        }

        optim.zero_grad(&mut g);
        // We can backpropagate directly from the vector! 
        // g.backward fills the initial gradient with 1.0s, which is perfectly valid 
        // for an element-wise loss function wrapper.
        g.backward(loss_id);
        optim.step(&mut g);
        
        g.tape.nodes.clear();
    }
    
    assert!(final_loss < 0.05, "Failed to overfit XOR on GPU!");
    println!("Phase 3 Bypass: SUCCESS! Linear layers, AdamW, and Autograd are fully operational in VRAM.");
}
