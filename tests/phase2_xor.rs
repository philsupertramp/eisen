use eisen::graph::Graph;
use eisen::nn::Module;
use eisen::nn::linear::Linear;
use eisen::nn::optim::AdamW;

#[test]
fn test_xor_mlp_overfit() {
    println!("\n=== Eisen Phase 2: XOR MLP Training ===");
    let mut g = Graph::default();
    
    // Architecture: 2 Inputs -> 8 Hidden -> 2 Outputs (Classes 0 and 1)
    let hidden_dim = 8;
    let l1 = Linear::new(&mut g, 2, hidden_dim, true);
    let l2 = Linear::new(&mut g, hidden_dim, 2, true);
    
    // Gather all parameters and initialize AdamW
    let mut params = Vec::new();
    params.extend(l1.params());
    params.extend(l2.params());
    
    let mut optim = AdamW::new(params, 0.1); // High learning rate for fast convergence
    
    // XOR Dataset (Batch size: 4)
    let x_data = vec![
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    ];
    let y_targets = vec![0, 1, 1, 0]; // XOR Truth Table

    let mut final_loss = f32::MAX;

    for epoch in 1..=50 {
        // --- FORWARD PASS ---
        // 1. Allocate inputs for this step
        let x_id = g.alloc(vec![4, 2], x_data.clone());
        
        // 2. Hidden Layer + SiLU Activation
        let h_id = l1.forward(&mut g, x_id);
        let act_id = g.silu(h_id);
        
        // 3. Output Layer (Logits)
        let logits_id = l2.forward(&mut g, act_id);
        
        // 4. Calculate Loss
        let loss_id = g.cross_entropy(logits_id, &y_targets);
        let current_loss = g.tensors[loss_id].data.as_cpu()[0];
        
        if epoch % 10 == 0 {
            println!("Epoch {:03} | Loss: {:.6}", epoch, current_loss);
        }
        final_loss = current_loss;

        // --- BACKWARD PASS ---
        optim.zero_grad(&mut g);
        g.backward(loss_id);
        
        // --- OPTIMIZATION STEP ---
        optim.step(&mut g);
        
        // --- LIFECYCLE MANAGEMENT ---
        // Crucial: Clear the Wengert list so we don't accumulate history across epochs!
        g.tape.nodes.clear();
    }
    
    println!("Final Loss: {:.6}", final_loss);
    
    // Phase 2 Milestone Assertion: The model must overfit the XOR problem!
    assert!(final_loss < 0.05, "Failed to overfit XOR. Loss is too high!");
}
