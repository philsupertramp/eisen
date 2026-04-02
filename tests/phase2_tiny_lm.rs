use eisen::graph::Graph;
use eisen::nn::embedding::Embedding;
use eisen::nn::linear::Linear;
use eisen::nn::optim::AdamW;
use eisen::nn::Module;

#[test]
fn test_embedding_text_generation() {
    println!("\n=== Eisen Phase 2: German Text Generation Test ===");
    let mut g = Graph::default();
    
    // Our mini German vocabulary
    let vocab = vec![
        "ich",          // 0
        "möchte",       // 1
        "ein",          // 2
        "eigenes",      // 3
        "deutsches",    // 4
        "sprachmodell", // 5
        "bauen",        // 6
        ".",            // 7
    ];
    let vocab_size = vocab.len();
    let hidden_dim = 16;
    
    // Architecture: Embedding -> Linear (Language Modeling Head)
    let emb = Embedding::new(&mut g, vocab_size, hidden_dim);
    let head = Linear::new(&mut g, hidden_dim, vocab_size, true);
    
    let mut params = Vec::new();
    params.extend(emb.params());
    params.extend(head.params());
    
    let mut optim = AdamW::new(params, 0.05); // AdamW Optimizer
    
    // Dataset: Predict the next word!
    // ich(0)->möchte(1)->ein(2)->eigenes(3)->deutsches(4)->sprachmodell(5)->bauen(6)->.(7)
    let x_data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let y_targets = vec![1,   2,   3,   4,   5,   6,   7];
    
    let mut final_loss = f32::MAX;

    // --- TRAINING LOOP ---
    println!("Training...");
    for epoch in 1..=200 {
        let x_id = g.alloc(vec![7], x_data.clone());
        let emb_id = emb.forward(&mut g, x_id);
        let logits_id = head.forward(&mut g, emb_id);
        
        let loss_id = g.cross_entropy(logits_id, &y_targets);
        final_loss = g.tensors[loss_id].data.as_cpu()[0];
        
        if epoch % 50 == 0 {
            println!("Epoch {:03} | Loss: {:.6}", epoch, final_loss);
        }

        optim.zero_grad(&mut g);
        g.backward(loss_id);
        optim.step(&mut g);
        g.tape.nodes.clear();
    }
    
    assert!(final_loss < 0.1, "Failed to learn the sequence.");
    println!("Training Complete. Model is ready to generate!\n");

    // --- INFERENCE / GENERATION LOOP ---
    print!("Generation: ");
    
    // 1. Seed the prompt with the first word: "ich" (ID: 0)
    let mut current_token_id = 0;
    print!("{} ", vocab[current_token_id]);

    // 2. Autoregressively generate the next 7 tokens
    for _ in 0..7 {
        // Allocate just the single current token
        let x_id = g.alloc(vec![1], vec![current_token_id as f32]);
        
        // Forward pass
        let emb_id = emb.forward(&mut g, x_id);
        let logits_id = head.forward(&mut g, emb_id);
        
        let logits = &g.tensors[logits_id].data;
        
        // Find the Argmax (Greedy Decoding)
        let mut best_score = std::f32::NEG_INFINITY;
        let mut best_token = 0;
        for (vocab_id, &score) in logits.as_cpu().iter().enumerate() {
            if score > best_score {
                best_score = score;
                best_token = vocab_id;
            }
        }
        
        // Print the predicted word
        print!("{} ", vocab[best_token]);
        
        // Update the current token for the next step in the loop
        current_token_id = best_token;
        
        // Clear the tape since we don't need gradients for inference!
        g.tape.nodes.clear();
    }
    println!("\n\nPhase 2 Practical Test: SUCCESS!");
}
