use eisen::graph::Graph;
use eisen::nn::linear::Linear;
use eisen::nn::embedding::Embedding;
use eisen::nn::optim::AdamW;
use eisen::nn::Module;

/// A Fixed-Window Neural Language Model (Bengio et al. 2003)
struct BengioLM {
    window_size: usize,
    hidden_dim: usize,
    embedding: Embedding,
    hidden: Linear,
    head: Linear,
}

impl BengioLM {
    fn new(g: &mut Graph, vocab_size: usize, window_size: usize, hidden_dim: usize) -> Self {
        let embedding = Embedding::new(g, vocab_size, hidden_dim);
        // We flatten the window of embeddings, so input is window * dim
        let hidden = Linear::new(g, window_size * hidden_dim, hidden_dim, true);
        let head = Linear::new(g, hidden_dim, vocab_size, true);

        Self {
            window_size,
            hidden_dim,
            embedding,
            hidden,
            head,
        }
    }
}

impl Module for BengioLM {
    fn forward(&self, g: &mut Graph, x_id: usize) -> usize {
        let batch_size = g.tensors[x_id].shape[0];
        
        // 1. Embed: [Batch, Window] -> [Batch, Window, Dim]
        let x = self.embedding.forward(g, x_id);
        
        // 2. Flatten: [Batch, Window, Dim] -> [Batch, Window * Dim]
        let flat_id = g.reshape(x, vec![batch_size, self.window_size * self.hidden_dim]);
        
        // 3. MLP: Linear -> SiLU -> Linear
        let h = self.hidden.forward(g, flat_id);
        let act = g.silu(h);
        
        // 4. Output Head (Logits)
        self.head.forward(g, act)
    }

    fn params(&self) -> Vec<usize> {
        let mut p = Vec::new();
        p.extend(self.embedding.params());
        p.extend(self.hidden.params());
        p.extend(self.head.params());
        p
    }
}

fn main() {
    println!("=== Eisen Example: German Bengio LM (Sliding Window) ===");
    let mut g = Graph::default();
    
    // Dataset: "der mensch ist was er isst ."
    let vocab = vec!["der", "mensch", "ist", "was", "er", "isst", "."];
    let vocab_size = vocab.len();
    let window_size = 2;
    let hidden_dim = 32;
    
    let model = BengioLM::new(&mut g, vocab_size, window_size, hidden_dim);
    let mut optim = AdamW::new(model.params(), 0.01);
    
    // (der, mensch) -> ist, etc.
    let x_data = vec![
        0.0, 1.0, // der, mensch
        1.0, 2.0, // mensch, ist
        2.0, 3.0, // ist, was
        3.0, 4.0, // was, er
        4.0, 5.0, // er, isst
    ];
    let y_targets = vec![2, 3, 4, 5, 6];
    let num_samples = 5;

    println!("Training on: 'der mensch ist was er isst .'");
    for epoch in 1..=300 {
        let x_id = g.alloc(vec![num_samples, window_size], x_data.clone());
        let logits_id = model.forward(&mut g, x_id);
        let loss_id = g.cross_entropy(logits_id, &y_targets);
        
        let loss = g.tensors[loss_id].data.as_cpu()[0];
        if epoch % 50 == 0 {
            println!("Epoch {:03} | Loss: {:.6}", epoch, loss);
        }

        optim.zero_grad(&mut g);
        g.backward(loss_id);
        optim.step(&mut g);
        g.tape.nodes.clear();
    }

    // Inference check
    let test_context = vec![1.0, 2.0]; // "der", "mensch"
    let x_test = g.alloc(vec![1, window_size], test_context);
    let logits_id = model.forward(&mut g, x_test);
    
    let predicted_id = g.tensors[logits_id].data.as_cpu().iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i).unwrap();
    
    println!("\nTest Context: 'der mensch'");
    println!("Prediction:   '{}'", vocab[predicted_id]);
    
    if vocab[predicted_id] == "ist" {
        println!("\nSUCCESS: Model correctly learned the grammatical dependency!");
    }
}
