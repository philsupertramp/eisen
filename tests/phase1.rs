use eisen::graph::Graph;

#[test]
fn test_broadcasting_and_reductions() {
    let mut g = Graph::default();
    
    // Matrix A: Shape [2, 3]
    let a_id = g.alloc(vec![2, 3], vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    ]);
    
    // Vector B: Shape [1, 3] (Will be broadcasted to [2, 3]!)
    let b_id = g.alloc(vec![1, 3], vec![
        10.0, 20.0, 30.0,
    ]);
    
    // Broadcasted Add: C = A + B => shape [2, 3]
    let c_id = g.add(a_id, b_id);
    
    // Transpose: D = C.T => shape [3, 2]
    let d_id = g.transpose(c_id, 0, 1);
    
    // Max: E = max(D, dim=1) => shape [3]
    let e_id = g.max(d_id, 1);
    
    // Sum: F = sum(E, dim=0) => shape []
    let f_id = g.sum(e_id, 0);
    
    // --- FORWARD ASSERTIONS ---
    assert_eq!(g.tensors[f_id].shape, Vec::<usize>::new(), "F should be a scalar (empty shape)");
    assert_eq!(g.tensors[f_id].data.as_cpu()[0], 75.0, "Forward pass computation mismatch");

    // Backward Pass
    g.backward(f_id);
    
    // --- BACKWARD ASSERTIONS ---
    assert_eq!(g.tensors[a_id].grad.as_cpu(), &vec![
        0.0, 0.0, 0.0, // Row 0 didn't win the max
        1.0, 1.0, 1.0, // Row 1 won the max
    ], "Gradient mismatch for Tensor A");

    assert_eq!(g.tensors[b_id].grad.as_cpu(), &vec![
        1.0, 1.0, 1.0
    ], "Gradient mismatch for broadcasted Tensor B");
}
