use eisen::graph::Graph;

#[test]
fn test_cpu_bmm_softmax_and_transpose_backward() {
    let mut g = Graph::default();

    let a_id = g.alloc(vec![1, 2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b_id = g.alloc(vec![1, 2, 3], vec![1.0, 0.0, -1.0, 2.0, 1.0, 0.0]);

    let scores_id = g.bmm(a_id, b_id, true);
    assert_eq!(g.tensors[scores_id].shape, vec![1, 2, 2]);

    let probs_id = g.softmax(scores_id);
    let probs = g.tensors[probs_id].data.as_cpu().clone();
    for row in 0..2 {
        let off = row * 2;
        let sum = probs[off] + probs[off + 1];
        assert!((sum - 1.0).abs() < 1e-5);
    }

    let weights_id = g.alloc(vec![1, 2, 2], vec![1.0, -0.5, 0.25, 2.0]);
    let weighted_probs = g.mul(probs_id, weights_id);
    let probs_loss = g.sum(g.sum(g.sum(weighted_probs, 2), 1), 0);

    let t_in = g.alloc(
        vec![1, 2, 2, 2],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    );
    let t_id = g.transpose_0213(t_in);
    assert_eq!(g.tensors[t_id].shape, vec![1, 2, 2, 2]);

    let t_loss = g.sum(g.sum(g.sum(g.sum(t_id, 3), 2), 1), 0);
    let total_loss = g.add(probs_loss, t_loss);
    g.backward(total_loss);

    let a_grad = g.tensors[a_id].grad.as_cpu();
    let b_grad = g.tensors[b_id].grad.as_cpu();
    let t_grad = g.tensors[t_in].grad.as_cpu();
    assert!(a_grad.iter().any(|v| v.abs() > 0.0));
    assert!(b_grad.iter().any(|v| v.abs() > 0.0));
    assert_eq!(t_grad, &vec![1.0; 8]);
}

#[test]
fn test_cpu_rope_and_flash_attention_reference_match() {
    let mut g = Graph::default();

    let q_id = g.alloc(vec![1, 2, 4], vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let k_id = g.alloc(vec![1, 2, 4], vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]);

    let rope_id = g.rope(q_id, 4);
    let rope_data = g.tensors[rope_id].data.as_cpu().clone();
    assert_eq!(rope_data[0..4], vec![1.0, 1.0, 1.0, 1.0]);

    let rope_loss = g.sum(rope_id, 0);
    g.backward(rope_loss);
    let q_grad = g.tensors[q_id].grad.as_cpu();
    assert!(q_grad.iter().any(|v| v.abs() > 0.0));

    let mut g_ref = Graph::default();
    g_ref.no_grad = true;

    let q = g_ref.alloc(vec![1, 2, 2], vec![1.0, 0.0, 0.5, 1.0]);
    let k = g_ref.alloc(vec![1, 2, 2], vec![1.0, 1.0, 0.0, 1.0]);
    let v = g_ref.alloc(vec![1, 2, 2], vec![0.0, 1.0, 2.0, 3.0]);

    let scores = g_ref.bmm(q, k, true);
    let scale_id = g_ref.alloc(vec![], vec![1.0 / (2.0f32).sqrt()]);
    let scaled = g_ref.mul(scores, scale_id);
    let probs = g_ref.softmax(scaled);
    let ref_id = g_ref.bmm(probs, v, false);
    let reference = g_ref.tensors[ref_id].data.as_cpu().clone();

    let flash_id = g_ref.flash_attention(q, k, v, 1.0 / (2.0f32).sqrt(), false);
    let flash = g_ref.tensors[flash_id].data.as_cpu().clone();

    assert_eq!(flash.len(), reference.len());
    for (f, r) in flash.iter().zip(reference.iter()) {
        assert!((f - r).abs() < 1e-5, "flash mismatch: {} vs {}", f, r);
    }
}
