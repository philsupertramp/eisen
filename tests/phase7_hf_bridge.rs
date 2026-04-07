use eisen::graph::Graph;
use eisen::tensor::Device;
use eisen::tools::huggingface::{write_llama_config, write_safetensors, LlamaConfig};
use std::fs;

#[test]
fn writes_llama_config_json() {
    let path = "target/test_phase7_config.json";
    let cfg = LlamaConfig {
        vocab_size: 32000,
        hidden_size: 384,
        intermediate_size: 1536,
        num_hidden_layers: 6,
        num_attention_heads: 6,
        max_position_embeddings: 256,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        tie_word_embeddings: false,
    };
    write_llama_config(path, &cfg).unwrap();
    let text = fs::read_to_string(path).unwrap();
    assert!(text.contains("\"model_type\": \"llama\""));
    assert!(text.contains("\"hidden_size\": 384"));
}

#[test]
fn writes_safetensors_file_with_header() {
    let mut g = Graph::new(Device::Cpu);
    let t0 = g.alloc(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let t1 = g.alloc(vec![3], vec![5.0, 6.0, 7.0]);

    let out = "target/test_phase7_model.safetensors";
    let named = vec![("a.weight".to_string(), t0), ("b.weight".to_string(), t1)];
    write_safetensors(&g, &named, out).unwrap();

    let bytes = fs::read(out).unwrap();
    let mut len_bytes = [0u8; 8];
    len_bytes.copy_from_slice(&bytes[0..8]);
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    let header = std::str::from_utf8(&bytes[8..8 + header_len]).unwrap();
    assert!(header.contains("\"a.weight\""));
    assert!(header.contains("\"dtype\":\"F32\""));
    assert!(header.contains("\"shape\":[2,2]"));
    assert!(header.contains("\"b.weight\""));

    let payload = &bytes[8 + header_len..];
    assert_eq!(payload.len(), (4 + 3) * 4);
}
