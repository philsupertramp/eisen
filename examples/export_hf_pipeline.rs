use serde::Deserialize;
use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use serde_json::json;

#[derive(Deserialize, Debug)]
struct RunManifest {
    hyperparams: Hyperparams,
}

#[derive(Deserialize, Debug)]
struct Hyperparams {
    hidden_dim: usize,
    num_heads: usize,
    ffn_dim: usize,
    num_layers: usize,
    seq_len: usize,
    num_kv_heads: usize,
    tie_weights: bool,
}

fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut transposed = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            transposed[c * rows + r] = data[r * cols + c];
        }
    }
    transposed
}

fn permute_rope_hf(data: &[f32], num_heads: usize, head_dim: usize, in_features: usize) -> Vec<f32> {
    let mut permuted = vec![0.0; data.len()];
    for h in 0..num_heads {
        for d in 0..(head_dim / 2) {
            let orig_row_1 = h * head_dim + 2 * d;
            let orig_row_2 = h * head_dim + 2 * d + 1;
            let hf_row_1 = h * head_dim + d;
            let hf_row_2 = h * head_dim + (head_dim / 2) + d;

            let orig_idx_1 = orig_row_1 * in_features;
            let orig_idx_2 = orig_row_2 * in_features;
            let hf_idx_1 = hf_row_1 * in_features;
            let hf_idx_2 = hf_row_2 * in_features;

            permuted[hf_idx_1..hf_idx_1 + in_features]
                .copy_from_slice(&data[orig_idx_1..orig_idx_1 + in_features]);
            permuted[hf_idx_2..hf_idx_2 + in_features]
                .copy_from_slice(&data[orig_idx_2..orig_idx_2 + in_features]);
        }
    }
    permuted
}

fn chunk_tensor(
    data: &[f32],
    offset: &mut usize,
    name: &str,
    shape_in: Vec<usize>,
    transpose: bool,
    is_q: bool,
    is_k: bool,
    hp: &Hyperparams,
) -> (String, Vec<usize>, Vec<f32>) {
    let size: usize = shape_in.iter().product();
    if *offset + size > data.len() {
        panic!("EOF reached unexpectedly while parsing tensor: {}", name);
    }
    
    let chunk = &data[*offset..*offset + size];
    *offset += size;

    let mut out_data = chunk.to_vec();
    let mut out_shape = shape_in.clone();

    if transpose {
        out_data = transpose_2d(&out_data, shape_in[0], shape_in[1]);
        out_shape.swap(0, 1);
    }

    let head_dim = hp.hidden_dim / hp.num_heads;
    if is_q {
        out_data = permute_rope_hf(&out_data, hp.num_heads, head_dim, out_shape[1]);
    } else if is_k {
        out_data = permute_rope_hf(&out_data, hp.num_kv_heads, head_dim, out_shape[1]);
    }

    (name.to_string(), out_shape, out_data)
}

fn as_json_array(shape: &[usize]) -> String {
    let inner: Vec<String> = shape.iter().map(|x| x.to_string()).collect();
    format!("[{}]", inner.join(","))
}

fn export_tokenizer(tokenizer_path: &str) {
    println!("Reading custom tokenizer from {}...", tokenizer_path);
    let mut content = String::new();
    File::open(tokenizer_path).expect("Missing tokenizer.model").read_to_string(&mut content).unwrap();
    let mut lines = content.lines();

    let num_special: usize = lines.next().unwrap().parse().unwrap();
    let mut special_tokens = Vec::new();
    for _ in 0..num_special {
        special_tokens.push(lines.next().unwrap().to_string());
    }

    let vocab_size: usize = lines.next().unwrap().parse().unwrap();
    let mut vocab = Vec::new();
    let mut vocab_dict = serde_json::Map::new();
    for i in 0..vocab_size {
        let v = lines.next().unwrap().replace("\\n", "\n");
        vocab.push(v.clone());
        vocab_dict.insert(v, json!(i));
    }

    let mut merges = Vec::new();
    for line in lines {
        let parts: Vec<usize> = line.split_whitespace().map(|s| s.parse().unwrap()).collect();
        if parts.len() == 3 {
            let p1 = &vocab[parts[0]];
            let p2 = &vocab[parts[1]];
            merges.push(format!("{} {}", p1, p2));
        }
    }

    let mut added_tokens = Vec::new();
    for st in special_tokens {
        if let Some(id) = vocab.iter().position(|x| x == &st) {
            added_tokens.push(json!({
                "id": id,
                "content": st,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
            }));
        }
    }

    let tokenizer_json = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": null,
        "pre_tokenizer": null, 
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": false,
            "vocab": vocab_dict,
            "merges": merges
        }
    });

    let file = File::create("tokenizer.json").unwrap();
    serde_json::to_writer_pretty(file, &tokenizer_json).unwrap();
    println!("Exported tokenizer.json!");
}

fn main() {
    println!("=== Eisen Safetensors & Tokenizer Converter ===");
    
    let manifest_file = File::open("data/run_manifest.json").expect("Missing data/run_manifest.json");
    let manifest: RunManifest = serde_json::from_reader(manifest_file).unwrap();
    let hp = manifest.hyperparams;

    export_tokenizer("data/german_tokenizer.model");

    let mut content = String::new();
    File::open("data/german_tokenizer.model").unwrap().read_to_string(&mut content).unwrap();
    let mut lines = content.lines();
    let num_special: usize = lines.next().unwrap().parse().unwrap();
    for _ in 0..num_special { lines.next(); }
    let vocab_size: usize = lines.next().unwrap().parse().unwrap();

    println!("Loaded Architectures: Layers={}, Hidden={}, Heads={}, KV_Heads={}, FFN={}, Vocab={}", 
             hp.num_layers, hp.hidden_dim, hp.num_heads, hp.num_kv_heads, hp.ffn_dim, vocab_size);

    println!("Reading raw weights from data/eisen_model.bin...");
    let mut bin_file = File::open("data/eisen_model.bin").expect("Missing data/eisen_model.bin");
    let mut buffer = Vec::new();
    bin_file.read_to_end(&mut buffer).unwrap();

    let raw_floats: Vec<f32> = buffer
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let mut offset = 0;
    let mut tensors = Vec::new();

    tensors.push(chunk_tensor(&raw_floats, &mut offset, "model.embed_tokens.weight", vec![vocab_size, hp.hidden_dim], false, false, false, &hp));

    for i in 0..hp.num_layers {
        tensors.push(chunk_tensor(&raw_floats, &mut offset, &format!("model.layers.{}.input_layernorm.weight", i), vec![hp.hidden_dim], false, false, false, &hp));
        
        tensors.push(chunk_tensor(&raw_floats, &mut offset, &format!("model.layers.{}.self_attn.q_proj.weight", i), vec![hp.hidden_dim, hp.hidden_dim], true, true, false, &hp));
        tensors.push(chunk_tensor(&raw_floats, &mut offset, &format!("model.layers.{}.self_attn.k_proj.weight", i), vec![hp.hidden_dim, (hp.hidden_dim / hp.num_heads) * hp.num_kv_heads], true, false, true, &hp));
        tensors.push(chunk_tensor(&raw_floats, &mut offset, &format!("model.layers.{}.self_attn.v_proj.weight", i), vec![hp.hidden_dim, (hp.hidden_dim / hp.num_heads) * hp.num_kv_heads], true, false, false, &hp));
        tensors.push(chunk_tensor(&raw_floats, &mut offset, &format!("model.layers.{}.self_attn.o_proj.weight", i), vec![hp.hidden_dim, hp.hidden_dim], true, false, false, &hp));
        
        tensors.push(chunk_tensor(&raw_floats, &mut offset, &format!("model.layers.{}.post_attention_layernorm.weight", i), vec![hp.hidden_dim], false, false, false, &hp));
        
        tensors.push(chunk_tensor(&raw_floats, &mut offset, &format!("model.layers.{}.mlp.gate_proj.weight", i), vec![hp.hidden_dim, hp.ffn_dim], true, false, false, &hp));
        tensors.push(chunk_tensor(&raw_floats, &mut offset, &format!("model.layers.{}.mlp.up_proj.weight", i), vec![hp.hidden_dim, hp.ffn_dim], true, false, false, &hp));
        tensors.push(chunk_tensor(&raw_floats, &mut offset, &format!("model.layers.{}.mlp.down_proj.weight", i), vec![hp.ffn_dim, hp.hidden_dim], true, false, false, &hp));
    }

    tensors.push(chunk_tensor(&raw_floats, &mut offset, "model.norm.weight", vec![hp.hidden_dim], false, false, false, &hp));

    if !hp.tie_weights && offset < raw_floats.len() {
        tensors.push(chunk_tensor(&raw_floats, &mut offset, "lm_head.weight", vec![hp.hidden_dim, vocab_size], true, false, false, &hp));
    }

    println!("All {} parameters parsed. Writing model.safetensors...", offset);

    let mut entries = BTreeMap::<String, (Vec<usize>, usize, usize)>::new();
    let mut payload = Vec::<u8>::new();

    for (name, shape, data) in tensors {
        let begin = payload.len();
        for v in data {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        let end = payload.len();
        entries.insert(name, (shape, begin, end));
    }

    let mut header = String::from("{");
    for (i, (name, (shape, begin, end))) in entries.iter().enumerate() {
        if i > 0 { header.push(','); }
        header.push_str(&format!("\"{}\":{{\"dtype\":\"F32\",\"shape\":{},\"data_offsets\":[{},{}]}}", 
            name, as_json_array(shape), begin, end));
    }
    header.push('}');
    while header.len() % 8 != 0 { header.push(' '); }

    let mut file = BufWriter::new(File::create("model.safetensors").unwrap());
    let header_len = header.len() as u64;
    file.write_all(&header_len.to_le_bytes()).unwrap();
    file.write_all(header.as_bytes()).unwrap();
    file.write_all(&payload).unwrap();
    file.flush().unwrap();

    println!("Writing config.json...");
    let config = json!({
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": vocab_size,
        "hidden_size": hp.hidden_dim,
        "intermediate_size": hp.ffn_dim,
        "num_hidden_layers": hp.num_layers,
        "num_attention_heads": hp.num_heads,
        "num_key_value_heads": hp.num_kv_heads,
        "max_position_embeddings": hp.seq_len,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "hidden_act": "silu",
        "tie_word_embeddings": hp.tie_weights,
        "initializer_range": 0.02,
        "torch_dtype": "float32"
    });

    let config_file = File::create("config.json").unwrap();
    serde_json::to_writer_pretty(config_file, &config).unwrap();

    println!("Done! You can now load `model.safetensors` and the tokenizer natively in Hugging Face.");
}
