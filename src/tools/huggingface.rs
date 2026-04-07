use crate::graph::Graph;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};

/// Minimal Llama-compatible config payload.
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
}

fn as_json_array(values: &[usize]) -> String {
    let items: Vec<String> = values.iter().map(|v| v.to_string()).collect();
    format!("[{}]", items.join(","))
}

/// Writes a tiny, dependency-free `config.json` that follows
/// Llama-style keys expected by `transformers`.
pub fn write_llama_config(path: &str, cfg: &LlamaConfig) -> std::io::Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    let body = format!(
        concat!(
            "{{\n",
            "  \"architectures\": [\"LlamaForCausalLM\"],\n",
            "  \"model_type\": \"llama\",\n",
            "  \"vocab_size\": {},\n",
            "  \"hidden_size\": {},\n",
            "  \"intermediate_size\": {},\n",
            "  \"num_hidden_layers\": {},\n",
            "  \"num_attention_heads\": {},\n",
            "  \"num_key_value_heads\": {},\n",
            "  \"max_position_embeddings\": {},\n",
            "  \"rms_norm_eps\": {},\n",
            "  \"rope_theta\": {},\n",
            "  \"bos_token_id\": 1,\n",
            "  \"eos_token_id\": 2,\n",
            "  \"pad_token_id\": 0,\n",
            "  \"hidden_act\": \"silu\",\n",
            "  \"tie_word_embeddings\": {},\n",
            "  \"initializer_range\": 0.02,\n",
            "  \"torch_dtype\": \"float32\"\n",
            "}}\n"
        ),
        cfg.vocab_size,
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_attention_heads,
        cfg.max_position_embeddings,
        cfg.rms_norm_eps,
        cfg.rope_theta,
        if cfg.tie_word_embeddings {
            "true"
        } else {
            "false"
        },
    );
    f.write_all(body.as_bytes())?;
    f.flush()
}

/// Export graph parameters into a valid `.safetensors` file.
///
/// `named_params` is an ordered list of (tensor_name, tensor_id).
pub fn write_safetensors(
    g: &Graph,
    named_params: &[(String, usize)],
    path: &str,
) -> std::io::Result<()> {
    let mut entries = BTreeMap::<String, (Vec<usize>, usize, usize)>::new();
    let mut payload = Vec::<u8>::new();

    for (name, tid) in named_params {
        let t = &g.tensors[*tid];
        let data = t.sync_to_cpu();
        let begin = payload.len();
        for v in &data {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        let end = payload.len();
        entries.insert(name.clone(), (t.shape.clone(), begin, end));
    }

    let mut header = String::from("{");
    for (i, (name, (shape, begin, end))) in entries.iter().enumerate() {
        if i > 0 {
            header.push(',');
        }
        header.push('"');
        header.push_str(name);
        header.push_str("\":{");
        header.push_str("\"dtype\":\"F32\",");
        header.push_str("\"shape\":");
        header.push_str(&as_json_array(shape));
        header.push(',');
        header.push_str("\"data_offsets\":[");
        header.push_str(&begin.to_string());
        header.push(',');
        header.push_str(&end.to_string());
        header.push(']');
        header.push('}');
    }
    header.push('}');
    while header.len() % 8 != 0 {
        header.push(' ');
    }

    let mut file = BufWriter::new(File::create(path)?);
    let header_len = header.len() as u64;
    file.write_all(&header_len.to_le_bytes())?;
    file.write_all(header.as_bytes())?;
    file.write_all(&payload)?;
    file.flush()
}
