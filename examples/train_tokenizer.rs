use eisen::data::tokenizer::BPETokenizer;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use rand::Rng;

/// The entry point for training our German BPE model on large datasets.
fn main() {
    // --- CONFIGURATION ---
    let input_path = "data/german_tiny_story_corpus.txt"; // From the Python extractor
    let output_tokenizer_path = "data/german_tokenizer.model";
    let cache_path = "data/german_sampled_training_text.txt"; 
    let target_vocab_size = 65536; 
    let sample_size_mb = 50 * 1024 * 1024;     

    // Define our atomic special tokens for GDPR masking and model logic
    let special_tokens = vec![
        "<|endoftext|>".to_string(),
        "<|pad|>".to_string(),
        "<EMAIL>".to_string(),
        "<PHONE>".to_string(),
        "<IBAN>".to_string(),
        "<|system|>".to_string(),
        "<|user|>".to_string(),
        "<|assistant|>".to_string(),
        "<|leichte_sprache|>".to_string()
    ];

    println!("=== Eisen Tokenizer Trainer ===");
    
    let combined_text = if Path::new(cache_path).exists() {
        println!("Recovering sampled state from {}...", cache_path);
        std::fs::read_to_string(cache_path).expect("Failed to read cache")
    } else {
        let mut file = File::open(input_path).expect("Could not open data file.");
        let file_size = file.metadata().expect("Could not get file metadata").len();
        
        let sample_bytes = sample_size_mb;
        let mut text = String::with_capacity(sample_bytes);
        
        println!("Dataset: {} ({:.2} GB)", input_path, file_size as f64 / 1e9);
        println!("Sampling strategy: 100 random chunks totalling {} MB", sample_size_mb);

        let mut rng = rand::thread_rng();
        let num_chunks = 100; 
        let chunk_size = sample_bytes / num_chunks;

        for i in 0..num_chunks {
            let offset = rng.gen_range(0..file_size.saturating_sub(chunk_size as u64));
            file.seek(SeekFrom::Start(offset)).unwrap();
            
            let mut buf = vec![0u8; chunk_size];
            let _ = file.read_exact(&mut buf);
            
            // Lossy conversion handles chunks cutting off mid-UTF8-char
            text.push_str(&String::from_utf8_lossy(&buf));
            
            if i % 20 == 0 { println!("Sampling progress: {}%", i); }
        }
        
        // CLEANUP: Filter out the DOC_ID boundaries so they don't pollute the BPE vocabulary
        println!("Stripping compliance metadata boundaries from sample...");
        let clean_text = text.lines()
            .filter(|line| !line.starts_with("### DOC_ID:"))
            .collect::<Vec<_>>()
            .join("\n");

        println!("Caching sample to {} for easy recovery...", cache_path);
        std::fs::write(cache_path, &clean_text).expect("Failed to cache sampled text");
        clean_text
    };

    // 3. Run BPE Training
    println!("\nStarting BPE merge logic (this is CPU intensive)...");
    let tokenizer = BPETokenizer::train(&combined_text, target_vocab_size, special_tokens);
    
    // 4. Serialize to disk
    println!("\nSaving tokenizer to {}...", output_tokenizer_path);
    tokenizer.save(output_tokenizer_path).expect("Failed to save tokenizer model");
    
    println!("=== Success! Tokenizer is ready for pre-tokenization. ===");
}
