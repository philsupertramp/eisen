use crate::data::tokenizer::BPETokenizer;
use rand::Rng;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;
use std::thread;

pub mod huggingface;

/// Converts a massive .txt file into a compact .bin file of u16 token IDs.
pub fn pre_tokenize(input_path: &str, output_path: &str, tokenizer: Arc<BPETokenizer>) {
    let text = std::fs::read_to_string(input_path).expect("Failed to read input");
    println!("Tokenizing {}...", input_path);

    let tokens = tokenizer.encode(&text);

    let file = File::create(output_path).expect("Failed to create output bin");
    let mut writer = BufWriter::new(file);

    println!("Writing {} tokens to binary...", tokens.len());
    for token in tokens {
        // Store as u16 (supports vocab up to 65,535)
        let bytes = (token as u16).to_le_bytes();
        writer.write_all(&bytes).expect("Write failed");
    }
    writer.flush().expect("Flush failed");
    println!("Pre-tokenization complete: {}", output_path);
}

/// Trains a tokenizer by sampling random chunks from a massive file
/// to ensure we get representative statistics without loading the whole file.
pub fn train_on_sampled_data(
    input_path: &str,
    output_tokenizer_path: &str,
    target_vocab_size: usize,
    sample_size_mb: usize,
) {
    let mut file = File::open(input_path).expect("Could not open data file");
    let file_size = file.metadata().expect("Could not get file metadata").len();

    let sample_bytes = sample_size_mb * 1024 * 1024;
    let mut combined_text = String::with_capacity(sample_bytes);

    println!(
        "Sampling {}MB from {}GB file...",
        sample_size_mb,
        file_size / (1024 * 1024 * 1024)
    );

    let mut rng = rand::thread_rng();
    let num_chunks = 100; // Sample 100 random locations
    let chunk_size = sample_bytes / num_chunks;

    for _ in 0..num_chunks {
        let offset = rng.gen_range(0..file_size.saturating_sub(chunk_size as u64));
        file.seek(SeekFrom::Start(offset)).unwrap();

        let mut buf = vec![0u8; chunk_size];
        let _ = file.read_exact(&mut buf);

        // Use lossy conversion to handle cases where we land in the middle of a UTF-8 char
        combined_text.push_str(&String::from_utf8_lossy(&buf));
    }

    println!("Training BPE on sampled text...");
    let tokenizer = BPETokenizer::train(&combined_text, target_vocab_size);

    println!("Saving tokenizer to {}...", output_tokenizer_path);
    tokenizer
        .save(output_tokenizer_path)
        .expect("Failed to save tokenizer");
    println!("Success!");
}

/// Converts a massive text file into a flat binary file of u16 token IDs.
/// Streams line-by-line and saves state to allow resuming after interruptions.
pub fn pre_tokenize_resumable(
    txt_input_path: &str,
    bin_output_path: &str,
    state_path: &str,
    tokenizer: Arc<BPETokenizer>,
) {
    let mut start_line = 0;

    // Recover state if execution was interrupted
    if Path::new(state_path).exists() {
        let state_str = std::fs::read_to_string(state_path).unwrap_or_default();
        if let Ok(line) = state_str.trim().parse::<usize>() {
            start_line = line;
            println!(
                "Recovered state! Resuming pre-tokenization from line {}...",
                start_line
            );
        }
    }

    let in_file = File::open(txt_input_path).expect("Failed to open text file");
    let reader = BufReader::new(in_file);

    // Open in append mode so we don't overwrite previous work when resuming
    let out_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(bin_output_path)
        .expect("Failed to open bin file");
    let mut writer = BufWriter::new(out_file);

    println!("Streaming and encoding text with BPE...");
    let mut tokens_written = 0;

    for (i, line) in reader.lines().enumerate() {
        if i < start_line {
            continue;
        } // Skip lines we already processed

        if let Ok(text) = line {
            if text.trim().is_empty() {
                continue;
            }

            let tokens = tokenizer.encode(&text);
            for token in tokens {
                let bytes = (token as u16).to_le_bytes();
                writer.write_all(&bytes).unwrap();
                tokens_written += 1;
            }
        }

        // Periodically save state and flush to disk
        if i % 10000 == 0 && i > start_line {
            writer.flush().unwrap();
            std::fs::write(state_path, i.to_string()).unwrap();
            println!(
                "Processed {} lines... (Appended {} tokens this session)",
                i, tokens_written
            );
        }
    }

    writer.flush().unwrap();
    std::fs::write(state_path, "DONE").unwrap();
    println!("Pre-tokenization completely finished!");
}

fn main() {
    // 1. Load the tokenizer you trained earlier
    let tokenizer = Arc::new(
        BPETokenizer::load("data/tokenizer.model")
            .expect("Train the tokenizer first with tools/train_tokenizer.rs!"),
    );

    // 2. Transform the Wikipedia/OSCAR dump securely
    pre_tokenize_resumable(
        "data/german_large_corpus.txt",
        "data/german_large_corpus.bin",
        "data/preprocess_state.txt",
        tokenizer,
    );
}
