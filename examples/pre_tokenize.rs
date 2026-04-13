use eisen::data::tokenizer::BPETokenizer;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write, Seek, SeekFrom, stdout};
use std::path::Path;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Instant;

// A Job contains: (chunk_id, text_to_tokenize, byte_offset_at_end_of_chunk)
type Job = Option<(usize, String, u64)>;
// A Result contains: (chunk_id, token_bytes, byte_offset_at_end_of_chunk)
type TokenResult = (usize, Vec<u8>, u64);

pub fn pre_tokenize_multithreaded(txt_input_path: &str, bin_output_path: &str, state_path: &str, tokenizer: Arc<BPETokenizer>) {
    let mut start_byte: u64 = 0;
    
    if Path::new(state_path).exists() {
        let state_str = std::fs::read_to_string(state_path).unwrap_or_default();
        if let Ok(byte_offset) = state_str.trim().parse::<u64>() {
            start_byte = byte_offset;
            println!("Recovered state! Instant seek to byte offset {}...", start_byte);
        }
    }

    let mut in_file = File::open(txt_input_path).expect("Failed to open text file");
    let total_bytes = in_file.metadata().expect("Failed to get metadata").len();
    in_file.seek(SeekFrom::Start(start_byte)).expect("Failed to seek in text file");
    
    // Determine thread count (fallback to 8 if detection isn't used)
    let num_workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    println!("Spawning {} BPE worker threads...", num_workers);

    // Channels for our Thread Pool
    let (job_tx, job_rx) = mpsc::sync_channel::<Job>(num_workers * 4);
    let job_rx = Arc::new(Mutex::new(job_rx));
    let (res_tx, res_rx) = mpsc::sync_channel::<TokenResult>(num_workers * 4);

    // 1. Spawn Worker Threads
    for _ in 0..num_workers {
        let rx_clone = Arc::clone(&job_rx);
        let tx_clone = res_tx.clone();
        let tok_clone = Arc::clone(&tokenizer);
        
        thread::spawn(move || loop {
            let job = {
                let lock = rx_clone.lock().unwrap();
                lock.recv().unwrap()
            };
            
            match job {
                Some((chunk_id, text, end_byte)) => {
                    let tokens = tok_clone.encode(&text);
                    let mut bytes = Vec::with_capacity(tokens.len() * 2);
                    for t in tokens {
                        bytes.extend_from_slice(&(t as u16).to_le_bytes());
                    }
                    tx_clone.send((chunk_id, bytes, end_byte)).unwrap();
                }
                None => break, // Shutdown signal received
            }
        });
    }

    // 2. Spawn Producer Thread (Reads disk and feeds workers)
    thread::spawn(move || {
        let mut reader = BufReader::new(in_file);
        let mut chunk_id = 0;
        let mut current_byte = start_byte;
        let mut buffer = String::with_capacity(2 * 1024 * 1024); // 2MB chunks

        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => {
                    // EOF: Send remaining buffer
                    if !buffer.is_empty() {
                        job_tx.send(Some((chunk_id, buffer, current_byte))).unwrap();
                    }
                    // Send shutdown signals to all workers
                    for _ in 0..num_workers { job_tx.send(None).unwrap(); }
                    break;
                }
                Ok(bytes_read) => {
                    buffer.push_str(&line);
                    current_byte += bytes_read as u64;

                    // When chunk reaches 2MB, dispatch it to the thread pool
                    if buffer.len() >= 2 * 1024 * 1024 {
                        job_tx.send(Some((chunk_id, buffer.clone(), current_byte))).unwrap();
                        buffer.clear();
                        chunk_id += 1;
                    }
                }
                Err(e) => panic!("Error reading file: {}", e),
            }
        }
    });
    drop(res_tx);
    // 3. Main Thread acts as the Writer (Ensures ordered output)
    let out_file = OpenOptions::new()
        .create(true).append(true).open(bin_output_path).unwrap();
    let mut writer = BufWriter::new(out_file);

    let mut next_expected_chunk = 0;
    let mut out_of_order_buffer: HashMap<usize, (Vec<u8>, u64)> = HashMap::new();
    
    let start_time = Instant::now();
    let mut last_update = Instant::now();
    let mut processed_bytes = start_byte;

    // Listen for results until the channel closes
    while let Ok((chunk_id, token_bytes, end_byte)) = res_rx.recv() {
        if chunk_id == next_expected_chunk {
            // Write the expected chunk
            writer.write_all(&token_bytes).unwrap();
            processed_bytes = end_byte;
            next_expected_chunk += 1;

            // Check if we can write any buffered chunks that arrived early
            while let Some((buffered_bytes, buf_end_byte)) = out_of_order_buffer.remove(&next_expected_chunk) {
                writer.write_all(&buffered_bytes).unwrap();
                processed_bytes = buf_end_byte;
                next_expected_chunk += 1;
            }
        } else {
            // Chunk arrived early, buffer it until its turn
            out_of_order_buffer.insert(chunk_id, (token_bytes, end_byte));
        }

        // --- CUSTOM PROGRESS BAR ---
        let now = Instant::now();
        if now.duration_since(last_update).as_millis() > 150 {
            let percent = (processed_bytes as f64 / total_bytes as f64).clamp(0.0, 1.0);
            let width: usize = 50;
            let filled = (percent * width as f64) as usize;
            let bar: String = std::iter::repeat('=').take(filled)
                .chain(std::iter::once('>'))
                .chain(std::iter::repeat(' ').take(width.saturating_sub(filled)))
                .collect();

            let elapsed = start_time.elapsed().as_secs();
            let rate = (processed_bytes - start_byte) as f64 / elapsed.max(1) as f64; 
            let eta = if rate > 0.0 { ((total_bytes - processed_bytes) as f64 / rate) as u64 } else { 0 };

            print!("\r[{}] {:.2}% | {:.2} / {:.2} GB | ETA: {}s ", 
                &bar[..width], percent * 100.0, processed_bytes as f64 / 1e9, total_bytes as f64 / 1e9, eta);
            stdout().flush().unwrap();
            last_update = now;

            // Periodically save state and flush to disk
            writer.flush().unwrap();
            std::fs::write(state_path, processed_bytes.to_string()).unwrap();
        }
    }

    writer.flush().unwrap();
    std::fs::write(state_path, "DONE").unwrap();
    println!("\nMulti-threaded Pre-tokenization completely finished! Wrote {} chunks.", next_expected_chunk);
}

fn main() {
    let tokenizer = Arc::new(
        BPETokenizer::load("data/tinystory_tokenizer.model")
            .expect("Train the tokenizer first with examples/train_tokenizer.rs!")
    );

    pre_tokenize_multithreaded(
        "data/german_tiny_story_corpus.txt", 
        "data/german_tinystory_corpus.bin", 
        "data/preprocess_state.txt",
        tokenizer
    );
}
