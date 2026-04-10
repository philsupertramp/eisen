use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::sync::mpsc::{sync_channel, Receiver};
use std::thread;
use std::sync::Arc;


// Note: We will need to move BPETokenizer out of the example and into src/data/tokenizer.rs 
// to use it here, but we will mock the interface for the dataloader structure.
use crate::data::tokenizer::BPETokenizer;

pub struct StreamingDataLoader {
    receiver: Receiver<(Vec<f32>, Vec<usize>)>, // Yields (x_batch_flat, y_batch_flat)
}

impl StreamingDataLoader {
    /// Spawns a background thread to read from a file (like an hf-mount virtual file),
    /// tokenize the text, and assemble batches asynchronously.
    pub fn new(
        file_path: String,
        tokenizer: Arc<BPETokenizer>,
        seq_len: usize,
        batch_size: usize,
        prefetch_batches: usize, // How many batches to buffer in RAM (e.g., 5)
    ) -> Self {
        // Create a bounded channel. If the buffer is full, the background thread sleeps, 
        // preventing CPU RAM exhaustion!
        let (tx, rx) = sync_channel(prefetch_batches);

        thread::spawn(move || {
            let file = match File::open(&file_path) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("Dataloader failed to open {}: {}", file_path, e);
                    return;
                }
            };
            
            // BufReader is crucial for streaming large hf-mount files without loading to RAM
            let reader = BufReader::new(file);
            
            let mut current_batch_x = Vec::with_capacity(batch_size * seq_len);
            let mut current_batch_y = Vec::with_capacity(batch_size * seq_len);
            let mut token_buffer = Vec::new();

            for line in reader.lines() {
                if let Ok(text) = line {
                    if text.trim().is_empty() { continue; }
                    
                    // Tokenize on the fly
                    let tokens = tokenizer.encode(&text);
                    token_buffer.extend(tokens);

                    // Process tokens into sliding windows
                    while token_buffer.len() > seq_len {
                        for s in 0..seq_len {
                            current_batch_x.push(token_buffer[s] as f32);
                            current_batch_y.push(token_buffer[s + 1]);
                        }
                        
                        // Advance the sliding window by 1 token
                        // (For less overlap, you could drain by `seq_len`)
                        token_buffer.drain(0..1); 

                        // If batch is full, send it to the main GPU thread
                        if current_batch_x.len() == batch_size * seq_len {
                            if tx.send((current_batch_x.clone(), current_batch_y.clone())).is_err() {
                                // The receiver was dropped (training stopped), exit thread cleanly
                                return; 
                            }
                            current_batch_x.clear();
                            current_batch_y.clear();
                        }
                    }
                }
            }
        });

        Self { receiver: rx }
    }

    /// Non-blocking pop from the prefetch queue.
    /// If the background thread is still reading/tokenizing, this will block until ready.
    pub fn next_batch(&self) -> Option<(Vec<f32>, Vec<usize>)> {
        self.receiver.recv().ok()
    }
}


/// A lightning-fast dataloader that buffers pre-tokenized u16 IDs.
/// Uses sequential non-overlapping chunking to max out disk throughput 
/// and improve gradient variance. Zero backward seeks!
pub struct BinaryDataLoader {
    reader: BufReader<File>,
    seq_len: usize,
    batch_size: usize,
    buffer: Vec<usize>,
    total_batches: usize,
}

impl BinaryDataLoader {
    pub fn new(file_path: &str, seq_len: usize, batch_size: usize) -> Self {
        let file = File::open(file_path).expect("Binary file not found");
        
        // Calculate the total number of batches we can yield in one pass
        let file_size = file.metadata().expect("Failed to read file metadata").len() as usize;
        let total_tokens = file_size / 2; // 2 bytes per u16 token
        let tokens_per_batch = batch_size * (seq_len + 1);
        let total_batches = total_tokens / tokens_per_batch;

        // Wrap the file in a massive 4MB BufReader for maximum sequential I/O
        let reader = BufReader::with_capacity(4 * 1024 * 1024, file);

        Self {
            reader,
            seq_len,
            batch_size,
            buffer: Vec::new(),
            total_batches,
        }
    }

    /// Returns the exact number of full batches available in a single pass
    pub fn total_batches(&self) -> usize {
        self.total_batches
    }

    pub fn next_batch(&mut self) -> Option<(Vec<f32>, Vec<usize>)> {
        let tokens_needed = self.batch_size * (self.seq_len + 1);

        // Refill the RAM buffer if we don't have enough tokens for a full batch
        if self.buffer.len() < tokens_needed {
            // Let's pull a massive chunk at once (e.g., 100 batches worth)
            let fetch_size = tokens_needed * 100;
            let mut byte_buf = vec![0u8; fetch_size * 2]; // 2 bytes per u16
            
            let bytes_read = self.reader.read(&mut byte_buf).unwrap_or(0);
            if bytes_read == 0 && self.buffer.len() < tokens_needed {
                return None; // EOF and not enough tokens left for a full batch
            }

            let new_tokens: Vec<usize> = byte_buf[..bytes_read]
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]) as usize)
                .collect();
            
            self.buffer.extend(new_tokens);
        }

        // If we still don't have enough after reading (end of file), bail out cleanly
        if self.buffer.len() < tokens_needed {
            return None; 
        }

        let mut x_batch = Vec::with_capacity(self.batch_size * self.seq_len);
        let mut y_batch = Vec::with_capacity(self.batch_size * self.seq_len);

        // Construct the batch using non-overlapping sequential blocks
        for b in 0..self.batch_size {
            let start_idx = b * (self.seq_len + 1);
            for s in 0..self.seq_len {
                x_batch.push(self.buffer[start_idx + s] as f32);
                y_batch.push(self.buffer[start_idx + s + 1]);
            }
        }

        // Drain the used tokens from the front of the buffer
        self.buffer.drain(0..tokens_needed);

        Some((x_batch, y_batch))
    }

    pub fn reset(&mut self) {
        // Seek to start and clear the internal buffers
        self.reader.seek(SeekFrom::Start(0)).expect("Failed to seek to start");
        self.buffer.clear();
    }
}
