// src/data/dataloader.rs

use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::sync::mpsc::{sync_channel, Receiver};
use std::thread;
use std::sync::Arc;

use crate::data::tokenizer::BPETokenizer;
use crate::data::fim::{FimConfig, apply_fim, mask_targets, IGNORE_INDEX};

// ─── StreamingDataLoader (unchanged) ─────────────────────────────────────────

pub struct StreamingDataLoader {
    receiver: Receiver<(Vec<f32>, Vec<usize>)>,
}

impl StreamingDataLoader {
    pub fn new(
        file_path: String,
        tokenizer: Arc<BPETokenizer>,
        seq_len: usize,
        batch_size: usize,
        prefetch_batches: usize,
    ) -> Self {
        let (tx, rx) = sync_channel(prefetch_batches);
        thread::spawn(move || {
            let file = match File::open(&file_path) {
                Ok(f) => f,
                Err(e) => { eprintln!("Dataloader failed to open {}: {}", file_path, e); return; }
            };
            let reader = BufReader::new(file);
            let mut current_batch_x = Vec::with_capacity(batch_size * seq_len);
            let mut current_batch_y = Vec::with_capacity(batch_size * seq_len);
            let mut token_buffer = Vec::new();

            for line in reader.lines() {
                if let Ok(text) = line {
                    if text.trim().is_empty() { continue; }
                    let tokens = tokenizer.encode(&text);
                    token_buffer.extend(tokens);

                    while token_buffer.len() > seq_len {
                        for s in 0..seq_len {
                            current_batch_x.push(token_buffer[s] as f32);
                            current_batch_y.push(token_buffer[s + 1]);
                        }
                        token_buffer.drain(0..1);

                        if current_batch_x.len() == batch_size * seq_len {
                            if tx.send((current_batch_x.clone(), current_batch_y.clone())).is_err() {
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

    pub fn next_batch(&self) -> Option<(Vec<f32>, Vec<usize>)> {
        self.receiver.recv().ok()
    }
}

// ─── BatchResult ─────────────────────────────────────────────────────────────

/// Output of `BinaryDataLoader::next_batch`.
///
/// `targets` already has `IGNORE_INDEX` (= `usize::MAX`) at positions that
/// should not contribute to the loss (FIM-masked positions).  When FIM is
/// disabled every entry is a real token ID.
pub struct BatchResult {
    pub x:       Vec<f32>,     // [batch * seq_len]  — token IDs as f32
    pub targets: Vec<usize>,   // [batch * seq_len]  — next-token IDs (or IGNORE_INDEX)
    pub has_masked: bool,      // true when at least one IGNORE_INDEX is present
}

// ─── BinaryDataLoader ─────────────────────────────────────────────────────────

/// Reads pre-tokenised u16 binary files and optionally applies FIM on-the-fly.
///
/// FIM is applied per-sequence (not per-batch) with the probability given in
/// `FimConfig::rate`.  The returned `targets` vector uses `IGNORE_INDEX` for
/// positions that should be excluded from the loss.
pub struct BinaryDataLoader {
    reader:      BufReader<File>,
    seq_len:     usize,
    batch_size:  usize,
    buffer:      Vec<usize>,
    total_batches: usize,
    fim:         Option<FimConfig>,
    rng_state:   u64,
}

impl BinaryDataLoader {
    pub fn new(file_path: &str, seq_len: usize, batch_size: usize) -> Self {
        let file = File::open(file_path).expect("Binary file not found");
        let file_size = file.metadata().expect("Failed to read file metadata").len() as usize;
        let total_tokens = file_size / 2;
        let tokens_per_batch = batch_size * (seq_len + 1);
        let total_batches = total_tokens / tokens_per_batch;
        let reader = BufReader::with_capacity(4 * 1024 * 1024, file);
        Self {
            reader,
            seq_len,
            batch_size,
            buffer: Vec::new(),
            total_batches,
            fim: None,
            rng_state: 1337,
        }
    }

    /// Enable FIM with the given config.
    pub fn with_fim(mut self, config: FimConfig) -> Self {
        self.fim = Some(config);
        self
    }

    /// Seed the internal xorshift64 RNG (used for FIM decisions).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng_state = seed;
        self
    }

    pub fn total_batches(&self) -> usize { self.total_batches }

    pub fn next_batch(&mut self) -> Option<BatchResult> {
        // We read seq_len+1 tokens per sequence so we can build (x, y) pairs.
        let tokens_needed = self.batch_size * (self.seq_len + 1);

        // Refill buffer from disk if needed.
        if self.buffer.len() < tokens_needed {
            let fetch_size = tokens_needed * 100;
            let mut byte_buf = vec![0u8; fetch_size * 2];
            let bytes_read = self.reader.read(&mut byte_buf).unwrap_or(0);
            if bytes_read == 0 && self.buffer.len() < tokens_needed {
                return None;
            }
            let new_tokens: Vec<usize> = byte_buf[..bytes_read]
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]) as usize)
                .collect();
            self.buffer.extend(new_tokens);
        }

        if self.buffer.len() < tokens_needed {
            return None;
        }

        let mut x_batch: Vec<f32>  = Vec::with_capacity(self.batch_size * self.seq_len);
        let mut y_batch: Vec<usize> = Vec::with_capacity(self.batch_size * self.seq_len);
        let mut has_masked = false;

        for b in 0..self.batch_size {
            let start = b * (self.seq_len + 1);
            // Raw sequence of seq_len+1 tokens; x = [0..seq_len], y = [1..seq_len+1]
            let raw_x: Vec<usize> = self.buffer[start..start + self.seq_len].to_vec();
            let raw_y: Vec<usize> = self.buffer[start + 1..start + self.seq_len + 1].to_vec();

            // Apply FIM with configured probability.
            let shift_val = self.xorshift_f32();
            let (final_x, final_y) = if let Some(fim) = &self.fim {
                if shift_val < fim.rate {
                    // Transform the x sequence and derive y from it.
                    // y[i] = x[i+1] after FIM reordering; the final position
                    // becomes the pad token's successor (ignored).
                    let (fx, mask) = apply_fim(&raw_x, fim, &mut XorShift64(&mut self.rng_state), self.seq_len);
                    // Build y from the FIM-transformed x
                    let mut fy: Vec<usize> = fx[1..].to_vec();
                    fy.push(IGNORE_INDEX); // last position has no successor

                    // Apply mask: positions where mask[i]=false → IGNORE_INDEX in y
                    // (shift by 1 because y[i] corresponds to predicting fx[i+1],
                    // so we mask y[i] when mask[i+1] is false or it's the fim_middle sentinel)
                    let masked_y = mask_targets(&fy, &mask);

                    has_masked = true;
                    (fx, masked_y)
                } else {
                    (raw_x, raw_y)
                }
            } else {
                (raw_x, raw_y)
            };

            for t in &final_x { x_batch.push(*t as f32); }
            y_batch.extend_from_slice(&final_y);
        }

        self.buffer.drain(0..tokens_needed);

        Some(BatchResult { x: x_batch, targets: y_batch, has_masked })
    }

    pub fn reset(&mut self) {
        self.reader.seek(SeekFrom::Start(0)).expect("Failed to seek to start");
        self.buffer.clear();
    }

    // ── internal tiny RNG ────────────────────────────────────────────────────

    fn xorshift_f32(&mut self) -> f32 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        // map to [0, 1)
        (self.rng_state >> 11) as f32 / (1u64 << 53) as f32
    }
}

// ─── Tiny RNG adapter for apply_fim ──────────────────────────────────────────

struct XorShift64<'a>(&'a mut u64);

impl<'a> rand::RngCore for XorShift64<'a> {
    fn next_u32(&mut self) -> u32 { self.next_u64() as u32 }
    fn next_u64(&mut self) -> u64 {
        *self.0 ^= *self.0 << 13;
        *self.0 ^= *self.0 >> 7;
        *self.0 ^= *self.0 << 17;
        *self.0
    }
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut v = self.next_u64();
        for (i, b) in dest.iter_mut().enumerate() {
            if i % 8 == 0 { v = self.next_u64(); }
            *b = (v >> (8 * (i % 8))) as u8;
        }
    }
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl rand::CryptoRng for XorShift64<'_> {}
