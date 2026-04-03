use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write, stdout};
use std::time::Instant;

#[derive(Clone)]
pub struct BPETokenizer {
    pub vocab: Vec<String>,
    pub merges: HashMap<(usize, usize), usize>,
}

impl BPETokenizer {
    pub fn train(text: &str, target_vocab_size: usize) -> Self {
        let mut vocab = Vec::new();
        let mut char_to_id = HashMap::new();
        let mut sequence = Vec::new();

        for ch in text.chars() {
            let ch_str = ch.to_string();
            let id = *char_to_id.entry(ch_str.clone()).or_insert_with(|| {
                let new_id = vocab.len();
                vocab.push(ch_str);
                new_id
            });
            sequence.push(id);
        }

        let mut merges = HashMap::new();
        let num_merges = target_vocab_size.saturating_sub(vocab.len());

        println!("BPE Training: Initial character vocab size: {}", vocab.len());
        println!("Starting {} merges...", num_merges);

        let start_time = Instant::now();
        let mut last_update = Instant::now();

        for step in 0..num_merges {
            let mut pair_counts = HashMap::new();
            for window in sequence.windows(2) {
                *pair_counts.entry((window[0], window[1])).or_insert(0) += 1;
            }

            if let Some((&best_pair, &count)) = pair_counts.iter().max_by_key(|&(_, count)| count) {
                if count < 2 { break; } 

                let new_id = vocab.len();
                let new_token = format!("{}{}", vocab[best_pair.0], vocab[best_pair.1]);
                vocab.push(new_token);
                merges.insert(best_pair, new_id);

                let mut new_sequence = Vec::with_capacity(sequence.len());
                let mut i = 0;
                while i < sequence.len() {
                    if i < sequence.len() - 1 && (sequence[i], sequence[i+1]) == best_pair {
                        new_sequence.push(new_id);
                        i += 2;
                    } else {
                        new_sequence.push(sequence[i]);
                        i += 1;
                    }
                }
                sequence = new_sequence;
                
                // --- CUSTOM PROGRESS BAR ---
                let now = Instant::now();
                if now.duration_since(last_update).as_millis() > 100 || step == num_merges - 1 {
                    let percent = (step as f64 / num_merges as f64).clamp(0.0, 1.0);
                    let width: usize = 40;
                    let filled = (percent * width as f64) as usize;
                    let bar: String = std::iter::repeat('=').take(filled)
                        .chain(std::iter::once('>'))
                        .chain(std::iter::repeat(' ').take(width.saturating_sub(filled)))
                        .collect();

                    let elapsed = start_time.elapsed().as_secs();
                    let rate = step as f64 / elapsed.max(1) as f64;
                    let eta = if rate > 0.0 { ((num_merges - step) as f64 / rate) as u64 } else { 0 };

                    print!("\r[{}] {:.1}% | Merge {}/{} | ETA: {}s ", 
                        &bar[..(width as usize)], percent * 100.0, step, num_merges, eta);
                    stdout().flush().unwrap();
                    last_update = now;
                }
            } else {
                break;
            }
        }
        println!(); // Clear line after progress bar

        Self { vocab, merges }
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut file = File::create(path)?;
        writeln!(file, "{}", self.vocab.len())?;
        for v in &self.vocab {
            let escaped = v.replace("\n", "\\n");
            writeln!(file, "{}", escaped)?;
        }
        for ((p1, p2), res) in &self.merges {
            writeln!(file, "{} {} {}", p1, p2, res)?;
        }
        Ok(())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let mut content = String::new();
        File::open(path)?.read_to_string(&mut content)?;
        let mut lines = content.lines();

        let vocab_size: usize = lines.next().unwrap().parse().unwrap();
        let mut vocab = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            let v = lines.next().unwrap().replace("\\n", "\n");
            vocab.push(v);
        }

        let mut merges = HashMap::new();
        for line in lines {
            let parts: Vec<usize> = line.split_whitespace().map(|s| s.parse().unwrap()).collect();
            if parts.len() == 3 {
                merges.insert((parts[0], parts[1]), parts[2]);
            }
        }

        Ok(Self { vocab, merges })
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut sequence = Vec::new();
        for ch in text.chars() {
            let ch_str = ch.to_string();
            if let Some(pos) = self.vocab.iter().position(|v| v == &ch_str) {
                sequence.push(pos);
            }
        }

        if sequence.is_empty() { return sequence; }

        loop {
            let mut best_pair = None;
            let mut min_rank = usize::MAX;

            for i in 0..sequence.len() - 1 {
                let pair = (sequence[i], sequence[i+1]);
                if let Some(&new_id) = self.merges.get(&pair) {
                    if new_id < min_rank {
                        min_rank = new_id;
                        best_pair = Some(pair);
                    }
                }
            }

            if let Some(pair) = best_pair {
                let new_id = self.merges[&pair];
                let mut new_sequence = Vec::with_capacity(sequence.len());
                let mut i = 0;
                while i < sequence.len() {
                    if i < sequence.len() - 1 && (sequence[i], sequence[i+1]) == pair {
                        new_sequence.push(new_id);
                        i += 2;
                    } else {
                        new_sequence.push(sequence[i]);
                        i += 1;
                    }
                }
                sequence = new_sequence;
            } else {
                break;
            }
        }
        sequence
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter().map(|&id| self.vocab.get(id).cloned().unwrap_or_default()).collect()
    }
}
