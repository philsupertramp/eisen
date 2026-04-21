use std::collections::{HashMap, BinaryHeap};
use std::fs::File;
use std::hash::{BuildHasherDefault, Hasher};
use std::io::{Read, Write, stdout};
use std::time::Instant;

// 13:19
// --- ZERO-DEPENDENCY FAST HASHER ---
// Rust's default SipHash is cryptographically secure but too slow for BPE.
// This is an FNV-1a hasher that speeds up pair lookups by ~40%.
pub struct FastHasher { hash: u64 }
impl Default for FastHasher {
    fn default() -> Self { Self { hash: 0xcbf29ce484222325 } }
}
impl Hasher for FastHasher {
    #[inline]
    fn finish(&self) -> u64 { self.hash }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.hash = (self.hash ^ (b as u64)).wrapping_mul(0x100000001b3);
        }
    }
}
pub type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FastHasher>>;

#[derive(Clone)]
pub struct BPETokenizer {
    pub vocab: Vec<String>,
    pub merges: FastMap<(usize, usize), usize>,
    pub special_tokens: Vec<String>, 
}

// A node for our heavily optimized O(1) doubly-linked list merge strategy
#[derive(Clone, Copy)]
struct ListNode {
    id: usize,
    prev: Option<usize>,
    next: Option<usize>,
    pair_prev: Option<usize>, // Intrusive list to completely eliminate Vec allocations
    pair_next: Option<usize>,
    valid: bool,
}

impl BPETokenizer {
    #[inline]
    fn remove_from_pair_list(
        idx: usize,
        nodes: &mut [ListNode],
        pair_heads: &mut FastMap<(usize, usize), usize>
    ) {
        let node = nodes[idx];
        if let Some(next_idx) = node.next {
            let pair = (node.id, nodes[next_idx].id);
            
            let is_head = pair_heads.get(&pair) == Some(&idx);
            if !is_head && node.pair_prev.is_none() { return; } // Not in the list
            
            if let Some(p) = node.pair_prev {
                nodes[p].pair_next = node.pair_next;
            } else if is_head {
                if let Some(n) = node.pair_next {
                    pair_heads.insert(pair, n);
                } else {
                    pair_heads.remove(&pair);
                }
            }
            if let Some(n) = node.pair_next {
                nodes[n].pair_prev = node.pair_prev;
            }
            nodes[idx].pair_prev = None;
            nodes[idx].pair_next = None;
        }
    }

    #[inline]
    fn add_to_pair_list(
        idx: usize,
        nodes: &mut [ListNode],
        pair_heads: &mut FastMap<(usize, usize), usize>
    ) {
        let next_idx = nodes[idx].next.unwrap();
        let pair = (nodes[idx].id, nodes[next_idx].id);

        nodes[idx].pair_prev = None;
        if let Some(&head) = pair_heads.get(&pair) {
            nodes[idx].pair_next = Some(head);
            nodes[head].pair_prev = Some(idx);
        } else {
            nodes[idx].pair_next = None;
        }
        pair_heads.insert(pair, idx);
    }

    pub fn train(text: &str, target_vocab_size: usize, special_tokens: Vec<String>) -> Self {
        let mut vocab = Vec::new();
        let mut char_to_id = FastMap::default();
        let mut sequence = Vec::new();

        // 1. Pre-register special tokens
        for st in &special_tokens {
            let id = vocab.len();
            vocab.push(st.clone());
            char_to_id.insert(st.clone(), id);
        }

        // 2. ATOMIC SCANNER with Metadata Skipping
        let mut head = 0;
        let text_len = text.len();

        while head < text_len {
            let tail = &text[head..];

            // Ignore our GDPR/Provenance boundaries
            if tail.starts_with("### DOC_ID:") {
                if let Some(newline_pos) = tail.find('\n') {
                    head += newline_pos + 1;
                } else {
                    head = text_len;
                }
                continue;
            }

            let mut matched_special = None;
            for st in &special_tokens {
                if tail.starts_with(st) {
                    matched_special = Some(st);
                    break;
                }
            }

            if let Some(st) = matched_special {
                let id = *char_to_id.get(st).unwrap();
                sequence.push(id);
                head += st.len();
            } else {
                let ch = tail.chars().next().unwrap();
                let ch_str = ch.to_string();
                let id = *char_to_id.entry(ch_str.clone()).or_insert_with(|| {
                    let new_id = vocab.len();
                    vocab.push(ch_str);
                    new_id
                });
                sequence.push(id);
                head += ch.len_utf8();
            }
        }

        // 3. FAST O(N log N) BPE ALGORITHM
        let mut nodes = Vec::with_capacity(sequence.len());
        for (i, &id) in sequence.iter().enumerate() {
            nodes.push(ListNode {
                id,
                prev: if i > 0 { Some(i - 1) } else { None },
                next: if i + 1 < sequence.len() { Some(i + 1) } else { None },
                pair_prev: None,
                pair_next: None,
                valid: true,
            });
        }

        let mut pair_counts: FastMap<(usize, usize), usize> = FastMap::default();
        let mut pair_heads: FastMap<(usize, usize), usize> = FastMap::default();

        // Initial population
        for i in 0..nodes.len() {
            if let Some(next_idx) = nodes[i].next {
                let pair = (nodes[i].id, nodes[next_idx].id);
                *pair_counts.entry(pair).or_insert(0) += 1;
                Self::add_to_pair_list(i, &mut nodes, &mut pair_heads);
            }
        }

        // Push to max-heap for O(1) best-pair extraction
        let mut heap: BinaryHeap<(usize, (usize, usize))> = BinaryHeap::new();
        for (&pair, &count) in &pair_counts {
            heap.push((count, pair));
        }

        let mut merges = FastMap::default();
        let num_merges = target_vocab_size.saturating_sub(vocab.len());

        println!("BPE Training: Initial vocab size (chars + special): {}", vocab.len());
        println!("Starting {} merges using hyper-fast intrusive list engine...", num_merges);

        let start_time = Instant::now();
        let mut last_update = Instant::now();

        for step in 0..num_merges {
            // Lazy Deletion: Find the real maximum pair
            let mut best_pair = None;
            while let Some((count, pair)) = heap.pop() {
                if let Some(&actual_count) = pair_counts.get(&pair) {
                    if count == actual_count {
                        if count >= 2 {
                            best_pair = Some(pair);
                        }
                        break;
                    }
                }
            }

            let pair = match best_pair {
                Some(p) => p,
                None => break, // No more pairs occurring >= 2 times
            };

            // Register the new token
            let new_id = vocab.len();
            let new_token = format!("{}{}", vocab[pair.0], vocab[pair.1]);
            vocab.push(new_token);
            merges.insert(pair, new_id);

            // Collect positions for the current pair using the Intrusive List
            let mut positions = Vec::new();
            let mut current = pair_heads.get(&pair).copied();
            while let Some(idx) = current {
                positions.push(idx);
                current = nodes[idx].pair_next;
            }
            
            // Clear state for the merged pair
            pair_heads.remove(&pair);
            pair_counts.remove(&pair); 

            for &idx in &positions {
                if !nodes[idx].valid { continue; }
                let next_idx = match nodes[idx].next {
                    Some(n) => n,
                    None => continue,
                };
                if nodes[idx].id != pair.0 || nodes[next_idx].id != pair.1 { continue; }
                if !nodes[next_idx].valid { continue; }

                // 1. Unlink ourselves from the old pair lists
                Self::remove_from_pair_list(idx, &mut nodes, &mut pair_heads);

                let prev_idx = nodes[idx].prev;
                let next_next_idx = nodes[next_idx].next;

                // 2. Unlink overlapping pairs and decrement counts
                if let Some(p) = prev_idx {
                    Self::remove_from_pair_list(p, &mut nodes, &mut pair_heads);
                    let old_pair = (nodes[p].id, nodes[idx].id);
                    if let Some(c) = pair_counts.get_mut(&old_pair) {
                        *c = c.saturating_sub(1);
                        heap.push((*c, old_pair));
                    }
                }
                if let Some(nn) = next_next_idx {
                    Self::remove_from_pair_list(next_idx, &mut nodes, &mut pair_heads);
                    let old_pair = (nodes[next_idx].id, nodes[nn].id);
                    if let Some(c) = pair_counts.get_mut(&old_pair) {
                        *c = c.saturating_sub(1);
                        heap.push((*c, old_pair));
                    }
                }

                // 3. Perform O(1) Linked List Merge
                nodes[idx].id = new_id;
                nodes[idx].next = next_next_idx;
                nodes[next_idx].valid = false;
                if let Some(nn) = next_next_idx {
                    nodes[nn].prev = Some(idx);
                }

                // 4. Add new touching pairs to lists and increment counts
                if let Some(p) = prev_idx {
                    Self::add_to_pair_list(p, &mut nodes, &mut pair_heads);
                    let new_pair = (nodes[p].id, new_id);
                    let c = pair_counts.entry(new_pair).or_insert(0);
                    *c += 1;
                    heap.push((*c, new_pair));
                }
                if let Some(nn) = next_next_idx {
                    Self::add_to_pair_list(idx, &mut nodes, &mut pair_heads);
                    let new_pair = (new_id, nodes[nn].id);
                    let c = pair_counts.entry(new_pair).or_insert(0);
                    *c += 1;
                    heap.push((*c, new_pair));
                }
            }

            // --- PROGRESS BAR ---
            let now = Instant::now();
            if now.duration_since(last_update).as_millis() > 50 || step == num_merges - 1 {
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
        }
        println!(); 

        Self { vocab, merges, special_tokens }
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut file = File::create(path)?;
        writeln!(file, "{}", self.special_tokens.len())?;
        for st in &self.special_tokens {
            writeln!(file, "{}", st)?;
        }
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

        let num_special: usize = lines.next().unwrap().parse().unwrap();
        let mut special_tokens = Vec::with_capacity(num_special);
        for _ in 0..num_special {
            special_tokens.push(lines.next().unwrap().to_string());
        }

        let vocab_size: usize = lines.next().unwrap().parse().unwrap();
        let mut vocab = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            let v = lines.next().unwrap().replace("\\n", "\n");
            vocab.push(v);
        }

        let mut merges = FastMap::default();
        for line in lines {
            let parts: Vec<usize> = line.split_whitespace().map(|s| s.parse().unwrap()).collect();
            if parts.len() == 3 {
                merges.insert((parts[0], parts[1]), parts[2]);
            }
        }

        Ok(Self { vocab, merges, special_tokens })
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut sequence = Vec::new();
        let mut head = 0;
        let text_len = text.len();

        while head < text_len {
            let tail = &text[head..];

            // Mirror the training skip behavior
            if tail.starts_with("### DOC_ID:") {
                if let Some(newline_pos) = tail.find('\n') {
                    head += newline_pos + 1;
                } else {
                    head = text_len;
                }
                continue;
            }

            let mut matched_special = None;
            for st in &self.special_tokens {
                if tail.starts_with(st) {
                    matched_special = Some(st);
                    break;
                }
            }

            if let Some(st) = matched_special {
                if let Some(pos) = self.vocab.iter().position(|v| v == st) {
                    sequence.push(pos);
                }
                head += st.len();
            } else {
                let ch = tail.chars().next().unwrap();
                let ch_str = ch.to_string();
                if let Some(pos) = self.vocab.iter().position(|v| v == &ch_str) {
                    sequence.push(pos);
                }
                head += ch.len_utf8();
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
