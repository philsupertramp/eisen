// src/data/fim.rs
//
// Fill-in-the-Middle (FIM) pretraining support.
//
// Two formats are supported:
//   PSM — <fim_prefix> | prefix | <fim_suffix> | suffix | <fim_middle> | middle
//   SPM — <fim_suffix> | suffix | <fim_prefix> | prefix | <fim_middle> | middle
//
// During training the loss is masked to zero for every token *before* the
// <fim_middle> sentinel; only the middle span is trained.  This matches the
// approach in "Efficient Training of Language Models to Fill in the Middle"
// (Bavarian et al., 2022).
//
// Usage in the dataloader:
//   1. Append 4 special tokens to the end of your vocabulary (see FimTokens).
//   2. Construct a FimConfig with those IDs.
//   3. Call apply_fim() on each raw token sequence ~50% of the time.

use rand::Rng;

// ─── Special-token bundle ─────────────────────────────────────────────────────

/// IDs for the four FIM control tokens.
/// These must be the *last* 4 entries in the vocabulary so that existing
/// BPE token IDs are undisturbed.
#[derive(Clone, Debug)]
pub struct FimTokens {
    pub prefix:  usize,  // <fim_prefix>
    pub suffix:  usize,  // <fim_suffix>
    pub middle:  usize,  // <fim_middle>
    pub pad:     usize,  // <fim_pad>  (used when the transformed seq is short)
}

impl FimTokens {
    /// Derive token IDs from the base vocab size: tokens are appended in order
    /// [prefix, suffix, middle, pad] so the new total vocab is `base + 4`.
    pub fn from_base_vocab(base_vocab_size: usize) -> Self {
        Self {
            prefix: base_vocab_size,
            suffix: base_vocab_size + 1,
            middle: base_vocab_size + 2,
            pad:    base_vocab_size + 3,
        }
    }

    pub fn vocab_overhead() -> usize { 4 }
}

// ─── Config ───────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct FimConfig {
    pub tokens: FimTokens,
    /// Fraction of samples to transform with FIM (0.0–1.0).  Typical: 0.5.
    pub rate: f32,
    /// Fraction of FIM samples that use SPM vs PSM.  Typical: 0.5.
    pub spm_rate: f32,
}

impl FimConfig {
    pub fn new(base_vocab_size: usize) -> Self {
        Self {
            tokens: FimTokens::from_base_vocab(base_vocab_size),
            rate: 0.5,
            spm_rate: 0.5,
        }
    }

    pub fn with_rate(mut self, rate: f32) -> Self {
        self.rate = rate;
        self
    }

    pub fn with_spm_rate(mut self, spm_rate: f32) -> Self {
        self.spm_rate = spm_rate;
        self
    }

    /// Total vocabulary size including FIM special tokens.
    pub fn full_vocab_size(&self, base: usize) -> usize {
        base + FimTokens::vocab_overhead()
    }
}

// ─── Core transformation ──────────────────────────────────────────────────────

/// Apply one FIM transformation to a raw token sequence.
///
/// Returns `(transformed_tokens, loss_mask)` where:
///   - `transformed_tokens` has at most `seq_len` tokens (truncated if needed,
///     padded with `fim_pad` if shorter).
///   - `loss_mask[i] == true` means the cross-entropy loss is computed at
///     position i; `false` means the position is skipped.
///
/// The loss is only active on the *middle span* (after the `<fim_middle>` token).
pub fn apply_fim<R: Rng>(
    tokens: &[usize],
    config: &FimConfig,
    rng: &mut R,
    seq_len: usize,
) -> (Vec<usize>, Vec<bool>) {
    let n = tokens.len();

    // Need at least 4 tokens to have non-empty prefix, middle, suffix.
    if n < 4 {
        let mut out = tokens.to_vec();
        out.resize(seq_len, config.tokens.pad);
        return (out.clone(), vec![true; out.len()]);
    }

    // Pick two split points strictly inside (0, n).
    let lo = rng.gen_range(1..n - 2);
    let hi = rng.gen_range(lo + 1..n - 1);

    let prefix = &tokens[..lo];
    let middle = &tokens[lo..hi];
    let suffix = &tokens[hi..];

    let use_spm = rng.r#gen::<f32>() < config.spm_rate;

    // Build the reordered sequence and a parallel boolean mask.
    let mut result: Vec<usize> = Vec::with_capacity(n + 3);
    let mut mask:   Vec<bool>  = Vec::with_capacity(n + 3);

    macro_rules! push_masked {
        ($tok:expr, $m:expr) => {
            result.push($tok);
            mask.push($m);
        };
    }
    macro_rules! push_slice {
        ($s:expr, $m:expr) => {
            for &t in $s {
                result.push(t);
                mask.push($m);
            }
        };
    }

    if use_spm {
        // <fim_suffix> suffix <fim_prefix> prefix <fim_middle> middle
        push_masked!(config.tokens.suffix, false);
        push_slice!(suffix,  false);
        push_masked!(config.tokens.prefix, false);
        push_slice!(prefix,  false);
        push_masked!(config.tokens.middle, false); // sentinel itself not trained
        push_slice!(middle,  true);                // ← only these tokens are trained
    } else {
        // <fim_prefix> prefix <fim_suffix> suffix <fim_middle> middle
        push_masked!(config.tokens.prefix, false);
        push_slice!(prefix,  false);
        push_masked!(config.tokens.suffix, false);
        push_slice!(suffix,  false);
        push_masked!(config.tokens.middle, false);
        push_slice!(middle,  true);
    }

    // Truncate or pad to seq_len
    result.truncate(seq_len);
    mask.truncate(seq_len);
    while result.len() < seq_len {
        result.push(config.tokens.pad);
        mask.push(false);
    }

    (result, mask)
}

// ─── Mask → ignore-index targets ──────────────────────────────────────────────

/// Convert a `(y_tokens, loss_mask)` pair into a `targets` vector where
/// positions masked out have value `IGNORE_INDEX` (used by `cross_entropy_masked`).
pub const IGNORE_INDEX: usize = usize::MAX;

pub fn mask_targets(y: &[usize], mask: &[bool]) -> Vec<usize> {
    y.iter()
        .zip(mask.iter())
        .map(|(&t, &m)| if m { t } else { IGNORE_INDEX })
        .collect()
}
