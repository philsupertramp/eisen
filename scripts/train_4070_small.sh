#!/usr/bin/env bash
set -euo pipefail

# RTX 4070 Laptop (8GB VRAM) SMALL-MODEL profile.
# Uses the same train_1b pipeline but with a much smaller architecture so the
# GPU can be utilized efficiently without extreme streaming overhead.
#
# Approx class: ~250-350M params depending on vocab and exact dims.

export EISEN_HIDDEN_DIM="${EISEN_HIDDEN_DIM:-1024}"
export EISEN_NUM_HEADS="${EISEN_NUM_HEADS:-16}"
export EISEN_FFN_DIM="${EISEN_FFN_DIM:-2816}"
export EISEN_NUM_LAYERS="${EISEN_NUM_LAYERS:-24}"

export EISEN_SEQ_LEN="${EISEN_SEQ_LEN:-256}"
export EISEN_MICRO_BATCH="${EISEN_MICRO_BATCH:-4}"
export EISEN_ACCUM_STEPS="${EISEN_ACCUM_STEPS:-8}"

export EISEN_VRAM_BUDGET_MB="${EISEN_VRAM_BUDGET_MB:-7000}"
export EISEN_ACTIVATION_RESERVE_MB="${EISEN_ACTIVATION_RESERVE_MB:-700}"

export EISEN_DTYPE="${EISEN_DTYPE:-''}"

if [ "${EISEN_DTYPE}" = "bf16" ]; then
  EISEN_DTYPE="--features bf16"
fi

echo "Launching SMALL profile on GPU:"
echo "  HIDDEN=$EISEN_HIDDEN_DIM HEADS=$EISEN_NUM_HEADS FFN=$EISEN_FFN_DIM LAYERS=$EISEN_NUM_LAYERS"
echo "  SEQ=$EISEN_SEQ_LEN MICRO_BATCH=$EISEN_MICRO_BATCH ACCUM=$EISEN_ACCUM_STEPS"
echo "  VRAM_BUDGET_MB=$EISEN_VRAM_BUDGET_MB RESERVE_MB=$EISEN_ACTIVATION_RESERVE_MB"

cargo run --release --example train_1b ${EISEN_DTYPE}
