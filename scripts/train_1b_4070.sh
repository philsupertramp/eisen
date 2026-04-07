#!/usr/bin/env bash
set -euo pipefail

# RTX 4070 Laptop (8GB VRAM) profile for train_1b.
# Tuned for stability first; increase MICRO_BATCH or VRAM_BUDGET_MB gradually
# if your specific laptop SKU/cooling allows more headroom.

export EISEN_SEQ_LEN="${EISEN_SEQ_LEN:-128}"
export EISEN_MICRO_BATCH="${EISEN_MICRO_BATCH:-2}"
export EISEN_ACCUM_STEPS="${EISEN_ACCUM_STEPS:-8}"
export EISEN_VRAM_BUDGET_MB="${EISEN_VRAM_BUDGET_MB:-6400}"
export EISEN_ACTIVATION_RESERVE_MB="${EISEN_ACTIVATION_RESERVE_MB:-900}"

echo "Launching train_1b with RTX 4070 Laptop profile:"
echo "  EISEN_SEQ_LEN=$EISEN_SEQ_LEN"
echo "  EISEN_MICRO_BATCH=$EISEN_MICRO_BATCH"
echo "  EISEN_ACCUM_STEPS=$EISEN_ACCUM_STEPS"
echo "  EISEN_VRAM_BUDGET_MB=$EISEN_VRAM_BUDGET_MB"
echo "  EISEN_ACTIVATION_RESERVE_MB=$EISEN_ACTIVATION_RESERVE_MB"

cargo run --release --example train_1b --features bf16
