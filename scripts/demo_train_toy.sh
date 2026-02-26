#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TOY_DIR="${TOY_DIR:-toy_data/train_demo}"
OUT_DIR="${OUT_DIR:-outputs/demo_stage1_toy}"
NUM_SAMPLES="${NUM_SAMPLES:-60}"
SEED="${SEED:-42}"

"$PYTHON_BIN" scripts/make_toy_data.py \
  --out_dir "$TOY_DIR" \
  --num_samples "$NUM_SAMPLES" \
  --seq_len 40 \
  --roi_dim 64 \
  --seed "$SEED"

CONFIG_PATH="configs/demo_stage1_toy.yaml"
"$PYTHON_BIN" train/train_stage1_raw.py --config "$CONFIG_PATH"

echo "[demo][ok] Stage-1 toy training done."
echo "[demo][ok] checkpoint: $OUT_DIR/checkpoints/best.pt"
