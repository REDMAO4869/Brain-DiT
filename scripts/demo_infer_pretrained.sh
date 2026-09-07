#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CKPT_PATH="${CKPT_PATH:-}"
if [[ $# -ge 1 && -z "$CKPT_PATH" ]]; then
  CKPT_PATH="$1"
  shift
fi
NUM_SAMPLES="${NUM_SAMPLES:-24}"
SEED="${SEED:-42}"

if [[ -z "$CKPT_PATH" ]]; then
  echo "[demo][error] checkpoint path is required." >&2
  echo "Usage 1: CKPT_PATH=/path/to/checkpoints/best.pt bash scripts/demo_infer_pretrained.sh" >&2
  echo "Usage 2: bash scripts/demo_infer_pretrained.sh /path/to/checkpoints/best.pt" >&2
  exit 1
fi

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "[demo][error] checkpoint not found: $CKPT_PATH" >&2
  exit 1
fi

ckpt_parent="$(basename "$(dirname "$CKPT_PATH")")"
if [[ "$ckpt_parent" == "checkpoints" ]]; then
  ckpt_tag_default="$(basename "$(dirname "$(dirname "$CKPT_PATH")")")"
else
  ckpt_tag_default="${ckpt_parent}_$(basename "$CKPT_PATH" .pt)"
fi
CKPT_TAG="${CKPT_TAG:-$ckpt_tag_default}"

read -r SEQ_LEN ROI_DIM TIMESTEP <<<"$($PYTHON_BIN - <<PY
import torch
ckpt = torch.load(r'''$CKPT_PATH''', map_location='cpu')
cfg = ckpt.get('config', {})
seq_len = int(cfg.get('seq_len', 40))
roi_dim = int(cfg.get('roi_dim', 424))
num_steps = int(cfg.get('diffusion', {}).get('num_steps', 1000))
timestep = min(100, max(1, num_steps // 2))
print(seq_len, roi_dim, timestep)
PY
)"

TOY_DIR="${TOY_DIR:-toy_data/infer_${CKPT_TAG}}"
OUT_DIR="${OUT_DIR:-outputs/demo_infer/${CKPT_TAG}}"
CFG_PATH="$OUT_DIR/extract_config.yaml"
mkdir -p "$OUT_DIR"

"$PYTHON_BIN" scripts/make_toy_data.py \
  --out_dir "$TOY_DIR" \
  --num_samples "$NUM_SAMPLES" \
  --seq_len "$SEQ_LEN" \
  --roi_dim "$ROI_DIM" \
  --seed "$SEED"

cat > "$CFG_PATH" <<YAML
output_dir: $OUT_DIR
stage1_ckpt_path: $CKPT_PATH
device: auto
noise_seed: 0

embedding:
  timestep: $TIMESTEP
  capture_layer: -1
  pool: mean
  noise_mode: per_subject

dataloader:
  batch_size: 4
  num_workers: 0

data:
  subject_col: Subject
  path_col: Path
  labels_csv: $TOY_DIR/labels/labels_downstream.csv
  labels_subject_col: Subject
  labels_label_col: target
  seq_len: $SEQ_LEN
  roi_dim: $ROI_DIM
  strict_seq_len: true
  crop_mode: center
  pad_mode: zeros
  path_prefix: null
  splits:
    train:
      csv: $TOY_DIR/splits/train.csv
    valid:
      csv: $TOY_DIR/splits/val.csv
    test:
      csv: $TOY_DIR/splits/test.csv
YAML

"$PYTHON_BIN" -m downstream.run --mode extract --config "$CFG_PATH"

echo "[demo][ok] Inference finished."
echo "[demo][ok] embeddings: $OUT_DIR/train_emb.npy $OUT_DIR/valid_emb.npy $OUT_DIR/test_emb.npy"
