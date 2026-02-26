#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODE="${MODE:-general}"
CONFIG_PATH="${CONFIG_PATH:-configs/stage3_raw_general.example.yaml}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

"$PYTHON_BIN" - <<"PY"
import importlib.util
mods = ["core.data", "core.diffusion", "downstream"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit("Missing required modules for downstream: " + ", ".join(missing))
PY

"$PYTHON_BIN" -m downstream.run --mode "$MODE" --config "$CONFIG_PATH" "$@"
