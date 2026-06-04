#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace}"
MODEL_DIR="${MODEL_DIR:-$ROOT/models/qwen35-base}"
SAE_DIR="${SAE_DIR:-$ROOT/sae}"
MODE="${MODE:-base}"
TEMPERATURE="${TEMPERATURE:-0.8}"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ROOT/venv/bin/python" ]]; then
    PYTHON="$ROOT/venv/bin/python"
  else
    PYTHON="python"
  fi
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

CLAMP="${CLAMP:-14:4310:0,14:1651:0,14:6970:0,14:11164:0,14:12327:0,16:2947:0,16:21861:0,20:52:0,20:18122:0,20:3356:0,20:571:0,20:30877:0,26:727:0,26:8920:0,37:10793:0,37:21049:0}"

export HF_HOME="${HF_HOME:-$ROOT/.hf_home}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "$SCRIPT_DIR"
exec "$PYTHON" god_chat.py \
  --model "$MODEL_DIR" \
  --sae-dir "$SAE_DIR" \
  --clamp "$CLAMP" \
  --temperature "$TEMPERATURE" \
  --mode "$MODE" \
  --plain-output \
  --stream-output \
  "$@"

