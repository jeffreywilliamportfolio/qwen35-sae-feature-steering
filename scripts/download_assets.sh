#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3.5-35B-A3B-Base}"
SAE_REPO="${SAE_REPO:-Qwen/SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50}"
MODEL_DIR="${MODEL_DIR:-$ROOT/models/qwen35-base}"
SAE_DIR="${SAE_DIR:-$ROOT/sae}"
SAE_LAYERS="${SAE_LAYERS:-11 14 16 20 26 33 37}"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ROOT/venv/bin/python" ]]; then
    PYTHON="$ROOT/venv/bin/python"
  else
    PYTHON="python"
  fi
fi

export HF_HOME="${HF_HOME:-$ROOT/.hf_home}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export MODEL_REPO SAE_REPO MODEL_DIR SAE_DIR SAE_LAYERS

mkdir -p "$MODEL_DIR" "$SAE_DIR" "$HF_HOME"

"$PYTHON" - <<'PY'
import os
from huggingface_hub import hf_hub_download, snapshot_download

model_repo = os.environ["MODEL_REPO"]
sae_repo = os.environ["SAE_REPO"]
model_dir = os.environ["MODEL_DIR"]
sae_dir = os.environ["SAE_DIR"]
layers = [int(x) for x in os.environ["SAE_LAYERS"].split()]
token = os.environ.get("HF_TOKEN") or None

print("download model", model_repo, "->", model_dir, flush=True)
snapshot_download(
    repo_id=model_repo,
    local_dir=model_dir,
    token=token,
    allow_patterns=[
        "*.json",
        "*.safetensors",
        "*.txt",
        "tokenizer.*",
        "vocab.*",
        "merges.txt",
    ],
    ignore_patterns=["original/*", "*.pth", "*.gguf"],
    max_workers=8,
)

print("download SAE layers", layers, "from", sae_repo, "->", sae_dir, flush=True)
for layer in layers:
    path = hf_hub_download(
        repo_id=sae_repo,
        filename=f"layer{layer}.sae.pt",
        local_dir=sae_dir,
        token=token,
    )
    print("SAE_DONE", path, flush=True)

print("ASSETS_DONE", flush=True)
PY

du -sh "$MODEL_DIR" "$SAE_DIR" 2>/dev/null || true
