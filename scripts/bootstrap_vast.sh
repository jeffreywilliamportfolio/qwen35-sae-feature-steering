#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

mkdir -p "$ROOT"
cd "$ROOT"

echo "[1/5] create venv"
"$PYTHON_BIN" -m venv "$ROOT/venv"
source "$ROOT/venv/bin/activate"
python -m pip install -U pip wheel setuptools

echo "[2/5] install torch"
python -m pip install -U torch --index-url "$TORCH_INDEX_URL"

echo "[3/5] install python dependencies"
python -m pip install -r "$REPO_DIR/requirements.txt"

echo "[4/5] verify CUDA"
python - <<'PY'
import torch, transformers, huggingface_hub
assert torch.cuda.is_available(), "CUDA is unavailable"
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("transformers", transformers.__version__)
print("huggingface_hub", huggingface_hub.__version__)
PY

echo "[5/5] download assets"
"$REPO_DIR/scripts/download_assets.sh"

echo "BOOTSTRAP_DONE"
