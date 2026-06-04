# Qwen3.5 35B SAE Feature Steering Chat

Interactive SAE feature steering for `Qwen/Qwen3.5-35B-A3B-Base`.

This repo reproduces the "god chat" style intervention: load the base 35B MoE model, load per-layer Qwen-Scope SAEs, register residual-stream hooks, and clamp selected SAE features during generation. The default launcher registers the known features but sets every target to `0`, so the model starts unsteered until you enable a feature in the REPL.

## Hardware

Tested target: a single high-VRAM NVIDIA GPU.

- Recommended: `96 GB+` VRAM.
- Tested classes: RTX PRO 6000 Blackwell 96 GB, H200 140 GB.
- Model download is roughly `67 GB`; the five default SAE files are roughly `2-3 GB`.

An 80 GB card may be tight for bf16 Hugging Face chat once model overhead, activations, and generation buffers are included.

## Quick Start

On a cloud GPU box:

```bash
git clone <your-new-repo-url>
cd qwen35-sae-feature-steering

python3 -m venv /workspace/venv
source /workspace/venv/bin/activate
python -m pip install -U pip wheel setuptools
python -m pip install -U torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt

export HF_TOKEN=hf_...
./scripts/download_assets.sh

./scripts/run.sh --max-new-tokens 768 --soft-max-new-tokens 384 \
  --repetition-penalty 1.08 --no-repeat-ngram-size 6
```

The REPL starts with all registered targets at zero:

```text
you> /clamp
you> /target 4310 1.75
you> Hello.
```

Useful live commands:

```text
/target FEAT N       set one feature floor
/target N            set all registered features to the same floor
/off                 set all floors to zero
/clamp               show registered features, floors, and last activations
/temp 0.8            set temperature
/topk 100            set top-k sampling
/reppen 1.08         set repetition penalty
/ngram 6             block repeated 6-token n-grams
/softmax 384         soft stop after a natural boundary past 384 generated tokens
/reset               clear history and roll the locked seed
/quit
```

## Paths

The helper scripts default to `/workspace`, matching Vast.ai style instances:

```text
/workspace/venv
/workspace/models/qwen35-base
/workspace/sae/layer14.sae.pt
```

For a local Linux box, override `ROOT`:

```bash
export ROOT="$PWD/.runtime"
python3 -m venv "$ROOT/venv"
source "$ROOT/venv/bin/activate"
python -m pip install -U torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
./scripts/download_assets.sh
./scripts/run.sh
```

## What Gets Downloaded

Model:

```text
Qwen/Qwen3.5-35B-A3B-Base
```

SAE family:

```text
Qwen/SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50
```

Default layers:

```text
14 16 20 26 37
```

## Default Feature Registry

The default launcher registers these layer/feature pairs with target `0`:

```text
14:4310   non-dual / God actuator
14:1651   tourism / attractions
14:6970   love / affection
14:11164  argument / opposition
16:2947   fear / timid / afraid
20:18122  golf
20:3356   criticism / arguments
20:571    apology / guilt / sorry
20:30877  anxiety / stress
26:8920   refute / correct / debunk
37:10793  em-dash / dash style
```

See [docs/FEATURES.md](docs/FEATURES.md) for provenance and example settings.

## Core Idea

For a feature `f` at layer `L`, the script reads the SAE activation:

```text
a_f = relu(resid_L . e_f + b_f)
```

Then, if `a_f` is below the configured floor, it adds the decoder direction:

```text
resid_L += max(target_f - a_f, 0) * d_f
```

This is an inference-time intervention. It does not train or edit the model weights.

## Notes

- The primary key is `(layer, feature_id)`. The same feature index at a different layer is a different SAE feature.
- The script defaults to base-mode prompting rather than chat-template prompting.
- Strong clamps can cause topic hijack or repetition. Start low and step upward.
- `--soft-max-new-tokens` is a soft stop; `--max-new-tokens` remains the hard runaway cap.
