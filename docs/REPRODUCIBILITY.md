# Reproducibility Notes

## Model and SAE Sources

Use the base model:

```text
Qwen/Qwen3.5-35B-A3B-Base
```

Use Qwen-Scope per-layer SAEs:

```text
Qwen/SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50
```

The default launcher expects:

```text
layer11.sae.pt
layer14.sae.pt
layer16.sae.pt
layer20.sae.pt
layer26.sae.pt
layer33.sae.pt
layer37.sae.pt
```

## Architecture Facts Used

The relevant Qwen3.5 A3B facts are:

- `35B` total parameters with about `3B` active per forward pass.
- `40` decoder layers.
- `256` routed experts.
- `8` routed experts selected per token, plus a shared expert path.

The chat script does not modify router logits. It only adds SAE decoder directions into the residual stream with PyTorch forward hooks.

## Residual Stream Convention

The SAEs are read on the residual stream for the corresponding decoder block layer. The release script uses a forward hook on the layer module output and computes:

```text
a_f = relu(resid_L . e_f + b_f)
resid_L += max(target_f - a_f, 0) * d_f
```

This means the target is a floor, not a forced exact activation. If the feature is already above the target, the hook does nothing for that feature.

## Determinism

The REPL starts with a locked seed. For sampled generation, the script derives a per-prompt seed from:

```text
base seed
prompt text
feature targets
temperature
top-p
top-k
repetition penalty
no-repeat n-gram size
```

Use `/reset` to clear history and roll the seed.

## Recommended Baseline Command

```bash
./scripts/run.sh --max-new-tokens 768 --soft-max-new-tokens 384 \
  --repetition-penalty 1.08 --no-repeat-ngram-size 6
```

Then enable one feature at a time:

```text
/target 4310 1.75
```

## What Counts as Reproduction

A useful reproduction should report:

- GPU model and VRAM.
- Torch, CUDA, Transformers, and Hugging Face Hub versions.
- Exact model repo and SAE repo.
- SAE layers downloaded.
- Clamp string or REPL commands.
- Temperature, top-p, top-k, repetition penalty, no-repeat n-gram size.
- Seed state and whether `/reset` was used.
