# Troubleshooting

## Out of Memory

Use a larger GPU or reduce generation length. A single 96 GB GPU is the practical target for this bf16 Hugging Face setup.

The model weights are roughly `67 GB`, and runtime overhead matters.

## Slow First Launch

First launch pays for model loading and SAE loading. Generation speed should be evaluated after the REPL is ready.

## "Fast path is not available"

Transformers may print a warning about a missing fast path or linear-attention package. This setup is intentionally pure PyTorch and does not require `fla`.

The warning is benign if generation proceeds.

## Output Repeats

Try:

```text
/reppen 1.08
/ngram 6
/topk 100
/temp 0.8
```

Also lower the clamp:

```text
/target 18122 1.0
```

Strong clamping can make the selected feature dominate the answer and increase loops.

## Output Invents a New User Turn

Base-mode prompting can sometimes continue the transcript. The script stops on common scaffold strings such as `User:` and `You:`, but a bare invented question can still appear.

For cleaner behavior, keep answers shorter with:

```bash
--soft-max-new-tokens 384 --max-new-tokens 768
```

## Thinking Blocks Appear

`/think off` disables thinking in future prompts and strips any emitted `<think>` blocks from the
display. `/think hide` only changes the display filter.

For HauhauCS-style chat runs, prefer the manual bare-close no-think prompt:

```bash
MODE=chat NO_THINK=1 NO_THINK_STYLE=bare-close ./scripts/run.sh --model-loader image-text
```

This renders `<|im_start|>assistant` followed by `</think>` and a blank line before generation,
matching the older HauhauCS/Q8 no-think experiment convention.

## Download Problems

Make sure `HF_TOKEN` is exported if the model or SAE repo requires authentication:

```bash
export HF_TOKEN=hf_...
```

Use:

```bash
export HF_XET_HIGH_PERFORMANCE=1
```

The older `HF_HUB_ENABLE_HF_TRANSFER` setting may print deprecation warnings in newer Hugging Face Hub versions.
