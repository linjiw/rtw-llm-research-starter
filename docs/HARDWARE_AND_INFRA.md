# Hardware and infra notes

## Local single-GPU profile

Good for first experiments:

- GPU: RTX 4090 / A10 / L4 / A5000-class, 24 GB.
- Model: Qwen2.5-0.5B-Instruct or Qwen2.5-1.5B-Instruct.
- Training: LoRA GRPO, short completions, `num_generations=4`.

## Larger profile

Good for more stable curves:

- GPU: A100/H100 80 GB or 2-4 GPUs.
- Model: 3B/7B.
- Training: larger batch, more generations, multi-seed experiments.

## Logging

The scaffold logs:

- `teacher_weights.jsonl`: teacher state, EMA scores, and reward weights.
- `reward_components.jsonl`: per-completion reward components.
- `generations.jsonl`: evaluation generations and verifier metrics.
- `metrics.json`: aggregate evaluation metrics.

Use W&B or TensorBoard for run comparison.

## Reproducibility checklist

- Save dataset generation seed.
- Save model name and adapter commit/hash where possible.
- Save exact command line.
- Save `teacher_weights.jsonl`.
- Evaluate all checkpoints with the same decoding settings.
- Run at least 3 seeds before reporting a serious result.
