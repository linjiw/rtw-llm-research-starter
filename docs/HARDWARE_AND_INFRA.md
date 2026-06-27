# Hardware and infra notes

## MacBook M4 local profile

Use Apple Silicon as the local research cockpit, not as the main experiment
engine:

- Good fit: verifier tests, dataset generation, reward/teacher debugging, CLI
  import checks, and small Hugging Face eval runs.
- Possible but not reliable enough for serious curves: very small SFT/GRPO
  compatibility smoke tests through PyTorch MPS.
- Not useful: CUDA Docker training, normal vLLM CUDA serving, or multi-seed
  paper-quality GRPO runs.

Apple Silicon uses MPS/Metal, not CUDA. Run local checks with:

```bash
source .venv/bin/activate
make m4-check
```

Run the first base-model eval smoke test locally with:

```bash
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
make eval-m4-smoke
```

If you try GRPO locally, treat it only as device compatibility validation:

```bash
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
export WANDB_DISABLED=true

python scripts/02_grpo_train.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --output_dir outputs/grpo_rtw_m4_tiny \
  --reward_strategy adaptive \
  --max_steps 5 \
  --batch_size 2 \
  --grad_accum 1 \
  --num_generations 2 \
  --max_prompt_length 512 \
  --max_completion_length 64 \
  --report_to none
```

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
