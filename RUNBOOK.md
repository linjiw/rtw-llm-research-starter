# Runbook

## 0. Verify package and data

```bash
pip install -e .
pytest -q
head -n 1 data/countdown/train.jsonl | jq .
```

## 1. Base eval

### MacBook M4 smoke eval

```bash
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
make m4-check
make eval-m4-smoke
```

Use this to validate prompt -> generation -> verifier -> metrics locally. It is
not a substitute for CUDA GRPO training.

### CUDA or CPU base eval

```bash
python scripts/03_eval.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --data_path data/countdown/test_in_dist.jsonl \
  --limit 100 \
  --output_dir outputs/eval_base_in_dist
```

## 2. First GRPO RTW smoke run

Run this on a CUDA/NVIDIA host for meaningful signal. On a MacBook M4, only run
the 5-step compatibility version from `docs/HARDWARE_AND_INFRA.md`.

```bash
WANDB_PROJECT=rtw-llm-countdown python scripts/02_grpo_train.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --output_dir outputs/grpo_rtw_smoke \
  --reward_strategy adaptive \
  --max_steps 100 \
  --num_generations 4
```

## 3. Baseline sweep

```bash
for STRATEGY in static manual random adaptive; do
  WANDB_PROJECT=rtw-llm-countdown python scripts/02_grpo_train.py \
    --model_name Qwen/Qwen2.5-0.5B-Instruct \
    --output_dir outputs/grpo_${STRATEGY}_seed0 \
    --reward_strategy ${STRATEGY} \
    --seed 0 \
    --max_steps 300 \
    --num_generations 4
done
```

## 4. Evaluate harness shift and OOD

```bash
RUN_DIR=outputs/grpo_adaptive_seed0
for SPLIT in test_in_dist test_ood_long test_ood_division; do
  for PROMPT_FIELD in prompt_high prompt_mid prompt_low; do
    python scripts/03_eval.py \
      --model_name Qwen/Qwen2.5-0.5B-Instruct \
      --adapter_path ${RUN_DIR} \
      --data_path data/countdown/${SPLIT}.jsonl \
      --prompt_field ${PROMPT_FIELD} \
      --output_dir ${RUN_DIR}/eval_${SPLIT}_${PROMPT_FIELD}
  done
done
```

## 5. Plot teacher weights

```bash
python scripts/04_analyze_results.py --run_dir outputs/grpo_adaptive_seed0
```
