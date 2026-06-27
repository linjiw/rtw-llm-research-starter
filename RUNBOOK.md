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

First verify the CUDA stack and save the environment preflight:

```bash
mkdir -p outputs
python - <<'PY' | tee outputs/cuda_env_preflight.txt
import sys
import torch
print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_version:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("gpu_count:", torch.cuda.device_count())
PY
```

```bash
WANDB_PROJECT=rtw-llm-countdown python scripts/02_grpo_train.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --train_path data/countdown/train.jsonl \
  --eval_path data/countdown/validation.jsonl \
  --output_dir outputs/grpo_rtw_cuda_smoke_50 \
  --reward_strategy adaptive \
  --seed 0 \
  --max_steps 50 \
  --num_generations 4
```

Immediately inspect health:

```bash
tail -n 5 outputs/grpo_rtw_cuda_smoke_50/reward_components.jsonl
tail -n 5 outputs/grpo_rtw_cuda_smoke_50/teacher_weights.jsonl
python scripts/04_analyze_results.py --run_dir outputs/grpo_rtw_cuda_smoke_50
python scripts/05_check_run_health.py \
  --run_dir outputs/grpo_rtw_cuda_smoke_50 | tee outputs/grpo_rtw_cuda_smoke_50/health.txt
```

Acceptance criteria for the 50-step smoke are modest: training completes without
NaNs, reward and teacher logs are populated, teacher weights move over time,
some non-brevity component is occasionally nonzero, and format reward has some
variance. If rewards stay flat and all non-brevity components remain zero, stop
GRPO and run the SFT harness warmup below.

The health report should explicitly show:

- `open_tag_rate`
- `close_tag_rate`
- `extractable_span_rate`
- `parseable_expression_rate`
- `allowed_numbers_rate`
- `allowed_ops_rate`
- `exact_correct_rate`
- `tag_only_rate`
- `parseable_but_wrong_rate`
- `correct_given_parseable`
- `reward_variance_nonzero_fraction`

## 2b. SFT harness warmup fallback

Use this if the CUDA smoke has no useful reward support.

```bash
WANDB_PROJECT=rtw-llm-countdown python scripts/01_sft_warmup.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --train_path data/countdown/train.jsonl \
  --eval_path data/countdown/validation.jsonl \
  --output_dir outputs/sft_harness_warmup_qwen05b_seed0 \
  --seed 0 \
  --max_steps 100
```

Evaluate the warmup:

```bash
python scripts/03_eval.py \
  --model_name outputs/sft_harness_warmup_qwen05b_seed0 \
  --data_path data/countdown/test_in_dist.jsonl \
  --output_dir outputs/eval_sft_harness_warmup_seed0 \
  --limit 128
```

Then run adaptive GRPO from the SFT checkpoint:

```bash
WANDB_PROJECT=rtw-llm-countdown python scripts/02_grpo_train.py \
  --model_name outputs/sft_harness_warmup_qwen05b_seed0 \
  --train_path data/countdown/train.jsonl \
  --eval_path data/countdown/validation.jsonl \
  --output_dir outputs/grpo_rtw_after_sft_300_seed0 \
  --reward_strategy adaptive \
  --seed 0 \
  --max_steps 300 \
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

## 6. v0.5 report

After the CUDA smoke and first adaptive/static pilot, fill in:

```bash
docs/v05_cuda_bootstrap_report.md
```

Keep primary reward, weighted auxiliary reward, and total reward separate in the
report so improvements in task correctness are not conflated with harness-only
progress.
