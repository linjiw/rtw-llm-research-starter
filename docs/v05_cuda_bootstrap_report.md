# v0.5 CUDA Bootstrap Report

Status: draft until CUDA runs are complete.

## Hardware

- Host:
- GPU:
- CUDA:
- PyTorch:
- Git commit:

## Dataset Manifest

- Train:
- Validation:
- Test in-distribution:
- OOD long:
- OOD division:
- Dataset generation seed:

## Training Configs

### Adaptive Smoke

- Run dir: `outputs/grpo_rtw_cuda_smoke_50`
- Model:
- Reward strategy: adaptive
- Max steps:
- Num generations:

### Adaptive Pilot

- Run dir: `outputs/grpo_rtw_cuda_pilot_300_seed0`
- Model:
- Reward strategy: adaptive
- Max steps:
- Num generations:
- Seed:

### Static Pilot

- Run dir: `outputs/grpo_static_cuda_pilot_300_seed0`
- Model:
- Reward strategy: static
- Max steps:
- Num generations:
- Seed:

## Base Eval Metrics

- Format:
- Parseable expression:
- Exact correct:
- Notes:

## Adaptive Smoke Health

Paste the output of:

```bash
python scripts/05_check_run_health.py --run_dir outputs/grpo_rtw_cuda_smoke_50
```

Key checks:

- Reward components populated:
- Reward variance nonzero fraction:
- Open tag rate:
- Extractable span rate:
- Parseable expression rate:
- Tag-only rate:
- Correct given parseable:
- Teacher weights moved:

Decision:

- Continue direct GRPO:
- Insert SFT harness warmup:

## Adaptive vs Static Curves

- Format acquisition speed:
- Parseability acquisition speed:
- First nonzero correctness step:
- Best validation exact correct:
- Tag-only / reward-hacking behavior:

## Sample Generations

### Before

```text
```

### After Adaptive

```text
```

### After Static

```text
```

## Teacher Weight Evolution

- Format:
- Valid expression:
- Uses numbers:
- Allowed ops:
- Brevity:

## Failure Analysis

- Main failure mode:
- Evidence:
- Next intervention:
