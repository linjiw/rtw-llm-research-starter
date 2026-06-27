# v0.5 CUDA Bootstrap Report

Status: draft until CUDA runs are complete.

## 1. Hardware / Environment

- Host:
- GPU:
- GPU count:
- CUDA:
- PyTorch:
- Python:
- Git commit:
- Preflight file: `outputs/cuda_env_preflight.txt`

## 2. Git Commit And Dataset Manifest

- Train:
- Validation:
- Test in-distribution:
- OOD long:
- OOD division:
- Dataset generation seed: 42
- Dataset card: `docs/DATASET_CARD.md`
- Commit:

## 3. Base M4 Eval Summary

- Run dir: `outputs/eval_m4_base_smoke`
- Format:
- Parseable expression:
- Exact correct:
- Primary observation:
- Representative raw prose failure:

```text
```

## 4. CUDA 50-Step Health Output

Paste the output of:

```bash
python scripts/05_check_run_health.py \
  --run_dir outputs/grpo_rtw_cuda_smoke_50 | tee outputs/grpo_rtw_cuda_smoke_50/health.txt
```

```json
```

## 5. Reward Decomposition Summary

- Primary reward mean:
- Weighted auxiliary reward mean:
- Total reward mean:
- Reward variance nonzero fraction:
- Open tag rate:
- Close tag rate:
- Extractable span rate:
- Parseable expression rate:
- Allowed numbers rate:
- Allowed ops rate:
- Exact correct rate:
- Tag-only rate:
- Parseable but wrong rate:
- Correct given parseable:

## 6. Teacher Weight Movement

- Weights moved:
- Format min / mean / max:
- Valid expression min / mean / max:
- Uses numbers min / mean / max:
- Allowed ops min / mean / max:
- Brevity min / mean / max:

## 7. Sample Generations Before / After

### Raw Prose Failure

```text
```

### Tag-Only Partial Success

```text
```

### Best Parseable Or Near-Parseable Example

```text
```

## 8. Failure Mode Classification

Choose the dominant failure mode:

- Tag farming:
- Parseable nonsense:
- Auxiliary dominance:
- Flat reward landscape:
- Other:

Evidence:

```text
```

## 9. Decision

- Continue direct 300-step GRPO:
- Insert SFT harness warmup first:
- Rationale:

## Training Config Appendix

### Adaptive Smoke

- Run dir: `outputs/grpo_rtw_cuda_smoke_50`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Reward strategy: adaptive
- Max steps: 50
- Num generations: 4
- Seed: 0

### Adaptive Pilot

- Run dir: `outputs/grpo_rtw_cuda_pilot_300_seed0`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Reward strategy: adaptive
- Max steps: 300
- Num generations: 4
- Seed: 0

### Static Pilot

- Run dir: `outputs/grpo_static_cuda_pilot_300_seed0`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Reward strategy: static
- Max steps: 300
- Num generations: 4
- Seed: 0

## Adaptive vs Static Curves

- Format acquisition speed:
- Parseability acquisition speed:
- First nonzero correctness step:
- Best validation exact correct:
- Tag-only / reward-hacking behavior:
