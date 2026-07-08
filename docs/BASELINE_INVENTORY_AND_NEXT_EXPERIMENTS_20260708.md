# Baseline Inventory and Next Experiments

Snapshot: 2026-07-08

## What We Have

### Environment and Infra

- CUDA training path works on `NVIDIA A10G`.
- Root EBS is expanded to `1024 GiB`; about `934 GiB` free after readiness runs.
- Tests and lint pass.
- Stable-RTW 60-step CUDA smoke passed health checks with no issues.

### Archived Research Results

The strongest archived result is v0.9B verifier-guided best-of-N over frozen
static and Stable-RTW checkpoints:

| split | method | N | reranked exact mean +/- std | selected valid mean +/- std | selected number F1 mean +/- std | reward hack mean |
|---|---|---:|---:|---:|---:|---:|
| validation | static | 8 | 0.067 +/- 0.050 | 0.540 +/- 0.092 | 0.928 +/- 0.023 | 0.453 |
| validation | Stable-RTW | 8 | 0.133 +/- 0.012 | 0.680 +/- 0.020 | 0.958 +/- 0.012 | 0.313 |
| test_in_dist | static | 8 | 0.120 +/- 0.020 | 0.667 +/- 0.070 | 0.950 +/- 0.030 | 0.320 |
| test_in_dist | Stable-RTW | 8 | 0.133 +/- 0.042 | 0.733 +/- 0.046 | 0.974 +/- 0.001 | 0.267 |

Paired v0.9B overlap:

| split | N | both | Stable-only | static-only | neither | p | delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 8 | 9 | 11 | 1 | 129 | 0.0063 | +0.067 +/- 0.042 |
| test_in_dist | 8 | 12 | 8 | 6 | 124 | 0.7905 | +0.013 +/- 0.050 |

Interpretation: Stable-RTW has a clean validation advantage under best-of-N.
Test-in-dist supports best-of-N as a general harness mechanism but is mixed for
Stable-vs-static.

Archived artifacts:

- `outputs/v09_seed_expansion_summary.csv`
- `outputs/v09_seed_expansion_paired.json`
- `outputs/v09_task_ids_validation_limit50.txt`
- `outputs/v09_task_ids_test_in_dist_limit50.txt`
- `docs/CURRENT_PROJECT_STATUS_AND_PAPER_ASSESSMENT.md`
- `docs/V09_BEST_OF_N_RERANKING_PLAN.md`

### Newly Collected Local Base Baseline

These runs use the same frozen v0.9 task IDs, prompt field, sampling seed,
temperature/top-p, and best-of-N script, but with `max_n=1` and no adapter.

| split | model | N | exact | valid expression | number F1 | reward hack | tokens | wall clock s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| validation | Qwen2.5-0.5B-Instruct base | 1 | 0.000 | 0.000 | 0.017 | 0.200 | 12689 | 297.2 |
| test_in_dist | Qwen2.5-0.5B-Instruct base | 1 | 0.020 | 0.020 | 0.098 | 0.320 | 12277 | 291.6 |

Artifacts:

- `outputs/bestofn/base_qwen05b_seed0_validation_limit50_n1`
- `outputs/bestofn/base_qwen05b_seed0_test_in_dist_limit50_n1`

## What We Do Not Have Locally

- Full v0.9B candidate-bank directories are not present locally.
- Full static v0.6b and Stable-RTW v0.6c trained checkpoints are not present locally.
- Local smoke checkpoints exist, but they are readiness artifacts, not research baselines.
- Base best-of-N at `N=4` or `N=8` is not collected yet.
- OOD best-of-N baselines are not collected under the v0.9B protocol.

## Practical Bottleneck Observed

The current HF generation engine is slow for best-of-N with `max_new_tokens=256`:

- Base validation `50 x 1` candidates took about 297 seconds.
- Base test-in-dist `50 x 1` candidates took about 292 seconds.

Before large candidate-bank sweeps, either keep diagnostics small or improve
generation throughput.

## Recommended Next Experiment Gate

Before designing a new algorithmic variant, fill the local reproducibility gap:

1. Reproduce one local seed-0 static checkpoint.
2. Reproduce one local seed-0 Stable-RTW checkpoint.
3. Evaluate both on frozen validation and test-in-dist task IDs at `N=1,4,8`.
4. Compare against the archived v0.9B aggregate and the newly collected base `N=1`.

This creates a local baseline ladder:

```text
base model
static shaping
Stable-RTW
new method candidate
```

Only after that should we spend budget on multi-seed or OOD extensions.

## Candidate Commands

Static seed-0 pilot:

```bash
source .env
WANDB_DISABLED=true uv run python scripts/02_grpo_train.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --train_path data/countdown/train.jsonl \
  --eval_path data/countdown/validation.jsonl \
  --output_dir outputs/checkpoints/grpo_static_seed0_300 \
  --reward_strategy static \
  --seed 0 \
  --max_steps 300 \
  --batch_size 2 \
  --grad_accum 8 \
  --num_generations 4 \
  --max_prompt_length 768 \
  --max_completion_length 256 \
  --report_to none
```

Stable-RTW seed-0 pilot:

```bash
source .env
WANDB_DISABLED=true uv run python scripts/02_grpo_train.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --train_path data/countdown/train.jsonl \
  --eval_path data/countdown/validation.jsonl \
  --output_dir outputs/checkpoints/grpo_stable_seed0_300 \
  --reward_strategy adaptive_stable \
  --seed 0 \
  --max_steps 300 \
  --batch_size 2 \
  --grad_accum 8 \
  --num_generations 4 \
  --max_prompt_length 768 \
  --max_completion_length 256 \
  --report_to none
```

Evaluation template:

```bash
source .env
uv run python scripts/07_best_of_n_rerank.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --adapter_path outputs/checkpoints/grpo_stable_seed0_300 \
  --engine hf \
  --device cuda \
  --prompt_field prompt \
  --limit 50 \
  --batch_size 8 \
  --max_n 8 \
  --n_values 1 4 8 \
  --max_new_tokens 256 \
  --temperature 0.7 \
  --top_p 0.95 \
  --seed 0 \
  --method stable_local \
  --training_seed 0 \
  --split validation \
  --data_path data/countdown/validation.jsonl \
  --task_ids_file outputs/v09_task_ids_validation_limit50.txt \
  --output_dir outputs/bestofn/stable_local_seed0_validation_limit50_n8
```

## Design Implication

The current evidence points to the bottleneck being legal candidate formation
and selection/rankability, not just scalar reward maximization. A new method
should be judged on:

- exact correctness under the strict verifier;
- selected valid-expression rate;
- number multiset F1;
- reward-hacking rate;
- oracle-vs-practical best-of-N gap;
- cost per exact result;
- generated token count and wall-clock time.

Do not count any output as correct unless it passes `src/rtw_llm/countdown.py`.
