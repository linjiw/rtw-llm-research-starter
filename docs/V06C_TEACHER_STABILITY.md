# v0.6c Teacher Stability Plan

## v0.6c Goal

Test whether a stability-constrained adaptive RTW teacher can close or exceed the v0.6b static reward baseline on Countdown legality and exact-correctness metrics, without changing the v0.6b reward surface or verifier semantics.

The experiment isolates teacher dynamics. We keep the model, dataset, reward components, primary verifier correctness, training length, generation count, and evaluation splits fixed. We change only the adaptive teacher policy by adding delayed adaptation, slower updates, legality-component floors, a numeric-distance cap, and weight-budget preservation.

Success means `adaptive_stable` improves over naive adaptive and matches or beats static on validation/test_in_dist valid_expression, exact_correct, and reward_hacking_candidate, while maintaining interpretable teacher weights that do not collapse into `numeric_distance_reward`.

## Objective

Stabilize adaptive RTW teacher dynamics without changing v0.6b reward semantics.

## Hypothesis

v0.6b static beats naive adaptive because the current adaptive teacher overreacts
to dense reward signals, suppresses legality components, and overweights numeric
distance before legal expression construction is stable.

## Controlled variables

```text
model: Qwen/Qwen2.5-0.5B-Instruct
dataset: data/countdown/*.jsonl
seed: 0
reward surface: v0.6b dense legality rewards
primary reward: exact verifier correctness
max_steps: 300
num_generations: 4
eval splits: validation, test_in_dist, test_ood_long, test_ood_division
```

## Changed variable

Only the teacher strategy changes:

```text
adaptive -> adaptive_stable
```

## adaptive_stable policy

```text
delay adaptation for first 50 teacher updates
smooth updates with alpha = 0.10
preserve total auxiliary weight budget near 1.20
floor valid_expression at 0.16
floor number_multiset_f1 at 0.18
floor allowed_ops at 0.12
cap numeric_distance_reward at 0.20
```

The reward components and verifier semantics are unchanged.

## Diagnostics

Health reports now include:

```text
weight_sum_final
weight_sum_mean
constraint_weight_mass_final
constraint_weight_mass_mean
numeric_distance_weight_final
numeric_distance_weight_mean
numeric_distance_to_constraint_ratio_final
numeric_distance_to_constraint_ratio_mean
teacher_update_l1_mean
teacher_update_linf_max
floor_hit_rate_by_component
cap_hit_rate_by_component
```

Expected v0.6c teacher behavior:

```text
weight_sum stays near 1.20
constraint_weight_mass stays high
numeric_distance_weight stays <= 0.20
numeric_distance_to_constraint_ratio stays below about 0.45
teacher update jumps are smaller than naive adaptive
```

## Experiment order

Run a 100-step smoke because 50 steps only tests the warmup phase:

```bash
RUN=outputs/grpo_rtw_v06c_adaptive_stable_cuda_smoke_100_seed0
mkdir -p "$RUN"

WANDB_PROJECT=rtw-llm-countdown .venv/bin/python scripts/02_grpo_train.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --train_path data/countdown/train.jsonl \
  --eval_path data/countdown/validation.jsonl \
  --output_dir "$RUN" \
  --reward_strategy adaptive_stable \
  --seed 0 \
  --max_steps 100 \
  --num_generations 4 \
  --report_to none \
  2>&1 | tee "$RUN/train.log"

.venv/bin/python scripts/05_check_run_health.py \
  --run_dir "$RUN" \
  | tee "$RUN/health_final.txt"
```

If the smoke passes, run the 300-step pilot:

```bash
RUN=outputs/grpo_rtw_v06c_adaptive_stable_cuda_pilot_300_seed0
mkdir -p "$RUN"

WANDB_PROJECT=rtw-llm-countdown .venv/bin/python scripts/02_grpo_train.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --train_path data/countdown/train.jsonl \
  --eval_path data/countdown/validation.jsonl \
  --output_dir "$RUN" \
  --reward_strategy adaptive_stable \
  --seed 0 \
  --max_steps 300 \
  --num_generations 4 \
  --report_to none \
  2>&1 | tee "$RUN/train.log"
```

## Comparison

Compare:

```text
static v0.6b 300 seed0
adaptive v0.6b 300 seed0
adaptive_stable v0.6c 300 seed0
```

Primary metrics:

```text
allowed_numbers
number_multiset_f1
valid_expression
exact_correct
reward_hacking_candidate
```

Decision:

```text
If v0.6c closes the gap to static, stabilization helped.
If v0.6c beats static on legality or exact correctness, expand seeds.
If v0.6c remains below static, stop floor/cap tuning and redesign teacher state.
```

## Local validation status

Implementation-local checks before CUDA:

```text
uv run pytest -q
uv run ruff check .
```

The CUDA smoke/final sections below must be filled only from actual run artifacts.

## 100-step CUDA smoke result

Status: pending until `outputs/grpo_rtw_v06c_adaptive_stable_cuda_smoke_100_seed0/health_final.txt` exists.

Required fields:

```text
reward rows:
teacher rows:
reward_variance_nonzero_fraction:
parseable_expression_rate:
allowed_numbers_rate:
allowed_ops_rate:
valid_expression_rate:
exact_correct_rate:
number_multiset_f1_mean:
weight_sum_final:
constraint_weight_mass_final:
numeric_distance_weight_final:
numeric_distance_to_constraint_ratio_final:
teacher_update_l1_mean:
teacher_update_linf_max:
issues:
```

## 300-step CUDA pilot result

Status: pending until `outputs/grpo_rtw_v06c_adaptive_stable_cuda_pilot_300_seed0/health_final.txt` exists.

## Held-out eval comparison

Status: pending until v0.6c validation/test_in_dist/test_ood_long/test_ood_division eval outputs exist.
