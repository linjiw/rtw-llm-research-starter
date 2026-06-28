# v0.6d Seed Expansion and Paper-Improvement Plan

> **For Hermes:** Use this as the next controlled experiment plan. Do not add new methods until the seed-1 tie-breaker is complete and documented.

## Goal

Determine whether the v0.6c `adaptive_stable` teacher is consistently competitive with or better than the v0.6b static reward schedule, rather than being a seed-0 accident.

## Current evidence from seed 0

v0.6c is a **moderate positive**:

- It clearly improves over naive adaptive v0.6b.
- It matches or slightly beats static v0.6b on several in-distribution metrics.
- It does not cleanly beat static on validation and OOD-division.
- OOD-long remains unsolved for all methods.

Seed-0 deltas for `adaptive_stable_v06c - static_v06b`:

| split | valid_expression | exact_correct | reward_hacking_candidate | allowed_numbers | number_f1 | allowed_ops |
|---|---:|---:|---:|---:|---:|---:|
| validation | -0.015 | +0.000 | +0.015 | -0.010 | +0.002 | +0.010 |
| test_in_dist | +0.005 | +0.005 | -0.005 | -0.015 | -0.001 | +0.025 |
| test_ood_long | +0.005 | +0.000 | -0.005 | -0.005 | -0.009 | +0.020 |
| test_ood_division | -0.010 | +0.000 | +0.010 | -0.010 | +0.007 | +0.010 |

For `reward_hacking_candidate`, lower is better, so negative deltas are good.

## Interpretation

The stability-constrained teacher did what it was designed to do mechanically:

```text
weight budget stayed fixed
constraint mass remained high
numeric_distance_reward was capped
teacher updates were smooth
```

Empirically, it moved the adaptive method from “worse than static” to “near-static / sometimes better.” That is paper-useful, but not yet a win.

## Hypothesis for seed expansion

If seed 1 shows the same pattern, the honest paper claim becomes:

> Stability constraints are necessary for adaptive RTW in verifier-based LLM post-training. Naive adaptive weighting underperforms static shaping, while stability-constrained adaptive weighting closes most of the gap and can improve selected in-distribution legality/reward-hacking metrics.

If seed 1 shows adaptive_stable beating static on most gates, the project can justify a full 3-seed comparison and a stronger claim.

If seed 1 shows adaptive_stable falling back below naive/static, the project should stop micro-tuning floors/caps and redesign teacher state.

## Non-goals

Do not do these before seed-1 tie-breaker:

- Do not add manual/random baselines.
- Do not add GACL/task curriculum.
- Do not change verifier semantics.
- Do not change dense v0.6b reward definitions.
- Do not change prompt templates or eval splits.
- Do not jump to coding-agent harnesses.

## Experiment matrix: v0.6d tie-breaker

Run exactly two systems first:

```text
static_v06b_seed1
adaptive_stable_v06c_seed1
```

Controlled variables:

```text
model: Qwen/Qwen2.5-0.5B-Instruct
dataset: data/countdown/*.jsonl
max_steps: 300
num_generations: 4
prompt_field: prompt
reward surface: v0.6b dense legality rewards
primary correctness: verifier exact_correct only
eval splits: validation, test_in_dist, test_ood_long, test_ood_division
```

Only changed variable:

```text
reward_strategy: static vs adaptive_stable
seed: 1
```

## Commands

### Static seed 1

```bash
RUN=outputs/grpo_static_v06b_dense_numbers_cuda_pilot_300_seed1
mkdir -p "$RUN"
WANDB_PROJECT=rtw-llm-countdown uv run python scripts/02_grpo_train.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --train_path data/countdown/train.jsonl \
  --eval_path data/countdown/validation.jsonl \
  --output_dir "$RUN" \
  --reward_strategy static \
  --seed 1 \
  --max_steps 300 \
  --num_generations 4 \
  --report_to none \
  2>&1 | tee "$RUN/train.log"

uv run python scripts/05_check_run_health.py \
  --run_dir "$RUN" \
  | tee "$RUN/health_final.txt"
```

### adaptive_stable seed 1

```bash
RUN=outputs/grpo_rtw_v06c_adaptive_stable_cuda_pilot_300_seed1
mkdir -p "$RUN"
WANDB_PROJECT=rtw-llm-countdown uv run python scripts/02_grpo_train.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --train_path data/countdown/train.jsonl \
  --eval_path data/countdown/validation.jsonl \
  --output_dir "$RUN" \
  --reward_strategy adaptive_stable \
  --seed 1 \
  --max_steps 300 \
  --num_generations 4 \
  --report_to none \
  2>&1 | tee "$RUN/train.log"

uv run python scripts/05_check_run_health.py \
  --run_dir "$RUN" \
  | tee "$RUN/health_final.txt"
```

### Eval fixed splits

For each run:

```bash
RUN=outputs/<run_dir>
CKPT=$(find "$RUN" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -n 1)
if [ -z "$CKPT" ]; then CKPT="$RUN"; fi

for SPLIT in validation test_in_dist test_ood_long test_ood_division; do
  uv run python scripts/03_eval.py \
    --model_name Qwen/Qwen2.5-0.5B-Instruct \
    --adapter_path "$CKPT" \
    --device cuda \
    --data_path "data/countdown/${SPLIT}.jsonl" \
    --output_dir "outputs/eval_<method>_300_seed1_${SPLIT}" \
    --batch_size 4 \
    --max_new_tokens 64
 done
```

Use method names:

```text
eval_static_v06b_300_seed1_${SPLIT}
eval_rtw_v06c_adaptive_stable_300_seed1_${SPLIT}
```

## Success gates

Primary seed-1 gates:

| Gate | adaptive_stable target |
|---|---|
| validation valid_expression | >= static - 0.01 |
| validation exact_correct | >= static |
| validation reward_hacking_candidate | <= static + 0.01 |
| test_in_dist valid_expression | >= static |
| test_in_dist reward_hacking_candidate | <= static |
| OOD-division valid_expression | >= static - 0.01 |
| OOD-long | no regression beyond 0.01; both are expected weak |

Outcome categories:

```text
Strong positive:
  adaptive_stable beats static on most validation/test_in_dist gates and ties exact_correct.

Moderate positive:
  adaptive_stable improves over naive adaptive pattern and stays within 0.01-0.02 of static.

Negative:
  adaptive_stable is materially worse than static on validation and test_in_dist.

Ambiguous:
  adaptive_stable improves legality but exact_correct drops, or improves ID while worsening OOD-division.
```

## Paper-improvement plan after seed 1

### If seed 1 is strong or moderate positive

Run full 3-seed comparison:

```text
static seeds 0,1,2
adaptive_stable seeds 0,1,2
```

Then add:

```text
adaptive naive seeds 0,1,2 only if needed for ablation narrative
manual/random only after static/adaptive_stable story is stable
```

Paper claim should be cautious:

> Stability-constrained adaptive reward weighting can match or improve static shaping on selected verifier-harness legality metrics, while naive adaptive weighting underperforms.

### If seed 1 is negative

Do not tune more floors/caps. Move to a better teacher design:

1. Add teacher state features for component improvement rates, not just EMA levels.
2. Penalize weight moves that reduce strict legality mass before primary correctness rises.
3. Treat numeric distance as conditional on high `number_multiset_f1` and `valid_expression` EMA.
4. Consider a two-phase teacher: legality-preservation phase, then numeric/correctness phase.

## Paper structure improvement

The paper should be written around the actual finding sequence:

1. **Problem:** LLM post-training happens inside harnesses; auxiliary rewards are brittle.
2. **Method:** RTW-style adaptive auxiliary reward teacher for verifier-based LLM tasks.
3. **Wind tunnel:** Countdown verifier harness with decomposed legality signals.
4. **Finding 1:** Dense legality rewards convert zero harness behavior into partial legality.
5. **Finding 2:** Naive adaptive weighting underperforms static due to teacher instability.
6. **Finding 3:** Stability constraints close much of the gap and may improve selected gates.
7. **Limit:** Exact correctness remains low; OOD-long remains unsolved.
8. **Next:** More robust teacher state, then GACL task curriculum.

This is a stronger paper than a forced success story because it exposes the design constraints needed for adaptive reward control to work.

## Done criteria for v0.6d

- [ ] static seed-1 training health exists.
- [ ] adaptive_stable seed-1 training health exists.
- [ ] four split evals exist for both methods.
- [ ] comparison table is added to this doc.
- [ ] decision is recorded: full 3-seed expansion vs teacher redesign.
- [ ] tests and ruff remain passing.
- [ ] results doc is committed.
