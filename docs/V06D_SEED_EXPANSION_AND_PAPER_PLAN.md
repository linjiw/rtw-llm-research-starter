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

## Interpretation before seed 1

The stability-constrained teacher did what it was designed to do mechanically:

```text
weight budget stayed fixed
constraint mass remained high
numeric_distance_reward was capped
teacher updates were smooth
```

Empirically, it moved the adaptive method from “worse than static” to “near-static / sometimes better.” That was paper-useful, but not yet a win.

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

## Seed-1 training health results

### Static seed 1

Run:

```text
outputs/grpo_static_v06b_dense_numbers_cuda_pilot_300_seed1
```

Health artifact:

```text
outputs/grpo_static_v06b_dense_numbers_cuda_pilot_300_seed1/health_final.txt
```

Training health:

```text
reward rows: 4800
teacher rows: 300
reward_variance_nonzero_fraction: 1.0
parseable_expression_rate: 0.7944
allowed_numbers_rate: 0.2248
allowed_ops_rate: 0.5869
valid_expression_rate: 0.2077
exact_correct_rate: 0.0254
number_multiset_f1_mean: 0.5408
issues: none
```

### adaptive_stable seed 1

Run:

```text
outputs/grpo_rtw_v06c_adaptive_stable_cuda_pilot_300_seed1
```

Health artifact:

```text
outputs/grpo_rtw_v06c_adaptive_stable_cuda_pilot_300_seed1/health_final.txt
```

Training health:

```text
reward rows: 4800
teacher rows: 300
reward_variance_nonzero_fraction: 1.0
parseable_expression_rate: 0.8054
allowed_numbers_rate: 0.2438
allowed_ops_rate: 0.5821
valid_expression_rate: 0.2215
exact_correct_rate: 0.0292
number_multiset_f1_mean: 0.5524
issues: none
```

Final adaptive_stable teacher mechanics:

```text
last format:                  0.1594
last valid_expression:        0.2791
last number_multiset_f1:      0.2044
last allowed_ops:             0.1977
last numeric_distance_reward: 0.2000
last brevity:                 0.1595
weight_sum_final:             1.2000
constraint_weight_mass_final: 0.6811
numeric_distance_ratio_final: 0.2936
teacher_update_l1_mean:       0.0011
numeric_distance cap hit rate: 0.7633
```

## Seed-1 held-out evaluation

| split | method | parse_ok | number_f1 | allowed_numbers | no_extra_numbers | all_required_numbers | allowed_ops | valid_expression | exact_correct | reward_hacking_candidate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | static_v06b | 0.955 | 0.276 | 0.060 | 0.315 | 0.070 | 0.300 | 0.050 | 0.005 | 0.915 |
| validation | adaptive_stable_v06c | 0.950 | 0.820 | 0.370 | 0.830 | 0.395 | 0.885 | 0.350 | 0.040 | 0.650 |
| test_in_dist | static_v06b | 0.920 | 0.291 | 0.045 | 0.280 | 0.055 | 0.330 | 0.045 | 0.005 | 0.885 |
| test_in_dist | adaptive_stable_v06c | 0.985 | 0.850 | 0.380 | 0.860 | 0.390 | 0.910 | 0.360 | 0.015 | 0.640 |
| test_ood_long | static_v06b | 1.000 | 0.626 | 0.050 | 0.675 | 0.075 | 0.850 | 0.050 | 0.010 | 0.950 |
| test_ood_long | adaptive_stable_v06c | 0.995 | 0.708 | 0.055 | 0.810 | 0.080 | 0.950 | 0.050 | 0.015 | 0.950 |
| test_ood_division | static_v06b | 0.995 | 0.765 | 0.235 | 0.750 | 0.245 | 0.890 | 0.235 | 0.025 | 0.760 |
| test_ood_division | adaptive_stable_v06c | 1.000 | 0.847 | 0.255 | 0.830 | 0.285 | 0.980 | 0.255 | 0.025 | 0.745 |

Seed-1 deltas for `adaptive_stable_v06c - static_v06b`:

| split | valid_expression | exact_correct | reward_hacking_candidate | allowed_numbers | number_f1 | allowed_ops |
|---|---:|---:|---:|---:|---:|---:|
| validation | +0.300 | +0.035 | -0.265 | +0.310 | +0.544 | +0.585 |
| test_in_dist | +0.315 | +0.010 | -0.245 | +0.335 | +0.559 | +0.580 |
| test_ood_long | +0.000 | +0.005 | +0.000 | +0.005 | +0.081 | +0.100 |
| test_ood_division | +0.020 | +0.000 | -0.015 | +0.020 | +0.082 | +0.090 |

For `reward_hacking_candidate`, lower is better.

## Two-seed aggregate: static vs adaptive_stable

Mean ± population standard deviation across seeds 0 and 1:

### Validation

| method | valid_expression | exact_correct | reward_hacking_candidate | allowed_numbers | number_f1 | allowed_ops |
|---|---:|---:|---:|---:|---:|---:|
| static_v06b | 0.232±0.182 | 0.025±0.020 | 0.750±0.165 | 0.250±0.190 | 0.565±0.289 | 0.593±0.292 |
| adaptive_stable_v06c | 0.375±0.025 | 0.042±0.002 | 0.625±0.025 | 0.400±0.030 | 0.837±0.017 | 0.890±0.005 |

### Test in-distribution

| method | valid_expression | exact_correct | reward_hacking_candidate | allowed_numbers | number_f1 | allowed_ops |
|---|---:|---:|---:|---:|---:|---:|
| static_v06b | 0.230±0.185 | 0.015±0.010 | 0.735±0.150 | 0.250±0.205 | 0.579±0.288 | 0.610±0.280 |
| adaptive_stable_v06c | 0.390±0.030 | 0.022±0.007 | 0.610±0.030 | 0.410±0.030 | 0.858±0.008 | 0.913±0.003 |

### OOD-long

| method | valid_expression | exact_correct | reward_hacking_candidate | allowed_numbers | number_f1 | allowed_ops |
|---|---:|---:|---:|---:|---:|---:|
| static_v06b | 0.048±0.003 | 0.013±0.002 | 0.952±0.003 | 0.053±0.002 | 0.680±0.053 | 0.900±0.050 |
| adaptive_stable_v06c | 0.050±0.000 | 0.015±0.000 | 0.950±0.000 | 0.053±0.002 | 0.716±0.008 | 0.960±0.010 |

### OOD-division

| method | valid_expression | exact_correct | reward_hacking_candidate | allowed_numbers | number_f1 | allowed_ops |
|---|---:|---:|---:|---:|---:|---:|
| static_v06b | 0.270±0.035 | 0.028±0.002 | 0.728±0.033 | 0.270±0.035 | 0.802±0.038 | 0.927±0.037 |
| adaptive_stable_v06c | 0.275±0.020 | 0.028±0.002 | 0.725±0.020 | 0.275±0.020 | 0.847±0.000 | 0.978±0.003 |

## v0.6d decision

Outcome category: **strong positive for seed-1 tie-breaker; proceed to full 3-seed comparison.**

The seed-1 result is stronger than seed 0 because static_v06b collapsed on validation and in-distribution legality while adaptive_stable_v06c remained stable. This suggests the stability-constrained teacher is not merely matching static; it may reduce seed sensitivity by preserving legality pressure and total auxiliary budget.

However, the paper claim should still remain cautious until seed 2 completes:

> Stability-constrained adaptive reward weighting improves robustness of verifier-harness legality acquisition compared with static shaping in this two-seed pilot, while exact correctness remains low and OOD-long remains unsolved.

## Next experiment: finish 3-seed comparison

Run:

```text
static_v06b_seed2
adaptive_stable_v06c_seed2
```

Use the same commands as seed 1 with `--seed 2` and output dirs:

```text
outputs/grpo_static_v06b_dense_numbers_cuda_pilot_300_seed2
outputs/grpo_rtw_v06c_adaptive_stable_cuda_pilot_300_seed2
outputs/eval_static_v06b_300_seed2_${SPLIT}
outputs/eval_rtw_v06c_adaptive_stable_300_seed2_${SPLIT}
```

If seed 2 is consistent with seed 1 or the two-seed mean, the paper can use a three-seed main table for static vs adaptive_stable and keep naive adaptive as a diagnostic ablation.

## Paper-improvement plan after seed 1

### Current paper claim after seed 1

The paper should now shift from “adaptive closes the gap” to a stronger but still careful claim:

> A naive adaptive RTW teacher underperforms static shaping, but a stability-constrained adaptive teacher improves robustness and reduces seed sensitivity in verifier-based LLM harness acquisition.

This is more compelling than the seed-0-only story because it adds a stability/variance argument, not only a mean-performance argument.

### If seed 2 confirms

Run or report:

```text
static seeds 0,1,2
adaptive_stable seeds 0,1,2
naive adaptive seed 0 as failure-mode ablation
```

Then add manual/random only if needed for reviewer-facing completeness.

### If seed 2 contradicts

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
6. **Finding 3:** Stability-constrained adaptive weighting improves robustness and may reduce seed sensitivity.
7. **Limit:** Exact correctness remains low; OOD-long remains unsolved.
8. **Next:** More robust teacher state, then GACL task curriculum.

This is a stronger paper than a forced success story because it exposes the design constraints needed for adaptive reward control to work.



## Seed-2 held-out evaluation

| split | method | valid_expression | exact_correct | reward_hacking_candidate | allowed_numbers | number_f1 | allowed_ops |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | static_v06b | 0.290 | 0.035 | 0.710 | 0.315 | 0.855 | 0.895 |
| validation | adaptive_stable_v06c | 0.430 | 0.055 | 0.570 | 0.435 | 0.845 | 0.920 |
| test_in_dist | static_v06b | 0.320 | 0.060 | 0.680 | 0.360 | 0.865 | 0.895 |
| test_in_dist | adaptive_stable_v06c | 0.420 | 0.015 | 0.580 | 0.430 | 0.860 | 0.925 |
| test_ood_long | static_v06b | 0.065 | 0.015 | 0.935 | 0.070 | 0.721 | 0.945 |
| test_ood_long | adaptive_stable_v06c | 0.035 | 0.015 | 0.965 | 0.035 | 0.726 | 0.970 |
| test_ood_division | static_v06b | 0.370 | 0.025 | 0.630 | 0.370 | 0.879 | 0.995 |
| test_ood_division | adaptive_stable_v06c | 0.245 | 0.030 | 0.755 | 0.245 | 0.847 | 0.970 |

For `reward_hacking_candidate`, lower is better.

## Final 3-seed aggregate: static vs adaptive_stable

Mean ± population standard deviation across seeds 0, 1, and 2.

### Validation

| method | valid_expression | exact_correct | reward_hacking_candidate | allowed_numbers | number_f1 | allowed_ops |
|---|---:|---:|---:|---:|---:|---:|
| static_v06b | 0.252±0.151 | 0.028±0.017 | 0.737±0.136 | 0.272±0.158 | 0.661±0.272 | 0.693±0.278 |
| adaptive_stable_v06c | 0.393±0.033 | 0.047±0.006 | 0.607±0.033 | 0.412±0.030 | 0.840±0.015 | 0.900±0.015 |

### Test in-distribution

| method | valid_expression | exact_correct | reward_hacking_candidate | allowed_numbers | number_f1 | allowed_ops |
|---|---:|---:|---:|---:|---:|---:|
| static_v06b | 0.260±0.157 | 0.030±0.023 | 0.717±0.125 | 0.287±0.175 | 0.675±0.271 | 0.705±0.265 |
| adaptive_stable_v06c | 0.400±0.028 | 0.020±0.007 | 0.600±0.028 | 0.417±0.026 | 0.859±0.007 | 0.917±0.006 |

### OOD-long

| method | valid_expression | exact_correct | reward_hacking_candidate | allowed_numbers | number_f1 | allowed_ops |
|---|---:|---:|---:|---:|---:|---:|
| static_v06b | 0.053±0.008 | 0.013±0.002 | 0.947±0.008 | 0.058±0.008 | 0.693±0.048 | 0.915±0.046 |
| adaptive_stable_v06c | 0.045±0.007 | 0.015±0.000 | 0.955±0.007 | 0.047±0.008 | 0.719±0.008 | 0.963±0.009 |

### OOD-division

| method | valid_expression | exact_correct | reward_hacking_candidate | allowed_numbers | number_f1 | allowed_ops |
|---|---:|---:|---:|---:|---:|---:|
| static_v06b | 0.303±0.055 | 0.027±0.002 | 0.695±0.053 | 0.303±0.055 | 0.828±0.047 | 0.950±0.044 |
| adaptive_stable_v06c | 0.265±0.022 | 0.028±0.002 | 0.735±0.022 | 0.265±0.022 | 0.847±0.000 | 0.975±0.004 |

## Final v0.6d decision

Outcome category: **positive but scoped**.

The 3-seed result supports the central paper point that stability-constrained adaptive reward weighting is more robust than static shaping for in-distribution legality acquisition. It does **not** support a blanket claim that adaptive_stable dominates static on every split.

Main positives:

```text
validation valid_expression:        +0.142 mean delta
validation exact_correct:           +0.018 mean delta
validation reward_hacking_candidate -0.130 mean delta
in-dist valid_expression:           +0.140 mean delta
in-dist reward_hacking_candidate:   -0.117 mean delta
```

Main caveats:

```text
in-dist exact_correct is lower for adaptive_stable by -0.010 mean delta
OOD-long remains unsolved and slightly worse on legality/reward-hacking
OOD-division favors static on valid_expression/reward_hacking_candidate
exact_correct remains low overall
```

The paper should therefore claim:

> Stability-constrained adaptive reward weighting substantially improves in-distribution legality robustness and reduces seed sensitivity relative to static shaping in a verifier-based Countdown harness, but it does not solve exact correctness or OOD generalization.

## Recommended next paper step

Do **not** add more reward micro-tuning before writing the paper skeleton. The result is now coherent enough for a paper draft:

1. Use static vs adaptive_stable 3-seed tables as the main result.
2. Use naive adaptive seed 0 as the failure-mode ablation showing why teacher stabilization matters.
3. Include teacher-weight trajectories as mechanistic evidence.
4. Treat OOD-long and OOD-division as honest limitations.
5. Put GACL / curriculum as future work unless a separate, controlled experiment is started.

If one more experiment is needed before drafting, prefer a **diagnostic prompt/length ablation** over another teacher-weight tweak, because completions remain max-length clipped and exact correctness is still low.

## Done criteria for v0.6d

- [x] static seed-1 training health exists.
- [x] adaptive_stable seed-1 training health exists.
- [x] four split evals exist for both methods.
- [x] comparison table is added to this doc.
- [x] decision is recorded: full 3-seed expansion vs teacher redesign.
- [x] tests and ruff remain passing.
- [x] results doc is committed.
