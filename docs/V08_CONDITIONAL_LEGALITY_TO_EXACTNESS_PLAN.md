# v0.8 Conditional Legality-to-Exactness Plan

> **For Hermes:** This is the next implementation plan after v0.7A. Do not start GPU training until the implementation is committed and local tests pass.

## Goal

Improve the paper and experiment by targeting the failure modes that v0.7A actually found, rather than continuing prompt/length tuning.

The dominant seed2 failure buckets are:

```text
missing_required_number
legal_but_wrong_value
illegal_extra_or_repeated_number
```

Therefore v0.8 should test whether a conditional training curriculum over auxiliary reward weights can move the model from:

```text
parseable expression -> complete legal number multiset -> exact target value
```

without changing the strict verifier definition.

## Why v0.8 is the right next experiment

v0.7A showed that simple inference-time prompt/length changes do not improve the Stable-RTW seed2 validation or in-distribution metrics:

```text
prompt_high vs prompt: no metric change
32 vs 64 max_new_tokens: no metric change
```

The model is not failing because evaluation uses 64 tokens instead of 32. It is failing because many completions either miss required numbers or produce legal-but-wrong expressions.

## Hypothesis

A conditional teacher that emphasizes number-multiset completion until number legality is reliable, then increases target-distance/exactness pressure, will reduce the two largest failure buckets:

```text
missing_required_number
legal_but_wrong_value
```

This should improve exact correctness more directly than more prompt or length tuning.

## Non-goals

Do not claim v0.8 replaces the current paper result unless it passes a controlled comparison.

Do not change:

- verifier semantics;
- dataset splits;
- model;
- base reward metrics;
- primary exact_correct definition;
- v0.6d static/adaptive_stable archival results.

## Proposed implementation

Add a new reward strategy:

```text
adaptive_phased  # paper name: Phased-RTW
```

Keep old strategies unchanged:

```text
adaptive
adaptive_stable
static
manual
random
```

### Phase logic with hysteresis

Use moving-average diagnostics already available to the teacher.

Phase A: legality acquisition

```text
condition: number_multiset_f1 EMA < 0.80 OR valid_expression EMA < 0.35
behavior:
  keep high minimum mass on number_multiset_f1 and valid_expression
  keep numeric_distance_reward capped at 0.15-0.20
  do not increase target/distance pressure aggressively
```

Phase B: exactness pressure

```text
enter condition: number_multiset_f1 EMA >= 0.80 AND valid_expression EMA >= 0.35
behavior:
  preserve legality floors
  allow numeric_distance_reward up to 0.25
  increase exactness-oriented pressure only after legal expression construction is stable
```



Hysteresis:

```text
enter Phase B only after the enter condition holds for K teacher updates
return to Phase A only if:
  number_multiset_f1 EMA < 0.75 OR valid_expression EMA < 0.30
  for K teacher updates
```

This prevents phase oscillation and makes the experiment interpretable.

Important: exact correctness remains the primary verifier reward in both phases.

### Diagnostics to log

Add to teacher weights log or health report:

```text
teacher_phase
phase_transition_step
phase_a_fraction
phase_b_fraction
missing_required_number_rate_proxy
legal_but_wrong_rate_proxy
```

If direct failure taxonomy is too expensive during training, compute those proxies from existing components:

```text
missing_required_number_proxy = parse_ok and uses_no_extra_numbers and not uses_all_required_numbers
legal_but_wrong_proxy = valid_expression and not exact_correct
```

## Tests

Add unit tests for:

1. `adaptive_phased` is accepted as a reward strategy.
2. Phase A keeps numeric distance capped and preserves number/valid-expression weights.
3. Phase B can increase numeric-distance pressure only after legality thresholds are met.
4. Existing `adaptive_stable` behavior is unchanged.
5. Health report includes phase diagnostics if phase logs exist.

## Experiment order

### Step 1: local tests

```bash
uv run pytest -q
uv run ruff check .
```

### Step 2: 100-step smoke

```bash
RUN=outputs/grpo_rtw_v08_adaptive_phased_cuda_smoke_100_seed0
rm -rf "$RUN"
mkdir -p "$RUN"
WANDB_PROJECT=rtw-llm-countdown uv run python scripts/02_grpo_train.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --train_path data/countdown/train.jsonl \
  --eval_path data/countdown/validation.jsonl \
  --output_dir "$RUN" \
  --reward_strategy adaptive_phased \
  --seed 0 \
  --max_steps 100 \
  --num_generations 4 \
  --report_to none \
  2>&1 | tee "$RUN/train.log"

uv run python scripts/05_check_run_health.py \
  --run_dir "$RUN" \
  | tee "$RUN/health_final.txt"
```

Smoke gate:

```text
reward_variance_nonzero_fraction == 1.0
issues == []
weight_sum_final near 1.20
numeric_distance_to_constraint_ratio below 0.45
phase diagnostics present
```

### Step 3: 300-step pilot only if smoke passes

```bash
RUN=outputs/grpo_rtw_v08_adaptive_phased_cuda_pilot_300_seed0
rm -rf "$RUN"
mkdir -p "$RUN"
WANDB_PROJECT=rtw-llm-countdown uv run python scripts/02_grpo_train.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --train_path data/countdown/train.jsonl \
  --eval_path data/countdown/validation.jsonl \
  --output_dir "$RUN" \
  --reward_strategy adaptive_phased \
  --seed 0 \
  --max_steps 300 \
  --num_generations 4 \
  --report_to none \
  2>&1 | tee "$RUN/train.log"
```

### Step 4: fixed eval splits

```bash
for SPLIT in validation test_in_dist test_ood_long test_ood_division; do
  uv run python scripts/03_eval.py \
    --model_name Qwen/Qwen2.5-0.5B-Instruct \
    --adapter_path outputs/grpo_rtw_v08_adaptive_phased_cuda_pilot_300_seed0/checkpoint-300 \
    --device cuda \
    --data_path "data/countdown/${SPLIT}.jsonl" \
    --output_dir "outputs/eval_rtw_v08_adaptive_phased_300_seed0_${SPLIT}" \
    --batch_size 4 \
    --max_new_tokens 64
 done
```

### Step 5: failure taxonomy

```bash
uv run python scripts/06_failure_taxonomy.py \
  outputs/eval_rtw_v08_adaptive_phased_300_seed0_validation/generations.jsonl \
  outputs/eval_rtw_v08_adaptive_phased_300_seed0_test_in_dist/generations.jsonl \
  --output outputs/v08_adaptive_phased_seed0_failure_taxonomy.json
```

## Decision gates

Compare v0.8 seed0 against adaptive_stable v0.6c seed0 and static seed0.

Primary win condition:

```text
validation exact_correct improves by >= +0.02 absolute
AND validation valid_expression does not drop by more than -0.03
AND missing_required_number failure rate decreases
AND reward_hacking_candidate does not increase by more than +0.03
AND number_multiset_f1 does not drop by more than -0.03
AND reward_variance_nonzero_fraction remains healthy
```

Secondary win condition:

```text
test_in_dist exact_correct improves or stays flat
AND reward_hacking_candidate does not worsen by more than +0.03
```

Negative outcome:

```text
valid_expression drops materially
OR reward_hacking_candidate increases
OR exact_correct does not improve while legality worsens
```

If v0.8 is negative, do not tune more teacher knobs immediately. Move to curriculum/search as the next paper-future-work branch.

## Paper integration

If v0.8 is positive, report it as a follow-up diagnostic improvement, not as the main result unless it is expanded to 3 seeds.

If v0.8 is negative or mixed, the paper is still strong:

```text
Stable-RTW solves legal-action-space robustness better than static shaping, but exact target search requires a separate mechanism.
```

## Done criteria

- [x] `adaptive_phased` implemented without changing existing strategies.
- [x] tests added and passing.
- [ ] 100-step smoke completed and health-checked.
- [ ] 300-step pilot completed only if smoke passes.
- [ ] fixed evals completed.
- [ ] failure taxonomy compared to v0.6d.
- [ ] decision recorded and committed.


## v0.8 seed0 300-step pilot result

Status: **completed and evaluated**.

Run:

```text
outputs/grpo_rtw_v08_adaptive_phased_cuda_pilot_300_seed0
```

Health artifact:

```text
outputs/grpo_rtw_v08_adaptive_phased_cuda_pilot_300_seed0/health_final.txt
```

Eval artifacts:

```text
outputs/eval_rtw_v08_adaptive_phased_300_seed0_validation
outputs/eval_rtw_v08_adaptive_phased_300_seed0_test_in_dist
outputs/eval_rtw_v08_adaptive_phased_300_seed0_test_ood_long
outputs/eval_rtw_v08_adaptive_phased_300_seed0_test_ood_division
```

Failure taxonomy comparison:

```text
outputs/v08_failure_taxonomy_seed0_compare.json
```

### Training health

| metric | value |
|---|---:|
| reward rows | 4800 |
| teacher rows | 300 |
| reward_variance_nonzero_fraction | 1.0000 |
| parseable_expression_rate | 0.7979 |
| allowed_numbers_rate | 0.2554 |
| allowed_ops_rate | 0.5840 |
| valid_expression_rate | 0.2340 |
| exact_correct_rate | 0.0285 |
| number_multiset_f1_mean | 0.5515 |
| issues | none |

Teacher diagnostics:

| metric | value |
|---|---:|
| phase_final | B |
| phase_a_fraction | 0.94 |
| phase_b_fraction | 0.06 |
| phase_switch_step_final | 282 |
| phase_flip_count_final | 1 |
| weight_sum_final | 1.2000 |
| constraint_weight_mass_final | 0.7135 |
| numeric_distance_weight_final | 0.1924 |
| numeric_distance_to_constraint_ratio_final | 0.2697 |
| teacher_update_l1_mean | 0.0018 |

Interpretation: Phase B **did activate**, but only very late, at teacher step 282. Therefore this run mostly tested a stronger legality-first teacher, with only a short exactness-pressure tail.

### Four-split evaluation: seed0

| split | method | valid_expression | exact_correct | reward_hacking_candidate | allowed_numbers | number_f1 | allowed_ops | uses_all_required | uses_no_extra |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | static_v06b | 0.415 | 0.045 | 0.585 | 0.440 | 0.853 | 0.885 | 0.470 | 0.835 |
| validation | Stable-RTW | 0.400 | 0.045 | 0.600 | 0.430 | 0.855 | 0.895 | 0.455 | 0.860 |
| validation | Phased-RTW | 0.370 | 0.045 | 0.630 | 0.395 | 0.835 | 0.880 | 0.425 | 0.835 |
| test_in_dist | static_v06b | 0.415 | 0.025 | 0.585 | 0.455 | 0.867 | 0.890 | 0.495 | 0.805 |
| test_in_dist | Stable-RTW | 0.420 | 0.030 | 0.580 | 0.440 | 0.866 | 0.915 | 0.450 | 0.840 |
| test_in_dist | Phased-RTW | 0.380 | 0.025 | 0.620 | 0.400 | 0.847 | 0.890 | 0.410 | 0.825 |
| test_ood_long | static_v06b | 0.045 | 0.015 | 0.955 | 0.055 | 0.733 | 0.950 | 0.075 | 0.750 |
| test_ood_long | Stable-RTW | 0.050 | 0.015 | 0.950 | 0.050 | 0.724 | 0.970 | 0.065 | 0.795 |
| test_ood_long | Phased-RTW | 0.030 | 0.015 | 0.970 | 0.030 | 0.715 | 0.965 | 0.045 | 0.825 |
| test_ood_division | static_v06b | 0.305 | 0.030 | 0.695 | 0.305 | 0.840 | 0.965 | 0.330 | 0.715 |
| test_ood_division | Stable-RTW | 0.295 | 0.030 | 0.705 | 0.295 | 0.847 | 0.975 | 0.325 | 0.770 |
| test_ood_division | Phased-RTW | 0.225 | 0.025 | 0.775 | 0.225 | 0.837 | 0.980 | 0.245 | 0.810 |

### Phased-RTW delta vs Stable-RTW

| split | Δ valid_expression | Δ exact_correct | Δ reward_hacking_candidate | Δ allowed_numbers | Δ number_f1 | Δ uses_all_required |
|---|---:|---:|---:|---:|---:|---:|
| validation | -0.030 | +0.000 | +0.030 | -0.035 | -0.020 | -0.030 |
| test_in_dist | -0.040 | -0.005 | +0.040 | -0.040 | -0.019 | -0.040 |
| test_ood_long | -0.020 | +0.000 | +0.020 | -0.020 | -0.009 | -0.020 |
| test_ood_division | -0.070 | -0.005 | +0.070 | -0.070 | -0.010 | -0.080 |

For reward_hacking_candidate, positive deltas are worse.

### Failure taxonomy: in-distribution splits

| split | method | missing_required_number | illegal_extra_or_repeated_number | legal_but_wrong_value | exact_correct |
|---|---|---:|---:|---:|---:|
| validation | static_v06b | 0.395 | 0.125 | 0.370 | 0.045 |
| validation | Stable-RTW | 0.430 | 0.105 | 0.355 | 0.045 |
| validation | Phased-RTW | 0.440 | 0.115 | 0.325 | 0.045 |
| test_in_dist | static_v06b | 0.350 | 0.175 | 0.390 | 0.025 |
| test_in_dist | Stable-RTW | 0.400 | 0.155 | 0.390 | 0.030 |
| test_in_dist | Phased-RTW | 0.425 | 0.155 | 0.355 | 0.025 |

### Decision against pre-registered v0.8 gate

Gate:

```text
exact_correct +0.02 absolute or better
valid_expression drop <= 0.03
number_multiset_f1 drop <= 0.03
missing_required_number decreases
reward_hacking_candidate does not materially worsen
reward_variance_nonzero_fraction remains healthy
```

Outcome: **gate failed**.

Evidence:

- exact_correct did not improve over Stable-RTW on validation: `+0.000`.
- exact_correct was worse on test_in_dist and test_ood_division: `-0.005`.
- validation valid_expression drop was at the allowed boundary: `-0.030`.
- test_in_dist and test_ood_division valid_expression dropped beyond the gate: `-0.040`, `-0.070`.
- missing_required_number did not decrease; it increased on validation and test_in_dist vs Stable-RTW.
- reward_hacking_candidate worsened on every split.
- reward variance remained healthy.

### v0.8 case classification

This is closest to **Case 2 / Case 3 hybrid**:

```text
Phase B activated late.
Legality mostly did not hold relative to Stable-RTW.
Exact correctness stayed flat or slightly worsened.
```

Because Phase B activated only for the final 6% of teacher updates, the clean interpretation is:

> Phased-RTW as configured mostly behaves as a stricter legality-first teacher. It reaches the exactness phase late, but the short Phase-B tail does not improve exact correctness and slightly weakens held-out legality/reward-hacking metrics relative to Stable-RTW.

### Scientific conclusion

Do **not** promote Phased-RTW to the main method.

This pilot is useful as a negative causal probe:

> Scalar reward phasing alone does not convert Stable-RTW's legal action-space acquisition into exact target success under this 300-step budget. The remaining exactness bottleneck likely requires mechanisms beyond reward weighting, such as verifier-guided candidate search, best-of-n reranking, exact-solver trace SFT, or curriculum over expression length/target distance.

### Recommended next step

Stop reward-teacher micro-tuning for the main paper. Keep Stable-RTW as the main contribution and use v0.8 as a limitation/follow-up probe.

If we run one more experiment, it should not be another scalar reward schedule. The best next controlled experiment is:

```text
verifier-guided best-of-n reranking over Stable-RTW generations
```

Reason: the model often generates parseable, high-number-F1, legal-but-wrong or near-legal expressions. Reranking/search tests whether the learned policy has useful candidates latent in its distribution that a harness-level verifier can select, without changing training or redefining correctness.
