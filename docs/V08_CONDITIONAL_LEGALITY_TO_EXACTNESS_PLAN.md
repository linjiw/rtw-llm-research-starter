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
adaptive_phased
```

Keep old strategies unchanged:

```text
adaptive
adaptive_stable
static
manual
random
```

### Phase logic

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
condition: number_multiset_f1 EMA >= 0.80 AND valid_expression EMA >= 0.35
behavior:
  preserve legality floors
  allow numeric_distance_reward up to 0.25
  increase exactness-oriented pressure only after legal expression construction is stable
```

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

- [ ] `adaptive_phased` implemented without changing existing strategies.
- [ ] tests added and passing.
- [ ] 100-step smoke completed and health-checked.
- [ ] 300-step pilot completed only if smoke passes.
- [ ] fixed evals completed.
- [ ] failure taxonomy compared to v0.6d.
- [ ] decision recorded and committed.
