# v0.6b Legality Reward Tracker

## Objective

Move Countdown GRPO from harness-surface compliance into legal expression space.

The v0.6 runs showed:

```text
format / answer tags: learned
expression parseability: learned
allowed operators: mostly learned
number multiset legality: still the bottleneck
exact correctness: nonzero but too sparse
```

## Implementation

Verifier-owned correctness remains unchanged:

```text
exact_correct = 1 only when the expression passes the verifier
```

New dense auxiliary diagnostics:

```text
number_precision
number_recall
number_multiset_f1
uses_no_extra_numbers
uses_all_required_numbers
operator_precision
operator_recall
evaluates_without_exception
numeric_distance_reward
```

The default teacher auxiliary keys now focus one wheel directly on dense number
matching:

```text
format
valid_expression
number_multiset_f1
allowed_ops
numeric_distance_reward
brevity
```

## Guardrails

- Do not count an answer as correct unless `exact_correct == 1`.
- Keep `uses_allowed_numbers` binary for strict all-number verifier compliance.
- Keep `primary_reward`, `aux_reward_weighted`, and `total_reward` separate.
- Treat dense number rewards as training wheels only.

## Next validation sequence

```text
1. Run unit tests when approved.
2. Run adaptive v0.6b 50-step smoke.
3. Run adaptive v0.6b 300 seed0.
4. Run static v0.6b 300 seed0.
5. Compare allowed_numbers_rate, number_multiset_f1, valid_expression, exact_correct.
```

## Decision gate

Continue the baseline matrix only if v0.6b improves number legality:

```text
allowed_numbers_rate > previous 0-2% range
valid_expression_rate > previous 0-2% range
exact_correct does not regress to flat zero
reward_variance_nonzero_fraction stays nonzero
```

## Smoke result: adaptive v0.6b 50 seed0

Run:

```text
outputs/grpo_rtw_v06b_dense_numbers_cuda_smoke_50_seed0
```

Focused tests passed before CUDA:

```text
tests/test_countdown.py: 16 passed
```

Final health:

```text
reward_rows: 800
teacher_steps: 50
reward_variance_nonzero_fraction: 1.0
number_precision_mean: 0.1020
number_recall_mean: 0.0725
number_multiset_f1_mean: 0.0792
uses_no_extra_numbers_rate: 0.0738
uses_all_required_numbers_rate: 0.0288
allowed_numbers_rate: 0.0200
allowed_ops_rate: 0.0900
valid_expression_rate: 0.0175
exact_correct_rate: 0.0000
numeric_distance_reward_mean: 0.0093
```

Teacher movement:

```text
format: 0.2405 -> 0.1287
valid_expression: 0.2450 -> 0.3424
number_multiset_f1: 0.2441 -> 0.3211
allowed_ops: 0.2450 -> 0.3196
numeric_distance_reward: 0.2446 -> 0.3459
brevity: 0.2407 -> 0.1310
```

Interpretation:

```text
v0.6b successfully creates a live dense legality signal.
The reward is not flat, teacher weights move toward legality components, and
invented constants no longer receive numeric-distance or operator credit.
However, 50 steps do not prove the full claim yet because exact correctness is
still zero and binary allowed-number compliance remains around the old 0-2% band.
```

Next proof step:

```text
Run adaptive v0.6b 300 seed0 before static/manual/random.
Use the same comparison metrics:
allowed_numbers_rate
number_multiset_f1_mean
valid_expression_rate
exact_correct_rate
reward_hacking_candidate
```
## Adaptive v0.6b 300-step CUDA pilot, seed 0

Run:

```text
outputs/grpo_rtw_v06b_dense_numbers_cuda_pilot_300_seed0
```

Final health:

```text
reward rows: 4800
teacher rows: 300
reward_variance_nonzero_fraction: 1.0
parseable_expression_rate: 0.8000
allowed_numbers_rate: 0.2444
allowed_ops_rate: 0.5769
valid_expression_rate: 0.2190
exact_correct_rate: 0.0246
number_precision_mean: 0.6075
number_recall_mean: 0.5269
number_multiset_f1_mean: 0.5507
uses_no_extra_numbers_rate: 0.4908
uses_all_required_numbers_rate: 0.3058
evaluates_without_exception_rate: 0.6723
numeric_distance_reward_mean: 0.0683
tag_only_rate: 0.1256
correct_given_parseable: 0.0307
```

Teacher movement:

```text
format: 0.2405 -> 0.0227
valid_expression: 0.2450 -> 0.2236
number_multiset_f1: 0.2441 -> 0.0879
allowed_ops: 0.2450 -> 0.0707
numeric_distance_reward: 0.2446 -> 0.3161
brevity: 0.2407 -> 0.0230
```

Interpretation:

```text
The adaptive v0.6b 300-step run strongly supports the training-side claim that
dense legality rewards move the model into legal expression space. Compared
with the v0.6 failure mode, allowed-number compliance, number-multiset F1, valid
expression rate, and exact verifier correctness all moved off the floor.

This does not yet complete the held-out proof. The next step is checkpoint eval
on validation, in-distribution test, OOD-long, and OOD-division splits, followed
by a static v0.6b 300-step seed-0 baseline under the same harness.
```
## Adaptive v0.6b held-out eval, checkpoint 300

Checkpoint:

```text
outputs/grpo_rtw_v06b_dense_numbers_cuda_pilot_300_seed0/checkpoint-300
```

Eval outputs:

```text
outputs/eval_rtw_v06b_300_seed0_validation
outputs/eval_rtw_v06b_300_seed0_test_in_dist
outputs/eval_rtw_v06b_300_seed0_test_ood_long
outputs/eval_rtw_v06b_300_seed0_test_ood_division
```

Summary:

```text
split              n    parseable  allowed_numbers  number_f1  valid_expression  exact_correct  reward_hacking_candidate
validation         200  0.9350     0.3700           0.8091     0.3450            0.0350         0.6550
test_in_dist       200  0.9850     0.3900           0.8403     0.3550            0.0300         0.6450
test_ood_long      200  1.0000     0.0500           0.6974     0.0500            0.0150         0.9500
test_ood_division  200  1.0000     0.2300           0.8269     0.2300            0.0250         0.7700
```

Interpretation:

```text
The held-out eval supports the v0.6b claim. The checkpoint preserves strict
verifier semantics while moving substantially into legal expression space on
validation and in-distribution test. OOD-long remains the weakest split: the
model is parseable and often uses partial legal number structure, but full
required-number use and valid expression rate collapse under longer tasks.

The next comparison should be static v0.6b 300 seed 0 under the same code,
model, dataset, seed, max steps, and generation settings. Do not patch
termination or max length before static, or the adaptive/static comparison will
be confounded.
```
## Static v0.6b 300-step baseline, seed 0

Run:

```text
outputs/grpo_static_v06b_dense_numbers_cuda_pilot_300_seed0
```

Final training health:

```text
reward rows: 4800
teacher rows: 300
reward_variance_nonzero_fraction: 1.0
parseable_expression_rate: 0.7808
allowed_numbers_rate: 0.2444
allowed_ops_rate: 0.5575
valid_expression_rate: 0.2248
exact_correct_rate: 0.0296
number_precision_mean: 0.5758
number_recall_mean: 0.5071
number_multiset_f1_mean: 0.5268
uses_no_extra_numbers_rate: 0.4585
uses_all_required_numbers_rate: 0.2998
evaluates_without_exception_rate: 0.6704
numeric_distance_reward_mean: 0.0710
tag_only_rate: 0.1456
correct_given_parseable: 0.0379
```

Held-out eval:

```text
split              n    parseable  allowed_numbers  number_f1  valid_expression  exact_correct  reward_hacking_candidate
validation         200  0.9600     0.4400           0.8531     0.4150            0.0450         0.5850
test_in_dist       200  0.9800     0.4550           0.8674     0.4150            0.0250         0.5850
test_ood_long      200  1.0000     0.0550           0.7327     0.0450            0.0150         0.9550
test_ood_division  200  0.9900     0.3050           0.8399     0.3050            0.0300         0.6950
```

Interpretation:

```text
Static v0.6b confirms that the dense legality reward surface is the main driver
of the v0.6 -> v0.6b improvement. Adaptive RTW does not beat static on seed 0.
Static is better on validation, in-distribution legality, and OOD-division.
The next method work should target teacher stability rather than adding more
baselines immediately.
```
