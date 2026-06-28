# v0.6b Adaptive vs Static Seed 0 Report

## Question

Does adaptive RTW add value beyond the improved v0.6b dense legality reward surface?

## Runs

```text
adaptive:
  outputs/grpo_rtw_v06b_dense_numbers_cuda_pilot_300_seed0

static:
  outputs/grpo_static_v06b_dense_numbers_cuda_pilot_300_seed0
```

Both runs used:

```text
model: Qwen/Qwen2.5-0.5B-Instruct
seed: 0
max_steps: 300
num_generations: 4
reward surface: v0.6b dense legality rewards
```

## Training health

| method | primary_reward_mean | aux_reward_weighted_mean | reward_variance_nonzero_fraction | number_multiset_f1_mean | allowed_numbers_rate | valid_expression_rate | exact_correct_rate | final weights |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| adaptive | 0.0246 | 0.3158 | 1.0000 | 0.5507 | 0.2444 | 0.2190 | 0.0246 | format 0.0227, valid 0.2236, number_f1 0.0879, ops 0.0707, distance 0.3161, brevity 0.0230 |
| static | 0.0296 | 0.6474 | 1.0000 | 0.5268 | 0.2444 | 0.2248 | 0.0296 | all components 0.2000 |

Training interpretation:

```text
Both runs learned legality and retained nonzero reward variance.
Static slightly exceeded adaptive on training exact correctness and valid-expression rate.
Adaptive had slightly higher number-multiset F1, but its final teacher weights heavily
suppressed format, brevity, number_f1, and allowed_ops while emphasizing numeric distance.
```

## Held-out eval

| split | method | parseable | number_multiset_f1 | allowed_numbers | uses_no_extra_numbers | uses_all_required_numbers | allowed_ops | valid_expression | exact_correct | reward_hacking_candidate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | adaptive | 0.9350 | 0.8091 | 0.3700 | 0.8050 | 0.3950 | 0.8600 | 0.3450 | 0.0350 | 0.6550 |
| validation | static | 0.9600 | 0.8531 | 0.4400 | 0.8350 | 0.4700 | 0.8850 | 0.4150 | 0.0450 | 0.5850 |
| test_in_dist | adaptive | 0.9850 | 0.8403 | 0.3900 | 0.8150 | 0.4000 | 0.8850 | 0.3550 | 0.0300 | 0.6450 |
| test_in_dist | static | 0.9800 | 0.8674 | 0.4550 | 0.8050 | 0.4950 | 0.8900 | 0.4150 | 0.0250 | 0.5850 |
| test_ood_long | adaptive | 1.0000 | 0.6974 | 0.0500 | 0.7850 | 0.0700 | 0.9650 | 0.0500 | 0.0150 | 0.9500 |
| test_ood_long | static | 1.0000 | 0.7327 | 0.0550 | 0.7500 | 0.0750 | 0.9500 | 0.0450 | 0.0150 | 0.9550 |
| test_ood_division | adaptive | 1.0000 | 0.8269 | 0.2300 | 0.8100 | 0.2450 | 0.9750 | 0.2300 | 0.0250 | 0.7700 |
| test_ood_division | static | 0.9900 | 0.8399 | 0.3050 | 0.7150 | 0.3300 | 0.9650 | 0.3050 | 0.0300 | 0.6950 |

## Interpretation

v0.6b is a clear reward-surface success:

```text
Both adaptive and static moved far beyond the v0.6 0-2% legality regime.
Dense number-multiset rewards shifted the model from parseable constants toward
constrained expression construction while strict correctness stayed verifier-backed.
```

Adaptive RTW is not yet a clear win:

```text
Static is better on validation and in-distribution legality.
Static is better on OOD-division legality and exact correctness.
OOD-long is essentially tied and remains the weakest split for both methods.
Adaptive does not beat static on the seed-0 v0.6b reward surface.
```

Most likely diagnosis:

```text
The current adaptive teacher overreacts to the dense reward surface.
By the end of training it puts high weight on numeric_distance_reward and
keeps valid_expression moderately high, but suppresses number_multiset_f1 and
allowed_ops too aggressively. Static keeps balanced pressure on all legality
components and generalizes better on this seed.
```

## Decision

Do not expand adaptive/static seeds yet as if adaptive won.

Do not patch termination before documenting this result.

Do not launch manual/random yet. The immediate scientific question is teacher quality, not another baseline.

## Recommended next step

Implement a v0.6c teacher-stability experiment:

```text
delayed teacher adaptation for the first 50 steps
weight smoothing / lower update rate
minimum floor for number_multiset_f1 and allowed_ops
cap numeric_distance_reward weight
keep primary correctness separate and unchanged
```

Then compare:

```text
static v0.6b 300 seed0
adaptive v0.6b 300 seed0
adaptive-stable v0.6c 300 seed0
```

Primary proof metrics:

```text
allowed_numbers
number_multiset_f1
valid_expression
exact_correct
reward_hacking_candidate
```

Secondary diagnostic:

```text
OOD-long by operand count, especially uses_all_required_numbers and valid_expression.
```

