# v0.9 Verifier-Guided Best-of-N Candidate Selection

> **For Hermes:** v0.9 is an inference-time harness experiment over frozen checkpoints. Do not retrain. Do not change verifier semantics. Do not change the main Stable-RTW claim.

## Goal

Test whether exact Countdown solutions are latent in the frozen Stable-RTW policy distribution and can be exposed by verifier-guided candidate selection.

## Hypothesis

Stable-RTW may already place correct or near-correct expressions somewhere in the policy distribution, but greedy/single-sample evaluation fails to select them.

This tests a different harness mechanism from v0.8:

```text
v0.8: training-time scalar reward phasing
v0.9: inference-time candidate generation + verifier-guided selection
```

## Controlled variables

Keep fixed:

```text
model: Qwen/Qwen2.5-0.5B-Instruct
checkpoint: frozen trained adapter
verifier: src/rtw_llm/countdown.py
correctness: exact_correct only from strict verifier
prompt_field: prompt
data splits: validation, test_in_dist first
max_new_tokens: 256
temperature: fixed per run
top_p: fixed per run
```

## Initial minimal diagnostic

Start with one checkpoint and two splits:

```text
checkpoint: Stable-RTW seed0
splits: validation, test_in_dist
N: 1,4,8,16,32
sampling: temperature 0.7, top_p 0.95
```

This is intentionally small. Expand only if it shows signal.

## Metrics

For each N:

```text
oracle_exact@N
reranked_exact@N
valid_expression@selected
allowed_numbers@selected
number_multiset_f1@selected
allowed_ops@selected
uses_all_required_numbers@selected
uses_no_extra_numbers@selected
reward_hacking_candidate@selected
cost_per_oracle_exact
cost_per_reranked_exact
```

## Selectors

### Oracle selector

Uses strict verifier exactness as a selector:

```text
choose any exact_correct candidate if present;
tie-break with practical legality score.
```

This answers:

> Does the policy distribution contain exact solutions at all?

### Practical selector

Does **not** use exact correctness as a feature. It scores candidates with:

```text
valid_expression
number_multiset_f1
uses_allowed_numbers
uses_allowed_ops
uses_all_required_numbers
uses_no_extra_numbers
numeric_distance_reward
brevity
- reward_hacking_candidate penalty
```

This answers:

> Can a non-oracle verifier-style selection harness improve the chosen answer?

Final exactness is still reported only after selection via the strict verifier.

## Decision cases

### Case A: oracle exact@N improves, practical reranking improves

Interpretation:

> Stable-RTW learns a useful candidate distribution, and verifier-guided selection converts legality into higher exact task success.

### Case B: oracle exact@N improves, practical reranking does not

Interpretation:

> Correct candidates exist, but the non-oracle selector cannot reliably identify them.

Next mechanism: better learned verifier or symbolic solver-assisted reranker.

### Case C: oracle exact@N does not improve much

Interpretation:

> The policy distribution lacks exact solutions; search/reranking is insufficient, and the next bottleneck is training-side reasoning or curriculum.

## Implementation

Script:

```text
scripts/07_best_of_n_rerank.py
```

Tests:

```text
tests/test_best_of_n_rerank.py
```

## Implementation update: cost-aware candidate bank

The v0.9 script now records cost and selection metadata in addition to metrics:

```text
wall_clock_seconds
tokens_generated
samples_per_task
cost_per_oracle_exact
cost_per_reranked_exact
completion_token_count per candidate
selected_by_practical_n
selected_by_oracle_n
```

The candidate bank stores:

```text
task_id / id
candidate_index
raw_generation
extracted_expression
exact_correct
valid_expression
allowed_numbers
number_f1
allowed_ops
numeric_distance_reward
reward_hacking_candidate
practical_score
selected_by_practical_n
selected_by_oracle_n
```

The selector score is intentionally simple and fixed:

```text
score =
  3.0 * valid_expression
+ 2.0 * uses_allowed_numbers
+ 1.5 * number_multiset_f1
+ 1.0 * uses_allowed_ops
+ 1.0 * numeric_distance_reward
- 2.0 * reward_hacking_candidate
```

Exact correctness is not used by the practical selector; it is only reported after selection.

## Status

- [x] Implement best-of-N candidate selection script.
- [x] Add unit tests for oracle and practical selector behavior.
- [x] Run minimal Stable-RTW seed0 validation/test_in_dist diagnostic.
- [x] Summarize result and decide whether to expand.


## Stage-0 implementation and tiny diagnostic result

Implementation commit:

```text
1c973cb Add v0.9 best-of-N reranking diagnostic
```

Validation:

```text
uv run pytest -q
37 passed

uv run ruff check .
All checks passed!
```

A first broad run with `limit=50` and `N=1,4,8,16,32` was intentionally stopped because HF sampling throughput was too slow for an interactive diagnostic. This is not a result artifact; it was a cost calibration.

Completed tiny diagnostic:

```text
checkpoint: Stable-RTW seed0
splits: validation, test_in_dist
limit: 10 examples per split
N: 1,4,8
temperature: 0.7
top_p: 0.95
max_new_tokens: 256
```

Artifacts:

```text
outputs/v09_bestofn_stable_seed0_validation_limit10
outputs/v09_bestofn_stable_seed0_test_in_dist_limit10
```

### Tiny diagnostic table

| split | N | oracle_exact@N | reranked_exact@N | reranked_valid_expression | reranked_number_f1 |
|---|---:|---:|---:|---:|---:|
| validation | 1 | 0.000 | 0.000 | 0.300 | 0.472 |
| validation | 4 | 0.100 | 0.100 | 0.700 | 0.963 |
| validation | 8 | 0.100 | 0.100 | 0.800 | 0.978 |
| test_in_dist | 1 | 0.100 | 0.100 | 0.300 | 0.696 |
| test_in_dist | 4 | 0.100 | 0.100 | 0.600 | 0.952 |
| test_in_dist | 8 | 0.100 | 0.100 | 0.600 | 0.963 |

### Interpretation

This tiny sample is not publication-scale, but it validates the mechanism and shows a useful signal:

- Best-of-N selection substantially improves selected legality metrics (`valid_expression`, `number_multiset_f1`).
- On validation, oracle/practical exact improves from `0.0` at N=1 to `0.1` at N=4/8.
- On test_in_dist, exact was already `0.1` at N=1 and stayed flat through N=8.
- Practical and oracle selectors matched in this tiny sample, suggesting the non-oracle legality/distance score can select the same exact candidate when one is present, at least for these examples.

### Decision

Do not make claims from `limit=10`. But the diagnostic is promising enough to justify a larger, cost-controlled v0.9 run.

Recommended next run:

```text
Stable-RTW seed0
validation + test_in_dist
limit: 50
N: 1,4,8
```

This removes the expensive N=16/32 settings while giving a less noisy estimate of whether oracle/practical exact@N moves with sampling. If that shows signal, expand to all 200 examples and then to static/Stable-RTW seed0 comparisons.
