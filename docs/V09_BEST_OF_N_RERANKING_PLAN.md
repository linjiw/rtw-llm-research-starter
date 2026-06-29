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

## Status

- [x] Implement best-of-N candidate selection script.
- [x] Add unit tests for oracle and practical selector behavior.
- [ ] Run minimal Stable-RTW seed0 validation/test_in_dist diagnostic.
- [ ] Summarize result and decide whether to expand.
