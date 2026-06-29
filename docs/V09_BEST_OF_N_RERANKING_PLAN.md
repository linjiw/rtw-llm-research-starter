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


## Stage-1 limit=50, N=1/4/8 diagnostic result

Status: **completed**.

Run:

```text
checkpoint: Stable-RTW seed0
splits: validation, test_in_dist
limit: 50 examples per split
N: 1,4,8
temperature: 0.7
top_p: 0.95
max_new_tokens: 256
```

Artifacts:

```text
outputs/v09_bestofn_stable_seed0_validation_limit50_n8
outputs/v09_bestofn_stable_seed0_test_in_dist_limit50_n8
```

### Results

| split | N | oracle_exact@N | reranked_exact@N | reranked_valid_expression | reranked_number_f1 | reward_hacking_candidate | tokens_generated | wall_clock_s_est | cost_per_oracle_exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 1 | 0.000 | 0.000 | 0.220 | 0.541 | 0.700 | 3,847 | 61.0 | inf |
| validation | 4 | 0.120 | 0.120 | 0.540 | 0.906 | 0.460 | 17,360 | 244.1 | 33.3 |
| validation | 8 | 0.140 | 0.140 | 0.680 | 0.945 | 0.300 | 34,528 | 488.3 | 57.1 |
| test_in_dist | 1 | 0.040 | 0.040 | 0.220 | 0.587 | 0.780 | 4,161 | 57.5 | 25.0 |
| test_in_dist | 4 | 0.060 | 0.060 | 0.520 | 0.916 | 0.480 | 15,711 | 229.9 | 66.7 |
| test_in_dist | 8 | 0.120 | 0.120 | 0.680 | 0.974 | 0.320 | 31,600 | 459.9 | 66.7 |

Notes:

- `inf` in the table means exact rate was zero, so finite cost-per-exact is undefined. The JSON artifact stores this as a large sentinel value.
- `wall_clock_s_est` is prefix-estimated from the max-N run, because candidates are sampled once at N=8 and N=1/4 are prefix subsets.

### Decision classification

This is **Case A** for the limit=50 diagnostic:

```text
oracle_exact@8 improves
reranked_exact@8 improves
```

Validation:

```text
exact@1: 0.00
oracle_exact@8: 0.14
reranked_exact@8: 0.14
```

Test-in-distribution:

```text
exact@1: 0.04
oracle_exact@8: 0.12
reranked_exact@8: 0.12
```

The practical selector matched oracle exactness at every measured N in this diagnostic. This suggests the simple non-oracle verifier-style score can recover exact candidates when they appear in the sampled distribution.

### Interpretation

This provides the first substantial evidence that Stable-RTW has useful latent exact candidates in its policy distribution, and that verifier-guided candidate selection can convert some of that latent capability into higher exact pass rates without retraining or redefining correctness.

The result also improves selected legality:

```text
validation valid_expression: 0.22 -> 0.68 from N=1 to N=8
validation number_f1:        0.541 -> 0.945
validation reward_hacking:   0.700 -> 0.300

test_in_dist valid_expression: 0.22 -> 0.68
test_in_dist number_f1:        0.587 -> 0.974
test_in_dist reward_hacking:   0.780 -> 0.320
```

### Cost observation

The HF sampling path remains expensive:

```text
validation:   488.3 seconds for 50 tasks × 8 candidates
test_in_dist: 459.9 seconds for 50 tasks × 8 candidates
```

This matters for the harness-engineering story: best-of-N improves exactness and legality, but it increases inference cost. Any paper result should report sample budget and wall-clock/token cost.

### Next step

This result is strong enough to expand, but keep the next expansion controlled:

```text
static_v06b seed0
Stable-RTW seed0
validation + test_in_dist
limit: 50
N: 1,4,8
```

Purpose: determine whether best-of-N is specifically enhanced by Stable-RTW's candidate distribution, or whether static shaping has similar latent exact candidates.

If Stable-RTW remains better under identical best-of-N harnessing, expand to seeds 0/1/2. If not, report v0.9 as a harness-level mechanism that helps both shaped policies rather than a Stable-RTW-specific advantage.


## Stage-2 static-vs-Stable seed0 limit=50 comparison

Status: **completed**.

Goal: test whether best-of-N exact recovery is a general property of the static/base shaped policy or whether Stable-RTW creates a more useful candidate distribution.

Controls:

```text
checkpoints: static_v06b seed0, Stable-RTW seed0
splits: validation, test_in_dist
limit: first 50 examples per split
N: 1,4,8 using prefixes from one N=8 candidate bank
temperature: 0.7
top_p: 0.95
max_new_tokens: 256
prompt_field: prompt
engine: HF
batch_size: 8
```

Artifacts:

```text
outputs/v09_bestofn_static_v06b_seed0_validation_limit50_n8
outputs/v09_bestofn_static_v06b_seed0_test_in_dist_limit50_n8
outputs/v09_bestofn_stable_seed0_validation_limit50_n8
outputs/v09_bestofn_stable_seed0_test_in_dist_limit50_n8
```

Pairing check:

```text
validation:   static and Stable used identical 50 task IDs
test_in_dist: static and Stable used identical 50 task IDs
```

### Main table

| split | method | N | exact@1 | oracle_exact@N | reranked_exact@N | selected_valid | selected_number_f1 | reward_hack ↓ | tokens | wall_clock_s | cost_per_exact ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | static_v06b | 1 | 0.00 | 0.00 | 0.00 | 0.24 | 0.584 | 0.60 | 6,712 | 145.8 | inf |
| validation | static_v06b | 4 | 0.00 | 0.02 | 0.02 | 0.48 | 0.916 | 0.52 | 24,600 | 583.2 | 200.0 |
| validation | static_v06b | 8 | 0.00 | 0.06 | 0.06 | 0.56 | 0.940 | 0.44 | 47,192 | 1,166.4 | 133.3 |
| validation | Stable-RTW | 1 | 0.00 | 0.00 | 0.00 | 0.22 | 0.541 | 0.70 | 3,847 | 61.0 | inf |
| validation | Stable-RTW | 4 | 0.00 | 0.12 | 0.12 | 0.54 | 0.906 | 0.46 | 17,360 | 244.2 | 33.3 |
| validation | Stable-RTW | 8 | 0.00 | 0.14 | 0.14 | 0.68 | 0.945 | 0.30 | 34,528 | 488.3 | 57.1 |
| test_in_dist | static_v06b | 1 | 0.02 | 0.02 | 0.02 | 0.26 | 0.662 | 0.68 | 5,015 | 187.2 | 50.0 |
| test_in_dist | static_v06b | 4 | 0.02 | 0.10 | 0.10 | 0.52 | 0.890 | 0.44 | 20,349 | 748.9 | 40.0 |
| test_in_dist | static_v06b | 8 | 0.02 | 0.10 | 0.10 | 0.74 | 0.972 | 0.26 | 43,045 | 1,497.9 | 80.0 |
| test_in_dist | Stable-RTW | 1 | 0.04 | 0.04 | 0.04 | 0.22 | 0.587 | 0.78 | 4,161 | 57.5 | 25.0 |
| test_in_dist | Stable-RTW | 4 | 0.04 | 0.06 | 0.06 | 0.52 | 0.916 | 0.48 | 15,711 | 229.9 | 66.7 |
| test_in_dist | Stable-RTW | 8 | 0.04 | 0.12 | 0.12 | 0.68 | 0.974 | 0.32 | 31,600 | 459.9 | 66.7 |

### Stable-minus-static delta table

| split | N | Δ oracle_exact | Δ reranked_exact | Δ selected_number_f1 | Δ reward_hack | interpretation |
|---|---:|---:|---:|---:|---:|---|
| validation | 1 | +0.00 | +0.00 | -0.043 | +0.10 | no single-sample exact difference |
| validation | 4 | +0.10 | +0.10 | -0.010 | -0.06 | Stable has clearly better best-of-4 exact recovery |
| validation | 8 | +0.08 | +0.08 | +0.005 | -0.14 | Stable retains a strong exact and anti-hacking advantage |
| test_in_dist | 1 | +0.02 | +0.02 | -0.075 | +0.10 | Stable single-sample exact is slightly higher but legality is noisier |
| test_in_dist | 4 | -0.04 | -0.04 | +0.025 | +0.04 | static has stronger early best-of-4 exact recovery |
| test_in_dist | 8 | +0.02 | +0.02 | +0.002 | +0.06 | Stable narrowly beats static on exact but not reward-hacking |

### Paired N=8 task overlap

| split | selector | both | Stable-only | static-only | neither |
|---|---|---:|---:|---:|---:|
| validation | oracle | 3 | 4 | 0 | 43 |
| validation | practical | 3 | 4 | 0 | 43 |
| test_in_dist | oracle | 4 | 2 | 1 | 43 |
| test_in_dist | practical | 4 | 2 | 1 | 43 |

### Paired uncertainty

The N=8 paired table should be treated as a mechanism diagnostic, not a benchmark-scale significance result. Validation has a suggestive paired advantage for Stable-RTW (`Stable-only=4`, `static-only=0`), but only four discordant examples; an exact two-sided McNemar/binomial test gives `p=0.125`, so this should be described as a **suggestive paired advantage**, not a statistically significant win. Test-in-distribution is clearly mixed/noisy (`Stable-only=2`, `static-only=1`; exact two-sided McNemar/binomial `p=1.0`).

### Sanity audit

Stable-RTW oracle/practical agreement was confirmed directly from the candidate banks:

| split | oracle_exact@8 tasks | reranked_exact@8 tasks | oracle-exact but practical-missed tasks |
|---|---:|---:|---:|
| validation | 7 | 7 | 0 |
| test_in_dist | 6 | 6 | 0 |

The selector implementation was also inspected: `practical_score()` uses validity, number legality, number multiset F1, operator legality, numeric-distance reward, and reward-hacking penalty; it does **not** use `exact_correct`.

### Decision classification

This is closest to **Case 1 on validation** and a mixed **Case 1/3 on test_in_dist**:

- Validation: Stable-RTW is clearly better than static under both oracle and practical best-of-N. At N=8, Stable solves 7/50 tasks vs static 3/50, and all static successes are included in the Stable success set.
- Test-in-distribution: both methods benefit from best-of-N; Stable is only slightly higher at N=8 (6/50 vs 5/50), while static is better at N=4 (5/50 vs 3/50). This is not a clean Stable-specific win on this split.

Overall interpretation: v0.9 remains a real positive signal for Stable-RTW, especially on validation, but the static control shows that verifier-guided best-of-N is also a broader harness effect. The strongest safe claim is that Stable-RTW creates a more useful candidate distribution on validation and remains competitive-to-slightly-better at N=8 on test_in_dist, while both policies can expose latent exact candidates under sampling.

Cost note: best-of-N increases inference-time compute and should be reported with sampling budget, generated tokens, wall-clock, and cost-per-exact. In this diagnostic, static generated more tokens and took longer than Stable at the same nominal N=8 budget, but this should **not** yet be framed as an intrinsic method-specific cost claim. The static/Stable wall-clock comparison is confounded by run conditions such as batch size, cache/loading state, adapter loading path, process state, and GPU utilization. The safe paper claim is that sampling and selection are part of the harness cost; a stronger claim that static is intrinsically more expensive would require a dedicated cost audit.

### Paper-ready v0.9 wording

We next tested whether exact solutions were latent in the policy distribution and could be recovered by inference-time harness selection. Using frozen checkpoints, we sampled up to eight candidates per task and applied a fixed non-oracle reranker based on expression validity, number legality, number multiset F1, operator legality, numeric distance, and reward-hacking penalty. On validation, Stable-RTW improved from 0/50 exact at N=1 to 7/50 exact at N=8, compared with 3/50 for static shaping; all static successes were contained within the Stable-RTW success set. On test-in-distribution, both methods benefited from best-of-N, with Stable-RTW reaching 6/50 exact at N=8 and static reaching 5/50. The practical reranker matched oracle exact@N in these diagnostics without using exact correctness as an input. These results suggest that verifier-guided candidate selection can convert latent exact candidates into task success, but also show that best-of-N is a broader harness effect rather than an exclusively Stable-RTW-specific improvement.

### Recommended next step

Run one controlled seed expansion before adding OOD splits:

```text
static_v06b seed0/1/2
Stable-RTW seed0/1/2
validation + test_in_dist
limit: 50
N: 1,4,8
same task IDs
same decoding config
same selector
```

Report `mean ± std`, paired deltas, Stable-only/static-only overlap, cost-per-exact, and tokens-per-exact. Only after this seed expansion should v0.9 move to `test_ood_long` and `test_ood_division`.


## v0.9B controlled seed expansion plan

Status: **implementation ready; runs pending**.

Experiment name: **v0.9B: Controlled Seed Expansion for Verifier-Guided Best-of-N**.

Question:

```text
Does Stable-RTW produce a better sampled candidate distribution than static shaping under the same verifier-guided best-of-N harness?
```

Frozen design:

```text
methods: static_v06b, Stable-RTW/adaptive_stable_v06c
seeds: 0, 1, 2
splits: validation, test_in_dist
limit: 50
N: 1,4,8 from max-N=8 prefixes
temperature: 0.7
top_p: 0.95
max_new_tokens: 256
batch_size: 8
prompt_field: prompt
sampling_seed: 0
```

Task-ID files are frozen from the seed0 candidate banks:

```text
outputs/v09_task_ids_validation_limit50.txt
outputs/v09_task_ids_test_in_dist_limit50.txt
```

Runner support added to `scripts/07_best_of_n_rerank.py`:

```text
--task_ids_file
--max_n
--n_values 1 4 8   # comma-separated form still accepted
--skip_if_complete
run_config.json writing
```

Aggregator added:

```text
scripts/08_summarize_v09_seed_expansion.py
```

Expected summary artifacts:

```text
outputs/v09_seed_expansion_summary.csv
outputs/v09_seed_expansion_paired.json
```

Validation gates before launching runs:

```text
uv run pytest -q
uv run ruff check .
```

Both passed at implementation time.


## v0.9B controlled seed expansion result

Status: **completed**.

Artifacts:

```text
outputs/v09b_bestofn_*_limit50_n8              # 12 run dirs
outputs/v09_seed_expansion_summary.csv
outputs/v09_seed_expansion_paired.json
```

Validation/invariant checks:

```text
12 / 12 run directories present
all candidate banks: 400 rows = 50 tasks x 8 candidates
metrics.json, summary.csv, and run_config.json present for every run
static/Stable task IDs match for every seed x split pair
oracle_exact@1 <= oracle_exact@4 <= oracle_exact@8 for every run
reranked_exact@N <= oracle_exact@N for every run
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q  # 42 passed
uv run ruff check .                               # passed
```

Note: the serial runner exited with status 1 only because the final `uv run pytest -q` auto-loaded a ROS Humble pytest plugin from the host environment and failed on missing `lark`. Re-running tests with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` passed. The experiment runs and aggregator had already completed successfully.

### Table 1: mean ± std across seeds

| split | method | N | oracle_exact | reranked_exact | selected_valid | selected_number_f1 | reward_hack ↓ | tokens | wall_clock_s | cost_per_exact ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | static | 1 | 0.007±0.012 | 0.007±0.012 | 0.187±0.092 | 0.533±0.163 | 0.687±0.081 | 5,669±1,125 | 131.9±50.6 | 50.0±0.0 |
| validation | Stable-RTW | 1 | 0.020±0.020 | 0.020±0.020 | 0.207±0.101 | 0.620±0.123 | 0.720±0.053 | 3,495±1,271 | 61.4±17.2 | 37.5±17.7 |
| validation | static | 4 | 0.027±0.012 | 0.027±0.012 | 0.413±0.083 | 0.827±0.095 | 0.493±0.023 | 23,096±1,488 | 527.5±202.3 | 166.7±57.7 |
| validation | Stable-RTW | 4 | 0.107±0.042 | 0.107±0.042 | 0.527±0.042 | 0.923±0.020 | 0.473±0.042 | 15,755±4,223 | 245.4±68.7 | 42.9±20.8 |
| validation | static | 8 | 0.067±0.050 | 0.067±0.050 | 0.540±0.092 | 0.928±0.023 | 0.453±0.081 | 44,882±2,098 | 1,055.0±404.7 | 200.0±176.4 |
| validation | Stable-RTW | 8 | 0.133±0.012 | 0.133±0.012 | 0.680±0.020 | 0.958±0.012 | 0.313±0.023 | 31,707±8,789 | 490.9±137.4 | 60.3±5.5 |
| test_in_dist | static | 1 | 0.027±0.012 | 0.027±0.012 | 0.187±0.064 | 0.542±0.133 | 0.700±0.053 | 4,902±378 | 116.1±61.6 | 41.7±14.4 |
| test_in_dist | Stable-RTW | 1 | 0.040±0.020 | 0.040±0.020 | 0.207±0.023 | 0.616±0.029 | 0.747±0.058 | 4,096±527 | 56.5±7.2 | 30.6±17.3 |
| test_in_dist | static | 4 | 0.107±0.012 | 0.107±0.012 | 0.480±0.040 | 0.873±0.066 | 0.473±0.042 | 20,720±677 | 464.5±246.3 | 37.8±3.8 |
| test_in_dist | Stable-RTW | 4 | 0.087±0.031 | 0.087±0.031 | 0.573±0.046 | 0.927±0.010 | 0.427±0.046 | 15,551±1,665 | 226.0±28.6 | 50.0±16.7 |
| test_in_dist | static | 8 | 0.120±0.020 | 0.120±0.020 | 0.667±0.070 | 0.950±0.030 | 0.320±0.053 | 43,071±1,253 | 929.1±492.6 | 67.9±11.5 |
| test_in_dist | Stable-RTW | 8 | 0.133±0.042 | 0.133±0.042 | 0.733±0.046 | 0.974±0.001 | 0.267±0.046 | 29,963±5,153 | 451.9±57.3 | 63.7±18.0 |

### Table 2: paired overlap across all three seeds

| split | N | both | Stable-only | static-only | neither | McNemar/binomial p | Δ reranked exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 1 | 0 | 3 | 1 | 146 | 0.6250 | +0.013±0.012 |
| validation | 4 | 4 | 12 | 0 | 134 | 0.0005 | +0.080±0.035 |
| validation | 8 | 9 | 11 | 1 | 129 | 0.0063 | +0.067±0.042 |
| test_in_dist | 1 | 2 | 4 | 2 | 142 | 0.6875 | +0.013±0.012 |
| test_in_dist | 4 | 8 | 5 | 8 | 129 | 0.5811 | -0.020±0.035 |
| test_in_dist | 8 | 12 | 8 | 6 | 124 | 0.7905 | +0.013±0.050 |

### Table 3: cost-aware Pareto at N=8

| split | method | reranked_exact@8 | tokens_per_exact ↓ | seconds_per_exact ↓ |
|---|---|---:|---:|---:|
| validation | static | 0.067 | 673,230 | 15,824.6 |
| validation | Stable-RTW | 0.133 | 237,800 | 3,681.6 |
| test_in_dist | static | 0.120 | 358,922 | 7,742.5 |
| test_in_dist | Stable-RTW | 0.133 | 224,725 | 3,389.5 |

These are observed harness costs in this run, not an intrinsic claim that static is slower. The cost result is still important because best-of-N trades inference compute for exactness, and this repo has already observed many clipped/max-length completions.

### Interpretation

v0.9B is **Case A on validation** and **Case B/mixed on test_in_dist**.

Validation is robust across seeds: Stable-RTW beats static at N=4 and N=8, practical reranking matches oracle exactness, and paired overlap strongly favors Stable-RTW (`Stable-only=11`, `static-only=1` at N=8; exact McNemar/binomial `p=0.0063`). This supports the claim that Stable-RTW improves the sampled candidate distribution and/or its rankability under the fixed verifier-style selector.

Test-in-distribution confirms best-of-N as a general harness mechanism but does **not** show a decisive Stable-specific advantage. Both methods improve from N=1 to N=8. Stable-RTW is slightly higher on mean reranked_exact@8 (`0.133` vs `0.120`) and cleaner on selected legality (`selected_valid`, `number_f1`, `reward_hack`), but paired overlap is mixed (`Stable-only=8`, `static-only=6`, `p=0.7905`). Static is also better at N=4 on exactness.

The practical reranker matched oracle exactness in every v0.9B aggregate shown here. That keeps the selector result clean: exact candidates are not merely latent, they are recoverable by the frozen non-oracle legality/distance/risk score in these diagnostics.

### Decision

Promote v0.9 as a real paper result, but with restrained language:

```text
Verifier-guided best-of-N is a general inference-time harness mechanism. Stable-RTW shows a robust best-of-N advantage on validation and a smaller/mixed advantage on test-in-distribution. The result supports the two-stage harness claim: training-time reward stability improves legal candidate formation/rankability, while inference-time verifier selection converts some latent exact candidates into task success. The benefit must be reported with sampling budget, paired uncertainty, generated tokens, wall-clock time, and cost-per-exact.
```

Do not claim universal Stable-RTW dominance under best-of-N. The in-distribution seed expansion is mixed enough that the honest claim is split-dependent: robust validation advantage, general harness benefit on test_in_dist.
