# v0.7 Prompt/Length Diagnostic and Paper Plan

> **For Hermes:** This is the next improvement loop after v0.6d. Do not change reward weights or verifier semantics in this loop.

## Goal

Improve the paper and identify the next experimental bottleneck after the v0.6d 3-seed result.

The v0.6d result is coherent enough for a paper skeleton, but exact correctness remains low and training completions were max-length clipped. Before another reward-controller tweak, run a diagnostic inference-time prompt/length ablation to determine whether failures are mainly:

1. **training/reward-side**: the model cannot construct correct expressions despite legality shaping;
2. **prompt/harness-side**: the model is sensitive to instruction format;
3. **decoding/length-side**: the model over-generates or clips, harming parseability and correctness.

## Current v0.6d claim

Supported, scoped claim:

> Stability-constrained adaptive reward weighting substantially improves in-distribution legality robustness and reduces seed sensitivity relative to static shaping in a verifier-based Countdown harness, but it does not solve exact correctness or OOD generalization.

Main 3-seed support:

```text
validation valid_expression:        +0.142 adaptive_stable - static
validation exact_correct:           +0.018
validation reward_hacking_candidate -0.130  (lower is better)
in-dist valid_expression:           +0.140
in-dist reward_hacking_candidate:   -0.117
```

Main caveats:

```text
in-dist exact_correct: adaptive_stable lower by -0.010
OOD-long: unsolved
OOD-division: static better on valid_expression and reward_hacking_candidate
all exact_correct rates remain low
training completions clipped at max_completion_length=256
```

## v0.7 experiment type

This is a **diagnostic eval-only ablation**. It does not create a new trained method and should not be used as a main-method result unless later repeated as a controlled training condition.

Keep fixed:

```text
model: Qwen/Qwen2.5-0.5B-Instruct
trained adapter: existing v0.6d checkpoints
verifier: src/rtw_llm/countdown.py
reward/correctness semantics: unchanged
eval data: data/countdown/validation.jsonl and data/countdown/test_in_dist.jsonl first
```

Vary only:

```text
prompt_field: prompt_low, prompt_mid, prompt_high, prompt
max_new_tokens: 32, 64, 128
```

Why these knobs:

- `prompt_low` tests reliance on tags/checklist.
- `prompt_mid` tests concise explicit constraints.
- `prompt_high` / `prompt` test current verbose harness.
- `32` tests whether concise decoding reduces repeated tags/prose.
- `64` is the current held-out eval setting.
- `128` tests whether eval truncation hides correct late answers.

## Stage 1: cheap probe

Run on the most informative pair first:

```text
adaptive_stable_v06c seed2
static_v06b seed2
splits: validation, test_in_dist
prompt_fields: prompt_low, prompt_mid, prompt_high, prompt
max_new_tokens: 32, 64, 128
```

Output naming:

```text
outputs/eval_v07_promptlen_<method>_seed2_<split>_<prompt_field>_tok<max_new_tokens>
```

## Stage 1 decision gates

A prompt/length setting is interesting if it improves `exact_correct` by at least 0.02 absolute without collapsing `valid_expression` by more than 0.03.

A prompt setting is unsafe if it increases `reward_hacking_candidate` by more than 0.05 or collapses tag/parse behavior.

If a setting wins for both static and adaptive_stable, the bottleneck is likely harness/decoding rather than teacher weighting.

If only adaptive_stable benefits, the teacher may have learned latent legality that requires a better inference harness.

If no setting helps exact correctness, the next experiment should target training-side reasoning, not prompting.

## Stage 2: expand only if Stage 1 finds a useful setting

If Stage 1 finds a useful setting, rerun it across all three seeds and all four splits:

```text
static_v06b seeds 0,1,2
adaptive_stable_v06c seeds 0,1,2
splits: validation, test_in_dist, test_ood_long, test_ood_division
best prompt/length setting only
```

Then report it as an **inference harness diagnostic**, not as a replacement main result.

## Paper improvement tasks

### Task 1: Update manuscript framing

Create a paper outline that treats the project as a stability study rather than a global adaptive-win claim.

Target file:

```text
docs/PAPER_OUTLINE.md
```

Required sections:

1. Abstract draft
2. Introduction
3. Related work
4. Countdown harness wind tunnel
5. Reward decomposition and RTW teacher
6. Experiments
7. Results
8. Failure analysis
9. Limitations
10. Future work

### Task 2: Add result table plan

Target file:

```text
docs/PAPER_OUTLINE.md
```

Required tables:

1. Base model vs v0.6b dense reward acquisition.
2. Naive adaptive vs static seed-0 failure mode.
3. Static vs adaptive_stable 3-seed main table.
4. Teacher mechanics table.
5. Prompt/length diagnostic table if v0.7 produces a useful finding.

### Task 3: Add limitations explicitly

Required limitations:

- exact correctness remains low;
- OOD-long remains unsolved;
- OOD-division does not favor adaptive_stable;
- Countdown is a wind-tunnel task, not a coding-agent harness;
- inference prompt/length diagnostics are not new trained methods;
- sample size remains small.



## Stage-1 actual execution note

The original full Stage-1 matrix was intentionally stopped after it became too slow for an interactive iteration. A focused finishable diagnostic was run instead:

```text
method: adaptive_stable_v06c seed2
splits: validation, test_in_dist
prompt_fields: prompt_high, prompt
max_new_tokens: 32, 64
```

Partial static diagnostics from the stopped broad run were also inspected for prompt sensitivity.

## v0.7A focused diagnostic results

Adaptive_stable seed2 prompt/length results:

| split | prompt | tok | valid_expression | exact_correct | reward_hacking_candidate | allowed_numbers | number_f1 | allowed_ops | parse_ok |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | prompt_high | 32 | 0.430 | 0.055 | 0.570 | 0.435 | 0.845 | 0.920 | 0.960 |
| validation | prompt_high | 64 | 0.430 | 0.055 | 0.570 | 0.435 | 0.845 | 0.920 | 0.960 |
| validation | prompt | 32 | 0.430 | 0.055 | 0.570 | 0.435 | 0.845 | 0.920 | 0.960 |
| validation | prompt | 64 | 0.430 | 0.055 | 0.570 | 0.435 | 0.845 | 0.920 | 0.960 |
| test_in_dist | prompt_high | 32 | 0.420 | 0.015 | 0.580 | 0.430 | 0.860 | 0.925 | 0.990 |
| test_in_dist | prompt_high | 64 | 0.420 | 0.015 | 0.580 | 0.430 | 0.860 | 0.925 | 0.990 |
| test_in_dist | prompt | 32 | 0.420 | 0.015 | 0.580 | 0.430 | 0.860 | 0.925 | 0.990 |
| test_in_dist | prompt | 64 | 0.420 | 0.015 | 0.580 | 0.430 | 0.860 | 0.925 | 0.990 |

Interpretation:

```text
Shorter decoding from 64 to 32 tokens did not change metrics for the normal/high prompts.
The prompt and prompt_high fields are effectively equivalent for these checkpoints.
This means the remaining exact-correctness bottleneck is not fixed by simple inference-time length truncation or by switching between the two high-information harness prompts.
```

Partial static prompt-sensitivity findings from the stopped broad run:

```text
prompt_low collapsed to all-zero metrics because it does not enforce the answer-tag contract.
prompt_mid produced almost no valid expressions and high reward-hacking rates at longer lengths.
prompt_high and prompt matched canonical static seed2 validation metrics exactly across 32/64/128 tokens.
```

This suggests the model has learned a narrow verifier-compatible harness contract: it needs the explicit high-information answer-tag prompt, but within that contract, 32 vs 64 tokens does not explain the exact-correctness bottleneck.

## Failure taxonomy diagnostic

Implemented:

```text
scripts/06_failure_taxonomy.py
```

Validated by:

```text
uv run pytest -q
uv run ruff check .
```

Output artifact:

```text
outputs/v07a_failure_taxonomy_seed2.json
```

Seed2 failure taxonomy on validation/test_in_dist:

| run | exact_correct | legal_but_wrong_value | missing_required_number | illegal_extra_or_repeated_number | parse_failure | illegal_operator | evaluation_error |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive_stable validation | 0.055 | 0.375 | 0.390 | 0.135 | 0.040 | 0.000 | 0.005 |
| adaptive_stable test_in_dist | 0.015 | 0.405 | 0.395 | 0.165 | 0.010 | 0.010 | 0.000 |
| static validation | 0.035 | 0.255 | 0.490 | 0.195 | 0.000 | 0.025 | 0.000 |
| static test_in_dist | 0.060 | 0.260 | 0.475 | 0.165 | 0.000 | 0.040 | 0.000 |

Key bottleneck:

```text
The largest failure buckets are missing_required_number and legal_but_wrong_value.
So the next experiment should not be another prompt/length tweak. It should target:
  1. using every required number exactly once;
  2. converting legal-but-wrong expressions into exact target solutions.
```

## v0.7 decision

Outcome: **diagnostic negative for prompt/length as a quick fix, positive for paper clarity.**

Do not expand v0.7 across all seeds. Use v0.7 as a limitation/failure-analysis result in the paper.

Next implementation/experiment direction:

```text
v0.8 should target the dominant failure modes directly:
  - missing required numbers;
  - legal but wrong target value;
  - answer discipline/termination only as a secondary issue.
```

Recommended v0.8 design:

1. Add a failure-taxonomy table to the paper.
2. Add a controlled training variant only if it changes one variable at a time.
3. Prefer a two-phase or conditional reward design:
   - phase A: enforce exact number multiset completion;
   - phase B: after number legality is high, increase target/exactness pressure.
4. Preserve strict verifier correctness and all v0.6d baselines.

## Done criteria

- [x] Stage-1 prompt/length probe completed.
- [x] Stage-1 results summarized in this doc.
- [x] Decision recorded: expand diagnostic or move to paper draft.
- [x] `docs/PAPER_OUTLINE.md` created.
- [x] Tests and ruff pass.
- [x] Docs committed.
