# Current Project Status and Paper Assessment

Snapshot time: `2026-06-28T03:24:54-04:00`

Repository: `/home/robotixx/rtw-llm-research-starter`

## Executive summary

This project is a promising research scaffold for **Reward Training Wheels (RTW) for harness-aware LLM post-training**, but the current empirical evidence is **not yet paper-ready for the strong adaptive-RTW claim**.

The main current finding is:

> Dense verifier-aligned auxiliary rewards successfully move a small LLM from essentially zero usable Countdown harness behavior into parseable, partially legal expression generation, while preserving strict verifier-backed primary correctness.

The current evidence does **not** yet support:

> Adaptive auxiliary reward weighting beats static/fixed reward shaping.

In the current v0.6b seed-0 comparison, the **static** reward schedule generally outperforms the **adaptive RTW** schedule on held-out legality and exact-correctness metrics.

## Current research goal

The intended paper-shaped goal is:

> Adaptive auxiliary reward weighting improves verifier-based harness acquisition in LLM post-training compared with fixed, manual, and random reward schedules.

The broader framing is strong:

> LLM agents are not just model weights. They are model plus harness: prompts, tools, validators, execution loops, state, guardrails, logging, and feedback. Therefore, post-training should treat harness components and verifier feedback as adaptive training signals.

The current controlled domain is **Countdown arithmetic**, chosen as a clean verifier-based wind tunnel before moving to coding-agent harnesses.

## Current repo state

The current branch is `main`.

At snapshot time, git status was:

```text
 M scripts/02_grpo_train.py
 M scripts/05_check_run_health.py
 M src/rtw_llm/countdown.py
 M src/rtw_llm/prompts.py
 M src/rtw_llm/teacher.py
 M tests/test_countdown.py
 M tests/test_teacher.py
?? docs/PROJECT_DESIGN.md
?? docs/V06B_ADAPTIVE_STATIC_SEED0_REPORT.md
?? docs/V06B_LEGALITY_TRACKER.md
?? docs/V06C_TEACHER_STABILITY.md
?? uv.lock
```

Important implication:

> The current research story depends on uncommitted code and documentation. Before treating any result as archival, commit the exact implementation, docs, and lockfile that produced it.

Suggested commit theme:

```text
Add v0.6b dense legality rewards and teacher-stability notes
```

## Validation performed

The test suite was run with:

```bash
uv run pytest -q
```

Observed result:

```text
22 passed in 0.01s
```

So the verifier/reward changes were passing at the time of inspection.

## Current experiment: v0.6b adaptive vs static, seed 0

Runs inspected:

```text
adaptive:
  outputs/grpo_rtw_v06b_dense_numbers_cuda_pilot_300_seed0

static:
  outputs/grpo_static_v06b_dense_numbers_cuda_pilot_300_seed0
```

Both used:

```text
model: Qwen/Qwen2.5-0.5B-Instruct
seed: 0
max_steps: 300
num_generations: 4
reward surface: v0.6b dense legality rewards
```

Both recorded audit commit:

```text
1815b8af70b8c57c25cadba9a39e4e3f2589ce68
```

## Training-side metrics

| Metric | Adaptive | Static |
|---|---:|---:|
| primary reward mean / exact correct | 0.0246 | 0.0296 |
| aux reward weighted mean | 0.3158 | 0.6474 |
| reward variance nonzero fraction | 1.0000 | 1.0000 |
| format | 0.9295 | 0.9298 |
| parseable expression | 0.8000 | 0.7808 |
| number multiset F1 | 0.5507 | 0.5268 |
| allowed numbers | 0.2444 | 0.2444 |
| allowed ops | 0.5769 | 0.5575 |
| valid expression | 0.2190 | 0.2248 |
| exact correct | 0.0246 | 0.0296 |

Interpretation:

- The v0.6b dense reward surface creates a live learning signal.
- Both adaptive and static move off the all-zero base-model regime.
- Adaptive is slightly better on training parseability and number F1.
- Static is slightly better on training valid-expression and exact-correctness rates.

## Held-out v0.6b eval metrics

| split | method | parse_ok | number_multiset_f1 | uses_allowed_numbers | uses_no_extra_numbers | uses_all_required_numbers | uses_allowed_ops | valid_expression | exact_correct | reward_hacking_candidate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | adaptive | 0.935 | 0.809 | 0.370 | 0.805 | 0.395 | 0.860 | 0.345 | 0.035 | 0.655 |
| validation | static | 0.960 | 0.853 | 0.440 | 0.835 | 0.470 | 0.885 | 0.415 | 0.045 | 0.585 |
| test_in_dist | adaptive | 0.985 | 0.840 | 0.390 | 0.815 | 0.400 | 0.885 | 0.355 | 0.030 | 0.645 |
| test_in_dist | static | 0.980 | 0.867 | 0.455 | 0.805 | 0.495 | 0.890 | 0.415 | 0.025 | 0.585 |
| test_ood_long | adaptive | 1.000 | 0.697 | 0.050 | 0.785 | 0.070 | 0.965 | 0.050 | 0.015 | 0.950 |
| test_ood_long | static | 1.000 | 0.733 | 0.055 | 0.750 | 0.075 | 0.950 | 0.045 | 0.015 | 0.955 |
| test_ood_division | adaptive | 1.000 | 0.827 | 0.230 | 0.810 | 0.245 | 0.975 | 0.230 | 0.025 | 0.770 |
| test_ood_division | static | 0.990 | 0.840 | 0.305 | 0.715 | 0.330 | 0.965 | 0.305 | 0.030 | 0.695 |

## Current empirical conclusion

Static v0.6b generally beats adaptive v0.6b on seed 0.

Static is better on:

- validation exact correctness
- validation valid expression
- validation number legality
- in-distribution valid expression
- OOD-division valid expression
- OOD-division exact correctness
- lower reward-hacking candidate rate

Adaptive is only narrowly competitive in a few areas, such as training parseability and OOD-long valid expression by a tiny margin.

Therefore:

> Do not claim that adaptive RTW works better than static shaping yet.

## Current strongest supported claim

The strongest currently supported claim is narrower:

> Dense verifier-aligned auxiliary rewards can teach a small LLM the surface and partial legality requirements of a Countdown verifier harness, moving it from zero usable behavior into parseable and partially legal expression construction.

This is meaningful because base CUDA evals showed all-zero behavior:

```text
format: 0.0
expression_parseable: 0.0
uses_allowed_numbers: 0.0
valid_expression: 0.0
exact_correct: 0.0
```

v0.6b GRPO then reached, on validation:

```text
parseable: 0.935-0.960
allowed_numbers: 0.370-0.440
valid_expression: 0.345-0.415
exact_correct: 0.035-0.045
```

## Diagnosis of adaptive underperformance

The adaptive teacher appears to overreact to dense reward components.

Final adaptive weights:

```text
format:                  0.0227
valid_expression:        0.2236
number_multiset_f1:      0.0879
allowed_ops:             0.0707
numeric_distance_reward: 0.3161
brevity:                 0.0230
```

Static weights remain:

```text
all components: 0.2000
```

Likely issue:

> The adaptive controller suppresses number-multiset and allowed-op pressure too aggressively while emphasizing numeric-distance reward too much.

This risks producing parseable, numerically close, but verifier-invalid expressions.

## Literature positioning

Relevant anchors found during inspection:

| Topic | Paper |
|---|---|
| RTW source | `Reward Training Wheels: Adaptive Auxiliary Rewards for Robotics Reinforcement Learning`, arXiv:2503.15724 |
| GACL source | `GACL: Grounded Adaptive Curriculum Learning with Active Task and Performance Monitoring`, arXiv:2508.02988 |
| earlier training-wheels RL idea | `Learning with Training Wheels: Speeding up Training with a Simple Controller for Deep Reinforcement Learning`, arXiv:1812.05027 |
| related LLM areas | RLVR, process supervision, curriculum learning for LLMs, verifier-based math/code training |

Novelty should be framed carefully.

Not novel enough:

> Auxiliary rewards help LLMs.

More defensible:

> RTW-style dynamic auxiliary reward weighting can be adapted to verifier-based LLM post-training, with decomposed traces exposing how harness skills are acquired or reward-hacked.

## Design strengths

The current design has strong scientific guardrails:

1. Strict verifier is the source of truth.
2. Primary correctness is separated from auxiliary reward.
3. Reward components are separately logged.
4. OOD splits exist:
   - in-distribution
   - longer tasks
   - division-enabled tasks
5. Harness levels exist:
   - `prompt_low`
   - `prompt_mid`
   - `prompt_high`
6. Failure modes are explicitly tracked:
   - tag farming
   - parseable nonsense
   - auxiliary dominance
   - harness overfitting
7. Tests pass.
8. Dataset is deterministic and verifier-backed.

## Design weaknesses / paper risks

Main blockers to a strong paper:

1. Only seed 0 has been evaluated for the core adaptive/static comparison.
2. Static currently beats adaptive.
3. Manual and random baselines are not complete for v0.6b.
4. Primary exact correctness is still low.
5. OOD-long remains very weak.
6. Harness-shift matrix has not yet been run.
7. There is no actual paper manuscript yet, only research docs.

## Paper-readiness verdict

Current status:

> Promising project, not paper-ready for the strong adaptive RTW claim.

| Area | Status |
|---|---|
| Research framing | Good |
| Code scaffold | Good |
| Verifier correctness | Good |
| Logging/diagnostics | Good |
| Dataset | Good for first study |
| Literature motivation | Plausible |
| Main adaptive claim | Not supported yet |
| Baseline matrix | Incomplete |
| Multi-seed evidence | Missing |
| Harness-shift evidence | Missing |
| Paper draft | Not started |

## Recommended next step: v0.6c teacher stability

Do **not** expand seeds yet as if adaptive won.

First implement and test a more stable adaptive teacher:

1. **Delayed adaptation**
   - Keep balanced initial weights for about the first 50 steps.

2. **Lower update rate**
   - Current `lr = 0.30` is probably too aggressive.
   - Try `lr = 0.05-0.10`.

3. **Floors for legality wheels**
   - Prevent `number_multiset_f1` and `allowed_ops` from collapsing too low.
   - Candidate floors:
     ```text
     number_multiset_f1: 0.15-0.20
     allowed_ops:        0.10-0.15
     ```

4. **Cap numeric-distance reward weight**
   - Numeric distance should not dominate legality.
   - Candidate cap:
     ```text
     numeric_distance_reward: 0.20-0.25
     ```

5. **Preserve primary correctness semantics**
   - Exact correctness remains verifier-owned.
   - Dense auxiliaries remain training wheels only.

## v0.6c comparison gate

Compare:

```text
static v0.6b 300 seed0
adaptive v0.6b 300 seed0
adaptive-stable v0.6c 300 seed0
```

v0.6c should beat or closely match static on:

| Metric | Target |
|---|---|
| validation valid_expression | beat or match static 0.415 |
| validation exact_correct | beat or match static 0.045 |
| test_in_dist valid_expression | beat or match static 0.415 |
| OOD-division valid_expression | beat or match static 0.305 |
| reward_hacking_candidate | lower than static |
| teacher weights | interpretable, no collapse into numeric distance |

Only if v0.6c works on seed 0 should the project expand to seeds 1 and 2.

## Suggested roadmap

1. Commit current v0.6b/v0.6c work.
2. Finish v0.6c teacher-stability implementation.
3. Run v0.6c seed 0 under identical settings.
4. Evaluate validation, in-distribution, OOD-long, and OOD-division splits.
5. If v0.6c beats or matches static, run 3 seeds for:
   - static
   - adaptive v0.6b
   - adaptive-stable v0.6c
6. Add manual/random baselines only after the adaptive teacher is credible.
7. Run harness-shift matrix after reward adaptation is stable.
8. Start actual manuscript draft.

## Suggested honest interim claim

Until v0.6c or multi-seed results improve, the honest claim is:

> Dense legality rewards create a usable verifier-aligned learning signal for Countdown harness acquisition. A naive RTW adaptive teacher underperforms balanced static shaping on seed 0, suggesting that adaptive reward controllers require stability constraints such as delayed adaptation, legality floors, and numeric-distance caps.

This is a useful research finding, but not yet the main paper result.
