# Current Project Status and Paper Assessment

Snapshot time: `2026-07-08T15:19:37-04:00`

Repository: `/home/robotixx/rtw-llm-research-starter`

Current local head at snapshot time: `cfd76bf` (`Document v0.9B seed expansion result`).

## Executive summary

This repo has moved from a raw RTW-for-LLMs starter into a paper-shaped Countdown harness study with a cleaner current result:

> **Stable-RTW improves candidate legality/rankability enough that a fixed verifier-guided best-of-N harness can recover more exact solutions on validation. On test-in-distribution, best-of-N is clearly useful as a general harness mechanism, but the Stable-vs-static advantage is smaller and mixed.**

The strongest current claim is **not** universal Stable-RTW dominance. The defensible framing is:

1. Dense verifier-aligned auxiliary rewards teach a small LLM to produce parseable, partially legal Countdown expressions under a strict verifier.
2. Stable/phase-protected RTW reduces teacher instability relative to earlier adaptive variants.
3. Inference-time verifier-guided best-of-N is a real harness mechanism: sampled exact candidates can be recovered by a fixed practical reranker that does **not** use exact correctness as an input.
4. Stable-RTW shows a robust validation advantage under best-of-N across seeds, while test-in-distribution remains a broader harness benefit with only a small/mixed Stable-specific edge.

## Repository state

At this snapshot, the branch was `main` with a clean working tree after local validation. The local branch contained 20 commits not yet pushed to `origin/main`, spanning v0.6c through v0.9B.

Important archival files:

| Purpose | Path |
|---|---|
| Best-of-N plan and results | `docs/V09_BEST_OF_N_RERANKING_PLAN.md` |
| Paper outline | `docs/PAPER_OUTLINE.md` |
| Research design | `docs/PROJECT_DESIGN.md` |
| v0.9B aggregate metrics | `outputs/v09_seed_expansion_summary.csv` |
| v0.9B paired overlap | `outputs/v09_seed_expansion_paired.json` |
| Frozen validation task IDs | `outputs/v09_task_ids_validation_limit50.txt` |
| Frozen test-in-dist task IDs | `outputs/v09_task_ids_test_in_dist_limit50.txt` |

## Validation performed at this snapshot

Commands:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
uv run ruff check .
```

Observed results:

```text
42 passed in 0.30s
All checks passed!
```

The pytest plugin autoload guard is intentional on this host: a ROS Humble pytest plugin can otherwise be imported and fail on missing `lark`, unrelated to this repo.

## Current empirical arc

### v0.6b: dense legality rewards work, naive adaptive RTW underperforms static

The initial dense-reward comparison established that verifier-aligned auxiliary rewards create usable learning signal. However, the first adaptive RTW controller overreacted to dense components and generally underperformed static shaping on seed 0.

Safe conclusion from v0.6b:

> Dense verifier-aligned auxiliary rewards can move a small LLM from near-zero usable Countdown behavior into parseable and partially legal expression construction, but naive adaptive weighting was not yet better than static shaping.

### v0.6c/v0.6d: Stable-RTW improves the adaptive story

The project then shifted from unconstrained adaptive weights to a more stable teacher/controller variant. This changed the paper story away from “any adaptive RTW beats static” and toward:

> Reward weighting needs stability constraints; RTW is useful when it preserves legality pressure instead of prematurely suppressing training-wheel components.

### v0.7/v0.8: prompt/teacher diagnostics refine the bottleneck

The prompt-length and Phased-RTW diagnostics narrowed the mechanism:

- exact correctness is not just a prompt-length artifact;
- teacher shaping helps legality and rankability more than it directly solves target search;
- repeated small teacher tweaks should stop once evidence points to search/reranking as the bottleneck.

### v0.9/v0.9B: verifier-guided best-of-N becomes the main positive result

v0.9 tested frozen checkpoints under an inference-time harness: sample up to N candidates and select using a fixed practical verifier-style score.

The practical selector uses legality/distance/risk features such as valid expression, number multiset F1, allowed numbers/operators, numeric distance reward, and reward-hacking penalty. It does **not** use exact correctness as an input. Exactness is only evaluated afterward by the strict verifier.

## v0.9B controlled seed expansion

Design:

```text
methods: static_v06b, Stable-RTW/adaptive_stable_v06c
seeds: 0, 1, 2
splits: validation, test_in_dist
limit: 50 examples per split
N: 1,4,8 from max-N=8 prefixes
temperature: 0.7
top_p: 0.95
max_new_tokens: 256
same frozen task IDs per split
same selector
```

Invariant checks completed:

```text
12 / 12 run directories present
all candidate banks: 400 rows = 50 tasks x 8 candidates
metrics.json, summary.csv, and run_config.json present for every run
static/Stable task IDs match for every seed x split pair
oracle_exact@1 <= oracle_exact@4 <= oracle_exact@8 for every run
reranked_exact@N <= oracle_exact@N for every run
```

### Key aggregate results

| split | method | N | reranked_exact mean ± std | selected_valid mean ± std | selected_number_f1 mean ± std | reward_hack ↓ |
|---|---|---:|---:|---:|---:|---:|
| validation | static | 8 | 0.067 ± 0.050 | 0.540 ± 0.092 | 0.928 ± 0.023 | 0.453 |
| validation | Stable-RTW | 8 | 0.133 ± 0.012 | 0.680 ± 0.020 | 0.958 ± 0.012 | 0.313 |
| test_in_dist | static | 8 | 0.120 ± 0.020 | 0.667 ± 0.070 | 0.950 ± 0.030 | 0.320 |
| test_in_dist | Stable-RTW | 8 | 0.133 ± 0.042 | 0.733 ± 0.046 | 0.974 ± 0.001 | 0.267 |

### Paired overlap across all three seeds

| split | N | both | Stable-only | static-only | neither | McNemar/binomial p | Δ reranked exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 4 | 4 | 12 | 0 | 134 | 0.0005 | +0.080 ± 0.035 |
| validation | 8 | 9 | 11 | 1 | 129 | 0.0063 | +0.067 ± 0.042 |
| test_in_dist | 4 | 8 | 5 | 8 | 129 | 0.5811 | -0.020 ± 0.035 |
| test_in_dist | 8 | 12 | 8 | 6 | 124 | 0.7905 | +0.013 ± 0.050 |

### Interpretation

v0.9B is **Case A on validation** and **mixed/general-harness on test_in_dist**.

Validation is the cleanest result: Stable-RTW beats static at N=4 and N=8, the practical reranker matches oracle exactness, and paired overlap strongly favors Stable-RTW at N=8 (`Stable-only=11`, `static-only=1`, `p=0.0063`).

Test-in-distribution confirms that best-of-N helps both methods. Stable-RTW has slightly higher mean reranked exact at N=8 (`0.133` vs `0.120`) and cleaner selected legality, but paired overlap is mixed (`Stable-only=8`, `static-only=6`, `p=0.7905`). Static is also better at N=4 on exactness.

## Current paper-ready claim

Use this restrained wording:

> Verifier-guided best-of-N is a general inference-time harness mechanism. Stable-RTW shows a robust best-of-N advantage on validation and a smaller/mixed advantage on test-in-distribution. The result supports a two-stage harness claim: training-time reward stability improves legal candidate formation/rankability, while inference-time verifier selection converts some latent exact candidates into task success. The benefit must be reported with sampling budget, paired uncertainty, generated tokens, wall-clock time, and cost-per-exact.

Avoid these overclaims:

- Stable-RTW universally dominates static shaping.
- best-of-N improvement is exclusively a Stable-RTW effect.
- observed wall-clock differences prove static is intrinsically slower.
- exact correctness improved because the selector used exact correctness; it did not.

## Recommended next gate

Before adding OOD splits or more seeds, do one of the following:

1. **Paper consolidation path:** freeze v0.9B as the main result, update the paper outline around the two-stage harness story, and add plots/tables from `outputs/v09_seed_expansion_summary.csv` and `outputs/v09_seed_expansion_paired.json`.
2. **Mechanism audit path:** inspect candidate banks for failure modes: valid-but-wrong arithmetic, missing/all-extra numbers, max-token clipping, selector near-misses, and cases where oracle exact exists but practical selector would fail under alternative scoring.
3. **Cost audit path:** rerun a tiny controlled latency/token audit with identical process/cache conditions before making any method-specific cost comparison.

Do **not** expand to all splits/seeds without preserving the current fixed task IDs, selector, sampling config, and cost accounting.
