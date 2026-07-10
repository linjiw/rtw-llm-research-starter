# Dataset card: Countdown-v2

Protocol ID: `countdown-dataset-v2`. Created for corrected, within-v2 RTW-LLM
experiments. This dataset does not overwrite or retroactively repair
`data/countdown/`, which remains the historical legacy-v1 artifact.

## Purpose

Countdown-v2 supports verifier-based SFT and RL experiments without semantic
split leakage or malformed operand counts. It is a finite frozen benchmark for
the shaping-vs-capability characterization and future Stable-RTW comparisons;
it is not sampled uniformly from a conceptual generator population.

## Frozen construction

Base seed: `20260710`. Proposal RNG: CPython 3.11 `random.Random`. Allocation
and row order use SHA-256 sort keys. Files are canonical UTF-8 JSONL with sorted
keys, fixed separators, and one LF per row.

| split | total | difficulty quotas | permitted use |
|---|---:|---|---|
| train | 5,000 | easy 900, medium 2,050, hard 2,050 | SFT/RL training |
| validation | 500 | easy 50, medium 225, hard 225 | method development |
| test_in_dist | 500 | easy 50, medium 225, hard 225 | one-shot confirmation |
| final_test_in_dist | 500 | easy 50, medium 225, hard 225 | sealed final paper read |
| test_ood_long | 500 | ood_long 500 | OOD confirmation |
| test_ood_division | 500 | ood_division 500 | OOD confirmation |

All 7,500 tasks have globally unique `(sorted numbers, target)` loose keys, a
stronger condition than exact semantic disjointness. In-distribution pools are
SHA-ordered before split allocation, so final-test rows are not the sequential
tail of duplicate rejection.

The existing easy specification has exactly 1,264 loose tasks; v2 uses 1,050.
This capacity prevents a balanced 5,000-row train split. The resulting quotas
are intentional and every analysis must report overall and macro-by-tier
metrics; the 50 easy rows in an evaluation split are descriptive on their own.

## Correctness and fields

The verifier in `src/rtw_llm/countdown.py` is the correctness source of truth.
Every stored `solution` and `completion` must pass it. Each row contains:

- versioned ID, split, difficulty, numbers, target, and ordered allowed ops;
- verifier-valid solution;
- current `prompt_low`, `prompt_mid`, `prompt_high`, and default prompt;
- SFT completion;
- generator protocol, operand count, proposal seed/index, and allocation seed.

The manifest binds the clean pre-generation code commit, source hashes,
runtime/recipe, artifact hashes, rejection statistics, and final-test policy.
It excludes itself from its artifact map and carries a digest over its core.

## Final-test access

`final_test_in_dist` is generated and hash-sealed but not released. Official
SFT, GRPO, eval, and best-of-N runners reject any input intersecting final IDs,
exact or loose semantic keys, or canonical row digests. This catches copied,
reserialized, subsetted, or appended rows.

Training access is never allowed. A future full final evaluation requires the
exact complete JSONL, no limit/task subset, and an explicit human-approved
release record bound to the dataset-manifest hash and the exact current frozen
Git commit. The lightweight script 03 is never release-authorized; final access
is restricted to strict-provenance best-of-N. V0.18 creates no release record
and evaluates no model on the final split.

## Generation and audit

From the reviewed clean code commit:

```bash
python scripts/18_generate_countdown_v2.py
python scripts/19_audit_countdown_v2.py
```

Generation refuses the legacy directory, any pre-existing v2 target, a dirty
worktree, or a non-CPython-3.11 runtime. It publishes by atomic directory rename.
The independent auditor hardcodes the frozen recipe and pinned legacy hashes,
checks verifier/prompt/schema/quota/disjointness/manifest/final-policy
conditions, and performs byte-identical replay.

## Intended analysis and comparability

- Primary: verifier-exact success over the full finite 500-task split.
- Secondary: macro average across easy/medium/hard and v0.17 task-clustered
  observed-panel inference.
- Training-seed generalization requires new true end-to-end seed runs.

V2 differs from legacy in leakage, malformed rows, prompt templates, train
size/mix, and evaluation size. A v2-minus-legacy difference is not a causal
method effect. Base, SFT-only, GRPO-only, static, Stable-RTW, and combined arms
needed by a claim must all be rerun within v2 at identical budgets.

## Limitations

- Narrow synthetic arithmetic task with construction artifacts.
- Unique-task sampling is first-arrival de-duplication, not uniform over the
  finite semantic space.
- Stored gold solutions make the procedural/technical runner seal important;
  repository access is not a cryptographic embargo against a malicious actor.
- One-shot test labels are forfeited if their outcomes change method choices.
