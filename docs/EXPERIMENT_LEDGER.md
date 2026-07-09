# Experiment Ledger

One row per experiment iteration under `docs/AUTORESEARCH_PROGRAM.md`.
Primary metric: paired `reranked_exact@8` on frozen validation task IDs.
Guardrails: test_in_dist exact, selected_valid, number F1, reward-hack rate, cost.

Historical results (v0.5–v0.9B, pre-ledger) are archived in `docs/V0*.md` and
summarized in `docs/CURRENT_PROJECT_STATUS_AND_PAPER_ASSESSMENT.md`. Current
best method: **Stable-RTW (`adaptive_stable`) + verifier-guided best-of-N**.

| id | date | hypothesis / change | primary result | guardrails | verdict | lesson |
|---|---|---|---|---|---|---|
| G0-base | 2026-07-08 | Base-model best-of-N rungs of local ladder (N=1 done, N=8 queued) | base N=1: valid 0.00/0.02, exact 0.00/0.02 (val/test) | — | baseline | base model has near-zero legality; all signal comes from post-training |
| G0-repro | 2026-07-08 | Local seed-0 static + Stable-RTW 300-step reproduction matches archived v0.9B ordering | (running) | (pending) | pending | — |
| v0.10-impl | 2026-07-09 | Task-curriculum controller/sampler implemented and validated (design + diff advisor-reviewed; 3 confirmed bugs fixed pre-GPU: queue-refill duplication, id-based grouping, fail-open dataset guard) | 59 tests pass, CPU dry-run healthy | — | infra | adversarial review before GPU spend caught silent-experiment-corruption bugs unit tests missed |
| v0.10-C2 | — | Adaptive difficulty sampling (C2) beats uniform Stable-RTW (C0) on paired validation reranked_exact@8 | queued behind Gate 0 (`scripts/run_v10_c2_pilot.sh`) | (pending) | pending | — |
