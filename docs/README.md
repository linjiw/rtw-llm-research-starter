# Docs index

Reading order for a fresh session: **operating docs first** (they say what is
true now and what to do next), then the design docs, then the archived
experiment rounds as needed.

## Operating docs (living — updated every iteration)

| doc | role |
|---|---|
| [RESEARCH_GOAL_AND_PLANS_20260709.md](RESEARCH_GOAL_AND_PLANS_20260709.md) | consolidated goal + design/implementation/experiment/validation plans (post-v0.13) |
| [NEXT_STEPS.md](NEXT_STEPS.md) | the current concrete plan: what runs now, what decision comes next |
| [AUTORESEARCH_PROGRAM.md](AUTORESEARCH_PROGRAM.md) | the research operating system: goal, frozen protocol, metric, loop (with advisor checkpoints), prioritized queue |
| [EXPERIMENT_LEDGER.md](EXPERIMENT_LEDGER.md) | one row per experiment: hypothesis, result, keep/discard, lesson |
| [STRATEGIC_REPIVOT_20260709.md](STRATEGIC_REPIVOT_20260709.md) | **current direction**: Paper 1 → shaping-vs-capability characterization; Paper 2 → agentic coding; kill list |
| [BOTTLENECK_DIAGNOSIS_20260709.md](BOTTLENECK_DIAGNOSIS_20260709.md) | 5-lens diagnosis (verified): exact-rate is generation-bound, not selection; reward-shaping can't cross the wall |
| [PAPER2_MICROCODE_TESTBED_SPEC.md](PAPER2_MICROCODE_TESTBED_SPEC.md) | recommended Paper-2 coding testbed (dense reward variance + live hacking surface) + framework fixes |
| [CURRENT_PROJECT_STATUS_AND_PAPER_ASSESSMENT.md](CURRENT_PROJECT_STATUS_AND_PAPER_ASSESSMENT.md) | empirical arc v0.6b→v0.9B and the paper-safe claim wording (pre-pivot; see STRATEGIC_REPIVOT for current) |

## Design docs (stable references)

| doc | role |
|---|---|
| [PROJECT_DESIGN.md](PROJECT_DESIGN.md) | research questions RQ1–RQ4, harness formalization, roadmap, failure modes |
| [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) | original v0.5/v0.6 experiment design |
| [PAPER_OUTLINE.md](PAPER_OUTLINE.md) | paper structure and claims |
| [LITERATURE_POSITIONING.md](LITERATURE_POSITIONING.md) | related work positioning |
| [NEXT_IDEAS.md](NEXT_IDEAS.md) | longer-horizon extensions (joint curricula, code harness, learned teacher) |
| [DATASET_CARD.md](DATASET_CARD.md) | Countdown dataset card |
| [HARDWARE_AND_INFRA.md](HARDWARE_AND_INFRA.md) | hardware notes (see also `outputs/STORAGE_LAYOUT.md`) |

## Experiment rounds (archival — plan → result → safe conclusion)

| round | doc | one-line outcome |
|---|---|---|
| v0.5 | [v05_cuda_bootstrap_report.md](v05_cuda_bootstrap_report.md) | CUDA GRPO learning signal confirmed |
| v0.6b | [V06B_ADAPTIVE_STATIC_SEED0_REPORT.md](V06B_ADAPTIVE_STATIC_SEED0_REPORT.md), [V06B_LEGALITY_TRACKER.md](V06B_LEGALITY_TRACKER.md) | dense legality rewards work; naive adaptive RTW underperforms static |
| v0.6c/d | [V06C_TEACHER_STABILITY.md](V06C_TEACHER_STABILITY.md), [V06D_SEED_EXPANSION_AND_PAPER_PLAN.md](V06D_SEED_EXPANSION_AND_PAPER_PLAN.md) | Stable-RTW (floors/caps/budget) fixes teacher instability |
| v0.7 | [V07_PROMPT_LENGTH_DIAGNOSTIC_AND_PAPER_PLAN.md](V07_PROMPT_LENGTH_DIAGNOSTIC_AND_PAPER_PLAN.md) | exactness not a prompt-length artifact |
| v0.8 | [V08_CONDITIONAL_LEGALITY_TO_EXACTNESS_PLAN.md](V08_CONDITIONAL_LEGALITY_TO_EXACTNESS_PLAN.md) | Phased-RTW no better; bottleneck is search/reranking, stop teacher tweaks |
| v0.9/B | [V09_BEST_OF_N_RERANKING_PLAN.md](V09_BEST_OF_N_RERANKING_PLAN.md) | verifier-guided best-of-N; Stable-RTW advantage attenuated to non-significant on the TRL 1.7 stack (see Gate 0 3-seed) |
| v0.10 | [V10_TASK_CURRICULUM_PLAN.md](V10_TASK_CURRICULUM_PLAN.md) | GACL task curriculum — DISCARD (mediator saturated); strike one |
| v0.12 | [V12_NUMBER_LEGALITY_REWARD_PLAN.md](V12_NUMBER_LEGALITY_REWARD_PLAN.md) | legality-weight envelope — DISCARD, strike two; retired reward-shaping-for-legality |
| v0.13 | [V13_SFT_WARMUP_LEGALITY_PLAN.md](V13_SFT_WARMUP_LEGALITY_PLAN.md) | SFT warmup (capability lever) — running; strong early signal (both walls moved at train time) |
| v0.14 | [V14_SEED_SEMANTICS_PROTOCOL_PLAN.md](V14_SEED_SEMANTICS_PROTOCOL_PLAN.md) | protocol gate — explicit legacy-v1/corrected-v2 SFT+GRPO seed semantics; legacy “3-seed” claims downgraded to stochastic repeats |
| v0.15 | [V15_RUN_PROVENANCE_MANIFEST_PLAN.md](V15_RUN_PROVENANCE_MANIFEST_PLAN.md) | protocol gate — content-addressed intent/result manifests across SFT, GRPO, and best-of-N |

## Snapshots (dated, superseded by operating docs)

| doc | note |
|---|---|
| [BASELINE_INVENTORY_AND_NEXT_EXPERIMENTS_20260708.md](BASELINE_INVENTORY_AND_NEXT_EXPERIMENTS_20260708.md) | baseline inventory that defined Gate 0; command templates still canonical |
