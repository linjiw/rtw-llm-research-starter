# Next Steps

Updated: 2026-07-09 (~17:15 UTC, after Gate 0 seeds 1/2 scored). This is the
concrete execution plan; the governing protocol is `AUTORESEARCH_PROGRAM.md`,
results land in `EXPERIMENT_LEDGER.md`.

## Where we are right now

- **Gate 0 complete at 3 seeds** (ledger `G0-seeds12`, seeds-1/2 section in
  `GATE0_LOCAL_LADDER_REPORT.md`): Stable-RTW directionally ahead of static
  on both splits at N=8 (val 0.100±0.020 vs 0.073±0.058; pooled discordants
  7-vs-3) but NOT significant (p=0.34) — the archived 11-vs-1/p=0.006
  advantage is attenuated on TRL 1.7. The robust cross-stack claim is
  **cost-per-exact** (stable ~0.58× tokens/wall-clock at equal-or-better
  exactness). ⚠ **ESCALATED TO HUMAN: whether to reword the paper's main
  claim** from "Stable-RTW improves best-of-N exactness" toward
  "stack-conditional exactness edge + robust 2× efficiency edge". No paper
  edits until answered.
- **v0.10 C2: DISCARD** (ledger `v0.10-C2`; results in
  `V10_TASK_CURRICULUM_PLAN.md`). Primary flat, cost regressed, mediator
  (group reward variance) already ~97% saturated. Curriculum theme closed at
  strike one — mechanism audit settled the revision question: NO revision.
- **Mechanism audit done** (`MECHANISM_AUDIT_LOCAL_BANKS_20260709.md`):
  ranking is not the bottleneck; number-set legality failures are 53–64% of
  trained candidates; clipping eats the rest. This defines v0.12.
- **GPU idle** as of ~17:05 UTC; infra-batchgen acceptance benchmark is the
  designated idle-window task (launched — see standing queue).

## Step 1 — v0.12 design: number-legality-targeted reward (in progress)

Findings (ledger row `audit-banks`): ranking is NOT the bottleneck (selector
near-misses = 0 on every bank — oracle == practical because exact candidates
always carry clean legality features); number-set legality failures are
53–64% of trained candidates; exactness is almost entirely easy-tier;
`no_answer_span` is 256-token clipping (v10c2 3× worse — its cost regression
bought truncation, not search).

**C2 revision question: settled, NO revision.** Difficulty-mix changes
cannot fix legality/truncation losses. Curriculum theme closed at strike
one with an informative mechanism.

## Step 2 — Next method iteration (design first, advisor-review, then GPU)

From the audit, in order of expected leverage (teacher-side, mutable):
1. **v0.12 number-legality-targeted reward** — attack the 60% class:
   raise the number_multiset_f1 floor/weight or gate aux reward on
   number-set legality in `adaptive_stable`. One variable; design doc
   before code (program §5), advisor review, then GPU (~3.5 h). Comparison:
   paired vs C0 (`grpo_stable_seed0_300` banks) — stable-vs-stable, so
   independent of the escalated paper-claim question.
2. **Truncation/close-tag shaping** — make capped rambles into scored
   attempts (v10c2 showed the cost of ignoring it).
Value-search help is premature until legality improves (legal-but-wrong is
only ~12% and mostly far from target).

## Step 3 — Human input needed (blocking paper edits only, not experiments)

The paper's main claim: 3-seed local evidence supports "directional exactness
edge, robust ~2× efficiency edge" rather than the archived "robust exactness
edge". Options: (a) keep the v0.9B claim scoped to its stack and add the
TRL 1.7 replication as a stack-sensitivity finding (honest and strengthens
the reproducibility story), (b) recenter the paper on cost-per-exact, (c)
spend more seeds to try to recover significance. Experiments continue on
v0.12 regardless.

## Standing queue after v0.10 (from AUTORESEARCH_PROGRAM.md §6)

1. v0.11 joint RTW × GACL (only if a v0.10 arm wins).
2. Throughput: **batched-generation path is implemented, default-off**
   (`--hf_gen_mode batched`, ledger row `infra-batchgen`,
   `docs/THROUGHPUT_BATCHED_BESTOFN_PLAN.md`). Remaining: run
   `scripts/09_benchmark_generation.py` (equivalence at ≥64 new tokens,
   distribution, timing at batch 8/16/32) in the first **idle-GPU** window —
   never alongside a training/eval job. Acceptance: ≥2× at batch 32.
   Loop mode stays the default for all v0.9/Gate-0-paired comparisons.
3. Multi-seed + OOD expansion for whatever method is best (frozen protocol).
4. Paper consolidation: plots/tables from the ledger + candidate banks.

## Standing rules (apply to every step)

- Commit before every CUDA run; one variable per iteration.
- Advisor checkpoints (program §5): design review before implementing, diff
  review before GPU spend.
- Nothing counts as correct unless it passes the verifier in
  `src/rtw_llm/countdown.py`; keep primary/aux/total rewards separately
  logged; frozen task IDs + sampling config for all v0.9-comparable evals.
