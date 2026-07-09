# Next Steps

Updated: 2026-07-09 (morning). This is the concrete execution plan; the
governing protocol is `AUTORESEARCH_PROGRAM.md`, results land in
`EXPERIMENT_LEDGER.md`.

## Where we are right now

- **Gate 0 (local baseline ladder) is running** on the A10G
  (`scripts/run_gate0_baseline_ladder.sh`, log
  `outputs/logs/gate0_ladder.log`). Static seed-0 training finished healthy
  (train-time diagnostics: open-tag 0.93, parseable 0.80, exact 0.029 — in
  family with v0.6b); Stable-RTW seed-0 is training now (~67% at last check).
  Remaining stages: 6 best-of-N evals (base/static/stable × validation/
  test_in_dist, ~25–30 min each at ~6 s/example) →
  `outputs/gate0_local_ladder_summary.csv` + `_paired.json`.
- **v0.10 task curriculum is implemented, advisor-reviewed, and committed**
  (`faf39ac`, `4455033`). Runner `scripts/run_v10_c2_pilot.sh` is ready and
  blocked only on the GPU.

## Step 1 — Score Gate 0 (when the ladder finishes; no GPU needed)

1. Read `outputs/gate0_local_ladder_summary.csv` and
   `outputs/gate0_local_ladder_paired.json`.
2. Pass criteria: local ladder ordering `base < static ≤ stable` on validation
   `reranked_exact@8`, directionally consistent with archived v0.9B
   (stable 0.133 vs static 0.067). Exact numbers will differ (new TRL/torch
   versions); direction is what matters.
3. Fill the `G0-repro` ledger row (keep/discard + lesson) and update the
   "current best local checkpoint" pointer in the ledger header.
4. If the ordering **fails** (stable ≤ static locally): stop, do not launch
   v0.10. Diagnose with `scripts/05_check_run_health.py` +
   `scripts/06_failure_taxonomy.py` on both checkpoints; the likely suspects
   are library-version drift (TRL 1.7 vs the v0.9B-era stack) or an unhealthy
   stable run. Escalate to the human if the archived result does not
   reproduce directionally — that would weaken the paper's baseline story.

## Step 2 — Launch v0.10 C2 (immediately after Gate 0 passes)

```bash
nohup ./scripts/run_v10_c2_pilot.sh > outputs/logs/v10_c2_pilot.log 2>&1 &
```

The runner self-gates: 60-step smoke → health check + curriculum-log
assertions (probs sum to 1, tier floor holds, no starved tier, probs move
after the 25-update delay) → 300-step pilot → best-of-N on frozen task IDs.
Budget: ~2.5 h train + ~1 h eval.

Watch for the smoke-specific failure: if `check_curriculum_log` aborts the
run, read `curriculum_state.jsonl` — tier collapse or frozen probs is a
controller bug, not a research result.

## Step 3 — Score v0.10 C2 vs C0 (the first curriculum verdict)

C0 = Gate 0 stable checkpoint (uniform sampling), C2 = adaptive curriculum.
Both: same machine, same commit family, same frozen protocol.

1. Paired per-task comparison on validation `reranked_exact@8`
   (`scripts/08_summarize_v09_seed_expansion.py` pattern, or direct candidate
   bank comparison), plus guardrails: test_in_dist, selected_valid,
   number F1, reward-hack rate, tokens/wall-clock.
2. Mediator check (this is the scientific payoff either way): compare
   `group_reward_std` / `batch_group_variance_fraction` between C0 and C2
   runs' `reward_components.jsonl`, and inspect tier occupancy in
   `curriculum_state.jsonl`. Three outcomes:
   - variance ↑ and exact ↑ → hypothesis supported (KEEP; proceed to C1 arm
     to show adaptivity matters, then seeds 1/2);
   - variance ↑ but exact flat → curriculum moves the mediator but the
     bottleneck is elsewhere — informative negative, record and apply the
     two-strike rule to the theme;
   - variance flat → controller not steering effectively; check tier
     occupancy before concluding anything about curricula.
3. Ledger row `v0.10-C2` + a short results section appended to
   `V10_TASK_CURRICULUM_PLAN.md` (plan → result → safe conclusion →
   overclaims to avoid, like the V06–V09 docs).

## Step 4 — Branch on the C2 verdict

- **KEEP:** run C1 (manual schedule) as the adaptivity ablation; then seeds
  1/2 for C2 under the frozen protocol; then draft the v0.10 section for the
  paper (two-curriculum story: reward curriculum + task curriculum).
- **DISCARD:** one revision attempt max (per the two-strike rule) — the most
  likely lever is the competence signal (τ/σ constants or the gate
  threshold), and only if tier occupancy shows the controller actually
  steered. Otherwise move to the next queue block: mechanism audit of Gate 0
  candidate banks (failure taxonomy: valid-but-wrong, clipping, selector
  near-misses) to find where exact candidates are lost.

## Standing queue after v0.10 (from AUTORESEARCH_PROGRAM.md §6)

1. v0.11 joint RTW × GACL (only if a v0.10 arm wins).
2. Throughput: vLLM or batched-generation path for best-of-N (~6 s/example
   is the experiment-rate bottleneck; a 2× win doubles iteration speed).
3. Multi-seed + OOD expansion for whatever method is best (frozen protocol).
4. Paper consolidation: plots/tables from the ledger + candidate banks.

## Standing rules (apply to every step)

- Commit before every CUDA run; one variable per iteration.
- Advisor checkpoints (program §5): design review before implementing, diff
  review before GPU spend.
- Nothing counts as correct unless it passes the verifier in
  `src/rtw_llm/countdown.py`; keep primary/aux/total rewards separately
  logged; frozen task IDs + sampling config for all v0.9-comparable evals.
