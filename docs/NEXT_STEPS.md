# Next Steps

Updated: 2026-07-09 (after the v0.10 C2 verdict). This is the concrete
execution plan; the governing protocol is `AUTORESEARCH_PROGRAM.md`, results
land in `EXPERIMENT_LEDGER.md`.

## Where we are right now

- **Gate 0 scored** (`GATE0_LOCAL_LADDER_REPORT.md`): infra PASS; method
  ordering (stable vs static) NOT CONFIRMED on 1 seed; stable keeps a ~2×
  cost advantage. Loss defaults pinned in `scripts/02_grpo_train.py`.
- **v0.10 C2 scored: DISCARD** (ledger row `v0.10-C2`, results section in
  `V10_TASK_CURRICULUM_PLAN.md`). Primary metric flat (val@8 0.10 vs 0.12,
  3-vs-4 discordants, p=1.0), cost regressed ~1.8× (longer completions).
  test_in_dist@8 showed 5-vs-0 discordants (p=0.062) — suggestive, but a
  guardrail split on one seed. Mechanistically: the controller steered as
  designed, but group reward variance (the mediator) was already ~97%
  saturated under Stable-RTW — difficulty mix is not the exact-search
  bottleneck at 0.5B. Theme has strike one; per the two-strike rule at most
  one revision (competence-signal retune), and only if the test_in_dist
  signal is judged worth ~3.5 h GPU.
- **Running now:** `scripts/run_gate0_seeds12.sh` →
  `outputs/logs/gate0_seeds12.log` (static+stable seeds 1/2, ~7 h; writes
  `outputs/gate0_local_ladder_seeds012_*`). Started ~07:48 UTC.
- **infra-batchgen landed** (default-off `--hf_gen_mode batched`); GPU
  acceptance gate still pending an idle window (see standing queue).

## Step 1 — Score Gate 0 seeds 1/2 (when the running job finishes)

1. Read `outputs/gate0_local_ladder_seeds012_summary.csv` and `_paired.json`
   (pools seeds 0/1/2 automatically — the seed-0 banks are already in the
   glob).
2. This settles the local baseline story with the same evidence standard as
   archived v0.9B (3 seeds, paired overlap):
   - stable > static locally → baseline story intact on the new stack; paper
     claims can cite local numbers;
   - indistinguishable or static ≥ stable → the v0.9B stable-vs-static
     advantage is stack-dependent. That reframes the paper: the robust claims
     become (a) shaping ≫ base, (b) best-of-N as harness mechanism, (c)
     stable's 2× cost advantage, (d) whatever v0.10 shows about task
     curricula on top of stable. **Escalate to the human before rewording the
     paper's main claim** (program §7 escalation rule).
3. Update `GATE0_LOCAL_LADDER_REPORT.md` with the 3-seed table and the
   ledger row `G0-seeds12`.

## Step 2 — Mechanism audit while the seeds-1/2 job holds the GPU (CPU-only)

C2's verdict was DISCARD and the mediator finding says difficulty mix is not
the bottleneck — so find out what is. Failure taxonomy over the local
candidate banks (Gate 0 base/static/stable + v10c2; all under
`outputs/bestofn/*_local_seed0_*`): valid-but-wrong vs unparseable vs
clipping-at-256 vs selector near-misses (oracle−practical gap is zero on
every local run so far — check that holds and why), where exact candidates
are lost, and whether C2's longer completions explain its higher capped
rate. Existing tool: `scripts/06_failure_taxonomy.py` (extend additively if
needed). Output: a short doc + queue update; this determines whether the
C2 revision (competence retune) is worth its GPU cost, and what v0.11
should even target. The Gate 0 banks already hint: valid on only ~13–17% of
candidates and `correct_given_parseable` ≈ 0.04 — candidate formation and
target search are both open bottlenecks.

## Step 3 — Decide the C2 revision question (after Steps 1–2)

One revision attempt max (two-strike rule). Case FOR: test_in_dist 5-vs-0
discordants (p=0.062) + better legality/hack guardrails. Case AGAINST:
primary flat, cost regressed 1.8×, mediator saturated. The mechanism audit
(Step 2) should settle it; if it stays ambiguous, default to NO revision and
move to the queue (the theme has one strike; spend GPU where the audit says
the bottleneck is).

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
