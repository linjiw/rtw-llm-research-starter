# Next Steps

Updated: 2026-07-09 (after Gate 0 completed). This is the concrete execution
plan; the governing protocol is `AUTORESEARCH_PROGRAM.md`, results land in
`EXPERIMENT_LEDGER.md`.

## Where we are right now

- **Gate 0 is done and scored** — see `GATE0_LOCAL_LADDER_REPORT.md`.
  Infra gate PASS (both trainings healthy; stable matches its archived range,
  val@8 = 0.12). Method-ordering gate NOT CONFIRMED on 1 seed: local static
  improved on the TRL 1.7 stack (val@8 = 0.14), paired stable-vs-static is
  1-vs-2 discordants, p = 1.0 — statistically indistinguishable. Stable keeps
  a ~2× token/wall-clock cost advantage at similar exactness.
- **Two GPU jobs are queued sequentially in one background task** (started
  2026-07-09 ~04:45 UTC):
  1. `scripts/run_v10_c2_pilot.sh` → `outputs/logs/v10_c2_pilot.log`
     (60-step smoke with curriculum-log gate → 300-step C2 pilot → frozen
     best-of-N; ~3.5 h);
  2. `scripts/run_gate0_seeds12.sh` → `outputs/logs/gate0_seeds12.log`
     (static+stable seeds 1/2 ladder to settle the stable-vs-static question
     locally; ~7 h; writes `outputs/gate0_local_ladder_seeds012_*`).
- Training-loss defaults (`loss_type=dapo`, `scale_rewards=group`, `beta=0`)
  are now pinned in `scripts/02_grpo_train.py` so future TRL bumps cannot
  silently change ladder dynamics; note the archive-era stack differed (KL
  penalty on), which is the leading explanation for the static shift.

## Step 1 — Score v0.10 C2 vs C0 (when the pilot finishes)

C0 = Gate 0 stable checkpoint (uniform sampling), C2 = adaptive curriculum.
Both: same machine, same commit family, same frozen protocol. Caveat from the
smoke: if `check_curriculum_log` aborted the run, read
`curriculum_state.jsonl` first — tier collapse or frozen probs is a
controller bug, not a research result.

1. Paired per-task comparison on validation `reranked_exact@8` (direct
   candidate-bank comparison against
   `outputs/bestofn/stable_local_seed0_*_limit50_n8`), plus guardrails:
   test_in_dist, selected_valid, number F1, reward-hack rate,
   tokens/wall-clock. One-seed caution learned from Gate 0: 50-task
   discordant counts of 1–3 are noise — state the count, not just the delta.
2. Mediator check (the scientific payoff either way): compare
   `group_reward_std` / `batch_group_variance_fraction` between C0 and C2
   `reward_components.jsonl`, and tier occupancy in `curriculum_state.jsonl`:
   - variance ↑ and exact ↑ → hypothesis supported (KEEP; proceed to C1 arm
     to show adaptivity matters, then seeds 1/2);
   - variance ↑ but exact flat → curriculum moves the mediator but the
     bottleneck is elsewhere — informative negative, record and apply the
     two-strike rule to the theme;
   - variance flat → controller not steering effectively; check tier
     occupancy before concluding anything about curricula.
3. Ledger row `v0.10-C2` + a results section appended to
   `V10_TASK_CURRICULUM_PLAN.md` (plan → result → safe conclusion →
   overclaims to avoid, like the V06–V09 docs).

## Step 2 — Score Gate 0 seeds 1/2 (when the second job finishes)

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

## Step 3 — Branch on the C2 verdict

- **KEEP:** run C1 (manual schedule) as the adaptivity ablation; then seeds
  1/2 for C2 under the frozen protocol; then draft the v0.10 section for the
  paper (two-curriculum story: reward curriculum + task curriculum).
- **DISCARD:** one revision attempt max (per the two-strike rule) — the most
  likely lever is the competence signal (τ/σ constants or the gate
  threshold), and only if tier occupancy shows the controller actually
  steered. Otherwise move to the next queue block: mechanism audit of the
  Gate 0 candidate banks (failure taxonomy: valid-but-wrong, clipping,
  selector near-misses) to find where exact candidates are lost. The Gate 0
  banks already hint at this: trained models produce valid expressions on
  only ~13–17% of candidates, and `correct_given_parseable` is ~0.04 — both
  candidate formation and target search remain open bottlenecks.

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
