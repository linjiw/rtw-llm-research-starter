# Next Steps

Updated: 2026-07-09 (~23:50 UTC, after the v0.13 verdict). This is the
concrete execution plan; the governing protocol is `AUTORESEARCH_PROGRAM.md`,
results land in `EXPERIMENT_LEDGER.md`.

## HEADLINE: v0.13 SFT-warmup → KEEP, decisively (ledger `v0.13-sft`)

Both splits, seed 0, frozen protocol, scored by the pre-registered
`scripts/12_score_v13.py`: easy candidate legality 0.22 → **1.00** (incl.
held-out-task subsets), P(exact|legal) 0.14 → **0.24–0.26**, oracle@8 = 
rerank@8 = **0.44/0.50** vs stable ~0.10 (≈5× the measured ~9% ceiling),
McNemar arm-only 16–23 vs base-only 0–2 against every stable seed, first-ever
medium/hard exacts (all novel expressions, zero verbatim-gold, non-overlap
tasks). Guardrails all green: diversity ROSE 2.4×, cost DOWN, clip 0.000,
GRPO not inert (+11 net oracle tasks over SFT-only), memorization bounded.

**Program implication:** the generation wall is a trainable-DATA gap, not a
0.5B architecture floor. Two reward-shaping strikes couldn't move what one
epoch of 2000 gold completions + GRPO moved 5×. This is the
shaping-vs-capability characterization's positive arm, with pre-registered
guardrails. "Hard = capability floor" is re-scoped to RL-from-base.

### v0.13 follow-up queue (updated ~00:30 UTC 2026-07-10)

1. ~~Score the SFT-only test arm~~ **DONE** — full ladder on test@8:
   base 0.02 → stable 0.10 → SFT-only 0.32 → SFT+GRPO 0.50. Both stages
   contribute on both splits; SFT supplies the bigger jump, GRPO adds
   +0.18–0.22 absolute.
2. **RUNNING (one background task, sequential):** v0.13 seeds 1/2
   (`scripts/run_v13_seeds12.sh` → `outputs/logs/v13_seeds12.log`; SFT +
   GRPO + frozen best-of-N per seed, then 3-seed scoring to
   `outputs/v13_score_seeds012_*.json`), followed by the OOD eval
   (`scripts/run_ood_eval.sh` → `outputs/logs/ood_eval.log`, includes the
   v13sft arm + mandatory base arm). ~6–8 h total.
3. **When they land:** fill ledger rows `v13-seeds12` and `ood-eval`; the
   v13 3-seed check is confirmatory (seed-0 discordants 16–23-vs-0–2 are far
   beyond seed noise, but the program standard applies); OOD is the open
   question — did SFT overfit the 3–5-number/4-op envelope?
4. **Paper integration** after seeds 1/2: update
   `CURRENT_PROJECT_STATUS_AND_PAPER_ASSESSMENT.md` with the
   shaping-vs-capability contrast (2 strikes vs 5×) — pending the escalated
   claim-wording decision for the framing.
5. **Then:** rank-2 harness-shift eval; Paper-2 base probe
   (`scripts/14_microcode_base_probe.py`) in the next idle window.

## Program direction (from `STRATEGIC_REPIVOT_20260709.md`)

The 3-pillar adaptivity thesis is NOT recoverable on Countdown-at-0.5B (2 of 3
pillars structurally inert). **Paper 1 pivots to a shaping-vs-capability
CHARACTERIZATION** (the bottleneck diagnosis IS the contribution), carrying the
cost result (0.58× tokens, confirmed to exceed cross-seed noise 2.7–3.3×) and
robustness (harness-shift/OOD) as honestly-scoped secondaries. **Paper 2 = the
same adaptive teacher on agentic CODING** (`PAPER2_MICROCODE_TESTBED_SPEC.md`)
is the north-star bet — it supplies the dense reward variance + live hacking
surface Countdown lacks. GPU is spent ONLY on ranks 1–3 below; ranks 4–5 are
CPU-forward. Kill list (protect budget): adversarial-init, more
reward/curriculum-shaping-for-legality, harder-Countdown, grammar-decoding-as-
pillar, prompt_low robustness arm, 1.5B sweep this cycle.

Two must-fix FRAMEWORK bugs found (ledger `framework-bug`): (a)
`CurriculumController.competence()` climbs on binary exact post-legality (part
of why v0.10 was inert) — fix to read a graded channel; (b)
`RTWRewardManager.__call__` is Countdown-coupled — generalize for Paper 2.

## Program reframing (from `BOTTLENECK_DIAGNOSIS_20260709.md`, all 5 lenses verified solid)

The exact-solution gap is a **generation** problem, not a selection problem:
- **Selection has ZERO headroom** — `reranked_exact@N == oracle_exact@N` in
  all 16 banks; 91.25% of lost tasks form no exact candidate at all. Retire
  reranker/selector work.
- Generation fails as a **legality wall** (81–87% illegal, number-multiset
  assembly ≈52% of candidates) then a **value-search wall**
  (P(exact|legal)≈0.14). Model ceiling: oracle_exact@8 ≈ 9%.
- **Tier collapse**: exact tasks easy 25% / medium 4% / hard 0.4%. Hard
  (5-number) is a capability floor at 0.5B, not a method gap. Tier-balanced
  eval splits dilute any easy-tier gain ~2/3 → why comparisons look like
  noise.
- **Reward hacking is not a driver** (primary reward never leaks to wrong
  answers; max incorrect total_reward 1.10 < 2.20 correct floor).
- **Paper claim that survives:** stable-vs-static is a **cost/efficiency**
  claim only (~0.58× tokens, ~half clip rate, p≈1e-14), NOT accuracy or
  robustness. Scope all method claims to easy-tier. (Still escalated — see
  Step 4.)

## Where we are right now

- **v0.12 legality-envelope: DISCARD → STRIKE TWO** (ledger `v0.12-legality`,
  results in `V12_...PLAN.md`). Legality rose as designed (val 0.13→0.19)
  and P(exact|legal) stayed flat (0.135→0.130); exact@8 within noise (val
  6-vs-6 p=1.0; the test 8-vs-2 p=0.031 is a C0-seed0 low-outlier artifact —
  stable 3-seed test = 2/6/8, v12's 8 is z=+1.1 inside range). **Exactly the
  diagnosis prediction: reward shaping moves assembly, not the value-search
  wall.** Retire reward-shaping-for-legality. One keeper: v12 halved clip
  rate (0.16→0.07) and cut tokens ~40% — an efficiency finding.
- **v0.13 SFT→GRPO code path built, advisor-reviewed, merged** (commit
  `6339354`): `02_grpo_train.py --init_adapter_path` continues an SFT LoRA;
  `01_sft_warmup.py --completion_only_loss`; verified end-to-end on CPU;
  80+4 tests pass. Launch is unblocked (v12 strike-two gate cleared).
- **Gate 0 3 seeds**, **v0.10 C2 DISCARD**, **infra-batchgen KEEP as
  tooling** — recorded.

## Step 1 — Launch v0.13 SFT warmup (top GPU bet; UNBLOCKED, GPU idle)

Rank-1 experiment from the diagnosis. **Data/capability lever — does NOT
consume the reward-shaping strike** (that theme is now retired anyway). Plan:
`docs/V13_SFT_WARMUP_LEGALITY_PLAN.md` (advisor-amended A1–A7, diff-reviewed).
- SFT data already exists: 2000 train completions, all verifier-exact.
- Sequence: SFT (light: ≤100 steps, lr ≤5e-5, `--completion_only_loss`, all
  tiers) → GRPO stable `--init_adapter_path <sft>` (seed42 as C0, same budget)
  → best-of-N on frozen IDs + SFT-only eval arm.
- **Primary metric (A2, well-powered):** candidate-level easy-tier legality
  rate (n≈136) + P(exact|legal). oracle@8/McNemar are directional only —
  score against the stable 3-SEED distribution (val 4-6, test 2-8), never a
  single C0 seed (the v12 test artifact is the cautionary tale).
- Guardrails (A5): distinct-legal-expr@8 vs C0 (Probe B: already ~0.98/task —
  diversity collapse is the top risk); cost; GRPO-not-inert (group variance).
- Predicted: if SFT ~doubles easy legality at flat P(exact|legal)≈0.18, easy
  oracle@8 ~25%→~35–40%. But note the v12 caution: legality already rose
  without moving exact, so the real test is whether SFT raises P(exact|legal)
  (genuine search capability), not just legality again.
- Next action: write `scripts/run_v13_sft_pilot.sh` (SFT → GRPO → evals,
  self-gating like the v12 runner), commit, launch under nohup.

## Step 3 — CPU probes (do now / interleave; no GPU)

- **Probe A (scope metric):** freeze an easy-tier and easy+medium
  sub-metric + per-tier paired McNemar over existing banks. The honest
  comparison surface; needed to score Step 2 cleanly. Does not manufacture
  significance (discordants unchanged) but makes easy-tier effects legible.
- **Probe B (generation headroom):** marginal-new-exact per candidate index
  1→8 and 256-clip recovery upper bound. Gates whether any
  generation-budget/decode GPU run (longer max_new_tokens, temperature) is
  worth it — likely small (clipping is a minority of the 91% no-exact loss).

## Step 4 — Human input needed (blocking paper edits only, not experiments)

The diagnosis sharpens the escalated question. The defensible stable-vs-static
claim is now **cost/efficiency only** (~0.58× tokens, ~half clip rate,
p≈1e-14), scoped to easy-tier; the accuracy edge is not significant and the
robustness/variance claim does not survive. Options: (a) keep the archived
v0.9B exactness claim scoped to its stack and present TRL 1.7 as a
stack-sensitivity + efficiency finding (honest, strengthens reproducibility),
(b) recenter the paper on cost-per-exact + the generation-bottleneck
characterization (which is itself a clean, defensible contribution),
(c) spend more seeds to try to recover exactness significance. Experiments
continue regardless.

## Standing queue (from AUTORESEARCH_PROGRAM.md §6, re-prioritized by the diagnosis)

1. **SFT warmup (Step 2)** — now the top method bet (was not in the old queue).
2. Generation-decode / longer max_new_tokens — only if Probe B shows headroom.
3. Adversarial non-uniform init (static-bad vs stable-bad) — the one regime
   where RTW adaptivity could beat a fixed schedule; defer until SFT settles.
4. Multi-seed + OOD expansion for whatever method is best (frozen protocol).
5. Paper consolidation: plots/tables from the ledger + candidate banks +
   the bottleneck characterization.

**Struck from the queue by the diagnosis:** selector/reranker engineering,
reward-hacking fixes, more hard-tier-exposure curricula, F1-shaped aux,
adaptive-weight tuning from uniform init, N>8 as a selection win (see
`BOTTLENECK_DIAGNOSIS_20260709.md` Discarded directions).

## Standing rules (apply to every step)

- Commit before every CUDA run; one variable per iteration.
- Advisor checkpoints (program §5): design review before implementing, diff
  review before GPU spend.
- Nothing counts as correct unless it passes the verifier in
  `src/rtw_llm/countdown.py`; keep primary/aux/total rewards separately
  logged; frozen task IDs + sampling config for all v0.9-comparable evals.
