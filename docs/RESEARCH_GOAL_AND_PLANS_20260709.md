# Research goal and master plans (design / implementation / experiment / validation)

Created: 2026-07-09 (~23:40 UTC), after the v0.13 KEEP verdict and while the
v13 seeds-1/2 → OOD chain runs on GPU. This document consolidates the program
into one statement of the goal and four explicit plans. It synthesizes (and
defers detail to): `STRATEGIC_REPIVOT_20260709.md` (direction),
`BOTTLENECK_DIAGNOSIS_20260709.md` (mechanism), `NEXT_STEPS.md` (live queue),
`PAPER2_MICROCODE_TESTBED_SPEC.md` (Paper-2 testbed), `EXPERIMENT_LEDGER.md`
(evidence), `AUTORESEARCH_PROGRAM.md` (operating loop).

> **Protocol correction added 2026-07-09 (v0.14 seed gate):** all results
> below were produced under `countdown-legacy-v1`. Historical Stable/static
> “seeds 0/1/2” did not vary the GRPO trainer seed and are legacy stochastic
> repeats, not true training seeds. v0.13 seeds 1/2 vary the SFT training-loop
> seed while retaining GRPO 42. Numerical observations remain useful, but
> cross-training-seed uncertainty, “above seed noise,” and confirmatory
> multi-seed language are provisional until every compared arm is rerun under
> `countdown-true-seeds-v2`. No new GPU experiment should start before the
> manifest, dataset-audit, and clustered-statistics gates are completed.

---

## 0. Status review (what is true right now)

**Verified results (ledger-backed):**

- **v0.13 SFT-warmup → GRPO: KEEP, decisive, independently re-verified.**
  Easy candidate legality 0.22 → 1.00; P(exact|legal) 0.14 → 0.24–0.26 — both
  generation walls moved for the first time. oracle@8 = rerank@8 = 0.44 val /
  0.50 test vs stable ~0.10 (≈5× the measured ~9% ceiling). McNemar 16–23-vs-0–2
  against every stable seed. ~90% of exact solutions are novel expressions
  (memorization bounded). Decomposition ladder (test@8): base 0.02 → RL-only
  0.10 → SFT-only 0.32 → SFT+GRPO 0.50 — both stages contribute.
- **Two pre-registered reward-shaping strikes failed** (v0.10 task curriculum,
  v0.12 legality envelope): shaping moves intermediate quantities (difficulty
  mix, legality) but not exact. Selection has **zero** headroom
  (reranked@N == oracle@N in all 16 banks).
- **Adaptivity is structurally inert on Countdown-at-0.5B**: GRPO groups ~97%
  variance-saturated, reward non-hackable by construction, controller barely
  leaves uniform. Post-SFT re-check: still 2 of 3 preconditions unmet — do NOT
  run adaptive arms on post-SFT Countdown.
- **Surviving stable-vs-static claim is cost/efficiency**: ~0.58× tokens at
  equal-or-better exact, gap 2.7–3.3× the cross-seed noise on both splits.
- **Paper-2 MicroCode CPU variance gate: GO** (held-out pass-rate std 0.393,
  live hacking surface observable); framework generalization landed
  (`f53cb33`); GPU base-model probe is the remaining gate.

**In flight (one background chain, sequential, ~6–8 h):** v13 seeds 1/2
(`outputs/logs/v13_seeds12.log`) then OOD eval (`outputs/logs/ood_eval.log`,
v13sft arm + mandatory base arm). Do not queue GPU work behind it without
checking `nvidia-smi`.

**Pending human decision (blocks paper-claim edits only):** main-claim
rewording. The v13 result strongly favors **option (b) — recenter Paper 1 on
the shaping-vs-capability characterization** with cost as the scoped
stable-vs-static secondary. The goal statement below assumes (b); flag at
sign-off if you want (a) or (c) instead.

---

## 1. Research goal

> **When does adaptive reward/curriculum control help RL post-training of LLMs
> on strict-verifier (agentic) tasks — and when is it structurally inert?**
>
> **Paper 1 (Countdown, now):** a *shaping-vs-capability characterization*.
> We show that on a strict-verifier task whose reward surface fails three
> adaptivity preconditions, (i) inference-time selection saturates, (ii) two
> pre-registered shaping levers (reward weights, task curriculum) move
> intermediate quantities but not task success, and (iii) a cheap capability
> lever (SFT on 2000 gold completions) moves success ≈5× under the identical
> frozen protocol. Secondary, honestly scoped: Stable-RTW's ~0.58× token cost
> at equal exactness, and harness-shift/OOD robustness as pre-registered
> (likely near-null) results.
>
> **Paper 2 (MicroCode, north star):** test the adaptivity thesis where the
> preconditions *hold by construction* — a test-driven function-synthesis task
> with dense within-group reward variance, a live hackable proxy channel
> (visible vs held-out tests), and a smooth difficulty ladder. Headline
> experiment: under a deliberately mis-weighted (TEMPTATION) budget, does the
> adaptive teacher spontaneously down-weight the gamed proxy while static gets
> hacked?

The three **adaptivity preconditions** (the conceptual spine linking the two
papers — Paper 1 derives them from failure, Paper 2 constructs a testbed that
satisfies them):

1. **Dense within-group reward variance** — the controller needs a gradient to
   read (Countdown: ~97% saturated; MicroCode: fractional held-out pass rate).
2. **A live proxy/hacking surface** — resistance is only testable if the hack
   is reachable (Countdown: inert by construction; MicroCode: hardcode visible
   I/O).
3. **A smooth difficulty/competence gradient** — curriculum needs a climbable
   ordering (Countdown: bimodal exact, tier collapse; MicroCode: per-test
   partial credit across 6 rungs).

---

## 2. Design plan

### 2.1 Paper 1 — claim architecture (evidence status in brackets)

| # | claim | evidence | status |
|---|---|---|---|
| C1 | Inference-time selection saturates: reranked@N == oracle@N; 91% of losses form no exact candidate | diag-bottleneck, 16 banks | done, verified |
| C2 | Shaping moves intermediates, not success: two pre-registered strikes (v0.10 curriculum, v0.12 legality envelope) | ledger rows + plan docs | done |
| C3 | A capability/data lever moves both walls ≈5× under the identical protocol; gains are novel-expression, held-out-task | v0.13-sft + v13-verify | seed 0 done; **seeds 1/2 in flight (confirmatory)** |
| C4 | Mechanism: why the adaptive controller is inert here (variance saturation, non-hackable budget, bimodal difficulty) → the preconditions | diag-bottleneck, strat-repivot, postsft-precondition | done |
| C5 | Cost: stable ≈0.58× tokens at equal exact, 2.7–3.3× above seed noise | rank-4 CPU audit | done |
| C6 | Robustness (scoped): harness-shift (prompt_mid vs high) and OOD, pre-registered, 3-seed interaction labeled underpowered | `HARNESS_OOD_ANALYSIS_CONTRACT.md` | OOD in flight; harness-shift queued |

Framing rule: never present C3 as an adaptivity win; it is the *positive arm of
the characterization*. "Hard = capability floor" is scoped to RL-from-base.

### 2.2 Paper 2 — testbed and experiment design (full spec in `PAPER2_MICROCODE_TESTBED_SPEC.md`)

- **Task:** emit one Python function from signature + docstring + K visible
  tests; graded against held-out tests (source of truth). Solvable by
  construction; verifier mirrors `countdown.py`'s contract.
- **Reward channels:** `held_out_all_pass` (PRIMARY, binary) +
  `held_out_pass_rate` (dense, diagnostic-only) + `visible_pass_rate` (the
  hackable proxy) + scaffold/anti-cheat channels (prune dead ones after the
  base probe).
- **Difficulty:** 6 rungs R0–R5 binned to 3 tiers; smoothness lives in the
  test set (fractional credit), plus `ood_*` held-out template families.
- **Headline experiment:** HONEST vs TEMPTATION budget arms × static vs
  adaptive_stable; metric = proxy−primary gap vs step. Pre-registered
  precondition: TEMPTATION-static must actually reach the hack signature at
  0.5B before any resistance claim.
- **Gates before GPU training:** CPU variance gate (PASSED) → GPU base-model
  probe (`scripts/14_microcode_base_probe.py`, launch-ready) → only then the
  full build + pilot.

### 2.3 Design decisions already made (do not relitigate)

Kill list stands: adversarial-init, reward/curriculum-shaping-for-legality,
harder-Countdown, grammar-decoding-as-pillar, prompt_low arm, 1.5B sweep,
selector/reranker work, N>8-as-selection-win. One narrow deferred probe:
legality-phase curriculum on post-SFT medium/hard *legality* (not exact).

---

## 3. Implementation plan

Ordered; CPU items can interleave with the running GPU chain. Advisor
checkpoints (design review before implementing, diff review before GPU) apply
to every non-trivial item.

### Paper 1 (mostly built — consolidation work)

| # | task | notes |
|---|---|---|
| I1 | ~~Consolidate the two v13 scorers~~ **DONE** (`44c37f0`) | `scripts/12_score_v13.py` is canonical; the duplicate was deleted. OOD scorer `15_score_ood.py` + harness-shift scorer `16_score_harness_shift.py` also landed (`7721447`) — E0/E1 are launch-AND-score ready. |
| I2 | Ledger + docs updates when the chain lands | rows `v13-seeds12`, `ood-eval`; refresh `NEXT_STEPS.md`. |
| I3 | Paper asset script | one script that regenerates every plot/table from the ledger + candidate banks + `outputs/v13_score_*` (claims C1–C6). Additive only; no frozen-component edits. |
| I4 | Rewrite `CURRENT_PROJECT_STATUS_AND_PAPER_ASSESSMENT.md` + `PAPER_OUTLINE.md` around the characterization | **blocked on the escalated claim sign-off**; draft-ready otherwise. |

### Paper 2 (build only what each gate unlocks)

| # | task | gate |
|---|---|---|
| I5 | Run the base probe (`14_microcode_base_probe.py`) | next GPU idle window |
| I6 | Template library 12 → ~20–40 (`microcode_gen.py`), randomized names, `ood_*` families | probe GO |
| I7 | Sandbox hardening: spawned persistent worker (never fork-after-CUDA), instruction-count budget, rlimit-mem/no-network; document residual risk honestly | probe GO |
| I8 | Teacher tables for new aux keys (STABLE_FLOORS/CAPS/target_weight_sum); restrict to adaptive_stable/static/manual/random | probe GO |
| I9 | Dataset card + tests per invariant #4 (reference-passes-held-out in CI, hardcode scores visible=1/primary=0, bit-stable re-verification, metamorphic cross-checks) | with I6–I8 |
| I10 | MicroCode eval/best-of-N path + **frozen Paper-2 protocol** (task IDs, sampling config, selector analog) — pre-registered BEFORE the pilot | before E4 |

Standing constraints: one variable per iteration; default-off flags; commit
before every CUDA run; Countdown defaults stay byte-identical.

---

## 4. Experiment plan

Sequential on the single A10G. Each experiment gets a pre-registered plan doc
(V0x pattern: hypothesis → metrics → decision rule → overclaims to avoid).

| id | experiment | arms / budget | primary metric | decision rule | status |
|---|---|---|---|---|---|
| E0 | v13 seeds 1/2 + OOD | SFT+GRPO seeds 1,2; OOD = v13sft + base + 6 existing ckpts | 3-seed oracle/rerank@8 distribution; OOD legality panel + `/`-adoption vs base | confirmatory: KEEP stands unless seeds collapse toward stable's ~0.10; OOD answers "did SFT overfit the envelope?" | **RUNNING** (~6–8 h) |
| E1 | Harness-shift (rank 2) | prompt_mid vs prompt_high, 6 existing ckpts, validation first (~5–6 GPU-h) | parseable-span `number_multiset_f1` degradation, per contract | consistent-sign 3-seed stable advantage → scoped robustness claim; else honest near-null **closes pillar 3** | queued after E0 |
| E2 | Paper-2 base probe (go/no-go) | eval-only, R0–R2 few-shot, 0.5B | held_out_pass_rate > 0, executes > 0, within-group std > 0 at init | GO → I6–I10; NO-GO → 1.5B or SFT-format-warmup fallback; if those fail → MiniPipe fallback | queued after E1 |
| E3 | Paper-1 consolidation (CPU) | plots/tables (I3), scorer consolidation (I1), paper edits (I4, after sign-off) | — | Paper 1 submission-ready draft | interleave now |
| E4 | MicroCode pilot | static vs adaptive_stable, HONEST budget, 300 steps, seed 0, frozen Paper-2 protocol | paired held_out_all_pass@8 + dense-variance health (did groups stay unsaturated?) | both healthy + non-degenerate → E5; adaptive-vs-static difference recorded but NOT the headline yet | gated on E2 GO |
| E5 | Hacking experiment (Paper-2 headline) | HONEST vs TEMPTATION × static vs adaptive_stable, seed 0 | proxy−primary gap (visible − held_out + no_hardcoding rate) vs step | pre-check: TEMPTATION-static must show the hack signature, else pillar 2 is inert-from-the-other-side (report as such); then thesis test = does adaptive close the gap? | gated on E4 |
| E6 | Seed expansion + OOD-template transfer for E4/E5 keepers | seeds 1/2, frozen protocol | 3-seed distribution + McNemar | program standard: multi-seed only for seed-0 survivors | gated on E5 |
| E7 (deferred) | Post-SFT legality-phase curriculum probe on med/hard legality | one arm, only if a GPU window is otherwise idle | med/hard candidate legality (NOT exact) | narrow; pre-register that exact is not expected to move | optional |

GPU-spend rule (unchanged): Paper-1 GPU only for E0/E1; everything else
Paper-2 or CPU. Two consecutive failures of a theme end the theme.

---

## 5. Validation plan

### 5.1 Protocol invariants (every experiment)

- Verifier (`countdown.py` / `microcode.py`) is the sole source of truth;
  reward components logged separately (primary/aux/total).
- Frozen task IDs, sampling config (temp 0.7, top-p 0.95, max_new_tokens 256,
  seed 0), loop-mode generation only for protocol comparisons (batched = 
  exploratory tooling only, never mixed into a comparison).
- Commit before CUDA; smoke run (60 steps) + `05_check_run_health.py` gate
  before/after every full run; one variable per iteration.
- Advisor checkpoints: adversarial design review before implementing,
  multi-angle diff review before GPU spend (this caught 3 silent-corruption
  bugs in v0.10 and the `import os` legality hole in MicroCode).

### 5.2 Statistical discipline

- Paired per-task McNemar counts, never mean±std alone; score against the
  **multi-seed distribution**, never a single seed (the v12 test-artifact
  lesson); discordants 1–5 on 50 tasks = noise.
- The statistical unit for any stable-vs-static *interaction* (harness-shift,
  OOD) is the 3-seed policy comparison — label it underpowered; never sell an
  n=2400-candidate framing.
- Pre-register metrics + decision rules BEFORE each run (the contract docs);
  a null/negative result is ledger-worthy and gets recorded, not respun.

### 5.3 Per-claim validation requirements

| claim | validation required before it goes in a paper |
|---|---|
| C3 (SFT 5×) | 3-seed confirmation (E0); memorization control mandatory for every SFT-lineage arm (overlap-task + verbatim-gold splits — already 90% novel at seed 0); OOD transfer answered either way (E0) |
| C5 (cost 0.58×) | done (noise-floor test passed); state as *observed*, not mechanistically attributed to adaptivity (near-uniform weights caveat) |
| C6 (robustness) | pre-registered contract only; near-null reported as pillar-3 closure |
| Paper-2 pillar 1 (adaptivity helps) | dense-variance precondition must be *measured live* during E4 (group reward std logged per step), not assumed from the CPU gate |
| Paper-2 pillar 2 (hack resistance) | TEMPTATION-static must demonstrably hack first; sandbox soundness limits stated (AST whitelist is defense-in-depth, not sound — no "resistance" headline until the sandbox question in the spec is settled) |

### 5.4 Continual-learning bookkeeping (program §7)

Every iteration ends with: ledger row → current-best pointer update if changed
→ queue reorder in `NEXT_STEPS.md` → memory update → commit. Escalate to human
only for: frozen-list changes, multi-day compute, main-claim changes.

---

## 6. Risks and open decision points

1. **Escalated claim wording** (the only human-blocking item): recommend (b),
   recenter on the characterization — v13 gives it a strong positive arm.
2. **E0 downside:** seeds 1/2 regress toward stable — KEEP survives only if the
   3-seed distribution stays far above 0.10; pre-committed to reporting either
   way. OOD collapse would scope C3 to the training envelope (still a valid
   characterization datapoint, not a kill).
3. **E2 NO-GO risk:** 0.5B floor at R0–R2 would recreate Countdown sparsity —
   fallbacks pre-declared (1.5B probe, SFT format-warmup, MiniPipe).
4. **Pillar-2 inert-from-the-other-side:** hack unreachable in 300 steps at
   0.5B — pre-registered as a reportable outcome, not a surprise.
5. **Parallel-session drift:** two Claude sessions have worked this repo;
   always `git log` + working-tree check before building (the duplicate-scorer
   incident, now resolved in `44c37f0`).
6. **North-star drift guard:** no Countdown micro-optimization beyond E0/E1;
   Countdown is a saturated wind tunnel — its remaining job is to finish
   Paper 1's evidence table.
