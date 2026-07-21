# Paper 1 outline — shaping-vs-capability characterization (rewritten 2026-07-20)

**This supersedes the pre-pivot "Stable-RTW method-win" outline** (archived in
git history at/ before commit `85c24a4`). Framing decision: **option (b)** —
recenter on the shaping-vs-capability characterization + generation-bottleneck
diagnosis, with cost and robustness as honestly-scoped secondaries (resolved
2026-07-20; was escalated-to-human). Evidence: `EXPERIMENT_LEDGER.md`,
`PAPER1_ASSETS.md` (auto-generated, `scripts/17`), numbers re-verified in
`RESEARCH_AGENDA_20260720.md`. Literature positioning: §related-work below,
citations verified in the agenda's Part 4.

---

## Working title

**"When is adaptive reward shaping structurally inert? A shaping-vs-capability
characterization of RL post-training on a strict-verifier task."**

Alt: *"Training wheels with nothing to grip: three preconditions for adaptive
reward/curriculum control in verifier-based LLM RL."*

## One-sentence claim

> On a strict-verifier task whose reward surface fails three measurable
> "adaptivity preconditions," inference-time selection saturates and two
> pre-registered reward/curriculum-shaping levers move intermediate quantities
> but not task success, while a cheap capability lever (SFT on gold
> completions) moves held-out success ~4–5× under the identical frozen
> protocol — locating the bottleneck in generation capability, not reward
> composition.

**Explicitly NOT** an adaptivity-wins paper. The contribution is the diagnostic
method + the mechanism (the preconditions) + the clean shaping-vs-capability
dissociation. Stable-RTW appears as a *cost* result and as the controller whose
structural inertness is explained, not as a success story.

## Abstract draft

Reinforcement learning with verifiable rewards (RLVR) is the dominant recipe for
post-training LLMs on tasks with strict checkers. A recurring design instinct is
to add dense auxiliary rewards and to *adapt* their weights (or the task
curriculum) during training — training wheels that should matter early and fade
as competence grows. We ask when this adaptive control actually helps, using a
controlled Countdown expression-generation harness with an AST/exact-rational
verifier as the sole source of truth. Through a five-lens diagnosis over 6,400
candidate records and eight checkpoints, we show: (1) inference-time
verifier-guided selection *saturates* — a reranker matches oracle exactness in
60/60 candidate banks, so ~91% of unsolved tasks form no correct candidate and
the bottleneck is generation, not selection; (2) two pre-registered shaping
levers — an adaptive reward-weight envelope and an adaptive task-difficulty
curriculum — move intermediate quantities (candidate legality 0.13→0.19,
difficulty mix) but leave exact success within noise, because success factors as
legality × P(exact | legal) and shaping does not touch the value-search wall;
(3) a cheap capability lever — supervised fine-tuning on ~2,000 gold completions
followed by the identical GRPO — moves held-out oracle success ~4.1–4.5× (from
~5/50 to ~20–24/50 across three seeds, ~90% novel expressions, non-overlapping
distributions). We explain *why* the adaptive controller is inert here through
three falsifiable **adaptivity preconditions**: dense within-group reward
variance (Countdown GRPO groups are ~97% variance-saturated), a live
reward-hacking surface (the auxiliary budget is capped below the primary reward,
so the reward is non-hackable by construction), and a smooth
difficulty/competence gradient (exact success is bimodal, difficulty tiers
collapse). We further show, via an assumption-free per-group oracle analysis,
that *no* auxiliary-weight controller of any strength — not merely our heuristic
one — can convert a wrong-preferred group into a correct-preferred one on this
reward surface. As secondary, honestly-scoped results: the stability-constrained
adaptive teacher reaches equal exactness at ~0.58× the tokens (2.7–3.3× the
cross-seed noise), and pre-registered harness-shift and out-of-distribution
probes yield near-null method differences. The preconditions are a portable,
cheap-to-measure diagnostic for when adaptive reward/curriculum control has
something to grip — and a principled bridge to a testbed (Paper 2) that
satisfies them by construction.

## Core reframing: from "which teacher wins" to "when does adaptivity bite"

The project began as a transfer of two robotics ideas — Reward Training Wheels
(RTW; adaptive auxiliary-reward weighting) and Grounded Adaptive Curriculum
Learning (GACL) — into LLM post-training, with a 3-pillar thesis that adaptive
control beats fixed schedules on sample-efficiency, reward-hacking-resistance,
and robustness. On Countdown-at-0.5B all three pillars proved structurally inert
(see §5). The honest and more valuable paper is the **diagnosis of why**, which
yields the preconditions and a clean shaping-vs-capability dissociation. This is
the paper.

## Contributions

1. **A five-lens generation-vs-selection diagnosis** showing exact-solution rate
   is capped by generation (legality wall → value-search wall → tier collapse),
   not selection. Method reusable for any strict-verifier RLVR task.
2. **A pre-registered shaping-vs-capability dissociation** under one frozen
   protocol: two shaping strikes (reward-weight envelope, task curriculum) vs
   one capability lever (SFT), held-out and memorization-controlled.
3. **Three falsifiable adaptivity preconditions** (dense within-group variance,
   live hackable proxy, smooth competence gradient) that predict a priori when
   adaptive reward/curriculum control is inert — with Countdown measured to fail
   all three, and a controller-independent oracle argument that reweighting
   cannot rescue a correct completion here.
4. **Honestly-scoped secondaries:** a reproducible cost result (stable ~0.58×
   tokens) and pre-registered near-null harness-shift/OOD robustness.

## Claim architecture (evidence status)

| # | claim | evidence | status |
|---|---|---|---|
| C1 | Selection saturates: reranked@N == oracle@N (60/60 banks); ~91% of losses form no exact candidate | diag-bottleneck; PAPER1_ASSETS C1 | done, verified |
| C2 | Shaping moves intermediates, not success (v0.10 curriculum, v0.12 legality — two pre-registered strikes) | ledger v0.10-C2, v0.12-legality | done |
| C3 | A capability lever moves both walls ~4.1–4.5× under identical protocol; gains held-out, ~90% novel expressions | v0.13-sft, v13-verify, v13-seeds12 | done, 3-seed |
| C4 | Mechanism: the three preconditions; controller inert because variance-saturated + non-hackable + bimodal; oracle cannot rescue a correct completion | diag-bottleneck, strat-repivot, postsft-precondition, **s5-bandit-replay** | done |
| C5 | Cost: stable ~0.58× tokens at equal exact, 2.7–3.3× above seed noise | rank-4 CPU audit; PAPER1_ASSETS C5 | done |
| C6 | Robustness (scoped): harness-shift + OOD pre-registered, 3-seed interaction labeled underpowered; near-null | HARNESS_OOD contract; PAPER1_ASSETS C6 | done |

Framing rule: **C3 is never presented as an adaptivity win** — it is the
positive arm of the characterization ("only a capability lever advances
exactness"). "Hard = capability floor" is scoped to RL-from-base.

## Verified numbers (for the tables — from PAPER1_ASSETS.md / bandit_replay.json)

- **C1:** reranked@8 == oracle@8 gap = 0 in **60/60** banks. Loss decomposition
  at N=8: oracle-selected 8.75%, misselected 0%, no-candidate 91.25%.
- **C2:** v0.12 legality 0.13→0.19, **P(exact|legal) flat 0.135→0.130**,
  exact@8 within noise. v0.10 curriculum val@8 0.10 vs 0.12, cost +1.8×.
- **C3:** easy legality 0.22→**1.00**; P(exact|legal) 0.235 (val)/0.257 (test).
  oracle@8=rerank@8: **val 22/18/21** (μ 20.3, 4.1×) vs stable 6/5/4 (μ 5.0);
  **test 25/21/26** (μ 24.0, 4.5×) vs stable 2/6/8 (μ 5.33). Non-overlapping.
- **C4:** GRPO groups ~97% variance-saturated; max incorrect total_reward 1.10 <
  2.20 correct floor (non-hackable); exact bimodal, tiers easy 25%/med 4%/hard
  0.4%. **S5 oracle:** reweighting flips top-1 in ~70% of groups but 0.0% among
  the ~9% groups with a correct completion; **0 rescue candidates** on all 7
  streams (correct completions score 1.0 on all six aux → total ~2.2 dominates).
- **C5:** stable tokens/candidate 0.578× (val) / 0.601× (test), gap/noise
  3.3×/2.7×.
- **C6:** harness-shift advantage +0.003…+0.044, sign-inconsistent 3/4 cells;
  OOD legality transfers across number-count (v13 6-num 0.35 vs stable 0.02) not
  operators (novel `/`-adoption 0.00); exact at floor.

## Related work (positioning — full citations in RESEARCH_AGENDA_20260720.md Part 4)

Four buckets; the project must CITE and OUT-POSITION, not claim novel mechanism:
1. **RLVR / GRPO recipe** — DeepSeekMath (GRPO), DeepSeek-R1, Tulu 3. We study a
   meta-question inside this recipe.
2. **Adaptive/auxiliary reward weighting** — Ng-Harada-Russell 1999 (PBRS),
   Hu et al. 2020 (bilevel shaping-weight optimization), **Min et al. 2024
   DynaOpt** (EXP3 bandit multi-reward reweighting for LLM RL), Lu et al. 2025
   (dynamic reward weighting). **Our reweighting mechanism is NOT novel** — we
   position our teacher as a lightweight controller in this lineage and claim
   only the strict-verifier + capped-budget + inertness-diagnosis combination.
3. **Curriculum / difficulty for LLM RL** — DAPO dynamic sampling, **VCRL
   (Jiang et al. 2025, group-reward-variance curriculum, POSITIVE on math)**,
   MMR1, CDAS, Online-Difficulty-Filtering. These converge on "keep within-group
   variance high"; our preconditions restate that mechanism as a *predictive
   test for when it fails*. Frame the v0.10 null as a boundary condition
   (bimodal/~0.97-saturated Countdown vs graded math), NOT a contradiction.
4. **Reward hacking / verifier selection** — Skalse 2022 (unhackability), Gao
   2023 (overoptimization), Brown 2024 (coverage vs selection), Snell 2024,
   Huang 2025. We sit at the strict-verifier corner these critiques exclude
   (why selection saturates). **Countdown-Code (Khalifa et al. 2026)** is a
   near-namesake RLVR reward-hacking testbed with the same proxy-vs-true split
   Paper 2 uses — cite prominently; our distinct question is adaptive mitigation.
5. **RL-vs-SFT capability** — Yue et al. 2025 (RL sharpens, distillation
   expands), Chu et al. 2025 (SFT memorizes, RL generalizes). Position C3 as
   "capability injection beats capability reweighting when the base lacks
   coverage" = Yue's distillation carve-out; NOT "SFT > RL."

## Paper structure

1. **Introduction** — the adaptive-shaping instinct; the meta-question (when
   does it bite); Countdown as a controlled microscope; preview the three
   findings + preconditions.
2. **Task & harness** — Countdown; the strict verifier (never relaxed); dense
   diagnostics as training wheels; the frozen protocol (task IDs, sampling,
   selector-never-sees-exactness).
3. **The adaptive machinery under test** — Stable-RTW (delay/EMA/floors/caps/
   budget) and the GACL curriculum, stated as the *methods being characterized*.
4. **Diagnosis** — C1 selection saturation; the exact = legality × P(exact|legal)
   factorization; tier collapse.
5. **Shaping vs capability** — C2 two strikes; C3 SFT lever; the dissociation
   under one protocol; memorization control.
6. **Mechanism: the three preconditions** — C4; measure each on Countdown; the
   S5 oracle argument (controller-independent). This is the conceptual core.
7. **Secondaries** — C5 cost; C6 harness-shift/OOD (pre-registered near-null).
8. **Bridge to Paper 2** — the preconditions predict where adaptivity *can* be
   tested; MicroCode satisfies all three by construction.
9. **Limitations** — 0.5B scope; 50-task power; on-policy scope of the oracle
   argument; cost not mechanistically attributed to the weight vector.

## Claims to avoid (unchanged discipline)

- Stable-RTW globally beats static (it does not; only cost survives multi-seed).
- Any adaptivity-wins framing for C3 (it is a capability lever).
- The reweighting mechanism is novel (Hu 2020 / DynaOpt / Lu 2025 pre-empt it).
- The variance-precondition contradicts VCRL (it is a boundary condition).
- Exact correctness is solved; OOD/harness robustness is improved.
- n=candidate framing for any 3-seed interaction (protocol-forbidden).
- Selection improved because the selector used exactness (it never does).
