# RTW-LLM research agenda: organize + literature + goals (2026-07-20)

Consolidation checkpoint after a ~9-day pause (working tree clean since commit
`f851cc8`, Jul 11). Purpose: organize **what we built / learned / analyzed**,
position the project against the **related literature** (deep web research,
8 fields, citations adversarially verified), and set **sharp investigation
questions + short/long-term goals**.

Method: a background workflow (21 subagents, ~911k tokens) ran two concurrent
tracks — internal-corpus readers over the code + result artifacts, and one
web-research + citation-verification agent pair per field — then a synthesis.
All load-bearing numbers below were **re-verified by hand** against
`outputs/paper1_assets.json`, `docs/EXPERIMENT_LEDGER.md`, and source
(`teacher.py`, `microcode_base_probe_easy.json`). Companion docs:
`RESEARCH_GOAL_AND_PLANS_20260709.md` (master plan), `EXPERIMENT_LEDGER.md`
(evidence), `STRATEGIC_REPIVOT_20260709.md` (direction).

> **One-line status:** Paper-1 empirical work is complete and points to a
> *shaping-vs-capability characterization* built on three falsifiable
> **adaptivity preconditions**; Paper-2 (MicroCode) is gated GO and is the
> north-star where those preconditions are constructed to *hold*, so the
> hack-resistance thesis can finally be tested. The literature says our
> **mechanism** is not novel but our **negative-result diagnostic framework**
> and the **adaptive-proxy-downweighting mitigation** are.

---

## Part 1 — What we BUILT

Production-quality core; MicroCode is an explicit prototype. The task-agnostic
seams are what make Paper 2 cheap.

| Component | What it is | Status |
|---|---|---|
| `countdown.py` | Task generator + **AST/`Fraction`-exact verifier** — the sole source of truth. Multiset-strict number use, allowed-op check, unary ops disallowed (can't smuggle negatives), robust partial-diagnostics on parse failure. `to_components()` emits binary signals **and** a dense diagnostic bank. | production, tested |
| `teacher.py` | `RTWTeacher`, six strategies. `adaptive_stable` = `adaptive` + delay (50 steps) + low LR (0.10) + double-EMA smoothing + per-key floors/caps + a **weight-sum budget** (1.20) via 32-round iterative redistribution. Signal is `need = 1 − ema_aux[key]` (`teacher.py:227`), so a *saturating* component decays — the exact lever the hack-resistance headline needs. I8b adds an optional per-key `init_weights` vector (default-off, byte-identical). | production, instrumented |
| `rewards.py` | `RTWRewardManager` — TRL reward fn. **Task-agnostic** via `scorer` + `example_fields` dispatch; logs primary/aux/total separately; computes **group-level reward std positionally** (the exact signal that proved ~97% variance saturation). Teacher + curriculum are observe-only hooks. | production |
| `curriculum.py` | GACL `CurriculumController` + `CurriculumSampler`. Two-channel competence: `gate_key` (legality phase) → `graded_key` (exact phase). `uniform` mode is **bit-identical** to TRL `RepeatSampler` (fairness). | production |
| `microcode.py` / `microcode_gen.py` | Testbed 2 verifier + 12-template R0–R5 generator. Extract-fn-by-name, static AST legality (scans full completion), **deterministic `sys.settrace` instruction budget** (not wall-clock). Primary = `held_out_all_pass`; `visible_pass_rate` = the hackable proxy. | **prototype** |
| `engine.py`, `trl_compat.py`, `prompts.py`, `07_best_of_n_rerank.py` | HF/vLLM wrappers (batched-safety hardened); TRL API-drift shims; harness-shift prompts; best-of-N harness whose `practical_score` selector is **provably barred from exact correctness**. | production |
| Scripts `00–18` + `run_*.sh` + `sim_e5_*` | Full paper pipeline: data → SFT → GRPO → eval → analysis → probes → Paper-1 tables/figures from one JSON source. CPU E5 sims drive the *real* imported teacher. | production |

---

## Part 2 — What we LEARNED (verified numbers)

The empirical arc **v0.6b → v0.9B → diagnosis → v0.13 → MicroCode gate**:

1. **Selection saturates (C1).** `reranked@8 == oracle@8` in **60/60** banks;
   at N=8, 91.25% of lost tasks form **no exact candidate**. The bottleneck is
   **generation, not selection** → reranker/selector work retired.
2. **Two pre-registered shaping strikes (C2).**
   - *Strike 1 (v0.10 GACL difficulty curriculum):* val@8 0.10 vs 0.12,
     cost +1.8× → DISCARD.
   - *Strike 2 (v0.12 legality reward):* candidate legality rose 0.13→0.19 but
     **P(exact|legal) flat 0.135→0.130**, exact within noise → DISCARD.
   - Both moved *intermediates*, neither moved *success*.
3. **Capability lever wins (C3).** v0.13 SFT(~2000 gold)+GRPO drove easy
   legality **0.22→1.00** and oracle@8 = rerank@8 to **val 20.3/50 (4.1×)**,
   **test 24.0/50 (4.5×)** vs stable ~5/50 — 3-seed **non-overlapping**
   (val 22/18/21; test 25/21/26). P(exact|legal) 0.235/0.257. **~90% of exact
   solutions are novel expressions on held-out tasks** → genuine construction
   transfer, not memorization.
4. **Controller is structurally INERT on Countdown (C4)** because all three
   **adaptivity preconditions** fail: (a) GRPO groups ~97% variance-saturated;
   (b) reward non-hackable **by construction** (max incorrect total 1.10 <
   2.20 primary floor); (c) exact is bimodal, tiers collapse (easy 25% /
   med 4% / hard 0.4%). Post-SFT does **not** revive it (exact still
   0.49/0.07/0.07; group std 0.205).
5. **Cost is the only surviving stable-vs-static claim (C5).** stable
   **0.578×/0.601×** tokens at equal exactness, **3.3×/2.7×** over cross-seed
   noise. Stated as *observed*, not mechanistically attributed to the weights.
6. **Robustness near-null (C6, pre-registered).** Harness-shift advantage
   +0.003…+0.044, sign-inconsistent in **3 of 4** cells → pillar 3 closed. OOD
   transfers across **number-count** (v13 6-num legality 0.35 vs stable 0.02)
   but **not operators** (novel `/`-adoption 0.00); exact at floor.
7. **MicroCode base probe: decisive GO (E2).** held_out_pass_rate 0.31,
   oracle@8 0.90, **`frac_groups_with_variance = 1.0`**, within-group std 0.41,
   smooth R0 0.22→R1 0.35 — the dense-variance precondition Countdown lacked.
8. **E5 teacher mechanism: GO + a saved GPU-hour.** CPU sim on the *real*
   teacher confirmed adaptive_stable down-weights a saturating proxy
   0.20→0.145 and routes mass to held_out (→0.308) — **and caught a
   thesis-inverting E5 design bug** (raised-floor TEMPTATION handcuffs the
   adaptive arm) before any GPU spend.

---

## Part 3 — What we ANALYZED (the method is itself a contribution)

The diagnostic discipline is publishable in its own right:

- **Measure saturation first.** Decompose a metric and prove headroom before
  spending GPU. The loss decomposition (91% no-candidate) retired reranking;
  oracle@N MLE extrapolation killed decode-tuning; the post-SFT precondition
  re-check killed "adaptivity-on-SFT" on CPU evidence alone.
- **Pre-registration is routine.** Metrics, statistical unit, and power
  caveats declared *before* runs — so near-nulls (pillar 3) were reported
  honestly and the v0.12 strike was pre-committed, not rationalized.
- **Paired stats with honest power.** McNemar discordant counts, never
  mean±std alone; the tier-scoped probe computed the ~19/5 split needed for
  p<0.05 and concluded **no easy-tier method can clear significance on the
  50-task protocol** — guarding the "candidate-count as statistical unit" trap.
- **CPU precondition-gating drives the *real* teacher**, never a
  reimplementation — which is how the E5 sim caught a design bug for free.
- **Verifier is sole truth; selection never peeks.** Non-hackability of
  Countdown reward is a *demonstrated* property; health checks enforce
  separate primary/aux/total logging every run.
- **Negative results are first-class** — inertness is converted into a
  mechanistic explanation, which *is* the paper.

---

## Part 4 — Literature positioning (8 fields, citations verified HIGH)

Full landscape with per-citation verification in the workflow output. Headline
conclusions (each citation spot-checked against primary arXiv sources):

**Origin transfer (faithful in spirit, loosened in mechanism — disclose):**
- RTW = Wang, Xu, Lu, Xiao, *Reward Training Wheels* (arXiv **2503.15724**,
  IROS 2025). Origin teacher is a **policy-gradient RL agent** (state = history
  of weights+rewards, reward = student primary); baselines show **Reward
  Randomization is detrimental** → *learned adaptation*, not weight movement,
  is what helped. Ours is a lightweight heuristic controller — a
  **simplification**, to be stated plainly.
- GACL = Wang, Xu, Stone, Xiao, *Grounded Adaptive Curriculum Learning*
  (arXiv **2508.02988**, IROS 2025). Uses a VAE task generator + PAIRED regret
  antagonist; ours is difficulty-bin sampling. **NOTE: our repo docs already
  correctly expand GACL as "Grounded Adaptive Curriculum Learning"** — the
  "generative adversarial" phrasing that surfaced in synthesis was a scout
  prompt artifact, **not** a doc error. No correction needed.

**Mechanism is established prior art — must cite & benchmark, cannot claim novel:**
- Hu et al. 2020 (bilevel shaping-weight optimization, NeurIPS) — canonical
  formalization of `primary + Σ w·aux`.
- Min et al. 2024 **DynaOpt** (arXiv 2403.13578) — **bandit-driven multi-reward
  reweighting for LLM RL**, the closest mechanistic analog to our teacher.
- Lu et al. 2025 (arXiv 2509.11452) — dynamic reward weighting in online LLM RL.
- Foundations: Ng-Harada-Russell 1999 (PBRS policy invariance), Devlin-Kudenko
  2012 (dynamic potentials sound), Du 2018 (gradient-similarity aux weighting).

**Concurrent work that PRE-EMPTS pieces — cite prominently, out-position:**
- **VCRL** (Jiang et al. 2025, arXiv 2509.19803) — uses **GRPO group-reward
  variance as a curriculum signal** with *positive* math results. Overlaps our
  variance-precondition instrument **and** the GACL lever. Frame our v0.10 null
  as *task-property-dependent* (bimodal/~0.97-saturated Countdown vs graded
  math), i.e. we map the boundary VCRL operates inside — not a contradiction.
  The broader difficulty-curriculum field (DAPO dynamic sampling, MMR1,
  Online-Difficulty-Filtering, CDAS) all converge on "keep within-group
  variance high" — our preconditions *restate their mechanism as a predictive
  test for when it fails*.
- **Countdown-Code** (Khalifa et al. 2026, arXiv 2603.07084) — near-namesake
  RLVR testbed with the **same proxy-test-pass vs true-correctness split** that
  MicroCode rests on. Studies *emergence/generalization* of hacking; our
  distinct question is **adaptive mitigation**. Cite prominently; ideally
  benchmark on their open env.

**Debates our findings land inside:**
- RL-elicits-vs-teaches (Yue et al. NeurIPS 2025; Spurious Rewards; 1-shot
  RLVR; Gandhi et al. *on Countdown*): our C1 (selection saturates, 91% form no
  candidate) is a concrete "nothing to sharpen" case. Position C3 as
  **"capability injection beats capability reweighting when the base lacks
  coverage"** — exactly the **distillation carve-out** Yue et al. concede
  genuinely expands reasoning; reconcile with "SFT memorizes, RL generalizes"
  (Chu et al.) by noting that slogan compares *equal-data* SFT vs RL.
- Verifier selection / test-time compute (Brown et al. 2024 coverage-vs-
  selection; Snell 2024; Huang et al. 2025): we sit at the **strict-verifier
  corner these critiques exclude**, which is *why* selection saturates — a
  crisp per-task confirmation, not a new phenomenon.
- Code-test reward hacking (ImpossibleBench, SpecBench, EvilGenie, Countdown-
  Code): visible-vs-held-out is well-established as *measurement*; **no one
  shows an adaptive reward-weight controller that spontaneously down-weights a
  gamed visible-test proxy during training** — that intervention is our gap.

---

## Part 5 — Investigation questions (next-phase targets)

**Q0 (central).** *When does adaptive reward/curriculum control help RL
post-training on strict-verifier tasks, and when is it structurally inert?* —
answered by the three preconditions; both the Countdown null and the MicroCode
positive are load-bearing evidence for one argument.

**Q1 (the thesis test — E4).** Under an **HONEST** reward budget on MicroCode,
does `adaptive_stable` beat `static` on paired `held_out_all_pass@8`, **and
does group-reward variance stay unsaturated *during* training** (not just at
init)? → E4 pilot, 300 steps, seed 0, frozen protocol; log `group_reward_std`
per step. First paired positive test of adaptivity the project can run.

**Q2 (the headline — E5).** Under a **proxy-overweighted (TEMPTATION)** budget,
does the adaptive teacher spontaneously down-weight the gamed visible-test
proxy in the **closed loop** while static gets hacked — and does
**TEMPTATION-static demonstrably hack first** so the test is winnable? →
TEMPTATION = proxy-overweight per-key init vector; adaptive proxy floor <0.10;
run the `hack_wins` pre-check *before* any resistance claim. Watch the
need-driven-not-hack-aware failure mode (a hack co-inflating held_out weakens
protection).

**Q3 (strongest reviewer threat).** Is Countdown's inertness a property of
**precondition failure** or of our **weak heuristic controller**? A reviewer
can argue the null reflects the controller (ours ≠ the origin's policy-gradient
teacher; DynaOpt/Lu already do learned reweighting). → Add a DynaOpt-style
contextual-bandit teacher (mutable, default-off); **first CPU-forward**: replay
logged Countdown reward-component streams through it. If it is *also* inert on
~97%-saturated groups, precondition failure dominates (cite RR-detrimental
precedent). GPU only if the CPU replay is ambiguous.

**Q4 (reconcile with VCRL).** Why does group-reward variance drive a *positive*
curriculum in VCRL (math) but is *inert* in Countdown — is the difference
exactly the variance-saturation fraction the preconditions predict? → Compute
the saturation fraction on a graded-math task alongside Countdown (~0.97) and
MicroCode (1.0); show v0.10-null and VCRL-positive fall on opposite sides of a
threshold. Positions GACL-inertness as a **boundary condition**, not a
contradiction.

**Q5 (framework generality).** Are the three preconditions jointly **necessary**
or does one dominate — is #2 (a live hackable proxy) required for adaptive-vs-
static to separate, or is dense variance (#1) enough? → Treat E4 (#1, no #2) vs
E5-TEMPTATION (#1+#2) as a built-in ablation.

---

## Part 6 — Short-term goals (this cycle; single A10G, kill-list respected)

**S1 — Ship the Paper-1 characterization draft (CPU-only).** Resolve the
escalated claim-wording (recommend **option b**: recenter on shaping-vs-
capability + generation bottleneck; carry cost 0.58× and near-null robustness
as scoped secondaries; present the archived v0.9B raw-exactness edge as
*stack-sensitive* since it didn't reproduce on TRL 1.7). Regenerate C1–C6 via
`scripts/17/18`; execute I4 (rewrite `CURRENT_PROJECT_STATUS` + `PAPER_OUTLINE`);
fold the literature citations + out-positioning into `LITERATURE_POSITIONING.md`
(add Hu 2020, DynaOpt, Lu 2025, VCRL, Countdown-Code, Gao 2022, Ng 1999, Yue
2025, Chu 2025). **Success:** every claim backed by committed JSON; near-nulls
unsoftened; all mandatory prior art cited & differentiated; no
mechanism-novelty overclaim.

**S2 — Build the MicroCode CPU stack I6–I10 + re-confirm gates.** Templates
12→~20–40 (randomized fn names, `ood_*` families); teacher tables for new aux
keys; dataset card + tests per invariant #4 (reference-passes-its-own-held-out
in CI; a known hardcode scores visible=1/primary=0; bit-stable re-verification;
metamorphic cross-checks); pre-register the **frozen Paper-2 protocol** (task
IDs, sampling, selector analog) *before* any pilot. **Success:** tests+ruff
pass; expanded-library mock-variance keeps non-zero within-group std; component
correlation matrix published (dead channels pruned).

**S3 — Harden the MicroCode sandbox (I7) + honest residual-risk statement.**
Spawned persistent worker (never fork-after-CUDA), keep the deterministic
instruction budget, add rlimit-mem / no-network / read-only tmpfs; document that
the AST whitelist is defense-in-depth, not sound. **Success:** a pre-registered
gate stating **no hack-RESISTANCE headline until the sandbox-soundness question
is settled**; pilot-level risk bounded.

**S4 — Run the E4 MicroCode HONEST pilot (first Paper-2 GPU).** static vs
adaptive_stable, HONEST budget, 300 steps, seed 0, frozen protocol; commit
before CUDA; 60-step smoke + health gate; log `group_reward_std` per step; score
paired `held_out_all_pass@8`. **Success:** both arms healthy + non-degenerate;
variance stays unsaturated *during* training → unlocks E5; degeneracy → ledger a
NO-GO and trigger the 1.5B / SFT-format-warmup fallback.

**S5 — Stand up a DynaOpt-style bandit teacher (CPU-forward).** Implement as a
default-off strategy; replay committed Countdown reward-component streams
offline. **Success:** a zero-GPU answer to "is the null just a weak controller?"
— either also-inert (precondition failure dominates) or it moves (refine, don't
overturn). Directly defuses Q3 / the top reviewer risk.

---

## Part 7 — Long-term goals

**L1 — Land the Paper-2 headline (E5, closed-loop).** An adaptive teacher
spontaneously down-weights a gamed visible-test proxy under TEMPTATION while
static gets hacked, measured by the held-out verifier — the reward-hacking
stress test the robotics RTW never ran (its budget, like Countdown's, was
non-hackable by construction). Gated on E4 health + the `hack_wins` pre-check +
sandbox soundness.

**L2 — Establish the three-preconditions framework as a portable, predictive
diagnostic.** Show that a cheap pre-measurement (variance-saturation fraction,
proxy hackability, gradient smoothness) predicts **both** the Countdown null
**and** the MicroCode positive *before* spending training GPU — turning a
negative result into a reusable tool and reconciling with VCRL's positive.

**L3 — Generalize from MicroCode to real agentic coding** (HumanEval/MBPP-scale
harnesses with realistic held-out suites) — the stated north star; elevates
from a constructed demonstration to a deployment-relevant claim.

**L4 — Close the controller-faithfulness gap** with ≥1 near-original learned
teacher (policy-gradient or bandit) alongside the heuristic controller —
defends the inertness claim and positions us honestly in the Hu/DynaOpt/Lu
lineage.

**L5 — Adaptive grounding schedule** — GACL explicitly leaves its fixed ε
(reference-vs-synthetic mixing) open; adapting ε from competence is a small,
invited extension once MicroCode's `ood_*` families and a non-inert curriculum
regime exist.

---

## Part 8 — Risks

- **MicroCode 0.5B floor beyond the probe** (E2 GO was 20 tasks R0–R2 few-shot;
  R3–R5 or zero-shot GRPO could re-create sparsity). Fallbacks pre-declared
  (1.5B, SFT format-warmup, MiniPipe).
- **Hack unreachable in 300 steps at 0.5B** — the `hack_wins` pre-check may fail,
  collapsing the E5 contrast. Pre-registered as reportable, not a surprise.
- **Sandbox soundness** — AST whitelist is not sound (`object.__subclasses__`→os);
  E5 executes adversarially-optimized code. No resistance headline until settled.
- **Need-driven ≠ hack-aware** — adaptive protects held_out only because its EMA
  is observed low; a hack co-inflating held_out weakens protection. Live
  closed-loop failure mode invisible to the CPU check.
- **Controller-strength confound (Q3)** — if the bandit comparison isn't run, the
  inertness claim is attributable to a weak controller. Prior art makes this
  objection likely → S5 is a priority, not optional.
- **Concurrent-work scooping** — VCRL (2025) and Countdown-Code (2026) are recent
  and fast-moving; frame contributions as *mitigation* + *diagnostic framework*,
  not testbed/phenomenon novelty.
- **Paper-1 main-claim exposure** — archived v0.9B raw-exactness edge did **not**
  reproduce on TRL 1.7 (G0 p=0.34/0.63); only cost survives multi-seed. The
  escalated claim-wording sign-off is still open (blocks I4).
- **Statistical power** — 50-task splits mean discordants 1–5 are noise; every
  stable-vs-static interaction is a 3-seed comparison, underpowered by design.
  The n=candidate framing is forbidden by protocol.
