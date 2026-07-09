# Strategic re-pivot: honest program direction (2026-07-09)

Method: 4-angle strategic workflow (harness-shift, OOD, thesis-salvage,
search-levers), each adversarially critiqued, synthesized. Triggered by the
bottleneck diagnosis showing 2 of 3 thesis pillars inert. This doc sets the
program direction; NEXT_STEPS holds the concrete queue.

## Honest situation (verified, not asserted)

The original 3-pillar adaptivity thesis is not recoverable as stated on
Countdown-at-0.5B:
- **Pillar 2 (reward-hacking resistance) is inert BY CONSTRUCTION**: the aux
  budget is capped below primary (`stable_target_weight_sum=1.20`, floors sum
  <0.5), so max incorrect total (1.10) can never reach the correct floor
  (2.20). A design lemma, not an empirical win.
- **Pillar 1 (adaptivity beats fixed) is unsupported**: the Stable-RTW
  controller barely leaves uniform (final weight L1 0.16, cross-seed std
  ~0.003), GRPO groups are ~97% variance-saturated, stable-vs-static exact is
  non-significant at 3 seeds (p=0.34). Only a COST differential survives.
- **Pillar 3 (robustness) never tested**, but the near-uniform-weights fact
  predicts a near-null stable-vs-static interaction there too.
- Root cause: a **generation-capability wall** (legality wall → P(exact|legal)
  =0.14 → oracle@8 ~9%, hard-tier at a 0.5B floor) that neither reward nor
  task-curriculum shaping can cross (v0.10, v0.12 both discarded).

## Verified confounds the critique surfaced (reshape all robustness work)

- **`extract_answer` falls back to full completion text** when no `<answer>`
  tag is present → `prompt_low` (which has no `<answer>` cue) contaminates
  BOTH format and number-assembly metrics. **Kill the prompt_low robustness
  arm**; use only prompt_mid-vs-prompt_high (both retain the cue).
- **Adversarial non-uniform-init is rigged by construction**: `_project_stable_
  weights` applies hardcoded floors, so a "bad init" is pulled back to the
  same place — not a fair test. **Killed.**
- **Frozen-uniform-controller cost ablation is incoherent**: the teacher's only
  output channel is the weight vector; near-uniform weights make
  frozen-uniform-stable ≡ static. **Killed**, replaced by the CPU noise-floor
  test (done — see below).
- **OOD contamination**: division/long first-50 contain trivial 1–2-number
  tasks → frozen with a len==5/len==6 filter (done).
- **Real eval cost is ~40–47 min/full-loop bank**, ~2× the workflow's initial
  estimate — budget accordingly.
- **3-seed interaction, not 400-candidate**: any harness/OOD stable-vs-static
  interaction's statistical unit is the 3-seed policy comparison
  (underpowered by design); do NOT sell an n=400-candidate framing as
  "well-powered".

## Recommended Paper 1 framing (the pivot)

**"Diagnosing why adaptive reward/curriculum shaping stalls on a
strict-verifier task: a shaping-vs-capability characterization."** Primary
contribution = the diagnostic method + mechanism finding, NOT an adaptivity
win:
1. Inference-time SELECTION saturates (reranked@N == oracle@N in all 16 banks).
2. Reward and task-difficulty shaping move intermediate quantities (legality,
   difficulty mix) but not exact, because exact = legality × P(exact|legal)
   and shaping can't touch the value-search wall (two pre-registered strikes).
3. The adaptive controller has almost nothing to bite on (variance-saturated
   groups, non-hackable-by-design reward) — the mechanism explaining WHY both
   pillars are structurally inert HERE.

Secondary, explicitly-scoped: **cost/efficiency** (stable ~0.58× tokens at
indistinguishable accuracy — now shown to exceed cross-seed noise, see below);
**robustness** (harness-shift + OOD as honestly-scoped directional near-nulls).
v0.13 SFT becomes the "only a capability lever advances exactness" evidence.
Do NOT frame as an adaptivity success; do NOT staple grammar-constrained
decoding on as a pillar (orthogonal to the thesis).

## Rank-4 result (CPU, done): the cost claim survives the noise floor

Cross-seed token-count from the 6 existing banks:

| split | static tok/cand (3 seeds) | stable tok/cand | ratio | gap/noise |
|---|---|---|---:|---:|
| validation | 134/115/101 (μ117 σ13) | 64/77/61 (μ68 σ7) | 0.58 | **3.3×** |
| test_in_dist | 119/111/93 (μ108 σ11) | 66/78/50 (μ65 σ11) | 0.60 | **2.7×** |

The gap is 2.7–3.3× the cross-seed noise on both splits → the ~0.58× cost
claim is a real, defensible efficiency result, not a single-seed artifact.
**No GPU needed.** (Caveat retained: not yet mechanistically attributed to
adaptivity — the near-uniform endpoint weights mean it likely comes from the
delay/EMA/low-LR training dynamics, not the weight vector; state as observed.)

## Re-prioritized queue (GPU spent ONLY on 1–3; 4–5 are CPU-forward)

1. **v0.13 SFT scoring** (in flight) — the only bet attacking the real wall.
   Load-bearing for the whole framing. Score with the memorization control
   (held-out vs overlap tasks). Exact materially above ~9% ceiling →
   capability-lever narrative confirmed; flat → wall is deeper, strengthens
   the Paper-2 pivot.
2. **Harness-shift eval**, prompt_mid-vs-prompt_high ONLY, on the 6 existing
   checkpoints (no retraining). Primary metric = parseable-span-restricted
   number_multiset_f1; interaction at 3-seed level, labeled underpowered.
   Stage validation-only first (~5–6 GPU-h).
3. **OOD legality/scope eval** on frozen OOD IDs (done) + a **base-model arm**
   (mandatory — `/` never in the 2000 fine-tune examples but the base has seen
   it; the question is whether RL narrowed a known operator). Report legality
   panel + `/`-adoption + truncation; exact as expected-floor.
4. **Cost mechanism** — done on CPU (above); a confirming multi-seed GRPO run
   is deferred and gated on need.
5. **Paper-2 agentic-coding testbed scoping** (CPU now): design a small
   verifier-based coding task WITH a real difficulty gradient and a
   hacking-prone (public-test-overfit) surface — the properties Countdown
   lacks. Generator + verifier + dataset card + tests (invariant #4). This is
   the north-star forward bet; no GPU to start.

## Kill list (protect the GPU budget)

Adversarial non-uniform init (rigged); any further reward/curriculum shaping
for legality (2 strikes); making Countdown harder (pushes below the 0.5B
floor); grammar-constrained decoding as a "pillar"; prompt_low robustness arm
(confounded); 3-seed interaction sold as well-powered; frozen-uniform cost
ablation (incoherent); a 2–4-run 1.5B model-size sweep this cycle (anecdote,
unverified memory headroom, relocates rather than answers the question).

## North-star path

**Pivot the contribution, don't keep hunting an adaptivity win on Countdown.**
Countdown-at-0.5B is a saturated wind tunnel: no smooth difficulty gradient,
~97% variance-saturated groups, reward non-hackable by design. Ship Paper 1 as
the shaping-vs-capability characterization (with cost + robustness as scoped
secondaries), and use the mechanism analysis as the principled bridge to
**Paper 2 = the same adaptive teacher pattern on agentic coding**, which
supplies exactly what Countdown lacks. If v0.13's capability lever also fails
to move exact, that STRENGTHENS the pivot (confirms the wall is a 0.5B/testbed
property). Guard against north-star drift into Countdown micro-optimizations.
