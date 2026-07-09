# v0.13: SFT warmup to attack the legality wall (top diagnosis bet)

Created: 2026-07-09. Status: design v1 (pre-advisor-review).
Grounding: rank-1 experiment in `BOTTLENECK_DIAGNOSIS_20260709.md`.

## Why this, why now

The diagnosis (5 lenses, all verified solid) says the exact-solution gap is a
**generation** problem: selection has zero headroom, and generation fails at a
**legality wall** (81–87% of trained candidates illegal, number-multiset
assembly ≈52% of all candidates) before a value-search wall
(P(exact|legal)≈0.14). Reward-shaping levers (v0.10, v0.12) do not move
legality-to-exactness because they cannot teach the *syntax* of a legal,
use-all-required expression — they only reweight a signal the model already
gets. SFT is a **capability/data lever** and, per the program's two-strike
rule, does NOT consume the reward-shaping-for-legality strike.

## Key fact established during design (verified)

**The SFT data already exists and is 100% verifier-exact.** Every one of the
2000 `data/countdown/train.jsonl` rows has a `completion` field of the form
`<reasoning>One valid expression is (EXPR). It evaluates to T ...</reasoning>
<answer>(EXPR)</answer>` and a matching `solution`; all 2000 pass
`countdown.score_completion` as exact (easy 667/667, medium 667/667, hard
666/666). `scripts/01_sft_warmup.py` already trains on `prompt + "\n" +
completion` with LoRA. So the experiment needs **no new data generation** —
this is cheaper and cleaner than the diagnosis assumed.

## Hypothesis

SFT warmup on legal, use-all-required gold expressions raises the
legal-candidate rate of the post-GRPO policy; at the observed
P(exact|legal)≈0.18 on easy, that converts to more exact candidates and lifts
paired best-of-N exactness — concentrated on easy tier.

Predicted (grounded): if SFT ~doubles easy legality (22%→~40%) at unchanged
P(exact|legal), easy oracle_exact@8 ~25%→~35–40% of tasks, pooled
oracle_exact@8 ~9%→~12–15%. Hard stays ~0 (search wall, capability floor at
0.5B). **The measurement plan must separate legality from P(exact|legal)** so
a "legality up, exact flat" outcome (the v10c2/v12 pattern) is diagnosed, not
mistaken for failure.

## Advisor review (v1 → amended). Verdict: PROCEED WITH AMENDMENTS

The design review verified C0's config (byte-identical to the pinned TRL-1.7
GRPO config, LoRA r16/α32/all-linear — so C0 reuse is fair) and resolved the
open decisions below. Seven amendments, all folded in.

### BLOCKING amendments

**A1 — The SFT→GRPO code path does not exist; without it the run silently
re-does C0.** `scripts/02_grpo_train.py` passes `model=args.model_name` (base
string) and always attaches a fresh zero-init LoRA. Fix (Option A, the
one-variable change): add `--init_adapter_path`; when set, load
`PeftModel.from_pretrained(base, sft_adapter, is_trainable=True)` and pass
`peft_config=None` (continue the SFT LoRA — same base, same rank/target as
C0, only the LoRA init differs). Trap: `from_pretrained` defaults to
inference/frozen — `is_trainable=True` is mandatory or GRPO updates nothing;
and `peft_config` MUST be None or TRL stacks a second adapter (→ different
capacity, a confound). Unit test: flag set ⇒ `peft_config is None` and adapter
trainable. Eval path stays byte-identical to C0 (base string + one adapter).
Rejected: merge-into-base (changes the frozen eval base string), stacked
double-LoRA (capacity confound).

**A2 — Decision rule was underpowered for its own predicted effect.** Frozen
val easy = 17 tasks; predicted easy oracle@8 25%→35–40% = +2–3 tasks =
McNemar discordants ~2–3, below the plan's own ">~10 lopsided" bar. A real
win would be discarded as noise (Probe A confirmed: even pooled easy is
p=0.308 at these counts). **Re-registered primary evidence = candidate-level
easy-tier legality rate** (17×8 = 136 candidates, well-powered to detect
22%→40%) **+ per-candidate P(exact|legal)** over the legal subset. Task-level
oracle@8 / McNemar rerank@8 are **directional/confirmatory only**, not gating.

### IMPORTANT amendments

**A3 — Completion-only loss.** `01_sft_warmup.py` trains on
`prompt + "\n" + completion` with no masking; ~65–75% of gradient lands on
prompt tokens (prompt ≈121–134 tok, completion ≈43–63 tok). Mask the prompt
(response template on `<reasoning>` / TRL completion-only loss) so the small
2000-example budget teaches expression construction, not prompt reproduction.

**A4 — Light SFT schedule.** Script defaults (300 steps, lr 2e-4 = 2.4 epochs
at 40× the GRPO LR) are NOT light and invite entropy collapse. Use ≤~100
steps, lr ≤5e-5, ~1 epoch, LoRA. Log and pin the SFT seed (default 0).

**A5 — Diversity-collapse guardrail with a threshold + decision.** best-of-N
lives on candidate variety; Probe B measured current legal diversity at
**0.98 distinct valid expressions/task** (already near 1!) — SFT sharpening
could push oracle@8 *below* C0. Pre-register: if distinct-legal-expr@8 (easy)
drops materially below C0, or oracle@8 drops while per-sample legality rises,
that is **DISCARD-with-diagnosis** ("SFT collapsed exploration"), not a null.

**A6 — GRPO-inert check + SFT-only eval arm.** GRPO groups are already ~97%
variance-saturated; if SFT sharpens further, groups go zero-variance and the
GRPO phase does nothing (result = SFT-only, misattributed). Check
`group_has_variance` / `batch_group_variance_fraction` vs C0 during health.
Add a cheap **SFT-only eval arm** (eval the SFT adapter directly, ~1h, no
train) to decompose SFT-alone vs SFT+GRPO — this is design-decision #3's
"third arm", as eval-only not a third training run.

**A7 — Sequencing / two-strike.** Score v0.12 FIRST (single sequential GPU;
v12's likely legality-up-exact-flat is the strike-two gate that authorizes
this pivot). Keep `GRPOConfig seed=42` identical to C0 (the `--seed` flag only
seeds the teacher — do NOT "fix" that plumbing in this run, it'd be a second
variable).

### Resolved low-stakes decisions

- Tier mix: **all tiers** for SFT (legality syntax transfers; hard gold
  targets are short ≤63 tok so cost is negligible; hard exact stays hopeless
  regardless), decision surface = easy tier.
- SFT target: **full gold completion** (the reasoning carries zero search
  content — verified 0/2000 have >2 sentences — so answer-only barely differs,
  and full completion keeps the `<reasoning>…</reasoning>\n<answer>…</answer>`
  format the verifier/eval expect). The "legal-looking-but-wrong inflates
  legality" worry IS the hypothesized mechanism and is guarded by the A2
  legality-vs-P(exact|legal) split.
- Two-strike classification: SFT is a §4-mutable capability/data lever, NOT
  a reward-shaping change — does not consume the reward-shaping strike. Holds.

## Protocol (frozen, per program §3)

SFT warmup (script 01, LoRA, **light: ≤100 steps / lr ≤5e-5 / ~1 epoch /
completion-only loss / seed logged**, all tiers) → GRPO stable (script 02,
300 steps, `--init_adapter_path` = SFT adapter, `GRPOConfig seed=42` as C0,
same budget) → best-of-N on frozen task IDs (loop mode, N=1/4/8, both splits).
Plus an SFT-only eval arm (eval the SFT adapter directly). Health + smoke gate
(incl. group-variance check) before the 300-step GRPO.

## Measurement (A2-re-registered; separation is the whole point)

Report vs C0 on frozen IDs, per tier and pooled:
1. **PRIMARY (well-powered): candidate-level easy-tier legality rate**
   (136 candidates) — does the wall move above the 13–19% band?
2. **PRIMARY: per-candidate P(exact | legal)** over the legal subset — did
   search stay ~0.14–0.18 (predicted), i.e. legality helped assembly not
   search?
3. Directional/confirmatory: oracle_exact@8 and paired McNemar rerank@8
   (report counts; do NOT gate on them — underpowered per A2/Probe A).
4. Guardrail (A5): distinct-legal-expressions-per-task at N=8 vs C0.
5. Guardrail: cost (tokens, clip rate) — must not regress like v10c2.
6. Check: selection still saturated (reranked@8 == oracle@8); GRPO not inert
   (group variance vs C0).

## Decision rule (A2-re-registered)

**KEEP** if candidate-level easy-tier legality rises materially above the
13–19% band AND P(exact|legal) does not regress AND no diversity/cost
regression (A5) — with oracle@8 directionally up. This is the well-powered
surface and would be the first lever to move the generation ceiling
(a capability intervention the reward-only methods could not achieve).
**DISCARD-with-diagnosis** if legality flat (SFT didn't take) OR diversity
collapsed (oracle@8 down, A5) OR legality up but P(exact|legal) flat AND
oracle@8 flat (legality was not the binding constraint → the value-search
wall binds; pivot to value-search levers, NOT decode — Probe B ruled decode
out). Task-level McNemar rerank@8 is reported but does not gate (underpowered
per A2/Probe A: ~17 easy val tasks ⇒ ~2–3 discordants even on full success).

## Ledger row (to fill)

id `v0.13-sft`, hypothesis above, verdict per rule. Either way this is a
high-information result: it tests whether the legality wall is the binding
constraint or a symptom of a deeper search-capability floor.

## Scoring control: memorization overlap (added mid-run 2026-07-09)

Early GRPO signal is strong — v13's first-300 training completions: parseable
0.94 (C0 cold 0.43), valid 0.58 (C0 0.01), train-time exact 0.12 (C0 0.00).
The SFT warmup moved BOTH walls at train time. But SFT trained on all 2000
gold solutions, and eval tasks overlap training by (numbers, target):

- **No task-ID overlap** (clean id split), but frozen-50 (numbers,target)
  overlap with train: **validation 9/50 (all easy), test_in_dist 6/50
  (5 easy, 1 medium).**

So a v13 eval gain concentrated on those 9/6 tasks is MEMORIZATION, not
capability transfer. **Scoring MUST partition the paired comparison:**
1. Overlap tasks (9 val / 6 test): expected to improve, but discount as
   memorization — report separately, do not credit to the method.
2. Held-out tasks (41 val / 44 test): a gain HERE is genuine
   legality/search-capability transfer — this is the real result.
3. The SFT-only arm on overlap tasks bounds pure memorization (no GRPO).

If the gain is only on overlap tasks → v13 is a memorization artifact, DISCARD
for the capability claim (though it still validates the SFT→GRPO machinery).
If held-out tasks improve → the legality/search wall genuinely moved, KEEP.
Given exact is easy-tier-only and 9 of ~17 easy val tasks are overlap, the
held-out easy pool is small (~8) — lean on candidate-level legality
(well-powered, A2) on held-out tasks as the primary evidence.

## Mid-training trajectory (2026-07-09, ~80% through GRPO) — SFT warmup HOLDS

Deciles of GRPO training completions, v13 (SFT→GRPO) vs C0 (cold stable):

| frac | v13 valid | v13 exact | C0 valid | C0 exact |
|---:|---:|---:|---:|---:|
| 0.0 | 0.59 | 0.11 | 0.01 | 0.00 |
| 0.4 | 0.75 | 0.22 | 0.20 | 0.02 |
| 0.9 | 0.89 | 0.22 | 0.39 | 0.07 |

Key reads (train-time; eval is the real test):
- The SFT warmup does NOT decay under GRPO — legality *climbs* 0.59→0.89
  (C0 tops at 0.39, so v13 >2× C0's legality ceiling) and train-exact holds
  ~0.19–0.23 (C0 never exceeds 0.07, so ~3× sustained). GRPO builds ON the
  warmup rather than washing it out.
- GRPO is NOT inert (A6 concern): group-variance fraction stays 0.82–0.98,
  so SFT did not collapse exploration.
- This is qualitatively unlike v0.10/v0.12, which never moved train-exact off
  the floor. First evidence a capability lever crosses the generation wall.
- CAVEAT: train-exact partly reflects the (numbers,target) overlap between SFT
  training data and eval tasks (9/50 val, 6/50 test). The load-bearing test is
  best-of-N on HELD-OUT frozen tasks (pending) — partition per the memorization
  control above. A held-out gain = genuine transfer; overlap-only = memorization.

## RESULTS (2026-07-09, seed 0; scored by `scripts/12_score_v13.py`)

Scoring artifacts: `outputs/v13_score_validation.json` (+ test_in_dist when
complete). Protocol integrity verified: loop mode, sampling seed 0, frozen
IDs, identical generation config to the stable 3-seed baseline banks.

### Validation (vs stable 3-seed baseline distribution)

**Primary surfaces (A2):**

| surface | v13 SFT+GRPO | v13 SFT-only | stable 3-seed |
|---|---:|---:|---:|
| easy candidate legality (all) | **1.000** (136/136) | 0.640 | 0.213–0.228, pooled 0.223 |
| easy candidate legality (held-out tasks) | **1.000** (64/64) | 0.625 | 0.224 |
| P(exact\|legal), all tiers | **0.235** (77/328) | 0.150 | 0.077–0.135 |

Both primaries moved, on held-out tasks, far beyond the baseline seed spread
(two-proportion p≈0 for legality; P(exact|legal) nearly doubles the best
baseline seed). **This is the first lever to move BOTH walls** — v0.10/v0.12
moved neither at eval (v12 moved legality only, 0.13→0.19; v13 hits 1.00).

**Directional/confirmatory:** oracle@8 = rerank@8 = **0.440** vs stable
0.10±0.02 (~4.9× the ~9% ceiling the diagnosis measured). Held-out-task
oracle@8 = 0.317; overlap-task = 1.000. McNemar vs every stable seed:
arm-only 16–18 vs base-only 0 (held-out-only: 11–12 vs 0) — lopsided beyond
anything previously observed in this program.

**Per-tier decomposition (the wall picture moved):**

| tier | legality v13 / C0 | P(exact|legal) v13 / C0 | oracle@8 v13 / C0 |
|---|---:|---:|---:|
| easy | 1.00 / 0.23 | 0.50 / 0.23 | 16/17 / 6/17 |
| medium | 0.87 / 0.14 | 0.05 / 0.00 | 3/15 / 0/15 |
| hard | 0.61 / 0.03 | 0.05 / 0.00 | 3/18 / 0/18 |

Medium and hard produce their FIRST-EVER exact solves in this program — and
every medium/hard exact expression was checked against all 2000 SFT gold
solutions: **all novel, zero verbatim, all on non-overlap tasks** (e.g.
`(16-(12+(12-4)-7))` on a held-out hard task). The "hard is a 0.5B capability
floor" claim from the diagnosis is now scoped: it was a floor for
*RL-from-base*; SFT+GRPO moves it slightly (3/18), though hard remains
search-limited (P(exact|legal)=0.05).

**Memorization control:** overlap-task exact candidates split 9 verbatim-gold
vs 29 novel; held-out easy legality (1.00) equals overlap legality — the
legality gain is format/assembly capability, not recitation. Verbatim
recitation exists but is a minority channel even on overlap tasks.

**SFT-only vs SFT+GRPO decomposition (A6):** SFT-only reaches 0.640 easy
legality / 0.220 oracle@8; GRPO on top adds +0.36 legality and +11 oracle
tasks net (12 added — 6 easy, 3 medium, 3 hard — 1 hard lost). Both stages
contribute; GRPO is NOT inert (group-variance 0.89 vs C0 0.97) and NOT
redundant. The two-stage story (SFT teaches the syntax of legality, GRPO
consolidates and extends the search) holds.

**Guardrails:**
- Diversity (A5, top pre-registered risk): distinct legal expr/task easy =
  **3.76** vs baseline 1.47–1.65 — diversity ROSE ~2.4×; no collapse.
- Cost: 55 tok/cand (stable ~64–68), clip rate 0.000 (stable ~0.16). Cheaper
  AND never truncated.
- Selection saturation persists (rerank@8 == oracle@8): consistent with the
  diagnosis; the selector remains lossless on the new distribution.

### Verdict (validation; test_in_dist pending)

**KEEP — decisively.** Every pre-registered KEEP condition is met with
margin; every pre-registered failure mode (diversity collapse, legality-only,
GRPO-inert, memorization-only) is affirmatively ruled out by its guardrail.

### test_in_dist (vs stable 3-seed baseline distribution)

Replicates and exceeds validation: easy legality **1.000** (120/120; held-out
80/80) vs pooled 0.219; P(exact|legal) **0.257** vs 0.101–0.160;
oracle@8 = rerank@8 = **0.500** (held-out-task 0.432, overlap 1.000);
McNemar arm-only 18–23 vs base-only 0–2 against every stable seed (even this
underpowered surface clears p<1e-4 at 18-vs-0); diversity 4.00 distinct legal
expr/task vs 1.33–1.73; cost 54 tok/cand, clip 0.000; verbatim-gold 5 vs
novel 27 on overlap tasks. Both splits agree: **KEEP.**
