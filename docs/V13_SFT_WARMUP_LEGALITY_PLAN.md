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
