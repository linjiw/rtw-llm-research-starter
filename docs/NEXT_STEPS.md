# Next Steps

Updated: 2026-07-09 (~18:45 UTC, after the 5-lens bottleneck diagnosis). This
is the concrete execution plan; the governing protocol is
`AUTORESEARCH_PROGRAM.md`, results land in `EXPERIMENT_LEDGER.md`.

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

- **v0.12 legality-envelope pilot TRAINING** (~74% at 18:40 UTC; ledger
  `v0.12-legality` pending). Pre-registered prediction (diagnosis): it moves
  legality but not exact. See Step 1 scoring rule.
- **Gate 0 complete at 3 seeds**, **v0.10 C2 DISCARD**, **infra-batchgen KEEP
  as tooling** — all recorded in the ledger.

## Step 1 — Score v0.12 when the pilot finishes (pre-registered rule)

Build a best-of-N bank on FROZEN task IDs first (v12 has none), method name
`v12legality` (do NOT let script 08's name-inference glob it as `stable`).
Then check in order:
1. Legal-candidate rate — does it rise clearly above the static/stable
   13–19% band?
2. P(exact | legal) — if legality rose but this stayed ~0.14–0.18, shaping
   helped assembly, not search (the expected outcome).
3. **Decisive:** oracle_exact@8 and paired McNemar rerank@8 vs stable C0 on
   frozen IDs. Signal requires discordants > ~10 with a lopsided split;
   3–5/50 is noise (Gate 0 p=0.34).
4. Per-tier: expect medium legality↑ / medium exact flat; hard exact ~0.
5. Confirm selection still saturated (reranked@8 == oracle@8) so any gain is
   attributed to generation.

**DECISION:** legality↑ + exact within noise ⇒ **strike TWO for
reward-shaping-for-legality ⇒ retire that lever, pivot GPU to Step 2 (SFT
warmup).** Exact clearly beyond noise ⇒ legality WAS the binding constraint,
reward shaping stays alive.

## Step 2 — SFT warmup on verifier-generated legal solutions (top next GPU bet)

Rank-1 experiment from the diagnosis. **Data/capability lever — does NOT
consume the reward-shaping strike.** Attacks the legality wall directly:
generate verifier-checked legal, use-all-required expressions (easy+medium)
via the existing generator, SFT warmup (script 01), THEN standard Stable-RTW
GRPO, then best-of-N on frozen IDs.
- Predicted: if SFT ~doubles easy legality (22%→~40%) at unchanged
  P(exact|legal)≈0.18, easy oracle@8 ~25%→~35–40%, pooled oracle_exact@8
  ~9%→~12–15% (concentrated on easy). Gains bounded by exact = legality ×
  P(exact|legal).
- Risk: may cut sampling diversity or teach legality without search
  (v10c2/v12 pattern) — measure legality AND P(exact|legal) separately.
- Cost ~4–5 h GPU. Design doc → advisor review → diff review → GPU
  (program §5), one variable.

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
