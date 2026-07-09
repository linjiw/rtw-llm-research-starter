# v0.12: legality-weight envelope (Stable-RTW variant)

Created: 2026-07-09. Status: design v2 (first design REDESIGNED by advisor
review; this version incorporates it).
Queue slot: from `MECHANISM_AUDIT_LOCAL_BANKS_20260709.md` — attack the
dominant failure class directly.

## Design-review history (why v2)

Design v1 proposed a new binary aux component `exact_number_multiset`
(= uses_all_required AND uses_no_extra). Advisor review REFUTED the premise
empirically, verified against the stable seed-0 validation bank (400
candidates): `valid_expression` = parse_ok AND exact multiset AND allowed_ops
AND numeric_ok (countdown.py:90-92) is a **strict subset** of the proposed
component (valid 52, exact-multiset 68, overlap 52, valid-only 0 — 96%
agreement). The sharp binary signal already exists and the stable teacher
already weights it at ~0.277 (final seed-0 weight; floor 0.16). The v1
component would add gradient only on the 4% exact-multiset-but-illegal-ops
candidates — and would reward disallowed ops there. Additional v1 flaws:
7-key init during the 50-step teacher delay is unprojected (7 × 0.20 = 1.40
aux mass vs 1.20 budget), and a 7th high-need key would absorb the budget
surplus that currently props up `format`/`brevity` — the only anti-clipping
signals, with clipping at 11% of candidates (v10c2's failure mode).

## Problem (from the audit, ledger `audit-banks`)

Number-set legality failures are 53–64% of all trained candidates
(missing-required 31–36%, extra/repeated 25–33% on static/stable banks).
Ranking is not the bottleneck (selector near-misses = 0); exact/legal
candidates are simply not formed.

## Hypothesis (v2)

The binary legality signal (`valid_expression`) exists but its weight
envelope is too small: the Stable-RTW controller can move it only within
[floor 0.16, budget-projected ~0.28]. If legal-candidate formation is the
binding constraint on best-of-N exactness, **letting the teacher allocate
substantially more mass to `valid_expression`** (floor 0.30, cap 0.45)
should raise the legal-candidate rate per bank, giving best-of-N more
verifier-legal candidates to search → paired validation `reranked_exact@8`
improves vs C0.

Mediator chain to log and check (in order): valid_expression rate in banks ↑
→ oracle_exact@8 ↑ → reranked_exact@8 ↑. The advisor flags the first link
(legality → exactness) as the most likely break given v10c2 lifted hard-tier
legality with zero exact gain; a break there is an informative negative that
would close the "reward shaping for legality" theme (second strike).

## Change (one variable, default-off)

New strategy name `adaptive_stable_v12` (aliasing `adaptive_stable` with a
different floors/caps table; `adaptive_stable` itself untouched — C0 and all
archived configs stay bit-identical):

```text
STABLE_FLOORS_V12 = STABLE_FLOORS | {"valid_expression": 0.30}
STABLE_CAPS_V12   = STABLE_CAPS   | {"valid_expression": 0.45}
```

- Six keys, same AUX_KEYS, same budget (`stable_target_weight_sum` 1.20),
  same delay/LR/EMA — only the valid_expression envelope moves. Floor sum
  becomes 0.03+0.16→0.30+0.18+0.12+0.00+0.02 = 0.65 < 1.20 (feasible).
- No new component; no change to `countdown.py`; eval path untouched.
- Verifier, selector, eval protocol, datasets: frozen.

Known redistribution effect (accepted, logged): raising valid_expression's
floor takes budget surplus from other components (format/brevity prop-up
shrinks). This is the experiment: shift mass toward the binding constraint.
The clipping guardrail below watches the anti-clipping side effect.

## Budget & protocol (frozen, per program §3)

Same as every ladder run: GRPO+LoRA 300 steps seed 0, then frozen best-of-N
(loop mode, N=1/4/8, frozen task IDs, both splits). Comparator: C0 =
`grpo_stable_seed0_300` banks (same machine, same stack). Smoke: 60 steps +
health gate before the 300-step run.

## Decision rule

Program §2 paired-discordant rule on validation `reranked_exact@8` is
decisive. Mechanism diagnostics (not KEEP criteria): number-set failure
classes drop, bank valid_expression rate rises.

Guardrails (regression on any → DISCARD regardless of primary):
- test_in_dist exact, selected number F1, reward-hack rate, tokens/wall-clock;
- **clipping**: capped-generation rate (len ≥ 256 tokens) and
  `no_answer_span` taxonomy class must not rise materially vs C0 (advisor
  finding: this design cuts format/brevity mass, the anti-clipping signals) —
  checked at the 60-step smoke gate AND the full run.

## Validation before GPU (program §5)

- Unit tests: v12 strategy resolves to the new floors/caps; projection
  respects them within budget; `adaptive_stable` behavior bit-identical.
- CPU dry-run of 02 with `--reward_strategy adaptive_stable_v12`;
  `teacher_weights.jsonl` shows valid_expression ≥ 0.30 post-delay.
- Advisor diff review, commit, then GPU.

## Success/failure recording

Ledger row `v0.12-legality`. KEEP → seeds 1/2 + consider stacking truncation
shaping next. DISCARD → second strike on "teacher-side reward shaping for
legality" → move to truncation/close-tag shaping (the other audit lever) or
the throughput/mechanism queue.

## Results (v0.12 pilot, seed 0, 2026-07-09) — DISCARD, strike two

Pilot completed clean (smoke gate passed, envelope [0.30,0.45] held, budget
1.20; 300-step healthy). Scored against C0 = `grpo_stable_seed0_300` on frozen
IDs, both splits. **The pre-registered prediction held: legality up, exact
flat.**

### Candidate-level (the well-powered surface)

| metric | split | C0 | v12 |
|---|---|---:|---:|
| legal-candidate rate | validation | 0.130 | **0.193** |
| legal-candidate rate | test_in_dist | 0.145 | **0.177** |
| P(exact \| legal) | validation | 0.135 | 0.130 |
| P(exact \| legal) | test_in_dist | 0.138 | 0.211 |
| mean tokens/cand | validation | 64 | **37** |
| 256-clip rate | validation | 0.16 | **0.07** |

The envelope widening **did** raise legal-candidate formation (+48% relative
on validation) and, notably, **halved the clip rate and cut tokens ~40%** —
more valid_expression weight suppressed rambling. But **P(exact|legal) stayed
flat** (val 0.135→0.130): the extra legal candidates did not convert to exact.

### Decisive metric (oracle_exact@8 / paired rerank@8 vs C0)

- validation: 6 vs 6 (paired 4-vs-4, p=1.0) — dead flat.
- test_in_dist: 8 vs 2 looks like p=0.031, BUT **C0-seed0 (2/50) is a low
  outlier** — stable across 3 seeds is 2/6/8 (mean 5.3). v12's 8 is z=+1.1,
  inside stable's own seed range (seed2 also = 8). So exact is **within
  noise**, not a v12 gain. (This is exactly the single-seed McNemar trap the
  diagnosis/Probe A warned about.)
- selection still saturated (reranked@8 == oracle@8) on both splits — any
  gain would have to come from generation, and it didn't.

### Verdict: DISCARD → strike two for reward-shaping-for-legality

Two consecutive reward-shaping attempts (v0.10 curriculum via mediator, v0.12
legality envelope) moved their intended intermediate quantity but not exact.
This is the diagnosis's central prediction: exact = legality ×
P(exact|legal≈0.14), and reward reweighting touches legality (and here,
usefully, cost) but not the value-search wall. **Retire the reward-shaping-for-
legality theme** (two-strike rule) and pivot GPU to v0.13 SFT warmup (a
capability/data lever). One genuine keeper: the `adaptive_stable_v12` envelope
is a strong *cost* reducer (−40% tokens, half the clipping) at equal exactness
— worth noting for the efficiency story, not the accuracy story.

Overclaims to avoid: "v12 improves OOD exactness" (the test signal is a C0
outlier artifact); "legality shaping fails" (it succeeded at legality and cost
— it just doesn't convert to exact, which is a different, informative result).
