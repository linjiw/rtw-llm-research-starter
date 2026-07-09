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
