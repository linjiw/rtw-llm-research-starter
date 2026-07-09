# Gate 0 Report: Local Baseline Ladder (seed 0)

Date: 2026-07-09. Runner: `scripts/run_gate0_baseline_ladder.sh` (commit
`d190f86` code state). Stack: TRL 1.7.0, transformers 5.12.1, torch 2.11
(cu130) — substantially newer than the archived v0.9B stack. Protocol: frozen
v0.9 task IDs, temp 0.7, top-p 0.95, max_new_tokens 256, sampling seed 0,
N=1/4/8 from max-N=8 banks, practical selector.

Artifacts: `outputs/gate0_local_ladder_summary.csv`,
`outputs/gate0_local_ladder_paired.json`, checkpoints under
`outputs/checkpoints/grpo_{static,stable}_seed0_300`, banks under
`outputs/bestofn/*_local_seed0_*`.

## Results (reranked exact, practical selector)

| split | N | base | static | Stable-RTW |
|---|---:|---:|---:|---:|
| validation | 1 | 0.00 | 0.06 | 0.00 |
| validation | 4 | 0.00 | 0.12 | 0.04 |
| validation | 8 | 0.00 | **0.14** | **0.12** |
| test_in_dist | 1 | 0.00 | 0.04 | 0.00 |
| test_in_dist | 4 | 0.06 | 0.06 | 0.04 |
| test_in_dist | 8 | 0.06 | **0.10** | **0.04** |

Paired stable-vs-static (seed 0):

| split | N | both | stable-only | static-only | p | Δ |
|---|---:|---:|---:|---:|---:|---:|
| validation | 8 | 5 | 1 | 2 | 1.0 | −0.02 |
| validation | 4 | 0 | 2 | 6 | 0.29 | −0.08 |
| test_in_dist | 8 | 2 | 0 | 3 | 0.25 | −0.06 |

Cost at validation N=8: stable 25.7k tokens / 1359 s; static 53.7k tokens /
2832 s (stable completions are ~half the length: 209 vs 428 chars/candidate).

## What reproduced and what did not

Reproduced:
- Infrastructure end-to-end: both 300-step trainings healthy (no NaNs, teacher
  logs correct, `reward_variance_nonzero_fraction=1.0`; stable teacher delay
  and weight evolution as designed — valid_expression rose to ~0.28, format
  and brevity decayed).
- The base ≪ trained legality ladder (base selected_valid 0.06 at val N=8 vs
  0.58/0.62 trained; base exact 0.00 on validation).
- Best-of-N as a general harness mechanism (exact rises with N for both
  trained methods) and practical selector = oracle on every run.
- Local stable ≈ archived stable (val@8 0.12 vs archived per-seed 0.12–0.14).

Did NOT reproduce (seed 0):
- The Stable-vs-static validation advantage. Archived seed-0 was Δ=+0.08
  (stable-only 4, static-only 0); local is Δ=−0.02 (1 vs 2). The gap closed
  from the static side: local static val@8 = 0.14 vs archived per-seed
  0.02–0.12 — **static shaping improved on the new library stack**, while
  stable stayed in its archived range.

## Safe conclusion

> On the current stack (TRL 1.7), a single local seed shows static shaping and
> Stable-RTW statistically indistinguishable under best-of-N (tiny discordant
> counts, p≈1.0), with Stable-RTW retaining a ~2× generation-cost advantage at
> similar exactness. The archived v0.9B stable advantage is not contradicted
> (it pooled 3 seeds; per-seed archived deltas ranged +0.02 to +0.10 on
> validation), but it is not confirmed on this stack either. Any new-stack
> method claim must be re-established with local multi-seed baselines.

Overclaims to avoid:
- "Stable-RTW advantage failed to replicate" — one seed, 1-vs-2 discordants
  cannot support that; per-seed noise in the archived data is of the same size.
- "Static is now as good as Stable-RTW" — same reason; and static costs ~2×
  tokens/wall-clock for the same exact count.
- Comparing local numbers directly against archived numbers as if same-stack.

## Gate verdict

- Environment/reproduction gate: **PASS** (healthy runs, ladder shape, working
  eval harness, fair C0 checkpoint exists for v0.10).
- Method-ordering gate (stable > static locally): **NOT CONFIRMED on 1 seed**
  — resolution requires local seeds 1/2 (the same evidence standard the
  archived claim used).

Decision taken (see `EXPERIMENT_LEDGER.md` and `NEXT_STEPS.md`): proceed with
v0.10 C2 (its comparison is stable-vs-stable — uniform sampling vs adaptive
curriculum on the same reward strategy, same stack, same machine — so it does
not depend on the static question), and queue the static/stable seeds-1/2
local expansion as the baseline-story diagnostic for the paper.
