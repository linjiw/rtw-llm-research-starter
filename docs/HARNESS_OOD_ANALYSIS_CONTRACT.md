# Pre-registered analysis contract: harness-shift & OOD evals (ranks 2–3)

Created: 2026-07-09. Pre-registered BEFORE running (per the strategic-repivot
critique: declare metrics + power caveats up front so a near-null is reported
honestly, not spun). Runners: `scripts/run_harness_shift_eval.sh`,
`scripts/run_ood_eval.sh`. Frozen protocol: v09 task IDs, temp 0.7/top_p 0.95/
seed 0, max_new_tokens 256, loop-mode generation, N=1/4/8, practical selector.

## Rank 2 — harness-shift (prompt_mid vs prompt_high)

Question: does adaptive shaping (stable) yield policies more robust to a terser
prompt than fixed (static)? Both prompts retain the `<answer>` cue.
**prompt_low is EXCLUDED** — it has no `<answer>` tag, and `extract_answer`
falls back to full completion text, contaminating format AND assembly metrics.

- **PRIMARY metric (well-powered): candidate-level, parseable-span-restricted
  `number_multiset_f1`** (and legality rate), computed over all 6×8×50 = 2400
  candidates per prompt field. The degradation prompt_high→prompt_mid, and
  whether stable degrades LESS than static, is the robustness signal.
- **Interaction (stable-vs-static robustness) is declared at the 3-SEED level**
  and **labeled underpowered by design** — the statistical unit is the 3-seed
  policy comparison, NOT the 2400 candidates (the recurring trap). Report the
  per-seed degradation deltas and their sign consistency; do not compute a
  candidate-level p-value for the interaction.
- Exact@8 / oracle@8 reported DESCRIPTIVELY only (the ~9% ceiling makes
  task-level McNemar ~2–3 discordants — noise).
- **Decision:** a consistent-sign stable robustness advantage across 3 seeds on
  parseable-span number_multiset_f1 → scoped robustness claim. Within-noise or
  sign-flipping → honest near-null that CLOSES pillar 3 (the strategic prior:
  near-uniform controller weights predict a near-null here too).

## Rank 3 — OOD (test_ood_division, test_ood_long)

Question: does RL narrow capability OOD, and does adaptive shaping generalize
better? test_ood_division = 5-number tasks with `/` (never in the 2000
fine-tune examples); test_ood_long = 6-number tasks.

- **BASE-MODEL ARM IS MANDATORY.** `/` is OOD for the fine-tune data, not for
  the pretrained base. The real question is whether RL (static/stable) NARROWED
  an operator the base knows — so every OOD metric is read relative to base.
- **PRIMARY signals (descriptive): legality panel** (valid_expression rate,
  number_multiset_f1) + **`/`-adoption rate on division** (fraction of
  candidates using `/`) + **truncation rate**. exact@8 reported as
  expected-floor (5–6-number tasks are below the 0.5B capability floor; a ~0
  result is the scope boundary, not a failure).
- No task-level significance claims (exact ~floor, 50 tasks). This is a SCOPE
  statement: "trained policies retain/lose legality OOD; exact is capped by the
  number-count capability floor."
- **Decision:** runs regardless as a scope boundary. Escalate to a
  harness×OOD 2×2 cross ONLY if rank-2 shows a live interaction (unlikely per
  the prior; budget nothing for the cross).

## Cost / sequencing

Eval-only, no retraining. Harness rank-2: 6 ckpts × 2 fields × 50 tasks × 8;
prompt_high banks for the frozen split may already exist from Gate 0 but under
un-suffixed dir names, so the runner re-generates into `harness_*_<field>` dirs
for unambiguous pairing (the prompt_field is now part of the sampling identity,
so `--skip_if_complete` will not cross-contaminate fields). Stage validation
first (~5–6 GPU-h) to read the effect before committing test_in_dist. OOD:
seed0 + base first to de-risk before seeds 1/2. All GPU-gated behind the
Paper-1 v0.13 scoring and the shared A10G.
