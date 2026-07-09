# v0.10 Plan: GACL-Style Task Curriculum for GRPO Post-Training

Status: DESIGN (2026-07-08). Queue position: first new-method experiment after
Gate 0 (`docs/AUTORESEARCH_PROGRAM.md` §6). Scored under the frozen v0.9B
protocol; primary metric is paired `reranked_exact@8` on frozen validation IDs.

## Hypothesis

> Training-time adaptive difficulty sampling (GACL-style) attacks the exact-
> search bottleneck that reward shaping alone did not solve (v0.7/v0.8
> evidence): concentrating rollouts in the tier where the policy currently has
> partial-but-not-saturated competence increases GRPO's within-group reward
> variance (more gradient signal per batch), producing more legal and more
> exact candidates at fixed step budget than uniform sampling.

Mechanistic rationale specific to GRPO: advantages are computed within a
prompt's generation group. Prompts that are too hard (all 4 generations
illegal/wrong → identical rewards) or too easy (all near-max) yield ~zero
advantage and waste the step. A curriculum that keeps batches in the
mid-success band directly raises `reward_batch_has_variance`, which we already
log — giving a measurable mediating variable, not just an outcome delta.

## What varies (one variable)

Only the training-time sampling distribution over the three existing dataset
difficulty tiers (easy=3 operands, medium=4, hard=5; 2000 train tasks,
balanced). Reward strategy is fixed at `adaptive_stable` (current best).
Datasets, verifier, selector, eval protocol: frozen. No new task generation in
v0.10 (synthetic task generation + grounding against a reference distribution
is deferred to v0.11 to keep this a single-variable experiment).

## Arms

| arm | `--task_curriculum` | description |
|---|---|---|
| C0 (control) | `uniform` | existing behavior (shuffled uniform; reproduces Gate 0 stable baseline) |
| C1 | `manual` | fixed schedule: easy-heavy → balanced → hard-heavy over 300 steps |
| C2 | `adaptive` | success-band targeting (below), ε-grounded to uniform |

Run order: C2 first (the bet), C1 only if C2 beats C0 (to show adaptivity
matters, not just non-uniformity). C0 = Gate 0 stable checkpoint (no re-run).

## C2 controller (CurriculumController)

*(Revised after advisor review, 2026-07-08 — verdict approve-with-changes.)*

State: per-tier EMAs of `valid_expression` and `correct` (both keys exist in
the components dict passed to `teacher.update`). EMA updates are
sample-weighted to avoid starvation noise on disfavored tiers: for a batch
with `k_d` completions of tier `d` and batch mean `m_d`,
`ema_d = beta_eff * ema_d + (1 - beta_eff) * m_d` with `beta_eff = β^k_d`
(β=0.9); a tier absent from the batch keeps its EMA. EMAs initialize from the
first observed batch mean, not 0.

Competence is **gated** (a flat `0.5*valid + 0.5*exact` score cannot
discriminate: a tier at valid=1.0/exact=0.0 would look "ideal"). Per tier:

```text
if valid_ema_d < 0.5:   c_d = valid_ema_d,  τ = 0.5,   σ = 0.25   # legality phase
else:                   c_d = exact_ema_d,  τ = 0.175, σ = 0.15   # exact phase
score_d = exp(-((c_d - τ)^2) / (2 σ^2))
p_d     = (1-ε) * score_d / Σ score + ε * uniform      ε = 0.2
p_d     = clip(p_d, p_min=0.10) then renormalize
```

Cold start: uniform for the first `curriculum_delay_updates=25` controller
updates (mirrors Stable-RTW's delay). All constants logged; per-update record
(EMAs, phases, probs, cumulative tier draws, unique-task coverage) appended to
`curriculum_state.jsonl` in the run dir.

C1 manual schedule (piecewise over teacher updates u of ~300-step run):
u<100: (0.6,0.3,0.1); 100≤u<200: (0.34,0.33,0.33); u≥200: (0.1,0.3,0.6).

## Integration (TRL 1.7.0)

Cadence arithmetic (verified against TRL 1.7.0 source with the repo's flags —
batch_size 2, grad_accum 8, num_generations 4, single GPU): defaults give
`generation_batch_size=16`, `steps_per_generation=8`, `num_iterations=1`, so
the sampler draws chunks of **4 unique prompts**, each chunk is yielded
`repeat_count=8` times, and **1 generation block = 1 reward-manager update =
1 optimizer step**. Delay=25 and the C1 breakpoints (100/200) are therefore in
optimizer-step units. This identity holds only under `num_iterations=1` and
`steps_per_generation == grad_accum` — assert all of these at startup and fail
loudly on drift.

- `src/rtw_llm/curriculum.py`: `CurriculumController` (+ config dataclass) and
  `CurriculumSampler`. The sampler mirrors `RepeatSampler`'s yield structure
  and `__len__` arithmetic exactly, but builds each 4-prompt chunk by sampling
  a tier ~ p_d per slot and popping from a **per-tier shuffled queue**
  (reshuffle on exhaustion) — without-replacement within tier *across* chunks,
  eliminating the repeat-exposure confound vs uniform (a concentrated tier of
  667 tasks sees ≤ ~1.8 epochs over 1200 draws; immediate cross-chunk repeats
  are otherwise possible). Each chunk is materialized **once** and re-yielded
  for all 8 repeats (re-drawing per repeat would burn RNG and corrupt tier
  logs silently). Controller state is read lazily per chunk; with
  `dataloader_num_workers=0` the sampler-vs-controller lag is effectively zero.
- `scripts/02_grpo_train.py`: `--task_curriculum {uniform,manual,adaptive}`
  (default `uniform` = exact current behavior; unit test asserts uniform mode
  reproduces `RepeatSampler`'s index sequence bit-for-bit at seed 0). For
  non-uniform, subclass `GRPOTrainer` overriding `_get_train_sampler` (guard:
  only for the train dataset) to return `CurriculumSampler`; the controller is
  updated inside `RTWRewardManager.score_batch` via an optional hook — the
  hook is observe-only and cannot alter reward values or logged components
  (AGENTS.md #3/#6).
- Additional logging (both arms): per-prompt-group reward std — the true GRPO
  mediator. The existing `reward_batch_reward_std` spans 4 different prompts
  and is nearly always nonzero; the hypothesis concerns within-group variance
  (4 generations of the same prompt). Groups are positional (consecutive
  `num_generations`-sized slices, matching TRL's layout) — id-based grouping
  would merge distinct groups that share a prompt id in one batch.
- Determinism: the sampler is seeded from `--seed` (dedicated `random.Random`;
  the controller itself has no RNG); reproducibility is conditional on
  identical GPU generations, since controller state depends on model outputs.
  Known limitations, enforced by startup guards: single-process only,
  `eval_strategy='no'`, `remove_unused_columns=False`, and no
  resume-from-checkpoint (controller state is not restored). The dataloader
  prefetches one batch, so controller-to-chunk influence lags one generation
  block; delay/schedule breakpoints are accurate to ±1 update.

## Validation ladder (before any GPU spend)

1. Unit tests: controller banding math (probs shift toward mid-competence
   tier; ε floor holds; delay holds; determinism), sampler yield structure
   equals RepeatSampler's shape for uniform mode, manual schedule breakpoints.
2. CPU dry-run: 2-step GRPO on a 30-row slice with `--task_curriculum
   adaptive` (tiny model), assert `curriculum_state.jsonl` populated and tier
   counts respond to injected competence.
3. Advisor review of design (before implementation) and of the diff (after).
4. GPU smoke 60 steps (after Gate 0 frees the A10G): health check + curriculum
   log sanity (no tier collapse; probs move).
5. Full 300-step C2 run + frozen best-of-N eval → ledger row.

## Decision rule (from program §2)

KEEP C2 iff paired validation `reranked_exact@8` vs C0 (Gate 0 stable local,
same machine, same protocol) shows a clear discordant-pair advantage and no
guardrail regression (test_in_dist, selected_valid, number F1, reward-hack,
cost). C0 reuse is fair only if the uniform-mode bit-for-bit sampler test
passes and both runs share the machine/TRL version; tier-dependent prompt
lengths change tokens-seen, so report the cost guardrail per arm. Secondary
evidence to report either way: per-tier competence curves, per-prompt-group
reward-std fraction (the hypothesized mediator), tier occupancy and
unique-task coverage over training.

Failure modes to watch: collapse-to-easy (curriculum farms legality on easy
tier → validation exact stagnates; ε floor + report per-tier occupancy),
mediator-without-outcome (variance up, exact flat → hypothesis wrong in an
informative way — record and stop the theme per the two-strike rule).

## Results (C2 pilot, seed 0, 2026-07-09)

Runner completed clean: smoke gate passed (60 updates, draws easy/med/hard =
94/82/68, probs moved after delay), 300-step pilot healthy (`health_final`
issues: none), both frozen best-of-N evals collected
(`outputs/bestofn/v10c2_local_seed0_*_limit50_n8`).

### Primary + paired (practical selector, vs C0 = grpo_stable_seed0_300)

| split | N | C0 | C2 | both | C2-only | C0-only | p | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 4 | 0.04 | 0.06 | 0 | 3 | 2 | 1.000 | +0.02 |
| validation | 8 | 0.12 | 0.10 | 2 | 3 | 4 | 1.000 | −0.02 |
| test_in_dist | 4 | 0.04 | 0.08 | 1 | 3 | 1 | 0.625 | +0.04 |
| test_in_dist | 8 | 0.04 | 0.14 | 2 | 5 | 0 | 0.062 | +0.10 |

Guardrails at N=8: selected_valid up on both splits (val 0.66 vs 0.58; test
0.72 vs 0.58); number F1 slightly down on validation (0.920 vs 0.953), up on
test (0.962 vs 0.933); selected reward-hack rate down (0.30 vs 0.42 val,
0.28 vs 0.40 test). **Cost regressed ~1.8×**: C2 completions are longer
(mean 116 vs 64 tokens/candidate; capped-at-256 rate 0.37 vs 0.17), so
val N=8 cost 46.3k tokens / 2806 s vs C0's 25.7k / 1359 s.

### Mediator + controller behavior

- Group reward variance (the hypothesized mediator) did NOT move: mean
  per-group reward std 0.231 (C2) vs 0.235 (C0); batch group-variance
  fraction 0.968 vs 0.969. Training exact rate 0.036 vs 0.030.
- The controller DID steer: tier probs moved off uniform (hard down-weighted
  to ~0.20 mid-run, recovering by end), cumulative draws easy/med/hard =
  453/423/328, 1204 unique tasks seen, easy tier reached the exact phase at
  ~update 205 while medium/hard stayed in legality phase.

### Safe conclusion

> On one seed, the adaptive task curriculum is statistically
> indistinguishable from uniform sampling on the primary metric (validation
> Δ at N=8 = −0.02, discordants 3-vs-4, p=1.0). The test_in_dist N=8 signal
> (5-vs-0 discordants, p=0.062, Δ=+0.10) is suggestive but is a guardrail
> split, not the primary, and comes with a ~1.8× token/wall-clock cost
> regression driven by longer completions. The curriculum steered sampling
> as designed but did not move the hypothesized mediator (group reward
> variance) — this is the "variance flat, controller steering" outcome:
> the bottleneck appears not to be difficulty mix at this model scale.

Overclaims to avoid: "task curriculum improves OOD/test exactness" (one
seed, guardrail split, p>0.05); "curriculum is cost-neutral" (it is not —
the cost guardrail regressed); "the mediator hypothesis is falsified"
(variance was already near-saturated: 97% of groups have nonzero variance
under Stable-RTW reward shaping, leaving the mediator little room to move).

### Verdict

Primary decision rule says **DISCARD** (no clear validation discordant
advantage; cost guardrail regressed). Per the two-strike rule the theme gets
at most one revision; the only revision worth considering is competence-
signal retuning (τ/σ, gate threshold) IF pursuing the test_in_dist signal is
judged worth ~3.5 h GPU — but note the primary-metric case for it is weak.
Recorded as ledger row `v0.10-C2`.
