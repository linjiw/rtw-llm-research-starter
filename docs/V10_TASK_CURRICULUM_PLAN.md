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
