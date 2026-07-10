# v0.19: complete within-v2 baseline reset

Status: **PRE-REGISTERED BEFORE IMPLEMENTATION OR ACCELERATOR RUNS**  
Protocol ID: `countdown-v19-within-v2-reset-v1`  
Date: 2026-07-10

Independent design review initially returned **BLOCK** on confirmation
contamination, post-hoc runtime selection, endpoint/inference ambiguity, and
prose-only test access. All blockers and moderate findings were incorporated;
the reviewer then returned **CLEAR** before implementation began.

## Decision and scope

Countdown-v2 passed its independent audit, but no legacy-v1 model bank can be
used as a causal baseline for it. V0.19 therefore restarts the complete
capability/shaping ladder inside v2. Legacy results remain historical,
hypothesis-generating evidence only.

The scientific matrix is:

| family | arm | trainable states | interpretation |
|---|---|---:|---|
| untrained | base | 1 | frozen-model floor |
| capability | SFT-only | 3 | supervised format/search capability |
| RL-only | GRPO-static | 3 | fixed reward shaping from a fresh LoRA |
| RL-only | GRPO-Stable | 3 | adaptive reward shaping from a fresh LoRA |
| SFT-initialized | SFT+GRPO-static | 3 | fixed shaping after supervised warmup |
| SFT-initialized | SFT+GRPO-Stable | 3 | adaptive shaping after supervised warmup |

Trainable states use true end-to-end seeds `0, 1, 2`. Each seed's SFT-only
checkpoint is shared as the initialization of its two SFT+GRPO arms. The base
model is evaluated once; duplicating the same frozen weights and sampling
stream would manufacture pseudo-seeds. Every v0.19 estimate is conditional on
the single frozen evaluation-sampling stream; this protocol does not estimate
sampling-seed uncertainty.

This protocol may characterize capability and efficiency, but it does not
claim that SFT and GRPO have equal total compute. Only within-family
static-versus-Stable contrasts have matched training budgets.

## Immutable inputs and runtime identity

- Dataset protocol: `countdown-dataset-v2`.
- Training file: `data/countdown_v2/train.jsonl` (5,000 rows).
- GRPO receives no evaluation dataset. The resolved trainer config must retain
  `eval_strategy=no`; confirm400 is forbidden from every training process.
- Model: `Qwen/Qwen2.5-0.5B-Instruct`.
- Hugging Face revision:
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Seed protocol: `countdown-true-seeds-v2`; SFT trainer seed, GRPO teacher seed,
  GRPO trainer seed, pre-model RNG, and curriculum RNG all equal the arm seed.
- Prompt field: `prompt`; `prompt_low`, `prompt_mid`, and `prompt_high` remain in
  every dataset row for future harness-shift work.
- Strict write-once provenance is mandatory for SFT, GRPO, and best-of-N.
- The protocol manifest fingerprints every Python source in `src/rtw_llm/`
  plus every registered runner/scorer. Environment capture, launch, scoring,
  confirmation, and test release all require a replay-eligible protocol, and
  every run commit must reproduce that exact transitive source fingerprint.
- Production arms must use one homogeneous, single-process, single-GPU stack.
  Before seed 0, a production-environment lock must be captured on that host,
  independently reviewed, committed, and bound into the protocol. It freezes
  the container/image digest when available; Python, PyTorch, Transformers,
  TRL, PEFT, Accelerate, Datasets, NumPy, Tokenizers and Safetensors versions;
  OS/architecture; CUDA runtime, driver, cuDNN and GPU model; process topology;
  deterministic settings; and precision-related environment variables. Every
  later arm must match it exactly. MPS smoke artifacts are infrastructure-only
  and cannot be pooled with CUDA.

The local planning environment was Python 3.11.14, PyTorch 2.11.0,
Transformers 5.12.1, TRL 1.7.0, PEFT 0.19.1, Accelerate 1.14.0, and Datasets
5.0.0. CUDA was unavailable; Apple MPS was available. Production results must
use a separately committed CUDA environment lock. The first completed arm is
not allowed to define the production environment after the fact.

## Frozen training budgets

### SFT

- Completion-only loss over all v2 training tiers.
- LoRA rank 16, alpha 32, dropout 0.05, all linear modules (existing runner).
- `max_steps=313`, batch size 2, gradient accumulation 8.
- This is one nominal pass at effective batch 16: `ceil(5000 / 16) = 313`.
- Learning rate `5e-5`; no metric-based early stopping.
- World size 1; sequence length 1024; no packing; frozen SHA row order
  (`shuffle_dataset=false`).
- Linear scheduler, zero warmup, zero weight decay, max gradient norm 1.0,
  gradient checkpointing on, no drop-last, save every 100 steps, and no resume.

### GRPO

- `max_steps=300`, batch size 2, gradient accumulation 8.
- Four generations per prompt; maximum completion length 256.
- Learning rate `5e-6`; uniform task sampling; no metric-based early stopping.
- World size 1; generation batch size 16; eight steps per generation block;
  one policy iteration; DAPO loss, group reward scaling, and beta 0.
- Linear scheduler, zero warmup, zero weight decay, max gradient norm 1.0,
  gradient checkpointing on, no drop-last, save every 100 steps, and no resume.
- The primary reward remains verifier-exact task success. Reward components,
  primary reward, auxiliary reward, total reward, teacher weights, clipping,
  token counts, and health diagnostics remain separately logged.
- Static arm: existing `reward_strategy=static`.
- Stable arm: existing `reward_strategy=adaptive_stable`.

RL-only arms attach a fresh seeded LoRA to the pinned base model. SFT+GRPO arms
continue the matching seed's SFT LoRA without stacking a second adapter. The
two reward strategies within an initialization family have identical optimizer
and generation budgets.

## Deterministic validation views

The frozen 500-task validation split is partitioned, not regenerated:

| view | easy | medium | hard | total | permitted use |
|---|---:|---:|---:|---:|---|
| `validation_dev100` | 10 | 45 | 45 | 100 | smoke, health, directional development |
| `validation_confirm400` | 40 | 180 | 180 | 400 | untouched until the arm code/config is frozen |

Within each tier, IDs are ordered by ascending SHA-256 of the UTF-8 byte string
`countdown-v19-validation-dev-v1\0<ID>`, with the ID as a deterministic tie
breaker. The first 10/45/45 IDs form `validation_dev100`; the complement forms
`validation_confirm400`. Each published ID file preserves the source
validation JSONL row order after membership selection. The implementation must
publish ordered ID files,
their SHA-256 hashes, a machine-readable protocol manifest, and tests proving
complete/disjoint coverage, quotas, source-file binding, and deterministic
replay. These views live outside `data/countdown_v2/` so the audited dataset
artifact is not mutated.

The confirm400 view—not a dev+confirm recombination—is the v0.19 validation
estimand. The dataset card's original full-500 recommendation remains the
general default; this preregistered partition is an explicit stricter override
for v0.19 because development and confirmation have different roles.

## Frozen evaluation signature

- Runner: strict-provenance `scripts/07_best_of_n_rerank.py`, HF engine.
- Generation mode: `batched`, batch size 16. This has a different sampling
  identity from the legacy loop path, but legacy comparability is not a v0.19
  estimand; every v0.19 state uses the same mode, ordering, and batch size.
- Prompt field: `prompt`.
- Sampling seed: `0` for the primary candidate bank.
- `temperature=0.7`, `top_p=0.95`, `max_new_tokens=256`.
- `max_n=8`, reported prefixes `N in {1, 4, 8}`.
- The exact ordered task-ID view is an identity-bound input.
- Every candidate is rescored through the verifier in
  `src/rtw_llm/countdown.py`; no stored or model-reported correctness label is
  accepted.
- Each candidate's ID, tier, number multiset/order, target, and allowed ops must
  exactly match its frozen source-validation row. Exact generated-token counts
  must lie in `[0, 256]`; a length finish is valid only at exactly 256 tokens.

Sampling-seed sensitivity, if later needed, is a separately registered
secondary experiment and is not a substitute for training seeds.

## Estimands and contrasts

Primary endpoint: verifier-exact **practical reranked success at N=8** over the
finite confirm400 task set. The practical selector is frozen and never reads
exact correctness. Oracle exact@8 is the generation diagnostic. Report overall
micro accuracy and the equal-weight macro average of easy, medium, and hard
accuracies. Legality, exact-given-legal, clipping, and reward trajectories are
diagnostic outcomes.

Preregistered paired contrasts, in order:

1. **Primary:** SFT+GRPO-Stable minus SFT+GRPO-static: adaptivity after SFT.
2. GRPO-Stable minus GRPO-static: adaptivity from the base initialization.
3. SFT-only minus base: supervised capability contribution.
4. SFT+GRPO-Stable minus SFT-only: effect of adding the specified Stable-GRPO
   stage, including its extra compute and data exposure.
5. SFT+GRPO-static minus SFT-only: effect of adding the specified static-GRPO
   stage, including its extra compute and data exposure.

The first two are the clean matched-budget method contrasts. Contrasts 3--5
are stage decompositions, not equal-total-compute comparisons.

Task is the sampling/analysis unit. Candidate rows are nested observations and
must never be treated as independent. For each trainable arm, aggregate the
three true seeds at the task level and report per-seed results, paired task
deltas, task-cluster bootstrap sensitivity intervals, and task-level sign-flip
tests. No candidate-pooled p-values are valid.

For every contrast, first average each arm's binary selected-exact outcome over
its observed true-seed runs within task, then form the paired task effect. The
shared base value is not relabeled as three base seeds. The primary contrast
uses a two-sided task sign-flip test at `alpha=0.05`: exact enumeration for at
most 20 nonzero task effects, otherwise 20,000 Monte Carlo draws with
`(extreme+1)/(draws+1)`, PCG64 seed 17. Its 95% task-cluster percentile
bootstrap also uses 20,000 draws and seed 17. A positive primary method claim
requires a positive point estimate, a two-sided sign-flip p-value below 0.05,
and a 95% bootstrap interval excluding zero. Contrasts 2--5 form a secondary
family controlled by Holm at familywise 0.05; raw and adjusted p-values are
both reported. No optional stopping or one-sided conversion is allowed.

These task-resampling quantities describe sensitivity across the finite task
panel conditional on the three selected training seeds and fixed sampling
stream. The task-by-seed product bootstrap is exploratory only. Three seeds do
not support a confirmatory training-seed-population claim.

Efficiency endpoints are secondary and precisely defined: generated completion
tokens per practical verifier-exact task at N=8, mean generated completion
tokens per task, completion-cap hit fraction, training wall-clock seconds, and
single-GPU hours. They are measured only within the production environment
lock. Cost-per-exact is infinite/undefined—not replaced with an epsilon—when an
arm has zero practical exact tasks.

## Hypotheses and decision rules

- H1: SFT raises legality and verifier-exact success versus the frozen base.
- H2: if Stable-RTW has an accuracy effect, it appears within an initialization
  family; otherwise its defensible value may be token/clip efficiency.
- H3: GRPO after SFT adds verifier-exact solutions beyond SFT-only rather than
  only reshaping auxiliary components.
- H4: medium/hard performance remains the capability stress test; easy-tier
  movement alone cannot be generalized to the full task distribution.

An arm is unhealthy if its run manifest fails verification, reward logs are
missing or conflate components, the primary reward disagrees with verifier
success, required group variance is absent, loss/gradients are non-finite,
artifacts are incomplete, or the task/sampling identity differs. An unhealthy
arm is repaired as an implementation issue and rerun from a new clean commit;
it is not silently dropped because of accuracy.

All healthy arms advance through all three seeds. There is no accuracy-based
early stopping and no winner selection on dev100. Dev100 may reveal broken
infrastructure or grossly inert training; any scientific config change after
seeing it creates a new protocol version and requires rerunning every affected
arm before confirm400 access.

## Execution gates

1. **Protocol gate:** independent design review; deterministic view artifacts;
   runner/scorer tests; full unit suite, Ruff, and compile checks.
2. **Clean-source gate:** review the exact diff, commit it, and require a clean
   worktree before any strict-provenance accelerator run.
3. **Local preflight:** tiny, separately named MPS/CPU jobs may validate model
   loading, one SFT update, one GRPO update, separate reward logging, adapter
   continuation, and best-of-N output. They are not scientific observations.
4. **Production seed 0:** after the environment lock is committed, train all
   five trainable states on one homogeneous CUDA stack, validate
   manifests/health, then evaluate dev100.
5. **Production seeds 1/2:** run regardless of dev accuracy if seed 0 is
   healthy; otherwise repair the protocol implementation before expansion.
6. **Confirmation-readiness seal:** the content-addressed dev100 seed-0 score,
   all 15 healthy training states, all six seed-0 dev banks, adapter-parent
   chains, configs, runtimes, source fingerprints, and analysis freeze are
   bound into a committed readiness record. Full validation and confirm IDs are
   technically inaccessible without it.
7. **Validation confirmation:** after the readiness seal, evaluate confirm400
   once per state and run the preregistered scorer. Its write-once score report
   and manifest bind all 16 candidate banks and training chains.
8. **One-shot test:** `test_in_dist` has a technical gate. The strict runner
   requires a future human-approved release record bound to the frozen protocol
   and scorer commit, exact complete test JSONL and ordered IDs, all 16 verified
   model-state experiment identities and provenance chains, and the exact
   sampling signature; subsets and limits are forbidden. The record is created
   only after methods, contrasts, analysis, and claim language are frozen. If
   its outcome changes any of those, its confirmation label is forfeited and
   that fact is logged. V0.19 implementation creates no such release record.
   The release must also bind and revalidate the complete content-addressed
   confirm400 score artifact; shallow adapter/result manifests are insufficient.
9. **Final test:** remains sealed. V0.19 creates no release record and performs
   no final-test read or evaluation.

OOD-long, OOD-division, alternate prompts, larger N, decoding sweeps, model
size sweeps, and MicroCode are outside this reset. They require later protocols
after the within-v2 ladder is complete.

## Compute budget and authorization boundary

The matrix contains 3 SFT jobs, 12 GRPO jobs, and 16 model states. Dev100 emits
12,800 candidates, confirm400 51,200, and test 64,000: 128,000 candidates if
the future test is released. Using the previously measured batched-HF speedups,
the planning estimate is 30--45 single-A10G-equivalent GPU-hours for training
plus dev/confirmation, excluding the unreleased test. The hard proposed cap is
60 single-A10G-equivalent hours and USD 150, whichever comes first; actual
production launch still requires a committed launch record naming the host,
environment-lock hash, and approved cap. The repository may autonomously run
bounded local preflights. Lack of CUDA is an infrastructure block, not a reason
to substitute scientifically different MPS runs or weaken the matrix.

Confirm400 has good sensitivity only to moderate aggregate task effects. Its
40 easy tasks make easy-tier inference descriptive; the macro summary must not
be described as a well-powered easy-tier claim.

## Required implementation outputs

- deterministic validation-view builder and manifest;
- fail-closed v0.19 matrix/runner specification with immutable arm identities;
- generic v0.19 scorer that verifies run manifests and candidate-bank identity;
- production environment-lock capture/validation and a matrix-wide check that
  runtime/full resolved configs match except for preregistered arm fields;
- provenance-chain checks from each evaluation label to the adapter training
  manifest and, for combined arms, to the exact SFT parent adapter;
- content-addressed dev-score and confirmation-score artifacts plus a committed
  confirmation-readiness record;
- fail-closed v0.19 one-shot test-release validation in the official runner;
- tests for view integrity, arm budgets, seed roles, final-test denial, manifest
  validation, task clustering, tier summaries, and malformed/missing arms;
- updated dataset card, next-steps gate, and experiment ledger;
- independent design and pre-compute diff-review records.
