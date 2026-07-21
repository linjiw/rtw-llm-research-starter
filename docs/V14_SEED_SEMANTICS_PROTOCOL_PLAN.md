# v0.14 protocol gate: explicit GRPO seed semantics

Created: 2026-07-09. Status: pre-registered infrastructure iteration; no GPU
run is authorized by this document.

## Problem

`scripts/02_grpo_train.py --seed N` currently seeds `RTWTeacher`, but the
value is not passed to `GRPOConfig`. TRL therefore uses its default trainer-
loop seed (`42`) for every run, and the curriculum sampler inherits that seed.
TRL 1.7 applies its internal seed only after model and fresh-LoRA construction,
so historical adapter initialization was not necessarily controlled by 42.
Historical Stable/static runs labeled seeds 0/1/2 are consequently legacy
repeat runs with teacher labels 0/1/2, a trainer-loop seed of 42, and
uncontrolled pre-trainer initialization—not independent end-to-end training
seeds. v0.13 seeds vary SFT and teacher seeds while keeping the GRPO trainer
loop at 42, as its plan explicitly required.

## Hypothesis

An explicit, fail-closed seed contract can preserve exact reproduction of the
archived protocol while making future true-seed experiments unambiguous.

## One-variable implementation

Change only seed plumbing in `scripts/01_sft_warmup.py` and
`scripts/02_grpo_train.py`:

- Keep `--seed` as the teacher seed for backward compatibility.
- Add `--trainer_seed`, default `42`.
- Add an explicit protocol identifier:
  - `countdown-legacy-v1`: requires trainer seed 42.
  - `countdown-true-seeds-v2`: requires trainer seed == teacher seed.
- For corrected-v2 only, apply the seed globally before any dataset, model, or
  adapter construction. Do not pre-seed legacy-v1: that would change archived
  fresh-LoRA initialization behavior.
- Pass the resolved trainer seed to `GRPOConfig` and assert the constructed
  config retains it.
- The curriculum sampler continues to inherit `train_args.seed`, so its seed
  becomes explicit under the same contract.
- SFT uses the same protocol identifier. Legacy-v1 preserves its historical
  pre-trainer LoRA RNG behavior; corrected-v2 pre-seeds before SFT model/LoRA
  construction and retains the requested seed in `SFTConfig`.

No verifier, reward, teacher algorithm, selector, dataset, frozen task ID,
training budget, or historical runner changes are allowed in this iteration.

## Acceptance criteria

1. CPU tests prove legacy resolution `(teacher=N, trainer=42)`.
2. CPU tests prove corrected resolution `(teacher=N, trainer=N)`.
3. Corrected-v2 pre-seeds before model construction; legacy-v1 does not.
4. Mislabeled combinations fail before model loading.
5. The generated `GRPOConfig.seed` equals the requested trainer seed.
6. Full pytest and Ruff pass.

## Interpretation rule

- Every existing artifact remains `countdown-legacy-v1`.
- Existing Stable/static “seeds” must be described as legacy repeats, not true
  training seeds.
- Future seed-generalization claims must use `countdown-true-seeds-v2` and
  rerun every compared arm under that protocol. Never mix v1/v2 banks in a
  method comparison.

## Decision

KEEP if all acceptance criteria pass without modifying frozen components.
Otherwise DISCARD and stop GPU work until the seed contract is repaired.

## Deferred, separate iterations

Run manifests, dataset-overlap auditing, and clustered statistical inference
remain separate additive iterations under the one-variable autoresearch rule.
