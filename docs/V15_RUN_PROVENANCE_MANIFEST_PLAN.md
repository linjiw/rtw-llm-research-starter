# v0.15 protocol gate: content-addressed run provenance

Created: 2026-07-09. Status: **KEEP — CPU acceptance passed**; no GPU run is
authorized until the remaining dataset/statistics gates close.

## Problem

Training outputs are ignored by Git and current run records do not bind a
result to its code state, explicit seed roles, resolved configuration, input
data, or produced evidence files. A directory containing `metrics.json` is not
enough to establish that a paper number came from the intended experiment.

## Hypothesis

A shared, fail-closed manifest contract across SFT, GRPO, and best-of-N can
prevent accidental output reuse and make every future corrected-v2 result
auditable without changing experimental behavior.

## Scope

Add one provenance module and integrate it behind `--strict_provenance` in:

- `scripts/01_sft_warmup.py`
- `scripts/02_grpo_train.py`
- `scripts/07_best_of_n_rerank.py`

Strict provenance is mandatory whenever a runner declares
`countdown-true-seeds-v2`; legacy-v1 remains backward compatible. Do not alter
the verifier, selector, dataset, reward, teacher, curriculum, sampling values,
or training values.

## Manifest contract

Each strict run writes:

1. `run_intent.json`, before model/trainer/engine construction. Its deterministic
   identity includes run kind, schema, Git commit and clean state, parsed CLI
   arguments, resolved configuration, explicit seed-role map, input file
   SHA-256/size, and model/adapter identifiers or local digests.
2. `run_result.json`, only after the run succeeds. It links the exact intent
   digest and records SHA-256/size for critical evidence artifacts.

Timestamps, elapsed time, PID, and storage location are observations, not part
of the experiment identity. Environment variables and secrets are never
serialized.

Local manifests under ignored `outputs/` are fail-closed execution records,
not durable evidence. A paper claim still requires publishing or committing a
compact bundle containing both manifests plus the listed small evidence files
or immutable artifact URLs.

## Fail-closed rules

- Strict runs require a clean tracked Git state.
- Intent/result files are atomically created and never overwritten.
- A non-empty output directory without a matching completed manifest is not
  reusable by a strict run.
- Existing manifests are verified by canonical digest; malformed or mismatched
  records abort before expensive compute.
- Result verification re-hashes every listed artifact; tampering fails.
- Strict best-of-N `--skip_if_complete` may skip only after manifest and
  artifact verification, never from legacy file-presence checks alone.
- An interrupted strict run gets a new output directory; automatic resume is
  out of scope because teacher/curriculum state is not fully resumable.

## Acceptance criteria

1. Canonical hashing is key-order invariant and byte/order sensitive.
2. Atomic no-overwrite and incomplete-run refusal are tested.
3. Dirty-tree, input/config mismatch, intent/result-link, and artifact-tamper
   failures are tested.
4. SFT, GRPO, and best-of-N serialize their distinct seed roles.
5. Strict intent is written before model/trainer/engine construction.
6. Full pytest and Ruff pass; independent diff review finds no blocker.

## Decision

KEEP if the contract passes all CPU acceptance tests without changing default
legacy execution semantics. Otherwise DISCARD and keep the GPU pause in place.

## Result

Implemented the shared v1 contract in `src/rtw_llm/provenance.py` and wired it
behind `--strict_provenance` in SFT, GRPO, and best-of-N. Corrected-v2 requires
strict provenance. Strict remote models require a full 40-hex Hugging Face
commit, and both model and tokenizer load that exact revision. Local
model/adapter identities require a real weight payload and hash supported
weight shards plus stable config/tokenizer/upstream-manifest files.

Intent is created before trainer/model/engine construction; result links the
intent and hashes the critical output evidence. Strict best-of-N skip requires
and verifies `candidates.jsonl`, `metrics.json`, `run_config.json`, and
`summary.csv`. Canonical experiment identity is invariant to output, dataset,
local model, and adapter storage paths once their content identities are
recorded.

CPU acceptance: **131 tests pass; Ruff clean.** Independent design and diff
reviews found five blockers (mutable revisions, omitted weight shards,
under-specified skip artifacts, and two storage-path leaks); all were fixed and
the final review cleared. Local ignored manifests are still not durable paper
evidence—their compact bundle must be committed or published after a run.
