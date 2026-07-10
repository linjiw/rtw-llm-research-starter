# v0.16 protocol gate: Countdown dataset integrity and leakage audit

Created: 2026-07-09. Status: pre-registered read-only audit; frozen legacy
datasets and task-ID files must not be modified.

## Question

Are the committed Countdown artifacts internally valid and reproducible, and
are they eligible for future `countdown-true-seeds-v2` paper comparisons?

## Scope

Add a deterministic CPU audit and tests. The audit reads committed JSONL data,
frozen ordered task-ID files, generator/prompt/verifier source, README, and
Makefile. It may write a deterministic report, but must not edit the verifier,
selector, generator, prompts, data, or task IDs.

## Canonical task identities

- Exact semantic key: sorted number multiset + integer target + sorted unique
  allowed operators. Operator presentation order is not task semantics.
- Loose diagnostic key: sorted number multiset + target, intentionally ignoring
  operators. Loose-only overlap is a warning, not exact leakage.

## Checks

1. Raw SHA-256, size, line count, and deterministic repo-relative path for
   every committed dataset and frozen-ID file.
2. Required schema/types, expected split/difficulty, globally unique IDs, and
   no within-split exact semantic duplicates.
3. Every stored solution and completion passes `countdown.py`, the correctness
   source of truth.
4. Exact and loose overlap matrices for all split pairs: shared key groups,
   affected records on each side, record-pair combinations, rates, and IDs.
5. Frozen IDs are ordered, unique, present in the intended split, and receive
   separate train-exposure and cross-frozen overlap accounting.
6. Stored `prompt_low/mid/high`, default prompt, and completion are compared
   byte-for-byte with current prompt/completion functions. Drift is provenance
   evidence, not permission to rewrite frozen inputs.
7. Replay the inferred legacy recipe (counts from committed files, base seed
   42 with split offsets) and compare every non-prompt field/order; report that
   README/Makefile advertise different counts.
8. Record HEAD plus raw hashes of generator, prompt, verifier, and audit source.

## Decision rules

Integrity FAIL: malformed schema/type, global duplicate ID, invalid stored
solution/completion, within-split semantic duplicate, or bad frozen ID.

Corrected-v2 eligibility FAIL: any exact cross-split semantic overlap.

WARN only: prompt/template byte drift with valid stored semantics, loose-only
overlap, noncanonical operator order, or an undocumented/mismatched generation
recipe.

The expected honest outcome is that legacy-v1 remains auditable and usable for
historical reproduction, while corrected-v2 is blocked until a human-approved,
globally disjoint dataset/version and untouched final test are created.

## Acceptance

- Synthetic tests cover clean data, reordered semantic duplicates, loose-only
  overlap, invalid verifier output, bad frozen IDs, prompt drift, and
  deterministic byte-identical reports.
- Full pytest and Ruff pass; independent diff review clears the implementation.
