# v0.18: corrected, globally disjoint Countdown-v2 data protocol

Created: 2026-07-10. Status: **source implementation complete and independently
reviewed; clean pre-generation commit pending**. Human approval was granted to create a new versioned dataset, fix the
generator defect, reserve a final test, lead the protocol decisions, and push
the completed work. Legacy files under `data/countdown/` remain immutable.

## Research question

Can the strong legacy SFT/capability signal and the Stable-RTW framework be
retested on deterministic, verifier-valid data without split leakage, malformed
difficulty rows, uncontrolled provenance, or repeated developmental use of the
final test?

This iteration creates data and audit infrastructure only. It does not train a
model or inspect final-test model performance.

## Version and files

- Dataset protocol: `countdown-dataset-v2`.
- New root: `data/countdown_v2/`; generation must refuse
  `data/countdown/` and any nonempty output directory.
- Legacy generator replay remains explicit through
  `random_solvable_task_legacy_v1`; the public `random_solvable_task` becomes
  the corrected implementation. `scripts/00_generate_countdown_dataset.py`
  must import and call the legacy symbol explicitly, and byte/order replay is
  tested through that script's `build_records`, not only through the alias.
- New deterministic generator: `scripts/18_generate_countdown_v2.py`.
- Independent audit: `scripts/19_audit_countdown_v2.py` and a tracked compact
  report under `docs/artifacts/`.
- Dataset card: `docs/COUNTDOWN_V2_DATASET_CARD.md`.

## Frozen split sizes and difficulty quotas

The task specifications themselves stay unchanged. Easy-task semantic capacity
is limited under the existing 1–12 number range, so a globally disjoint balanced
5,000-row train split is impossible. The frozen quotas preserve 900 easy
training tasks while allocating more gold supervision to the observed
medium/hard bottleneck.

The exact easy loose-key capacity is **1,264**. The global quota of 1,050 uses
83.1% of that finite capacity (train alone uses 71.2%). Generation must check
this bound before proposing tasks and fail if its frozen proposal budget is
exhausted. Inference targets this finite frozen benchmark, not a hypothetical
uniform generator population: first-arrival de-duplication is not uniform over
unique semantic tasks.

| split | total | easy | medium | hard | purpose |
|---|---:|---:|---:|---:|---|
| train | 5,000 | 900 | 2,050 | 2,050 | SFT/RL training only |
| validation | 500 | 50 | 225 | 225 | method development |
| test_in_dist | 500 | 50 | 225 | 225 | one confirmation after method freeze |
| final_test_in_dist | 500 | 50 | 225 | 225 | sealed final paper read only |
| test_ood_long | 500 | — | — | — | six-number OOD confirmation |
| test_ood_division | 500 | — | — | — | division OOD confirmation |

Total: 7,500 tasks. Base seed: `20260710`.

## Generation and allocation

1. Fix the leftover-node defect: any failed/oversized combination invalidates
   the whole attempt, and every returned solution must contain exactly
   `n_numbers` leaves and pass `src/rtw_llm/countdown.py`.
2. Generate one global pool per difficulty using fixed, recorded pool seeds.
   Maintain a single global loose key set `(sorted numbers, target)`; this is
   stronger than exact semantic disjointness and also prevents cross-split
   reuse under a different operator set.
3. Deterministically order each in-distribution difficulty pool, then allocate
   the frozen quotas. This avoids making the final split a sequential tail of
   increasingly rare tasks.
4. Deterministically order records within each split, write full ordered ID
   files, and write a manifest containing recipe, source commit, source hashes,
   counts, rejection statistics, and hashes of every generated artifact.
5. The final split is generated and hash-sealed now. A shared data-access guard
   is added to official SFT, GRPO, eval, and best-of-N runners. Training and
   training-loop evaluation canonicalize every input row and reject any
   intersection with final IDs, semantic keys, or canonical row digests. This
   catches copies, whitespace reserialization, subsets, and final rows appended
   to another file. The lightweight `scripts/03_eval.py` is never release
   authorized. Final model evaluation is allowed only through strict-provenance
   best-of-N and additionally requires the exact complete final JSONL and
   ordered-ID hashes plus an explicit release
   record bound to the dataset-manifest digest and a frozen methods/claims/
   analysis commit; current HEAD must equal that commit. No release record is
   created in v0.18. Its policy is
   `NO_MODEL_EVALUATION_UNTIL_METHODS_CLAIMS_ANALYSIS_AND_STOPPING_RULES_ARE_FROZEN`.

### Frozen generation constants

- Difficulty generation order: `easy`, `medium`, `hard`, `ood_long`,
  `ood_division`.
- Pool targets: 1,050; 2,725; 2,725; 500; 500 respectively.
- Proposal seeds: base seed plus offsets 0, 1, 2, 3, 4 respectively.
- Allocation-order seeds: base seed plus offsets 100, 101, 102 for easy,
  medium, hard.
- In-distribution slice order for every SHA-ordered pool: `train`,
  `validation`, `test_in_dist`, `final_test_in_dist`.
- Within-split order seed offsets: train=200, validation=201,
  test_in_dist=202, final_test_in_dist=203, test_ood_long=204,
  test_ood_division=205.
- Maximum proposals: 250,000 per pool. Exhaustion is a hard failure.
- Seed derivation is integer addition only; Python `hash()` and set iteration
  order are forbidden.
- Allocation and file order use SHA-256 sort keys over the explicit seed,
  domain label, and canonical semantic key, with the canonical key as tie-break.
- Proposal RNG is pinned to CPython 3.11 `random.Random` and recorded in the
  manifest. JSONL is canonical UTF-8 JSON with sorted keys, fixed separators,
  and exactly one LF per record.

Pool construction is deliberately order-dependent only where hard and
OOD-division loose keys can collide; the frozen generation order above defines
ownership. Allocation *within* each in-distribution pool is exchangeable after
SHA ordering, so final-test tasks are not the rejection tail.

### Manifest and archive identity

The manifest is not self-referential. It records clean pre-generation source
commit **C**, raw source hashes, recipe/runtime, artifact hashes for every JSONL
and ID file **excluding the manifest**, and a canonical digest over the
manifest core excluding that digest. The post-generation audit hashes the
manifest itself. The later data/archive commit **D** is recorded in the ledger,
not rewritten into the manifest. Audit at D verifies current source hashes
still equal those generated from C.

## Audit and acceptance gates

The independent v2 audit must fail closed unless all conditions hold:

- exact split counts and quotas;
- required schema/types and globally unique IDs;
- exact operand count/operator set for every difficulty;
- every stored solution and completion passes the source-of-truth verifier;
- stored low/mid/high/default prompts and completions match current functions;
- zero within-split duplicates and zero exact **and loose** overlap across all
  split pairs;
- ordered ID files exactly match their split and have verified hashes;
- manifest core digest, artifact/source hashes, generator version, base seed,
  frozen proposal/order constants, runtime, and final-test policy match;
- deterministic replay regenerates byte-identical JSONL, ID lists, and manifest
  payload (excluding no fields; there are no timestamps);
- legacy dataset **and frozen-ID** hashes equal hardcoded values pinned from the
  committed v0.16 artifact; the mutable artifact is not trusted as expectation.

The v2 auditor independently hardcodes the file list, counts, quotas, expected
operand counts, exact ordered operator lists, base seed, generation constants,
and policy string. It may import the source-of-truth verifier and prompt
functions, but it must not trust recipe constants from the generator or
manifest. It checks within-split loose duplicates as well as exact duplicates
and cross-split overlap.

Tests must cover the three-node leftover failure, exact operand-count property,
legacy replay preservation, global allocation/disjointness, deterministic
bytes, write-once/refuse-legacy behavior, manifest tampering, loose-only
collisions, byte-identity detection after a copied final file, release-record
binding, and malformed rows. Full pytest, Ruff,
shell syntax, independent diff review, and the real 7,500-row audit must pass.

## Archival sequence and stopping rule

1. Commit reviewed generator/audit code before generating data.
2. Generate into a temporary sibling directory from clean CPython 3.11 source
   commit C, then atomically rename. Refuse any pre-existing target, including
   an empty directory; no partial directory can be accepted.
3. Run the audit without evaluating any model on `final_test_in_dist`.
4. If any acceptance condition fails, do not patch generated rows manually;
   fix code, delete only the uncommitted v2 output, recommit, and regenerate.
5. Commit the generated data, manifest, audit artifact, and result notes; push
   all protocol-hardening commits to the remote.

GPU remains paused. The next GPU ladder requires its own pre-registration after
this dataset audit passes.

## Comparability and future estimands

V2 simultaneously fixes leakage/malformed rows, updates prompts, changes train
size and difficulty mix, and enlarges evaluation. Therefore no v2-minus-legacy
number is a causal one-variable method effect. Every claimed baseline must be
rerun within v2 under identical budgets: base, SFT-only, GRPO-only static and
Stable-RTW, and SFT+GRPO static/Stable as required by the final claims.

The future primary accuracy estimand is verifier-exact success over the full
500-task finite split, with a macro average across easy/medium/hard reported
alongside it and v0.17 task-clustered analysis. The 50 easy rows per split are
descriptive/underpowered alone. `validation` is developmental;
`test_in_dist` is one-shot confirmation and cannot trigger method changes
without forfeiting that label; `final_test_in_dist` remains technically blocked
until the release contract is satisfied. Training-seed generalization remains
limited by the number of new true end-to-end seeds.
