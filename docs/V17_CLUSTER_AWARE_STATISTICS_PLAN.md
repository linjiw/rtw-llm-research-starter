# v0.17 protocol gate: task-clustered statistical inference

Created: 2026-07-09. Status: **completed — KEEP as protocol infrastructure;
legacy clustered estimates unavailable because raw banks are absent**. No
model training or evaluation generation was authorized by this iteration.

## Question

Can the existing analysis path stop treating correlated candidates and
repeated run panels as independent observations, while preserving auditable
descriptive summaries of the historical results?

## Unit of analysis and estimand

The independent unit is a canonical semantic task, not a candidate and not a
task-by-run cell. A task key is the sorted number multiset, integer target, and
sorted unique allowed-operator set. Duplicate semantic tasks, missing cells,
unequal candidate counts, or mismatched task panels fail closed.

For arm `A`, baseline `B`, task `t`, and the observed runs in each arm:

```text
D_t   = mean_r A[t, r] - mean_r B[t, r]
Delta = mean_t D_t
```

`Delta` is an equal-task contrast conditional on the observed run panels. It
does not estimate a training-seed population unless a separate true-seed
design supports that claim. Candidate legality is reduced to one rate per
task/run before inference. Selected exactness is one verifier-derived binary
value per task/run.

## Inference

1. A deterministic percentile bootstrap resamples whole task rows, carrying
   every run value together. Default: 20,000 draws, NumPy PCG64, recorded seed.
2. A two-sided task-level sign-flip test operates on `D_t`. It enumerates every
   assignment when there are at most 20 nonzero task effects and otherwise
   uses deterministic Monte Carlo with the finite-sample `(extreme+1)/(B+1)`
   correction.
3. `P(exact | legal)` remains explicitly post-treatment and non-causal. Its
   difference uses a task-cluster ratio bootstrap over exact-and-legal
   numerators and legal denominators. Zero observed denominators make the
   estimand unavailable; zero-denominator bootstrap draws are counted and
   dropped, never silently replaced. Independent design review added a
   fail-closed support rule: if more than 1% of draws have an undefined ratio,
   the interval is withheld and only a labeled descriptive estimate remains.
4. An optional task-by-seed product bootstrap is allowed only for paired grids
   produced by `countdown-true-seeds-v2` with at least three true seeds. Three
   seeds are labeled `exploratory_underpowered_seed_generalization`; no
   confirmatory p-value is reported. Every contributing run must also pass the
   v0.15 completed-manifest verification, including matching seed-role labels.

All intervals and p-values are exploratory for legacy-v1 because the audited
dataset is reused, leaky across splits, and development decisions have already
seen its test split.

## Script integration

- `scripts/08_summarize_v09_seed_expansion.py`: add task-clustered summaries
  reconstructed from raw candidate banks. Keep per-run paired 2x2 tables.
  Move the old cross-run pooled McNemar calculation under an object named
  `legacy_pseudoreplicated_descriptive_only` with `inference_valid=false`.
- `scripts/12_score_v13.py`: compare easy-tier legality using task-level
  candidate rates and compare conditional exactness with the cluster ratio
  bootstrap. Move the old candidate-pooled z-test under the same explicit
  invalid-legacy label. An explicit `--combine_arms_as` surface stacks the
  v13 arm runs for the declared observed-panel estimand while retaining each
  individual run's descriptors.
- If raw candidate banks are unavailable, report clustered inference as
  unavailable. Never reconstruct p-values from aggregate counts.

Every compared bank must share one persisted evaluation-protocol signature:
sampling seed, temperature, top-p, completion cap, prompt field, engine/device,
HF generation mode and relevant batch size, split, N/task counts, data hash,
and ordered-task-file hash. Strict runs additionally verify their completed
manifests. The tracked legacy v0.9 aggregate is metadata-migrated so its pooled
p-values remain visible only inside the invalid legacy diagnostic object.

## Acceptance tests

- identical arms and all-positive task effects;
- correlated candidate outcomes count as tasks, not `tasks * candidates`;
- task-row joint resampling and deterministic golden results;
- exact and Monte Carlo sign-flip paths;
- missing cells, duplicate semantic tasks, and unequal candidate counts fail;
- zero legal denominators are explicit;
- task-by-seed inference rejects legacy/false seed declarations;
- both scripts tag the former pooled tests as invalid and expose no generic
  pooled p-value as valid inference.
- mismatched evaluation signatures, fractional task identities, sparse legal
  support, and an unrequested/malformed arm panel fail closed.

## Claim limits and decision rule

This gate may repair analysis code and withdraw invalid inferential labels. It
cannot validate the legacy-v1 dataset, recover absent raw evidence, establish
causal adaptivity, or support a training-seed population claim. KEEP only if
the full test suite and Ruff pass and an independent diff review finds no
remaining inference-corruption bug. GPU remains paused regardless of outcome.

## Result

- Added `src/rtw_llm/cluster_stats.py`: deterministic whole-task bootstrap,
  exact/Monte Carlo task sign-flip tests, sparse-safe conditional-ratio
  bootstrap, and an explicitly exploratory true-seed product bootstrap.
- Both analysis scripts now fail closed on semantic duplicates, missing cells,
  unequal candidate N, incomplete or mismatched evaluation signatures, and
  unverified corrected-v2 seed claims.
- `scripts/12_score_v13.py` now reports a separately requested combined arm
  panel and calls all historical run labels “observed runs,” not true seeds.
- The tracked `outputs/v09_seed_expansion_paired.json` was metadata-migrated:
  cross-run pooled p-values remain visible only inside
  `legacy_pseudoreplicated_descriptive_only` with `inference_valid=false`.
- No raw v0.9/v0.13 candidate banks are present in this checkout. Therefore no
  clustered legacy interval was fabricated; regenerated summaries will return
  `{available:false, analyses:[]}` until verified raw banks are restored.
- Acceptance: **170 tests passed**, Ruff clean, shell syntax and diff checks
  clean. Independent adversarial review returned CLEAR after all seven verified
  findings were fixed.

Verdict: **KEEP** the analysis infrastructure. This closes the statistics-code
gate but does not unblock corrected-v2 GPU work: v0.16 still blocks on the
leaky/defective legacy dataset and requires a separate human-approved v2 data
protocol.
