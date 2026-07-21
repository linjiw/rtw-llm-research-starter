# V0.19 independent pre-compute diff review

Date: 2026-07-10  
Protocol: `countdown-v19-within-v2-reset-v1`  
Reviewed manifest core:
`d4d175ed6368f6a5f4d4ccddfdd77848bd1980cf973f1108c3d9eed9124db8ca`

## Verdict

**CLEAR — no remaining confirmed pre-compute blocker.**

The first diff review returned BLOCK on four silent-corruption paths:

1. production did not enforce the registered transitive source snapshot;
2. confirm400 could be reached without a technical readiness gate;
3. the scorer could verify an altered task payload under the expected ID;
4. one-shot test release did not require the complete confirmatory score.

All four were fixed and independently re-reviewed. The cleared snapshot:

- fingerprints all 31 registered/package Python sources and replay-checks each
  run commit;
- blocks full validation and confirm IDs until a committed readiness record
  binds all 15 healthy training states, six seed-0 dev banks, adapter ancestry,
  runtime/config identity, and the content-addressed dev score;
- requires CUDA for registered dev, confirm, and test evaluation;
- matches every candidate task payload to its frozen source row, recomputes
  verifier metrics, validates candidate grids and exact token/finish metadata,
  and keeps task-cluster/Holm inference fail-closed;
- requires the healthy training matrix, all 16 confirmation states, and the
  content-addressed confirm400 score before a future one-shot test release;
- keeps the bounded local preflight to train data and two dev IDs only.

Acceptance evidence: protocol replay `ELIGIBLE`; 211 tests passed; Ruff,
compile, and diff checks clean. This record authorizes only the bounded local
preflight. Production still requires a separately captured, committed CUDA
environment lock and launch record.
