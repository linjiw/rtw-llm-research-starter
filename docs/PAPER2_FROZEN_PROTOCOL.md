# Paper-2 (MicroCode) frozen protocol — pre-registration

Created 2026-07-20 (S2/I10-c). This is the **pre-registered, frozen** protocol
for the MicroCode experiments. It mirrors the Countdown frozen-protocol
discipline (`AUTORESEARCH_PROGRAM.md` §3–4). **Nothing in the "Frozen" section
may change during autonomous iterations** — a change is a human-escalation item.

Grounding: `PAPER2_MICROCODE_TESTBED_SPEC.md` (design), `DATASET_CARD_MICROCODE.md`
(dataset), `RESEARCH_GOAL_AND_PLANS_20260709.md` §4 (E4/E5).

## Frozen — dataset & task IDs

- Dataset committed under `data/microcode/` by
  `scripts/30_generate_microcode_dataset.py` (seed 42): train 1998,
  validation 201, test_in_dist 201, test_ood_compose 200, test_ood_transform 200.
- Frozen eval task IDs (50/split, tier-balanced, deterministic):
  `data/microcode/frozen_microcode_task_ids_{validation,test_in_dist}_limit50.txt`.
  Verified subsets of the committed jsonl.
- The generator is FROZEN (30 templates, 24 train + ood_compose/ood_transform).

## Frozen — sampling config (best-of-N eval)

```text
temperature      0.7
top_p            0.95
max_new_tokens   256
sampling seed    0
N values         1, 4, 8   (max_n = 8)
prompt field     prompt_high
generation mode  loop-mode ONLY for protocol comparisons (batched = exploratory)
```

## Frozen — verifier & selector

- Verifier = `src/rtw_llm/microcode.py` (`verify_completion`), sole source of
  truth. Primary success = `held_out_all_pass` (`correct` == `exact_correct`).
- Practical selector = `microcode_practical_score` (frozen weights:
  valid_expression 3.0, runs_without_error 1.0, visible_pass_rate 2.0,
  no_hardcoding_heuristic 1.0). It uses ONLY deployment-observable features and
  **never** reads `held_out_pass_rate` / `correct` / `exact_correct`
  (test-guarded). `visible_pass_rate` IS a selection feature (the realistic
  deployment signal); the analysis separates proxy-selection from held-out truth
  post hoc, and the SAME selector is used for both arms so it is never a variable.
- Oracle selection (upper bound) uses `held_out_all_pass` directly, reported
  separately from the practical selector (as in Countdown).

## Frozen — teacher aux set & HONEST budget (E4)

- `MICRO_AUX_KEYS = [valid_expression, runs_without_error, visible_pass_rate,
  no_hardcoding_heuristic]` (finalized by `scripts/31_microcode_aux_prune.py`;
  scaffold/collinear channels pruned; `held_out_pass_rate` is diagnostic-only,
  never a weighted wheel).
- HONEST floors `MICRO_STABLE_FLOORS` (visible_pass_rate floor 0.05 < the ~0.295
  crossover so adaptive can down-weight the proxy), target weight sum 0.80.
- Strategies restricted to adaptive_stable / static / manual / random
  (adaptive_phased excluded — Countdown-coupled). Curriculum
  graded_key=`held_out_pass_rate`, gate_key=`valid_expression`.

## Frozen — experiment arms & metrics

- **E4 (HONEST pilot):** static vs adaptive_stable, HONEST budget, 300 steps,
  seed 0. ONE variable = the teacher strategy. Primary = paired
  `held_out_all_pass@8` (McNemar discordants) on frozen validation IDs.
  Health gate: `group_reward_std` logged per step must confirm within-group
  variance stays UNSATURATED during training (not just at init) — the live
  precondition-#1 check. Decision: both arms healthy + non-degenerate → unlocks
  E5; degeneracy → ledger NO-GO + 1.5B / SFT-format-warmup fallback.
- **Headline metric (E4 diagnostic, E5 primary):** proxy−primary gap =
  `visible_pass_rate − held_out_pass_rate` vs step (+ `no_hardcoding` firing
  rate). Measured by the held-out verifier throughout.
- **Statistical unit** for any static-vs-adaptive interaction = the seed-level
  policy comparison (underpowered by design; label it; never an n=candidate
  framing).

## DEFERRED to the E5 pre-registration (NOT frozen here — need advisor input)

These are the E5-headline knobs and are deliberately left open (they gate E5,
not E4):
- The TEMPTATION proxy-overweight per-key **init/static weight vector**
  (visible_pass_rate init ~0.30–0.35 proxy-dominant; true-signal channels ~0.10;
  held_out floor ≥0.10). Requires the `init_weights` vector (I8b, done).
- The `hack_wins` fairness pre-check (TEMPTATION-static must demonstrably reach
  the hack signature at 0.5B in ~300 steps BEFORE any resistance claim).
- **Sandbox hardening (S3/I7): PARTIALLY DONE.** The spawn-worker + RLIMIT_AS
  memory wall + parent-crash isolation is BUILT and tested
  (`microcode_sandbox.py`; ledger `s3-sandbox`) — this covers the accidental-
  DoS threat and is turned ON for E4/E5 GPU runs (`sandbox="worker"`). What
  remains before an E5 hacking-RESISTANCE headline is the SOUND-ESCAPE question:
  the AST whitelist is not sound (C-level gadget → os), so OS-level isolation
  (nsjail/firejail/gVisor/seccomp) is still required before claiming resistance
  to a model that actively escapes. Until then, scope any E5 claim to
  reward-channel behavior (does adaptive down-weight the proxy), NOT security.

## Eval runner (I10-b, built at E4 launch)

The best-of-N eval will reuse the task-agnostic helpers from
`scripts/07_best_of_n_rerank.py` (which is on the FROZEN list — must NOT be
edited) via a new `scripts/21_microcode_best_of_n.py` that re-implements locally
only the Countdown-global-bound pieces (selected-metric keys, summarize/evaluate)
and defines the MicroCode selector import. Built when E4 launches (it needs the
model); the selector + frozen IDs + this protocol are the pre-registration.
