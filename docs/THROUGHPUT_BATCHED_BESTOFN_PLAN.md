# Throughput: batched HF generation for best-of-N (infra iteration)

Created: 2026-07-09. Status: design advisor-reviewed (10 findings; 3 blocking
amendments folded in below and marked [ADV-n]).
Queue slot: AUTORESEARCH_PROGRAM.md §6 "Supporting audits — Throughput".
Interleaved while Gate 0 occupies the GPU (~4 h of evals remaining).

## Problem

`HFEngine.generate` (src/rtw_llm/engine.py) iterates prompts one at a time;
the `--batch_size 8` flag in `scripts/07_best_of_n_rerank.py` only chunks the
candidate list, so each chunk is 8 *sequential* `model.generate` calls.
Measured cost: ~47 s per batch of 8 ≈ 5.9 s/candidate at 256 new tokens.
A full frozen eval (50 tasks × 8 candidates) ≈ 40 min; a 6-eval ladder ≈ 4 h.
The A10G sits at 1.3 GB / 23 GB during eval — decode is memory-bandwidth-bound
for a 0.5B model, so true batching should recover a large factor.

## Hypothesis

True batched generation (left-padded, attention-masked, one `generate` call
per chunk) gives ≥2× wall-clock reduction on the frozen best-of-N eval **at
batch 32** (the batch size we would adopt), with no change to eval semantics
(verifier, selector, metrics, candidate bookkeeping untouched). [ADV-10: the
bar sits at the adoption batch size, not at batch 8 where it is trivial.]

## Key facts established during design review (verified on this stack)

- Qwen2.5-0.5B-Instruct's `generation_config.json` silently injects
  `repetition_penalty=1.1` and `top_k=20` into every `generate` call. The
  **effective** frozen sampling distribution has always been
  temp 0.7 / top_p 0.95 / top_k 20 / repetition_penalty 1.1. All archived
  loop-mode artifacts (v0.9B, Gate 0) were produced under it. Do NOT
  "clean up" these values — that would change the frozen protocol. [ADV-4]
- The tokenizer's default pad token `<|endoftext|>` (151643) is **in the
  model's `eos_token_id` list** `[151645, 151643]`, and transformers'
  `RepetitionPenaltyLogitsProcessor` counts left-pad tokens as "already
  generated" (it never sees the attention mask). Padding with EOS therefore
  systematically suppresses termination on padded rows (~up to 7× at
  temp 0.7) — a directional, prompt-length-correlated distribution shift,
  not just different RNG draws. [ADV-1, BLOCKING]
- With the frozen config (`max_n=8`, `--batch_size 8`) the harness expands
  candidates task-major, so every batch is 8 copies of the same prompt —
  zero padding. Padding effects only activate at batch ≥16 with mixed tasks.
  [ADV-2]

## Change (one variable, default-off)

1. `src/rtw_llm/engine.py`: batched path in `HFEngine`, opt-in
   (`gen_mode="batched"`), default `loop` byte-identical to today:
   - pad with `<|fim_pad|>` (151662) — a registered special token that is
     neither in `eos_token_id` nor ever generated; assert at init that the
     chosen pad ID is not in the effective EOS list [ADV-1];
   - `padding_side="left"` set once at init [ADV-10];
   - tokenize the chunk with `padding=True`, pass `attention_mask`; single
     `model.generate`; new tokens for row *i* = `gen[i][padded_input_len:]`
     (uniform padded length; position_ids derive from the attention mask in
     transformers 5.12.1 — verified); decode `skip_special_tokens=True`.
   - Resolve and expose the **effective** generation config (incl. inherited
     top_k / repetition_penalty) so callers can record it. [ADV-4]
2. `scripts/07_best_of_n_rerank.py`:
   - flag `--hf_gen_mode {loop,batched}`, default `loop`; recorded in
     `run_config.json` and `metrics.json`, plus the effective generation
     config [ADV-4];
   - **sampling identity in batched mode = (seed, hf_gen_mode, batch_size)**:
     batched RNG consumption depends on batch size, so `batch_size` is part
     of the candidate-bank identity (loop mode is chunking-invariant) [ADV-6];
   - harden `is_complete`/`--skip_if_complete`: compare stored
     `run_config.json` against the requested args (hf_gen_mode, batch_size
     when batched, model, adapter, seed, temperature, top_p); refuse to skip
     on mismatch (missing `hf_gen_mode` in old artifacts ⇒ legacy loop)
     [ADV-7, closes a pre-existing silent-mixup hole].
3. `scripts/08_summarize_v09_seed_expansion.py` (additive): paired-overlap
   raises on `hf_gen_mode`/`batch_size`-identity mismatch between arms
   (legacy configs without the key count as loop). [ADV-6]
4. New `scripts/09_benchmark_generation.py`:
   - **two-tier equivalence** [ADV-5]: (i) same-token-length prompt batch,
     greedy, CPU fp32 → loop vs batched must match token-for-token (pure
     batching numerics, no padding involved); (ii) mixed-length batch →
     run with safe pad vs pad=EOS; a mismatch that disappears under the safe
     pad confirms the ADV-1 mechanism. Any mismatch is investigate-and-
     explain, never auto-excused as nondeterminism;
   - **distribution check** (the statistic ADV-1 corrupts): completion-length
     and EOS-termination-rate distributions, loop vs batched, mixed-length
     batches [ADV-2];
   - **timing**: requires idle GPU (assert via nvidia-smi), ≥1 warmup
     generate, median of ≥3 repeats per (mode, batch) cell, batch 8/16/32
     [ADV-8].
5. `scripts/03_eval.py` and training untouched (default path unreachable
   change surface for 03).

## Isolation from the in-flight Gate 0 ladder [ADV-3, BLOCKING]

Gate 0's runner spawns a **fresh `07` process per eval stage** from this
checkout. Editing `engine.py`/`07` in the working tree would split the
ladder across code states. Therefore: implement and test in a **git
worktree**; merge into `main` only after Gate 0's last `07` invocation
completes. Even a "byte-identical default path" is an unverified claim until
the diff review passes — no exceptions.

## What this deliberately does NOT change

- Verifier, selector, frozen task IDs, sampling config (including the newly
  documented effective top_k/repetition_penalty) — untouched.
- No new packages. vLLM deferred: installing it into the live `.venv` risks
  the training stack, and `VLLMEngine` lacks LoRA support. Revisit only if
  batching lands <2×, as its own iteration in a separate venv.

## Comparability rules (scientific caveats)

- Paired per-task comparisons are valid only within the same sampling
  identity: loop vs loop, or batched-at-B vs batched-at-B. [ADV-6]
- Gate 0 artifacts are loop-mode ⇒ the v0.10 C2 eval pairs against them and
  **must stay loop-mode**. Nothing about the v0.10 plan changes.
- Wall-clock / cost-per-exact guardrails are only comparable within a mode
  (the `n/max_n` wall-clock scaling assumes per-candidate linearity, which
  batching breaks); token counts remain cross-mode comparable. [ADV-9]
- Batched mode becomes the protocol default for *new* comparisons only after
  acceptance passes, and any comparison mixing old artifacts re-collects
  both arms under the same identity.

## Validation plan (gate before use)

CPU, in the worktree, before merge:
- `uv run pytest -q` + `ruff check .`; new unit tests: left-pad slice helper,
  pad-not-in-EOS assertion, flag recorded in run_config, `is_complete`
  refuses on mode/identity mismatch, early-EOS right-padded row does not
  inflate `completion_token_count` [ADV-10], default path unchanged.
- CPU smoke of `09`: tier-(i) greedy equivalence exact at fp32.

GPU, first idle window after Gate 0 (do not contend with C2 stages) [ADV-8]:
- `09` timing + tier-(ii) equivalence + distribution checks at batch 8/16/32.
- Distributional sanity **at batch 32** [ADV-2]: one batched base-model
  validation eval (50 tasks, N=8) vs Gate 0's loop-mode base eval —
  aggregate rates (valid, exact@8, oracle exact@8, F1) within binomial
  noise; completion-length / EOS-rate distributions compared explicitly.

Acceptance: ≥2× speedup at batch 32, tier-(i) equivalence exact, no
unexplained tier-(ii) mismatch, distribution checks clean, no OOM at 32.
Failure → keep `loop` default, record in ledger, consider vLLM iteration.

## Ledger row (to fill)

id `infra-batchgen`, hypothesis as above, verdict KEEP if acceptance holds
(batched becomes the recommended mode for *new* non-mixed comparisons),
DISCARD otherwise. Either way the default stays `loop` until the row is
filled. Also record the ADV-4 discovery (effective sampling config includes
top_k=20 / repetition_penalty=1.1 from the model's generation_config) in the
ledger and note it in AUTORESEARCH_PROGRAM.md §3 as a clarification of what
"frozen sampling config" has always meant on this stack.
