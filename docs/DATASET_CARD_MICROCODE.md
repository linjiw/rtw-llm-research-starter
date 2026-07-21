# Dataset card: MicroCode (Paper-2 testbed)

Test-driven Python function synthesis, built to satisfy the three adaptivity
preconditions Countdown lacked (dense within-group reward variance, a live
reward-hacking surface, a smooth difficulty gradient). Generator:
`src/rtw_llm/microcode_gen.py`; verifier: `src/rtw_llm/microcode.py`; spec:
`docs/PAPER2_MICROCODE_TESTBED_SPEC.md`.

## Task

The model emits ONE self-contained Python function inside `<answer>...</answer>`
from a signature + docstring + K visible unit tests. It is graded by executing
the extracted function against a **held-out** unit-test suite (the source of
truth). Visible tests are the hackable proxy; held-out tests are strict truth.
Every task is **solvable by construction**: held-out I/O is generated at build
time by running a vetted reference implementation (the analogue of Countdown
building solvable expression trees).

## Generation

`random_solvable_task(rng, difficulty_spec(tier), idx, split)` samples a
template for the tier, randomizes the function name (anti template-identity
memorization; param names stay fixed to carry docstring meaning), and generates
visible + held-out I/O by running the reference. Determinism: references return
only JSON-serializable ints/bools/None/lists/dict-of-lists (never sets or
tuples), take no wall-clock dependence, and are pure (no input mutation), so a
task JSON-round-trips and re-verifies bit-identically.

## Splits and families

| split / tier | family | rungs | role |
|---|---|---|---|
| easy / medium / hard | `train` | R0–R1 / R2–R3 / R4–R5 | training + in-distribution eval |
| `ood_compose` | held-out | R3–R4 | recombination of known primitives (transfer) |
| `ood_transform` | held-out | R3–R5 | novel output shapes (transfer) |

`train` = 24 templates (4 per rung R0–R5). OOD families are **never** drawn by a
train spec (enforced by `_template_pool` + tests) — the MicroCode analogue of
Countdown's `test_ood_*`.

## Difficulty ladder

6 rungs R0–R5 binned to 3 tiers (`RUNG_TIER`). Smoothness lives in the **test
set**, not the label: per-held-out-test fractional credit (`held_out_pass_rate`)
means a rung where the model nails the happy path but misses edge cases still
yields graded reward and non-zero within-group variance — no rung is a 0/1 cliff.

## Reward channels (`to_components`)

- **`correct` = `exact_correct` = `held_out_all_pass`** — PRIMARY (binary; all
  held-out tests pass; the verifier is the sole source of truth).
- **`held_out_pass_rate`** — the dense fractional truth signal. DIAGNOSTIC ONLY:
  it is the primary via `correct` and must **never** be a weighted training
  wheel (the model would optimize the truth channel directly).
- **`visible_pass_rate`** — the HACKABLE PROXY (hardcoding visible I/O drives it
  to 1.0 while held-out collapses). The channel the E5 TEMPTATION arm over-weights.
- Weighted training wheels (teacher aux set, `MICRO_AUX_KEYS`, finalized by the
  prune probe `scripts/31_microcode_aux_prune.py`): `valid_expression` (legality
  gate), `runs_without_error`, `visible_pass_rate`, `no_hardcoding_heuristic`.
- Pruned as dead/collinear scaffold (present in `to_components` for diagnostics
  but NOT weighted): `has_extractable_answer_span`, `format`, `syntax_parses`,
  `defines_target_signature`, `imports_safe` (all perfectly collinear with
  `valid_expression`), and `brevity` (always 1.0 under the 600-char budget).

## Teacher / curriculum wiring (Paper-2)

- Strategies restricted to `adaptive_stable` / `static` / `manual` / `random`.
  **`adaptive_phased` is NOT used** — it hardcodes Countdown component names
  (`number_multiset_f1` etc.) in its phase constraints.
- Curriculum `graded_key = "held_out_pass_rate"` (the fractional competence
  signal), `gate_key = "valid_expression"` — set via config, NOT by changing
  Countdown defaults. Using the binary `correct` as the graded key would
  re-inherit Countdown's bimodality (the `framework-bug` lesson).

## Intended use

- GRPO/RLOO post-training with the held-out verifier reward (E4 HONEST pilot).
- The E5 hacking experiment: HONEST vs TEMPTATION budget × static vs
  adaptive_stable — does the adaptive teacher spontaneously down-weight the
  gamed visible-test proxy?
- Transfer eval across the held-out `ood_*` families.

## Not intended use

- A general code-generation benchmark (it is a deliberately narrow micro-testbed).
- Any hacking-**RESISTANCE** headline until the sandbox-soundness question below
  is settled.

## Known limitations & sandbox risk (honest)

- **Sandbox (two epistemic statuses — do not blur):** A hardened out-of-process
  path exists (`microcode_sandbox.py`, S3/I7): a SPAWNED (never fork-after-CUDA)
  persistent worker with an `RLIMIT_AS` memory wall + `RLIMIT_FSIZE`, a private
  builtins namespace, and `except BaseException` guards. This provides **SOUND
  memory-DoS + parent-crash isolation** — an allocation bomb becomes a
  deterministic crash verdict in the child (verified: caught in ~0.04s by
  RLIMIT_AS, backstop untouched), and a child segfault/OOM-kill cannot take down
  the trainer. Select it with `verify_completion(..., sandbox="worker")`;
  `"inprocess"` (default) is the byte-identical path for the deterministic CPU
  test/repro suite. **Turn the worker ON for any GPU GRPO run (E4/E5)** — honest
  accidental resource exhaustion is intent-independent.
- **It is NOT a sound escape sandbox.** The AST import/name whitelist is
  defense-in-depth only, defeatable by a C-level gadget (`object.__subclasses__`
  → os); there is **zero sound containment for filesystem/network escape**.
  OS-level isolation (nsjail/firejail/gVisor/seccomp container) is deferred and
  documented, not built. No hacking-**RESISTANCE** headline may rest on sandbox
  security; the E5 claim is scoped to reward-channel behavior only.
- **Verdict determinism is bit-stable for the ALLOCATION class, not the
  compute-hang class.** The `sys.settrace` line budget (100k) does not bound
  non-allocating C-level loops (e.g. `sum(range(10**12))` = 1 trace tick); those
  fall to a wall-clock liveness backstop, whose firing is contention-dependent
  (a NON-deterministic verdict). Every backstop firing is counted
  (`SandboxWorker.backstop_firings`); a run with count>0 has a qualified
  bit-stability claim.
- Templates are simple list/dict/int transforms; not representative of
  real-world code complexity.
- The `no_hardcoding_heuristic` is a cheap AST smell detector, not a sound
  anti-cheat measure.
