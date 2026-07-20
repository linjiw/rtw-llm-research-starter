# S3 / I7 — MicroCode execution-sandbox hardening (design)

Pre-registration (design stage). Created 2026-07-20. Program loop:
DESIGN → **ADVISE** → IMPLEMENT → VALIDATE → RECORD. No sandbox code lands until
an adversarial review of this design is folded in. Grounding:
`PAPER2_MICROCODE_TESTBED_SPEC.md` step 2 + open questions, `DATASET_CARD_MICROCODE.md`
sandbox-risk section, `PAPER2_FROZEN_PROTOCOL.md` (S3 is the HARD gate before E5).

## Why (and precisely when)

MicroCode executes model-generated Python to grade it. Today `_run_one`
(`microcode.py:110`) `exec()`s the extracted function **in the trainer process**
under a `sys.settrace` instruction budget. That budget makes the pass/fail
verdict **deterministic** (bit-stable under CPU contention — the design's core
requirement) and bounds infinite loops, but it does NOT bound:
- **memory** — one line `x = [0]*10**10` executes few traced lines but exhausts RAM;
- **process integrity** — a segfault via a C-extension gadget takes down the trainer;
- **escape** — the AST whitelist is defense-in-depth, not sound
  (`().__class__.__bases__[0].__subclasses__()` → os), so truly hostile code can
  reach the filesystem/network from inside the trainer process.

**Scope of the gate (important):** E4 (HONEST budget) has **no** incentive for
the model to hack/exploit, so the in-process prototype is acceptable there (the
E2 base probe already ran model code in-process safely). Hardening is the
pre-registered gate before **E5**, where the TEMPTATION budget actively rewards
gaming. So S3 must be READY before E5 but is **default-off** so E4 is unchanged.

## ADVISOR AMENDMENTS (2026-07-20 — empirically verified on-box; folded below)

The adversarial review verified mechanics on this host and found flaws that
change the design. Must-fix, ranked:

1. **Persistent-worker state leak (FLAW).** A shared interpreter lets candidate
   N poison candidate N+1 via the **real shared `builtins`** (`ns['__builtins__']
   is builtins` — verified) and via **`sys.modules`** (an honest
   `import math; math.sqrt=...` changes a later candidate's verdict — non-
   determinism AND contamination). Also `exit()`/`SystemExit` is NOT caught by
   `except Exception` (verified) and in the in-process path would **kill the
   trainer**. FIX: per-case reset — restore a post-warmup `sys.modules`
   snapshot (or forbid ALL imports in the worker; templates are pure); inject a
   **private copy** of builtins into `ns`, never the shared module; add
   `__builtins__`, `exit`, `quit` to the forbidden names; broaden the guard to
   `except BaseException` so a verdict is ALWAYS produced and never unwinds into
   the trainer.
2. **RLIMIT_AS determinism is conditional (IMPROVE).** It IS the right tool
   (per-process VA cap → deterministic `MemoryError`, independent of contention/
   OOM-killer/other tenants) — but only if the persistent worker's baseline
   `VmSize` is FROZEN: eager-import the whole allowed-module whitelist + a
   warm-up alloc, set `MALLOC_ARENA_MAX=1`, single-threaded worker, call
   `setrlimit` ONCE after warmup. Cap = `frozen_baseline + measured_max_reference_peak
   + margin` (MEASURED across the template library, recorded in the ledger — not
   guessed). Also set `RLIMIT_NPROC` (fork-bomb) + `RLIMIT_FSIZE`. Use a raw
   `Pipe`, NOT `multiprocessing.Queue` (SIGKILL mid-write leaks a semaphore and
   can wedge the parent). NOTE: `MemoryError` is caught IN the worker (worker
   survives, reports (False,False)); only SIGKILL/segfault closes the pipe.
3. **Compute-hang class is NOT bit-stable, and the line budget misses it
   (verified: `sum(range(2_000_000))` = 1 trace tick).** A non-allocating
   C-level loop escapes BOTH the settrace budget AND RLIMIT_AS; only the
   wall-clock backstop stops it, and its firing point is contention-dependent →
   non-deterministic verdict. FIX the CLAIM, not just the code: state that the
   verdict is bit-stable for the **allocation class**, NOT the compute-hang
   class; LOG every backstop firing; if count>0 in an E5 run the bit-stability
   claim for that run is qualified. (This is a pre-existing property of the
   line-budget, surfaced honestly.)
4. **Default mp start method here is `fork` (verified).** Pin
   `mp.get_context("spawn")` at the call site; never `set_start_method`
   globally; forbid forkserver. Spawn-AFTER-CUDA is safe (fresh interpreter);
   only fork-after-CUDA corrupts the context. The worker module must NEVER
   transitively import torch/transformers/trl — add a test that imports it in a
   subprocess and asserts `torch not in sys.modules`. Own the worker + pipes
   only in the main process (close-on-exec) so a DataLoader fork can't inherit
   the pipe FDs.
5. **Gate logic (FLAW): E4 needs the worker too.** "Malice ⇒ worker" conflates
   intent with accident. Thousands of stochastic 0.5B completions will include
   honest-but-pathological code (in-body `[0]*10**9`, default-arg bombs that
   allocate BEFORE settrace is active, C-level compute, `exit()`). FIX: the gate
   is **"model code executes during a GPU GRPO run ⇒ worker ON"** — turn the
   worker on for E4's GPU run too. Keep `"inprocess"` ONLY for the deterministic
   CPU test/repro suite (byte-identical reproduction of the CPU variance gates),
   never for a live training run.
6. **Drop "no-network" as a claim (honesty).** The whitelist blocking `socket`/
   `os` is defeated by the same gadget already conceded unsound → network
   containment is ZERO, not "best-effort". State the honest split: the sandbox
   provides SOUND memory-DoS + parent-crash isolation (RLIMIT_AS + spawn +
   separate address space); it provides NOTHING sound for escape/fs/network
   (defense-in-depth only). Two different epistemic statuses — do not blur.

## Design (default-off; amendments above are authoritative where they conflict)

A **persistent spawned worker** process that executes test cases out-of-process:

1. **Spawn, never fork** — `multiprocessing.get_context("spawn")`. Critical: the
   trainer initializes CUDA; a `fork()` after CUDA init corrupts the CUDA
   context in the child. Spawn starts a fresh interpreter. (This is why the
   spec says "spawned … never fork-after-CUDA".)
2. **Persistent, not per-call** — spawn ONE worker, feed it
   `(fn_src, fn_name, args, expected)` over a pipe, read back `(ran, correct)`.
   Per-testcase spawn (~30–50 ms) × 7 tests × N candidates × 300 steps is
   prohibitive; a persistent worker amortizes it.
3. **Resource caps set once in the child at startup:**
   - `resource.setrlimit(RLIMIT_AS, (cap, cap))` — address-space cap turns an
     allocation bomb into a deterministic `MemoryError` (same input → same
     error regardless of contention → still bit-stable). This is the main gap
     the instruction budget cannot cover.
   - NO `RLIMIT_CPU` for the verdict — CPU-time is wall-clock-ish and
     non-deterministic under trainer contention; the **instruction budget stays
     the authoritative liveness/verdict mechanism** inside the child.
   - Best-effort `no-network`: not sound in-process; documented as a residual,
     nsjail/firejail deferred (stated honestly, not claimed).
4. **Verdict determinism unchanged** — the child runs the SAME `_run_one`
   logic (settrace instruction budget); the subprocess only adds a memory wall
   and process isolation. A child that dies (OOM-kill, segfault, RLIMIT) → the
   manager reads a closed pipe → returns `(False, False)` = crash verdict
   (deterministic for `MemoryError`; a segfault verdict is also stable for a
   given input) → **respawns** the worker for the next case.
5. **Wall-clock is a LIVENESS BACKSTOP only, never the verdict** — a
   `pipe.recv()` timeout kills+respawns a genuinely hung child (should be
   unreachable given the instruction budget) and is logged as an anomaly; it
   does NOT decide pass/fail (would break bit-stability). If it ever fires, that
   is a bug to investigate, not a normal path.

## Default-off wiring

- `verify_completion(..., sandbox: str = "inprocess")` — `"inprocess"` (default)
  = today's byte-identical path (all 146 tests + the E2 probe reproducibility
  unchanged); `"worker"` = the hardened path. Select via `RTWRewardManager`
  config for E5.
- The worker module is new (`src/rtw_llm/microcode_sandbox.py`); `microcode.py`
  imports it only when `sandbox="worker"`.

## Tests (VALIDATE)

- **Equivalence:** for a battery of correct/partial/hack/crash/infinite-loop
  completions, `"worker"` returns the SAME `(ran, correct)` verdicts as
  `"inprocess"` (the hardening must not change grading).
- **Memory wall:** an allocation-bomb completion (`return [0]*10**12`) →
  crash verdict under `"worker"` (deterministic), and does NOT OOM the test
  process (proving isolation).
- **Determinism:** the same completion re-graded N times under `"worker"` gives
  identical verdicts (bit-stable), incl. across a worker respawn.
- **Spawn-not-fork:** assert the worker uses the spawn context (guards the
  CUDA-safety invariant).
- **Escape smell (best-effort):** a completion that tries `open()`/filesystem is
  already blocked by static legality; add a defense-in-depth test that even if
  it executed, the worker's isolation contains it — documented as
  defense-in-depth, NOT a soundness claim.

## Honest residual risk (goes in the card + any E5 write-up)

The AST whitelist + spawn + rlimit-mem is **defense-in-depth for a single-host
research pilot, not a sound sandbox.** A determined escape via a C-level gadget
is not provably blocked without OS-level isolation (nsjail/firejail/gVisor/
container with seccomp). The E5 pre-registration must state this and scope the
"hacking-resistance" claim to the *reward-channel* behavior (does adaptive
down-weight the gamed proxy), NOT to sandbox security.

## Not in scope

nsjail/firejail/container isolation (OS-level, deferred — documented, not built);
network namespace isolation; any change to the deterministic instruction-budget
verdict; any change to the default in-process path (E4 unaffected).
