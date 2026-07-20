"""S3/I7: hardened out-of-process execution sandbox for MicroCode.

Executes UNTRUSTED model-generated functions in a persistent SPAWNED worker with
a memory wall, so an accidental (or adversarial) allocation bomb / crash cannot
take down the trainer. Default-OFF: microcode.verify_completion(sandbox=...)
selects "inprocess" (byte-identical prototype) vs "worker" (this module).

Design + advisor amendments: docs/S3_SANDBOX_HARDENING_PLAN.md. Key invariants:
- SPAWN, never fork (fork-after-CUDA corrupts the CUDA context; the default mp
  method on Linux is fork, so the context is pinned explicitly). This module
  must NEVER import torch/transformers/trl (a spawned child re-imports it).
- The pass/fail VERDICT stays the deterministic sys.settrace instruction budget
  (bit-stable for the allocation class). RLIMIT_AS adds a deterministic memory
  wall (per-process VA cap → MemoryError, independent of contention). Non-
  allocating C-level compute bombs escape both and fall to a wall-clock LIVENESS
  backstop, which is a NON-deterministic verdict — logged, never silent.
- Per-case isolation: a private builtins copy in the exec namespace (never the
  shared module); the worker forbids imports (templates are pure), closing the
  sys.modules baseline-drift + poisoning vectors; guard is except BaseException
  so SystemExit/MemoryError always yield a verdict, never unwind.
- Raw Pipe (not Queue: a SIGKILL'd Queue writer leaks a semaphore).

Usage note (spawn requirement): because the worker is SPAWNED, the calling
entrypoint must be importable — i.e. guarded by `if __name__ == "__main__":`
(all scripts here are; pytest modules are). Driving it from `python -c '...'`
prints a harmless re-import traceback from the child and is not a supported
entrypoint.

SOUNDNESS (honest): this gives SOUND memory-DoS + parent-crash isolation. It
provides NOTHING sound for escape / filesystem / network — the AST whitelist is
defense-in-depth only, defeatable by a C-level gadget. No hacking-RESISTANCE
headline rests on sandbox security (see the dataset card + E5 pre-registration).
"""
from __future__ import annotations

import builtins as _builtins
import multiprocessing as mp
import os
import resource
import sys
from typing import Any

# Memory address-space cap for a child (measured: baseline VmSize ~241 MiB,
# reference peak alloc ~0.03 MiB → 640 MiB is far above any legitimate reference
# yet turns an allocation bomb into a deterministic MemoryError well before it
# can pressure the shared box). Recorded in the ledger.
_RLIMIT_AS_BYTES = 640 * 1024 * 1024
_RLIMIT_NPROC = 0          # no child processes (fork-bomb guard); 0 = no new procs
_RLIMIT_FSIZE = 0          # no file writes
_RECV_TIMEOUT_S = 10.0     # LIVENESS backstop only (kills a hung child); NOT the verdict
_INSTRUCTION_BUDGET = 100_000


class _BudgetExceeded(Exception):
    pass


def _safe_builtins() -> dict:
    """A private builtins mapping without import/exec/eval/open/exit. Injected
    per-exec so a candidate cannot poison the shared builtins module."""
    forbidden = {
        "__import__", "eval", "exec", "compile", "open", "input",
        "exit", "quit", "globals", "vars", "memoryview",
    }
    d = {k: getattr(_builtins, k) for k in dir(_builtins) if k not in forbidden}
    return d


def _run_one_isolated(fn_src: str, fn_name: str, args: list, expected: Any) -> tuple[bool, bool]:
    """Same deterministic verdict logic as microcode._run_one, but with a
    private builtins namespace and BaseException guard. Runs inside the worker."""
    ns: dict[str, Any] = {"__builtins__": _safe_builtins()}
    try:
        exec(compile(fn_src, "<microcode>", "exec"), ns)  # noqa: S102 - sandboxed worker
    except BaseException:
        return (False, False)
    fn = ns.get(fn_name)
    if not callable(fn):
        return (False, False)

    count = [0]

    def tracer(frame, event, arg):
        if event == "line":
            count[0] += 1
            if count[0] > _INSTRUCTION_BUDGET:
                raise _BudgetExceeded
        return tracer

    sys.settrace(tracer)
    try:
        out = fn(*args)
        ran = True
    except _BudgetExceeded:
        return (False, False)
    except BaseException:
        return (False, False)
    finally:
        sys.settrace(None)
    return (ran, out == expected)


def _worker_main(conn) -> None:
    """Child entrypoint (spawned). Sets resource caps ONCE, then serves
    (fn_src, fn_name, args, expected) -> (ran, correct) until the pipe closes."""
    # Freeze baseline determinism knobs before setting the AS cap.
    os.environ["MALLOC_ARENA_MAX"] = "1"
    try:
        resource.setrlimit(resource.RLIMIT_AS, (_RLIMIT_AS_BYTES, _RLIMIT_AS_BYTES))
    except (ValueError, OSError):
        pass
    for lim, val in ((resource.RLIMIT_FSIZE, _RLIMIT_FSIZE),):
        try:
            resource.setrlimit(lim, (val, val))
        except (ValueError, OSError):
            pass
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg is None:
            return
        fn_src, fn_name, args, expected = msg
        try:
            result = _run_one_isolated(fn_src, fn_name, tuple(args), expected)
        except BaseException:
            result = (False, False)
        try:
            conn.send(result)
        except BaseException:
            return


class SandboxWorker:
    """Persistent spawned worker. run_one() returns (ran, correct); a dead child
    (OOM-kill / segfault / hang past the liveness backstop) yields a crash
    verdict (False, False) and respawns."""

    def __init__(self) -> None:
        self._ctx = mp.get_context("spawn")   # NEVER fork (CUDA-safety)
        self.proc = None
        self.conn = None
        self.backstop_firings = 0
        self._spawn()

    def _spawn(self) -> None:
        parent, child = self._ctx.Pipe()      # raw duplex Pipe (not Queue)
        proc = self._ctx.Process(target=_worker_main, args=(child,), daemon=True)
        proc.start()
        child.close()                          # parent owns only its end
        self.proc, self.conn = proc, parent

    def run_one(self, fn_src: str, fn_name: str, args, expected) -> tuple[bool, bool]:
        try:
            self.conn.send((fn_src, fn_name, list(args), expected))
        except BaseException:
            self._respawn()
            return (False, False)
        if not self.conn.poll(_RECV_TIMEOUT_S):
            # liveness backstop (should be unreachable via the instruction
            # budget; reachable by non-allocating C-level compute). NON-
            # deterministic verdict → logged, then kill+respawn.
            self.backstop_firings += 1
            self._respawn()
            return (False, False)
        try:
            return self.conn.recv()
        except (EOFError, OSError):
            self._respawn()               # child died (OOM-kill / segfault)
            return (False, False)

    def _respawn(self) -> None:
        try:
            if self.proc and self.proc.is_alive():
                self.proc.kill()
                self.proc.join(timeout=2.0)
        except BaseException:
            pass
        self._spawn()

    def close(self) -> None:
        try:
            self.conn.send(None)
        except BaseException:
            pass
        try:
            self.proc.join(timeout=2.0)
            if self.proc.is_alive():
                self.proc.kill()
        except BaseException:
            pass


# Module-level lazily-created singleton worker (one per process).
_WORKER: SandboxWorker | None = None


def run_one_worker(fn_src: str, fn_name: str, args, expected) -> tuple[bool, bool]:
    global _WORKER
    if _WORKER is None:
        _WORKER = SandboxWorker()
    return _WORKER.run_one(fn_src, fn_name, args, expected)
