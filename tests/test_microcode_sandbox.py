"""S3/I7: tests for the hardened MicroCode execution sandbox.

Pre-registered guards from docs/S3_SANDBOX_HARDENING_PLAN.md (advisor-amended).
The worker path must (a) match the in-process verdicts exactly, (b) turn an
allocation bomb into a deterministic crash verdict without OOMing the parent,
(c) be deterministic across respawns, (d) use spawn (never fork), (e) never
import torch.
"""
import subprocess
import sys

import pytest

from rtw_llm.microcode import verify_completion

TASK = {
    "fn_name": "count_greater",
    "visible_tests": [(([1, 5, 3], 2), 2), (([], 0), 0)],
    "held_out_tests": [(([4, 4, 4], 4), 0), (([7, 7, 8, 2], 6), 3), (([10], 100), 0)],
}


def wrap(code: str) -> str:
    return f"<answer>\n{code}\n</answer>"


CORRECT = wrap("def count_greater(nums, threshold):\n    return sum(1 for x in nums if x > threshold)")
PARTIAL = wrap("def count_greater(nums, threshold):\n    return sum(1 for x in nums if x >= threshold)")
CRASH = wrap("def count_greater(nums, threshold):\n    return undefined_name")
HACK = wrap("def count_greater(nums, threshold):\n    if list(nums) == [1, 5, 3]:\n        return 2\n    return 0")


@pytest.mark.parametrize("comp", [CORRECT, PARTIAL, CRASH, HACK])
def test_worker_verdicts_match_inprocess(comp):
    a = verify_completion(comp, TASK, sandbox="inprocess").to_components()
    b = verify_completion(comp, TASK, sandbox="worker").to_components()
    for k in ["correct", "held_out_pass_rate", "visible_pass_rate", "runs_without_error"]:
        assert a[k] == b[k], (k, a[k], b[k])


def test_worker_memory_bomb_is_crash_not_parent_oom():
    # An allocation bomb inside the function body must yield a crash verdict
    # (RLIMIT_AS -> MemoryError in the CHILD) and must NOT take down this test
    # process. static_legality allows this (no forbidden import/name).
    bomb = wrap("def count_greater(nums, threshold):\n    x = [0] * (10 ** 12)\n    return len(x)")
    r = verify_completion(bomb, TASK, sandbox="worker").to_components()
    assert r["correct"] == 0.0
    assert r["held_out_pass_rate"] == 0.0
    # the test process is still alive and can grade a normal completion after
    ok = verify_completion(CORRECT, TASK, sandbox="worker").to_components()
    assert ok["correct"] == 1.0


def test_worker_is_deterministic_across_repeats_and_respawn():
    # Re-grade the same completions (and force a respawn via the bomb between)
    # -> identical verdicts (bit-stable for the allocation class).
    r1 = verify_completion(CORRECT, TASK, sandbox="worker").to_components()
    verify_completion(wrap("def count_greater(n, t):\n    return [0]*(10**12)"), TASK, sandbox="worker")
    r2 = verify_completion(CORRECT, TASK, sandbox="worker").to_components()
    assert r1 == r2


def test_worker_uses_spawn_not_fork():
    from rtw_llm.microcode_sandbox import SandboxWorker
    w = SandboxWorker()
    try:
        # the context must be the spawn context (CUDA-safety invariant)
        assert w._ctx.get_start_method() == "spawn"
    finally:
        w.close()


def test_sandbox_module_does_not_import_torch():
    # A spawned child re-imports the worker module; it must not drag in torch/
    # transformers/trl (would re-init CUDA in the child, waste VRAM, risk hang).
    code = (
        "import sys; import rtw_llm.microcode_sandbox; "
        "bad=[m for m in ('torch','transformers','trl') if m in sys.modules]; "
        "print(bad); assert not bad, bad"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
