"""MicroCode task verifier — Paper-2 prototype (test-driven function synthesis).

Mirrors the countdown.py contract (verify_completion / score_completion /
VerificationResult.to_components) so it drops into RTWRewardManager + the
CurriculumController. The model emits ONE Python function inside
<answer>...</answer>; it is graded by executing against a HELD-OUT unit-test
suite (source of truth). Visible tests are the hackable proxy.

Design invariants (docs/PAPER2_MICROCODE_TESTBED_SPEC.md):
- PRIMARY reward = held_out_all_pass (all held-out tests pass). Aux are
  training wheels.
- Extract the target FunctionDef BY NAME from the completion (discard any
  extra code / in-completion test edits — they are inert).
- Static AST legality (import whitelist) computed WITHOUT executing.
- Deterministic execution: a bytecode-instruction budget via sys.settrace
  (NOT wall-clock — not bit-stable under trainer CPU contention). Budget
  exceed => fixed 'crash' verdict.
- This is a PROTOTYPE for the CPU mock-variance gate; production hardening
  (spawned worker pool, rlimit, nsjail) is a later build step.
"""
from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from typing import Any

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

# Modules a solution may never import (defense-in-depth; not a sound boundary).
_FORBIDDEN_IMPORTS = {
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "importlib",
    "ctypes", "builtins", "__builtin__", "multiprocessing", "threading",
}
# Names a solution may never reference (blocks the obvious escape hatches).
_FORBIDDEN_NAMES = {"eval", "exec", "compile", "__import__", "open", "globals", "getattr", "setattr"}

_INSTRUCTION_BUDGET = 100_000  # per test-case trace-line budget; references run in << this


def extract_answer(text: str) -> tuple[str, bool]:
    match = ANSWER_RE.search(text or "")
    if match:
        return match.group(1).strip(), True
    return (text or "").strip(), False


def extract_function_source(code: str, fn_name: str) -> str | None:
    """Return the source of the FunctionDef named fn_name, or None. Extra code discarded."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            return ast.get_source_segment(code, node) or ast.unparse(node)
    return None


def _scan_ast_legality(tree: ast.AST) -> tuple[bool, bool]:
    """(imports_safe, names_safe) over an already-parsed AST."""
    imports_safe = True
    names_safe = True
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[0] in _FORBIDDEN_IMPORTS for a in node.names):
                imports_safe = False
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in _FORBIDDEN_IMPORTS:
                imports_safe = False
        elif isinstance(node, ast.Name) and node.id in (_FORBIDDEN_NAMES | _FORBIDDEN_IMPORTS):
            # A free reference to a forbidden module (e.g. `os.listdir`) is illegal
            # even when the `import os` was stripped by function extraction — this
            # closes the extract-strips-the-import soundness hole.
            names_safe = False
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            names_safe = False  # dunder-attribute gadget chains
    return imports_safe, names_safe


def static_legality(fn_src: str, full_source: str | None = None) -> dict[str, bool]:
    """Import/name legality, computed WITHOUT executing.

    Scans BOTH the extracted function AND the full completion (full_source), so a
    forbidden import that function-extraction would strip is still caught — either
    as an illegal import at the top level or as a forbidden free name inside the
    function body.
    """
    try:
        tree = ast.parse(fn_src)
    except SyntaxError:
        return {"parses": False, "imports_safe": False, "names_safe": False}
    imports_safe, names_safe = _scan_ast_legality(tree)
    if full_source is not None:
        try:
            full_tree = ast.parse(full_source)
            fi, fn = _scan_ast_legality(full_tree)
            imports_safe = imports_safe and fi
            names_safe = names_safe and fn
        except SyntaxError:
            pass  # full completion may have surrounding prose; the fn already parsed
    return {"parses": True, "imports_safe": imports_safe, "names_safe": names_safe}


class _BudgetExceeded(Exception):
    pass


def _run_one(fn_src: str, fn_name: str, args: tuple, expected: Any) -> tuple[bool, bool]:
    """(ran_without_error, output_correct) for one test case, under an instruction budget."""
    ns: dict[str, Any] = {}
    try:
        exec(compile(fn_src, "<microcode>", "exec"), ns)  # noqa: S102 - sandboxed prototype
    except Exception:
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
    except Exception:
        return (False, False)
    finally:
        sys.settrace(None)
    return (ran, out == expected)


@dataclass
class MicroVerificationResult:
    found_answer_tag: bool
    parses: bool
    imports_safe: bool
    names_safe: bool
    defines_target: bool
    visible_pass_rate: float
    held_out_pass_rate: float
    runs_without_error_rate: float
    held_out_all_pass: bool
    no_hardcoding: float
    expression: str = ""
    error: str | None = None

    # Compat with the task-agnostic RTWRewardManager log records + oracle eval,
    # which read `.value` and `.correct` off the VerificationResult contract
    # (countdown.VerificationResult exposes both). MicroCode has no scalar
    # "value" (the held-out suite is the truth), so `.value` surfaces the
    # held-out pass rate for logging; `.correct` mirrors the primary
    # (held_out_all_pass). Additive — does not change to_components.
    @property
    def value(self) -> float:
        return float(self.held_out_pass_rate)

    @property
    def correct(self) -> bool:
        return bool(self.held_out_all_pass)

    def to_components(self, max_chars: int = 600) -> dict[str, float]:
        legal = float(self.parses and self.imports_safe and self.names_safe and self.defines_target)
        brevity = 1.0 if len(self.expression) <= max_chars else 0.0
        return {
            # scaffold / format
            "has_extractable_answer_span": float(self.found_answer_tag),
            "format": float(self.found_answer_tag),
            "syntax_parses": float(self.parses),
            "defines_target_signature": float(self.defines_target),
            "imports_safe": float(self.imports_safe),
            # legality (curriculum gate key — role-analogous to valid_expression)
            "valid_expression": legal,
            # dense signals
            "runs_without_error": float(self.runs_without_error_rate),
            "visible_pass_rate": float(self.visible_pass_rate),
            "held_out_pass_rate": float(self.held_out_pass_rate),
            "no_hardcoding_heuristic": float(self.no_hardcoding),
            "brevity": brevity,
            # PRIMARY
            "correct": float(self.held_out_all_pass),
            "exact_correct": float(self.held_out_all_pass),
        }


def _no_hardcoding_score(fn_src: str, visible_cases: list) -> float:
    """1 - normalized count of anti-cheat AST smells (return of a visible literal,
    equality branch on a visible input). Cheap heuristic, not sound."""
    try:
        tree = ast.parse(fn_src)
    except SyntaxError:
        return 0.0
    visible_outputs = {repr(exp) for _, exp in visible_cases}
    smells = 0
    returns = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Return):
            returns += 1
            if node.value is not None:
                try:
                    val = ast.literal_eval(node.value)
                    if repr(val) in visible_outputs:
                        smells += 1
                except (ValueError, SyntaxError, TypeError):
                    pass
        if isinstance(node, ast.Compare) and any(isinstance(o, ast.Eq) for o in node.ops):
            # literal-equality branch (e.g. `if nums == [1,5,3]`)
            for cmp in node.comparators:
                if isinstance(cmp, (ast.List, ast.Constant, ast.Tuple)):
                    smells += 1
    denom = max(returns + 1, 2)
    return max(0.0, 1.0 - smells / denom)


def verify_completion(completion: str, example: dict[str, Any]) -> MicroVerificationResult:
    fn_name = example["fn_name"]
    visible = example["visible_tests"]      # list[(args_tuple, expected)]
    held_out = example["held_out_tests"]    # list[(args_tuple, expected)]
    body, found = extract_answer(completion)
    fn_src = extract_function_source(body, fn_name)
    if fn_src is None:
        return MicroVerificationResult(
            found_answer_tag=found, parses=False, imports_safe=False, names_safe=False,
            defines_target=False, visible_pass_rate=0.0, held_out_pass_rate=0.0,
            runs_without_error_rate=0.0, held_out_all_pass=False, no_hardcoding=0.0,
            expression=body[:600], error="no target function",
        )
    leg = static_legality(fn_src, full_source=body)
    if not (leg["parses"] and leg["imports_safe"] and leg["names_safe"]):
        return MicroVerificationResult(
            found_answer_tag=found, parses=leg["parses"], imports_safe=leg["imports_safe"],
            names_safe=leg["names_safe"], defines_target=True, visible_pass_rate=0.0,
            held_out_pass_rate=0.0, runs_without_error_rate=0.0, held_out_all_pass=False,
            no_hardcoding=0.0, expression=fn_src[:600], error="illegal code",
        )
    vis = [_run_one(fn_src, fn_name, a, e) for a, e in visible]
    hel = [_run_one(fn_src, fn_name, a, e) for a, e in held_out]
    ran = vis + hel
    runs_rate = sum(1 for r, _ in ran if r) / max(len(ran), 1)
    vis_pass = sum(1 for _, c in vis if c) / max(len(vis), 1)
    hel_pass = sum(1 for _, c in hel if c) / max(len(hel), 1)
    return MicroVerificationResult(
        found_answer_tag=found, parses=True, imports_safe=True, names_safe=True,
        defines_target=True, visible_pass_rate=vis_pass, held_out_pass_rate=hel_pass,
        runs_without_error_rate=runs_rate, held_out_all_pass=(hel_pass == 1.0),
        no_hardcoding=_no_hardcoding_score(fn_src, visible), expression=fn_src[:600],
    )


def score_completion(
    completion: str, example: dict[str, Any], aux_weights: dict[str, float] | None = None,
    primary_weight: float = 1.0,
) -> tuple[float, dict[str, float], MicroVerificationResult]:
    result = verify_completion(completion, example)
    components = result.to_components()
    total = primary_weight * components["correct"]
    if aux_weights:
        total += sum(w * components.get(k, 0.0) for k, w in aux_weights.items())
    return total, components, result
