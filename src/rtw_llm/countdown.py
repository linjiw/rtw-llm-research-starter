"""Countdown task generation, parsing, verification, and reward components."""
from __future__ import annotations

import ast
import operator
import random
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.IGNORECASE | re.DOTALL)
ANSWER_OPEN_RE = re.compile(r"<answer>", flags=re.IGNORECASE)
ANSWER_CLOSE_RE = re.compile(r"</answer>", flags=re.IGNORECASE)

OP_SYMBOLS: dict[type[ast.operator], str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
}

OP_FUNCS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}


@dataclass(frozen=True)
class VerificationResult:
    expression: str
    found_answer_tag: bool
    parse_ok: bool
    uses_all_numbers: bool
    uses_allowed_ops: bool
    numeric_ok: bool
    correct: bool
    value: str | None
    error: str | None = None
    contains_open_answer_tag: bool | None = None
    contains_close_answer_tag: bool | None = None
    numbers_used: tuple[int, ...] = ()
    number_precision: float = 0.0
    number_recall: float = 0.0
    number_multiset_f1: float = 0.0
    uses_no_extra_numbers: bool = False
    uses_all_required_numbers: bool = False
    operator_precision: float = 0.0
    operator_recall: float = 0.0
    evaluates_without_exception: bool = False
    numeric_distance_reward: float = 0.0

    def to_components(self, max_chars: int = 600) -> dict[str, float]:
        brevity = 1.0 if len(self.expression) <= max_chars else 0.0
        open_tag = (
            self.found_answer_tag
            if self.contains_open_answer_tag is None
            else self.contains_open_answer_tag
        )
        close_tag = (
            self.found_answer_tag
            if self.contains_close_answer_tag is None
            else self.contains_close_answer_tag
        )
        has_span = self.found_answer_tag
        format_score = (float(open_tag) + float(close_tag) + float(has_span)) / 3.0
        return {
            "contains_open_answer_tag": float(open_tag),
            "contains_close_answer_tag": float(close_tag),
            "has_extractable_answer_span": float(has_span),
            "format": format_score,
            "parse_ok": float(self.parse_ok),
            "expression_parseable": float(self.parse_ok),
            "uses_numbers": float(self.uses_all_numbers),
            "uses_allowed_numbers": float(self.uses_all_numbers),
            "number_precision": float(self.number_precision),
            "number_recall": float(self.number_recall),
            "number_multiset_f1": float(self.number_multiset_f1),
            "uses_no_extra_numbers": float(self.uses_no_extra_numbers),
            "uses_all_required_numbers": float(self.uses_all_required_numbers),
            "allowed_ops": float(self.uses_allowed_ops),
            "uses_allowed_ops": float(self.uses_allowed_ops),
            "operator_precision": float(self.operator_precision),
            "operator_recall": float(self.operator_recall),
            "evaluates_without_exception": float(self.evaluates_without_exception),
            "numeric_distance_reward": float(self.numeric_distance_reward),
            "valid_expression": float(
                self.parse_ok and self.uses_all_numbers and self.uses_allowed_ops and self.numeric_ok
            ),
            "brevity": brevity,
            "correct": float(self.correct),
            "exact_correct": float(self.correct),
        }


def extract_answer(text: str) -> tuple[str, bool]:
    """Extract text inside <answer>...</answer>, falling back to full text."""
    match = ANSWER_RE.search(text or "")
    if match:
        return match.group(1).strip(), True
    return (text or "").strip(), False


def answer_tag_signals(text: str) -> dict[str, bool]:
    """Return dense format signals for the required answer tag contract."""
    raw = text or ""
    return {
        "contains_open_answer_tag": bool(ANSWER_OPEN_RE.search(raw)),
        "contains_close_answer_tag": bool(ANSWER_CLOSE_RE.search(raw)),
        "has_extractable_answer_span": bool(ANSWER_RE.search(raw)),
    }


def _collect_numbers(node: ast.AST) -> list[int]:
    nums: list[int] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant):
            if not isinstance(child.value, int) or isinstance(child.value, bool):
                raise ValueError(f"Only integer constants are allowed; got {child.value!r}")
            nums.append(int(child.value))
    return nums


def _eval_ast(node: ast.AST, allowed_ops: set[str]) -> Fraction:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body, allowed_ops)

    if isinstance(node, ast.Constant):
        if not isinstance(node.value, int) or isinstance(node.value, bool):
            raise ValueError(f"Only integer constants are allowed; got {node.value!r}")
        return Fraction(int(node.value), 1)

    if isinstance(node, ast.BinOp):
        op_symbol = OP_SYMBOLS.get(type(node.op))
        if op_symbol is None:
            raise ValueError(f"Operator {type(node.op).__name__} is not allowed")
        if op_symbol not in allowed_ops:
            raise ValueError(f"Operator {op_symbol!r} is not in allowed_ops={sorted(allowed_ops)}")
        left = _eval_ast(node.left, allowed_ops)
        right = _eval_ast(node.right, allowed_ops)
        if op_symbol == "/":
            if right == 0:
                raise ZeroDivisionError("division by zero")
            return left / right
        if op_symbol == "+":
            return left + right
        if op_symbol == "-":
            return left - right
        if op_symbol == "*":
            return left * right

    # Disallow unary operators so models cannot smuggle negative constants as numbers.
    raise ValueError(f"Unsupported syntax: {ast.dump(node, include_attributes=False)}")


def _ops_in_ast(node: ast.AST) -> set[str]:
    ops: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.BinOp):
            symbol = OP_SYMBOLS.get(type(child.op))
            if symbol is None:
                ops.add(type(child.op).__name__)
            else:
                ops.add(symbol)
    return ops


def _number_match_scores(used: list[int], required: list[int]) -> dict[str, float | bool]:
    """Dense multiset diagnostics without changing the binary verifier contract."""
    used_counter = Counter(used)
    required_counter = Counter(required)
    matched = sum((used_counter & required_counter).values())
    precision = matched / max(1, sum(used_counter.values()))
    recall = matched / max(1, sum(required_counter.values()))
    f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
    return {
        "number_precision": float(precision),
        "number_recall": float(recall),
        "number_multiset_f1": float(f1),
        "uses_no_extra_numbers": not bool(used_counter - required_counter),
        "uses_all_required_numbers": not bool(required_counter - used_counter),
    }


def _operator_scores(
    ops: set[str],
    allowed_ops: set[str],
    parse_ok: bool,
    required_number_count: int,
) -> dict[str, float]:
    """Dense operator diagnostics. Countdown requires only allowed ops, not all ops."""
    if not parse_ok:
        return {"operator_precision": 0.0, "operator_recall": 0.0}
    if not ops:
        score = 1.0 if required_number_count <= 1 else 0.0
        return {"operator_precision": score, "operator_recall": score}
    allowed_used = sum(1 for op in ops if op in allowed_ops)
    fraction = allowed_used / max(1, len(ops))
    return {"operator_precision": float(fraction), "operator_recall": float(fraction)}


def _numeric_distance_reward(
    value: Fraction | None,
    target: int,
    number_multiset_f1: float,
) -> float:
    """Dense target-distance reward gated by number legality progress."""
    if value is None:
        return 0.0
    distance = abs(value - Fraction(target, 1))
    distance_reward = float(Fraction(1, 1) / (Fraction(1, 1) + distance))
    return float(max(0.0, min(1.0, number_multiset_f1)) * distance_reward)


def verify_expression(
    expression: str,
    numbers: list[int],
    target: int,
    allowed_ops: list[str],
    found_answer_tag: bool = True,
    contains_open_answer_tag: bool | None = None,
    contains_close_answer_tag: bool | None = None,
) -> VerificationResult:
    """Verify a Countdown expression with exact rational arithmetic."""
    expr = (expression or "").strip()
    allowed_set = set(allowed_ops)

    if not expr:
        return VerificationResult(
            expr,
            found_answer_tag,
            False,
            False,
            False,
            False,
            False,
            None,
            "empty",
            contains_open_answer_tag,
            contains_close_answer_tag,
        )

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        return VerificationResult(
            expr,
            found_answer_tag,
            False,
            False,
            False,
            False,
            False,
            None,
            str(exc),
            contains_open_answer_tag,
            contains_close_answer_tag,
        )

    try:
        nums = _collect_numbers(tree)
        number_scores = _number_match_scores(nums, numbers)
        uses_all = Counter(nums) == Counter(numbers)
        ops = _ops_in_ast(tree)
        operator_scores = _operator_scores(
            ops,
            allowed_set,
            parse_ok=True,
            required_number_count=len(numbers),
        )
        uses_allowed = ops.issubset(allowed_set) and (bool(ops) or len(numbers) <= 1)
        value = _eval_ast(tree, allowed_set)
        numeric_ok = True
        correct = uses_all and uses_allowed and value == Fraction(target, 1)
        value_str = str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
        return VerificationResult(
            expr,
            found_answer_tag,
            True,
            uses_all,
            uses_allowed,
            numeric_ok,
            correct,
            value_str,
            None,
            contains_open_answer_tag,
            contains_close_answer_tag,
            tuple(nums),
            float(number_scores["number_precision"]),
            float(number_scores["number_recall"]),
            float(number_scores["number_multiset_f1"]),
            bool(number_scores["uses_no_extra_numbers"]),
            bool(number_scores["uses_all_required_numbers"]),
            operator_scores["operator_precision"],
            operator_scores["operator_recall"],
            True,
            _numeric_distance_reward(
                value,
                target,
                float(number_scores["number_multiset_f1"]),
            ),
        )
    except Exception as exc:  # verifier must be robust to arbitrary model text
        nums: list[int] = []
        ops: set[str] = set()
        uses_all = False
        uses_allowed = False
        number_scores = _number_match_scores(nums, numbers)
        operator_scores = _operator_scores(
            ops,
            allowed_set,
            parse_ok=True,
            required_number_count=len(numbers),
        )
        try:
            nums = _collect_numbers(tree)
            number_scores = _number_match_scores(nums, numbers)
            uses_all = Counter(nums) == Counter(numbers)
            ops = _ops_in_ast(tree)
            operator_scores = _operator_scores(
                ops,
                allowed_set,
                parse_ok=True,
                required_number_count=len(numbers),
            )
            uses_allowed = ops.issubset(allowed_set) and (bool(ops) or len(numbers) <= 1)
        except Exception:
            pass
        return VerificationResult(
            expr,
            found_answer_tag,
            True,
            uses_all,
            uses_allowed,
            False,
            False,
            None,
            str(exc),
            contains_open_answer_tag,
            contains_close_answer_tag,
            tuple(nums),
            float(number_scores["number_precision"]),
            float(number_scores["number_recall"]),
            float(number_scores["number_multiset_f1"]),
            bool(number_scores["uses_no_extra_numbers"]) and bool(nums),
            bool(number_scores["uses_all_required_numbers"]),
            operator_scores["operator_precision"],
            operator_scores["operator_recall"],
            False,
            0.0,
        )


def verify_completion(completion: str, example: dict[str, Any]) -> VerificationResult:
    expr, has_tag = extract_answer(completion)
    signals = answer_tag_signals(completion)
    return verify_expression(
        expr,
        numbers=list(example["numbers"]),
        target=int(example["target"]),
        allowed_ops=list(example["allowed_ops"]),
        found_answer_tag=has_tag,
        contains_open_answer_tag=signals["contains_open_answer_tag"],
        contains_close_answer_tag=signals["contains_close_answer_tag"],
    )


def score_completion(
    completion: str,
    example: dict[str, Any],
    aux_weights: dict[str, float] | None = None,
    primary_weight: float = 1.0,
) -> tuple[float, dict[str, float], VerificationResult]:
    """Return total reward, components, and verification details."""
    result = verify_completion(completion, example)
    components = result.to_components()
    breakdown = reward_breakdown(components, aux_weights, primary_weight)
    return breakdown["total_reward"], components, result


def reward_breakdown(
    components: dict[str, float],
    aux_weights: dict[str, float] | None = None,
    primary_weight: float = 1.0,
) -> dict[str, float]:
    """Split reward into primary, weighted auxiliary, and total terms."""
    primary_reward = float(components.get("correct", 0.0))
    primary_reward_weighted = float(primary_weight) * primary_reward
    aux_reward_weighted = 0.0
    for key, weight in (aux_weights or {}).items():
        aux_reward_weighted += float(weight) * float(components.get(key, 0.0))
    return {
        "primary_reward": primary_reward,
        "primary_reward_weighted": primary_reward_weighted,
        "aux_reward_weighted": float(aux_reward_weighted),
        "total_reward": float(primary_reward_weighted + aux_reward_weighted),
    }


# ------------------------- Synthetic task generation -------------------------

@dataclass(frozen=True)
class ExprNode:
    value: Fraction
    expr: str
    numbers: tuple[int, ...]


def _combine(a: ExprNode, b: ExprNode, op: str) -> ExprNode | None:
    if op == "+":
        return ExprNode(a.value + b.value, f"({a.expr}+{b.expr})", a.numbers + b.numbers)
    if op == "-":
        return ExprNode(a.value - b.value, f"({a.expr}-{b.expr})", a.numbers + b.numbers)
    if op == "*":
        return ExprNode(a.value * b.value, f"({a.expr}*{b.expr})", a.numbers + b.numbers)
    if op == "/":
        if b.value == 0:
            return None
        return ExprNode(a.value / b.value, f"({a.expr}/{b.expr})", a.numbers + b.numbers)
    raise ValueError(op)


def random_solvable_task_legacy_v1(
    rng: random.Random,
    n_numbers: int,
    allowed_ops: list[str],
    min_number: int = 1,
    max_number: int = 20,
    min_target: int = 1,
    max_target: int = 999,
    max_attempts: int = 10_000,
) -> dict[str, Any]:
    """Replay the historical generator, including its leftover-node defect."""
    for _ in range(max_attempts):
        leaves = [rng.randint(min_number, max_number) for _ in range(n_numbers)]
        nodes = [ExprNode(Fraction(n, 1), str(n), (n,)) for n in leaves]
        rng.shuffle(nodes)
        while len(nodes) > 1:
            i = rng.randrange(len(nodes))
            a = nodes.pop(i)
            j = rng.randrange(len(nodes))
            b = nodes.pop(j)
            if rng.random() < 0.5:
                a, b = b, a
            op = rng.choice(allowed_ops)
            combined = _combine(a, b, op)
            if combined is None:
                break
            # Keep generated expressions numerically sane.
            if abs(combined.value) > max_target * 5:
                break
            nodes.append(combined)
        if len(nodes) != 1:
            continue
        final = nodes[0]
        if final.value.denominator != 1:
            continue
        target = int(final.value)
        if not (min_target <= target <= max_target):
            continue
        numbers = list(final.numbers)
        rng.shuffle(numbers)
        # Validate after shuffling numbers.
        vr = verify_expression(final.expr, numbers, target, allowed_ops)
        if vr.correct:
            return {
                "numbers": numbers,
                "target": target,
                "allowed_ops": allowed_ops,
                "solution": final.expr,
            }
    raise RuntimeError("Failed to generate a solvable task. Try relaxing bounds.")


def random_solvable_task(
    rng: random.Random,
    n_numbers: int,
    allowed_ops: list[str],
    min_number: int = 1,
    max_number: int = 20,
    min_target: int = 1,
    max_target: int = 999,
    max_attempts: int = 10_000,
) -> dict[str, Any]:
    """Generate a verifier-valid task using exactly the requested operand count."""
    for _ in range(max_attempts):
        leaves = [rng.randint(min_number, max_number) for _ in range(n_numbers)]
        nodes = [ExprNode(Fraction(n, 1), str(n), (n,)) for n in leaves]
        rng.shuffle(nodes)
        attempt_failed = False
        while len(nodes) > 1:
            i = rng.randrange(len(nodes))
            a = nodes.pop(i)
            j = rng.randrange(len(nodes))
            b = nodes.pop(j)
            if rng.random() < 0.5:
                a, b = b, a
            op = rng.choice(allowed_ops)
            combined = _combine(a, b, op)
            if combined is None or abs(combined.value) > max_target * 5:
                attempt_failed = True
                break
            nodes.append(combined)
        if attempt_failed or len(nodes) != 1:
            continue
        final = nodes[0]
        if len(final.numbers) != n_numbers:
            continue
        if final.value.denominator != 1:
            continue
        target = int(final.value)
        if not (min_target <= target <= max_target):
            continue
        numbers = list(final.numbers)
        rng.shuffle(numbers)
        vr = verify_expression(final.expr, numbers, target, allowed_ops)
        if vr.correct:
            return {
                "numbers": numbers,
                "target": target,
                "allowed_ops": allowed_ops,
                "solution": final.expr,
            }
    raise RuntimeError("Failed to generate a solvable task. Try relaxing bounds.")


def difficulty_spec(difficulty: str) -> dict[str, Any]:
    """Default generation parameters by difficulty."""
    if difficulty == "easy":
        return {"n_numbers": 3, "allowed_ops": ["+", "-"], "max_number": 12, "max_target": 60}
    if difficulty == "medium":
        return {"n_numbers": 4, "allowed_ops": ["+", "-", "*"], "max_number": 15, "max_target": 250}
    if difficulty == "hard":
        return {"n_numbers": 5, "allowed_ops": ["+", "-", "*"], "max_number": 20, "max_target": 700}
    if difficulty == "ood_long":
        return {"n_numbers": 6, "allowed_ops": ["+", "-", "*"], "max_number": 20, "max_target": 999}
    if difficulty == "ood_division":
        return {"n_numbers": 5, "allowed_ops": ["+", "-", "*", "/"], "max_number": 20, "max_target": 999}
    raise ValueError(f"Unknown difficulty: {difficulty}")
