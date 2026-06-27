"""Prompt templates for harness-shift experiments."""
from __future__ import annotations

from typing import Iterable


def ops_to_text(allowed_ops: Iterable[str]) -> str:
    return " ".join(allowed_ops)


def make_prompt(numbers: list[int], target: int, allowed_ops: list[str], level: str = "high") -> str:
    """Create a Countdown prompt at a requested harness informativeness level.

    Levels:
      - low: minimal task instruction.
      - mid: includes constraints and answer tags.
      - high: includes explicit validator-style checklist.
    """
    nums = ", ".join(str(n) for n in numbers)
    ops = ops_to_text(allowed_ops)

    if level == "low":
        return (
            f"Use the numbers [{nums}] to make {target}. "
            f"Allowed operations: {ops}. Give the expression."
        )

    if level == "mid":
        return (
            "Solve this arithmetic puzzle.\n"
            f"Numbers: [{nums}]\n"
            f"Target: {target}\n"
            f"Allowed operations: {ops}\n"
            "Use every number exactly once. Return only one expression inside "
            "<answer>...</answer>."
        )

    if level == "high":
        return (
            "You are solving a verifiable arithmetic task.\n\n"
            "Rules:\n"
            f"1. Use each number in this multiset exactly once: [{nums}]\n"
            f"2. Use only these operators: {ops}\n"
            "3. Parentheses are allowed. No new constants are allowed.\n"
            f"4. The expression must evaluate exactly to the target: {target}\n"
            "5. Put the final expression in <answer>...</answer>.\n\n"
            "Bad answer example: <answer>target</answer> because it invents a constant.\n"
            "Good answer format example: <answer>(1+2)*3</answer>\n\n"
            "Now solve the task."
        )

    raise ValueError(f"Unknown harness level: {level}")


def make_sft_completion(solution: str, target: int) -> str:
    return (
        "<reasoning>One valid expression is "
        f"{solution}. It evaluates to {target} while following the task constraints.</reasoning>\n"
        f"<answer>{solution}</answer>"
    )
