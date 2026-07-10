#!/usr/bin/env python
"""Replay the historical Countdown-v1 dataset generator.

New datasets must use scripts/18_generate_countdown_v2.py. This script retains
the legacy leftover-node behavior solely for byte/order reproducibility.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rtw_llm.countdown import (  # noqa: E402
    difficulty_spec,
    random_solvable_task_legacy_v1,
)
from rtw_llm.data import write_jsonl  # noqa: E402
from rtw_llm.prompts import make_sft_completion  # noqa: E402


def assert_safe_legacy_replay_output(out_dir: Path, repo_root: Path = ROOT) -> Path:
    output = out_dir if out_dir.is_absolute() else repo_root / out_dir
    output = output.resolve()
    frozen = (repo_root / "data/countdown").resolve()
    if output == frozen or frozen in output.parents:
        raise ValueError(f"Refusing to overwrite frozen legacy evidence under {frozen}")
    return output


def make_legacy_prompt(
    numbers: list[int], target: int, allowed_ops: list[str], level: str
) -> str:
    """Frozen prompt templates used by the committed legacy-v1 JSONL bytes."""
    nums = ", ".join(str(number) for number in numbers)
    ops = " ".join(allowed_ops)
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


def build_records(count: int, split: str, difficulties: list[str], seed: int) -> list[dict]:
    rng = random.Random(seed)
    records = []
    seen = set()
    i = 0
    while len(records) < count:
        difficulty = difficulties[len(records) % len(difficulties)]
        spec = difficulty_spec(difficulty)
        task = random_solvable_task_legacy_v1(rng, **spec)
        key = (tuple(sorted(task["numbers"])), task["target"], tuple(task["allowed_ops"]))
        if key in seen:
            continue
        seen.add(key)
        rec_id = f"{split}_{difficulty}_{i:06d}"
        rec = {
            "id": rec_id,
            "split": split,
            "difficulty": difficulty,
            "numbers": task["numbers"],
            "target": task["target"],
            "allowed_ops": task["allowed_ops"],
            "solution": task["solution"],
            "prompt_low": make_legacy_prompt(
                task["numbers"], task["target"], task["allowed_ops"], level="low"
            ),
            "prompt_mid": make_legacy_prompt(
                task["numbers"], task["target"], task["allowed_ops"], level="mid"
            ),
            "prompt_high": make_legacy_prompt(
                task["numbers"], task["target"], task["allowed_ops"], level="high"
            ),
            "prompt": make_legacy_prompt(
                task["numbers"], task["target"], task["allowed_ops"], level="high"
            ),
            "completion": make_sft_completion(task["solution"], task["target"]),
            "metadata": {
                "n_numbers": len(task["numbers"]),
                "harness_levels": ["low", "mid", "high"],
                "source": "synthetic_by_construction",
            },
        }
        records.append(rec)
        i += 1
    rng.shuffle(records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("outputs/legacy_replay/countdown"),
    )
    parser.add_argument("--train", type=int, default=5000)
    parser.add_argument("--valid", type=int, default=500)
    parser.add_argument("--test", type=int, default=500)
    parser.add_argument("--ood", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir = assert_safe_legacy_replay_output(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train = build_records(args.train, "train", ["easy", "medium", "hard"], args.seed)
    valid = build_records(args.valid, "validation", ["easy", "medium", "hard"], args.seed + 1)
    test = build_records(args.test, "test_in_dist", ["easy", "medium", "hard"], args.seed + 2)
    ood_long = build_records(args.ood, "test_ood_long", ["ood_long"], args.seed + 3)
    ood_division = build_records(args.ood, "test_ood_division", ["ood_division"], args.seed + 4)

    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "validation.jsonl", valid)
    write_jsonl(args.out_dir / "test_in_dist.jsonl", test)
    write_jsonl(args.out_dir / "test_ood_long.jsonl", ood_long)
    write_jsonl(args.out_dir / "test_ood_division.jsonl", ood_division)

    print(f"Wrote dataset to {args.out_dir.resolve()}")
    for name, records in [
        ("train", train),
        ("validation", valid),
        ("test_in_dist", test),
        ("test_ood_long", ood_long),
        ("test_ood_division", ood_division),
    ]:
        print(f"  {name}: {len(records)}")


if __name__ == "__main__":
    main()
