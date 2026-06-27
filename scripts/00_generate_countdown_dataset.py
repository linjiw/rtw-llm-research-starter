#!/usr/bin/env python
"""Generate a synthetic Countdown-style dataset for RTW-LLM."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rtw_llm.countdown import difficulty_spec, random_solvable_task  # noqa: E402
from rtw_llm.data import write_jsonl  # noqa: E402
from rtw_llm.prompts import make_prompt, make_sft_completion  # noqa: E402


def build_records(count: int, split: str, difficulties: list[str], seed: int) -> list[dict]:
    rng = random.Random(seed)
    records = []
    seen = set()
    i = 0
    while len(records) < count:
        difficulty = difficulties[len(records) % len(difficulties)]
        spec = difficulty_spec(difficulty)
        task = random_solvable_task(rng, **spec)
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
            "prompt_low": make_prompt(task["numbers"], task["target"], task["allowed_ops"], level="low"),
            "prompt_mid": make_prompt(task["numbers"], task["target"], task["allowed_ops"], level="mid"),
            "prompt_high": make_prompt(task["numbers"], task["target"], task["allowed_ops"], level="high"),
            "prompt": make_prompt(task["numbers"], task["target"], task["allowed_ops"], level="high"),
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
    parser.add_argument("--out_dir", type=Path, default=Path("data/countdown"))
    parser.add_argument("--train", type=int, default=5000)
    parser.add_argument("--valid", type=int, default=500)
    parser.add_argument("--test", type=int, default=500)
    parser.add_argument("--ood", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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
