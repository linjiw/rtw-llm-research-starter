#!/usr/bin/env python
"""S2/I10-a: generate + COMMIT the MicroCode dataset (Paper-2).

Analogue of scripts/00 for Countdown. Materializes data/microcode/{train,
validation,test_in_dist}.jsonl + test_ood_compose/test_ood_transform.jsonl so
the frozen Paper-2 task IDs (I10-c) resolve against a committed artifact.

Determinism: references return only JSON-serializable ints/bools/None/lists/
dict-of-lists, so records round-trip and re-verify bit-identically (guarded by
tests). Each record carries the full verifier + harness schema.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rtw_llm.data import write_jsonl  # noqa: E402
from rtw_llm.microcode_gen import difficulty_spec, random_solvable_task  # noqa: E402


def _dedup_key(task: dict) -> tuple:
    # fn names are randomized, so dedup on template + the actual test content.
    return (task["template"], repr(task["visible_tests"]), repr(task["held_out_tests"]))


def build_records(count: int, split: str, tier: str, seed: int) -> list[dict]:
    rng = random.Random(seed)
    records, seen = [], set()
    i = 0
    guard = 0
    spec = difficulty_spec(tier)
    while len(records) < count and guard < count * 200 + 1000:
        guard += 1
        task = random_solvable_task(rng, spec, i, split)
        key = _dedup_key(task)
        i += 1
        if key in seen:
            continue
        seen.add(key)
        records.append(task)
    rng.shuffle(records)
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=Path("data/microcode"))
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--valid", type=int, default=200)
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--ood", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # train draws all train tiers (interleave easy/medium/hard by seeding each);
    # build separately per tier then merge for a balanced train mix.
    train = []
    for k, tier in enumerate(["easy", "medium", "hard"]):
        train += build_records(args.train // 3, "train", tier, args.seed + k)
    random.Random(args.seed).shuffle(train)
    valid = (build_records(args.valid // 3 + 1, "validation", "easy", args.seed + 10)
             + build_records(args.valid // 3 + 1, "validation", "medium", args.seed + 11)
             + build_records(args.valid // 3 + 1, "validation", "hard", args.seed + 12))
    test = (build_records(args.test // 3 + 1, "test_in_dist", "easy", args.seed + 20)
            + build_records(args.test // 3 + 1, "test_in_dist", "medium", args.seed + 21)
            + build_records(args.test // 3 + 1, "test_in_dist", "hard", args.seed + 22))
    ood_compose = build_records(args.ood, "test_ood_compose", "ood_compose", args.seed + 30)
    ood_transform = build_records(args.ood, "test_ood_transform", "ood_transform", args.seed + 31)

    splits = {
        "train": train, "validation": valid, "test_in_dist": test,
        "test_ood_compose": ood_compose, "test_ood_transform": ood_transform,
    }
    for name, recs in splits.items():
        write_jsonl(args.out_dir / f"{name}.jsonl", recs)
    print(f"Wrote MicroCode dataset to {args.out_dir.resolve()}")
    for name, recs in splits.items():
        print(f"  {name}: {len(recs)}")


if __name__ == "__main__":
    main()
