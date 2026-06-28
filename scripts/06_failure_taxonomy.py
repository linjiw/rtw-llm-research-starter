#!/usr/bin/env python
"""Summarize failure modes from eval generations.jsonl artifacts."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def classify(metrics: dict[str, Any], completion: str) -> str:
    """Return a single primary failure class for a generated completion."""
    if float(metrics.get("exact_correct", 0.0)) > 0.0:
        return "exact_correct"
    if float(metrics.get("has_extractable_answer_span", 0.0)) <= 0.0:
        return "no_answer_span"
    if float(metrics.get("parse_ok", 0.0)) <= 0.0:
        return "parse_failure"
    if float(metrics.get("uses_no_extra_numbers", 0.0)) <= 0.0:
        return "illegal_extra_or_repeated_number"
    if float(metrics.get("uses_all_required_numbers", 0.0)) <= 0.0:
        return "missing_required_number"
    if float(metrics.get("uses_allowed_ops", 0.0)) <= 0.0:
        return "illegal_operator"
    if float(metrics.get("evaluates_without_exception", 0.0)) <= 0.0:
        return "evaluation_error"
    if float(metrics.get("valid_expression", 0.0)) > 0.0:
        return "legal_but_wrong_value"
    if "</answer>" in completion and completion.count("</answer>") > 1:
        return "repeated_answer_tag"
    return "other_invalid"


def summarize(path: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    metric_sums: defaultdict[str, float] = defaultdict(float)
    n = 0
    examples: dict[str, dict[str, Any]] = {}

    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            n += 1
            metrics = rec.get("metrics", {})
            completion = rec.get("completion", "") or ""
            label = classify(metrics, completion)
            counts[label] += 1
            examples.setdefault(
                label,
                {
                    "id": rec.get("id"),
                    "numbers": rec.get("numbers"),
                    "target": rec.get("target"),
                    "completion": completion[:240],
                    "expression": metrics.get("expression"),
                    "value": metrics.get("value"),
                    "error": metrics.get("error"),
                },
            )
            for key in [
                "valid_expression",
                "exact_correct",
                "reward_hacking_candidate",
                "uses_allowed_numbers",
                "number_multiset_f1",
                "uses_allowed_ops",
                "parse_ok",
            ]:
                metric_sums[key] += float(metrics.get(key, 0.0))

    return {
        "path": str(path),
        "n": n,
        "failure_counts": dict(counts),
        "failure_rates": {k: v / n for k, v in sorted(counts.items())} if n else {},
        "metric_means": {k: v / n for k, v in sorted(metric_sums.items())} if n else {},
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="generations.jsonl files to summarize")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    results = [summarize(Path(path)) for path in args.paths]
    payload: dict[str, Any] = {"results": results}
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")


if __name__ == "__main__":
    main()
