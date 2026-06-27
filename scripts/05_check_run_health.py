#!/usr/bin/env python
"""Summarize early GRPO run health from reward and teacher logs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from rtw_llm.analysis import load_jsonl_df


IGNORED_NONZERO_COMPONENTS = {"brevity"}


def summarize_reward_components(path: Path) -> tuple[dict, list[str]]:
    if not path.exists():
        return {"exists": False}, [f"Missing reward log: {path}"]

    df = load_jsonl_df(path)
    if df.empty:
        return {"exists": True, "n": 0}, [f"Reward log is empty: {path}"]

    components = pd.json_normalize(df["components"])
    component_means = components.mean(numeric_only=True).to_dict()
    component_stds = components.std(numeric_only=True).fillna(0.0).to_dict()
    component_nonzero = (components > 0).mean(numeric_only=True).to_dict()
    reward_std = float(df["reward"].std()) if len(df) > 1 else 0.0

    components_with_variance = sorted(
        name for name, value in component_stds.items() if float(value) > 1e-9
    )
    nonzero_non_brevity = sorted(
        name
        for name, value in component_nonzero.items()
        if name not in IGNORED_NONZERO_COMPONENTS and float(value) > 0
    )

    issues = []
    if reward_std == 0.0:
        issues.append("Total reward has zero variance.")
    if not nonzero_non_brevity:
        issues.append("No non-brevity reward component is nonzero yet.")
    if "format" not in components_with_variance:
        issues.append("Format reward has no observed variance yet.")

    return (
        {
            "exists": True,
            "n": int(len(df)),
            "reward_mean": float(df["reward"].mean()),
            "reward_std": reward_std,
            "component_means": {k: float(v) for k, v in component_means.items()},
            "component_stds": {k: float(v) for k, v in component_stds.items()},
            "component_nonzero_rates": {k: float(v) for k, v in component_nonzero.items()},
            "components_with_variance": components_with_variance,
            "nonzero_non_brevity_components": nonzero_non_brevity,
        },
        issues,
    )


def summarize_teacher_weights(path: Path) -> tuple[dict, list[str]]:
    if not path.exists():
        return {"exists": False}, [f"Missing teacher log: {path}"]

    df = load_jsonl_df(path)
    if df.empty:
        return {"exists": True, "n": 0}, [f"Teacher log is empty: {path}"]

    weights = pd.json_normalize(df["weights"])
    deltas = (weights.iloc[-1] - weights.iloc[0]).abs() if len(weights) > 1 else weights.iloc[0] * 0
    moving = sorted(name for name, value in deltas.items() if float(value) > 1e-9)
    min_weight = float(weights.min(numeric_only=True).min())
    max_weight = float(weights.max(numeric_only=True).max())

    issues = []
    if len(weights) > 1 and not moving:
        issues.append("Teacher weights did not change over multiple updates.")
    if min_weight < 0.0 or max_weight > 1.0:
        issues.append(f"Teacher weights outside expected [0, 1] range: {min_weight}, {max_weight}.")

    return (
        {
            "exists": True,
            "n": int(len(df)),
            "first_step": int(df["step"].iloc[0]),
            "last_step": int(df["step"].iloc[-1]),
            "first_weights": {k: float(v) for k, v in weights.iloc[0].to_dict().items()},
            "last_weights": {k: float(v) for k, v in weights.iloc[-1].to_dict().items()},
            "moving_weights": moving,
            "min_weight": min_weight,
            "max_weight": max_weight,
        },
        issues,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default="outputs/grpo_rtw_cuda_smoke_50")
    parser.add_argument("--fail_on_issue", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    reward_summary, reward_issues = summarize_reward_components(run_dir / "reward_components.jsonl")
    teacher_summary, teacher_issues = summarize_teacher_weights(run_dir / "teacher_weights.jsonl")
    report = {
        "run_dir": str(run_dir),
        "reward_components": reward_summary,
        "teacher_weights": teacher_summary,
        "issues": reward_issues + teacher_issues,
    }
    print(json.dumps(report, indent=2))

    if args.fail_on_issue and report["issues"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
