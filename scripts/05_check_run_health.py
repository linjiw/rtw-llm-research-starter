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
EPS = 1e-12


def component_mean(components: pd.DataFrame, name: str, fallback: str | None = None) -> float:
    if name in components:
        return float(components[name].mean())
    if fallback and fallback in components:
        return float(components[fallback].mean())
    return 0.0


def component_series(
    components: pd.DataFrame,
    name: str,
    fallback: str | None = None,
) -> pd.Series:
    if name in components:
        return components[name]
    if fallback and fallback in components:
        return components[fallback]
    return pd.Series([0.0] * len(components), index=components.index)


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
    split_reward_summary = {}
    for key in ["primary_reward", "primary_reward_weighted", "aux_reward_weighted", "total_reward"]:
        if key in df:
            split_reward_summary[f"{key}_mean"] = float(df[key].mean())
            split_reward_summary[f"{key}_std"] = float(df[key].std()) if len(df) > 1 else 0.0

    open_tag_rate = component_mean(components, "contains_open_answer_tag")
    close_tag_rate = component_mean(components, "contains_close_answer_tag")
    extractable_span_rate = component_mean(components, "has_extractable_answer_span", "format")
    parseable_expression_rate = component_mean(components, "expression_parseable", "parse_ok")
    allowed_numbers_rate = component_mean(components, "uses_allowed_numbers", "uses_numbers")
    allowed_ops_rate = component_mean(components, "uses_allowed_ops", "allowed_ops")
    exact_correct_rate = component_mean(components, "exact_correct", "correct")
    number_precision_mean = component_mean(components, "number_precision")
    number_recall_mean = component_mean(components, "number_recall")
    number_multiset_f1_mean = component_mean(components, "number_multiset_f1")
    uses_no_extra_numbers_rate = component_mean(components, "uses_no_extra_numbers")
    uses_all_required_numbers_rate = component_mean(components, "uses_all_required_numbers")
    operator_precision_mean = component_mean(components, "operator_precision")
    operator_recall_mean = component_mean(components, "operator_recall")
    evaluates_without_exception_rate = component_mean(components, "evaluates_without_exception")
    numeric_distance_reward_mean = component_mean(components, "numeric_distance_reward")
    tag_only_rate = extractable_span_rate - parseable_expression_rate
    span_series = component_series(components, "has_extractable_answer_span", "format")
    parseable_series = component_series(components, "expression_parseable", "parse_ok")
    tag_only_observed_rate = float(
        ((span_series > 0) & (parseable_series == 0)).mean()
        if len(components)
        else 0.0
    )
    parseable_but_wrong_rate = parseable_expression_rate - exact_correct_rate
    correct_given_parseable = exact_correct_rate / max(parseable_expression_rate, EPS)

    if "reward_batch_has_variance" in df:
        reward_variance_nonzero_fraction = float(
            df.groupby("reward_batch_index")["reward_batch_has_variance"].max().mean()
            if "reward_batch_index" in df
            else df["reward_batch_has_variance"].mean()
        )
    elif "reward_batch_index" in df:
        batch_stds = df.groupby("reward_batch_index")["reward"].std().fillna(0.0)
        reward_variance_nonzero_fraction = float((batch_stds > 1e-9).mean())
    else:
        reward_variance_nonzero_fraction = float(reward_std > 1e-9)

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
            **split_reward_summary,
            "diagnostic_ratios": {
                "open_tag_rate": open_tag_rate,
                "close_tag_rate": close_tag_rate,
                "extractable_span_rate": extractable_span_rate,
                "parseable_expression_rate": parseable_expression_rate,
                "allowed_numbers_rate": allowed_numbers_rate,
                "allowed_ops_rate": allowed_ops_rate,
                "exact_correct_rate": exact_correct_rate,
                "number_precision_mean": number_precision_mean,
                "number_recall_mean": number_recall_mean,
                "number_multiset_f1_mean": number_multiset_f1_mean,
                "uses_no_extra_numbers_rate": uses_no_extra_numbers_rate,
                "uses_all_required_numbers_rate": uses_all_required_numbers_rate,
                "operator_precision_mean": operator_precision_mean,
                "operator_recall_mean": operator_recall_mean,
                "evaluates_without_exception_rate": evaluates_without_exception_rate,
                "numeric_distance_reward_mean": numeric_distance_reward_mean,
                "tag_only_rate": tag_only_rate,
                "tag_only_observed_rate": tag_only_observed_rate,
                "parseable_but_wrong_rate": parseable_but_wrong_rate,
                "correct_given_parseable": correct_given_parseable,
                "reward_variance_nonzero_fraction": reward_variance_nonzero_fraction,
            },
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
    weight_sums = weights.sum(axis=1)
    constraint_columns = [c for c in ["valid_expression", "number_multiset_f1", "allowed_ops"] if c in weights]
    if constraint_columns:
        constraint_weight_mass = weights[constraint_columns].sum(axis=1)
    else:
        constraint_weight_mass = pd.Series([0.0] * len(weights), index=weights.index)
    if "numeric_distance_reward" in weights:
        numeric_distance_weight = weights["numeric_distance_reward"]
    else:
        numeric_distance_weight = pd.Series([0.0] * len(weights), index=weights.index)
    numeric_distance_to_constraint_ratio = numeric_distance_weight / constraint_weight_mass.clip(lower=EPS)
    if len(weights) > 1:
        update_deltas = weights.diff().abs().iloc[1:]
        teacher_update_l1_mean = float(update_deltas.sum(axis=1).mean())
        teacher_update_linf_max = float(update_deltas.max(axis=1).max())
    else:
        teacher_update_l1_mean = 0.0
        teacher_update_linf_max = 0.0

    floor_hit_rate_by_component: dict[str, float] = {}
    cap_hit_rate_by_component: dict[str, float] = {}
    if "diagnostics" in df:
        diagnostics = pd.json_normalize(df["diagnostics"])
        for column in diagnostics.columns:
            if column.startswith("floor_hits."):
                floor_hit_rate_by_component[column.split(".", 1)[1]] = float(diagnostics[column].fillna(False).mean())
            if column.startswith("cap_hits."):
                cap_hit_rate_by_component[column.split(".", 1)[1]] = float(diagnostics[column].fillna(False).mean())

    per_component = {
        column: {
            "min": float(weights[column].min()),
            "max": float(weights[column].max()),
            "mean": float(weights[column].mean()),
        }
        for column in weights.columns
    }

    issues = []
    strategies = set(df["strategy"].dropna().astype(str)) if "strategy" in df else set()
    if len(weights) > 1 and not moving and strategies != {"static"}:
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
            "per_component": per_component,
            "weight_sum_final": float(weight_sums.iloc[-1]),
            "weight_sum_mean": float(weight_sums.mean()),
            "constraint_weight_mass_final": float(constraint_weight_mass.iloc[-1]),
            "constraint_weight_mass_mean": float(constraint_weight_mass.mean()),
            "numeric_distance_weight_final": float(numeric_distance_weight.iloc[-1]),
            "numeric_distance_weight_mean": float(numeric_distance_weight.mean()),
            "numeric_distance_to_constraint_ratio_final": float(numeric_distance_to_constraint_ratio.iloc[-1]),
            "numeric_distance_to_constraint_ratio_mean": float(numeric_distance_to_constraint_ratio.mean()),
            "teacher_update_l1_mean": teacher_update_l1_mean,
            "teacher_update_linf_max": teacher_update_linf_max,
            "floor_hit_rate_by_component": floor_hit_rate_by_component,
            "cap_hit_rate_by_component": cap_hit_rate_by_component,
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
