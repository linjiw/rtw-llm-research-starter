#!/usr/bin/env python
"""Summarize v0.9B seed-expansion best-of-N runs."""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import pandas as pd

from rtw_llm.cluster_stats import (
    ClusterInferenceError,
    evaluation_protocol_signature,
    require_complete_evaluation_signature,
    require_matching_evaluation_signatures,
    semantic_task_key,
    stack_task_runs,
    task_clustered_difference,
    task_seed_product_bootstrap_difference,
)
from rtw_llm.provenance import verify_completed_run
from rtw_llm.seed_protocol import TRUE_SEED_PROTOCOL


def finite_or_inf(value: float) -> float | str:
    if value >= 1e11:
        return "inf"
    return float(value)


def exact_mcnemar_p(stable_only: int, static_only: int) -> float:
    """Two-sided exact McNemar/binomial p-value over discordant pairs."""
    n = stable_only + static_only
    if n == 0:
        return 1.0
    k = min(stable_only, static_only)
    p = 2.0 * sum(math.comb(n, i) * 0.5**n for i in range(k + 1))
    return float(min(1.0, p))


def infer_from_name(path: Path) -> dict[str, Any]:
    name = path.name
    method = "stable" if "stable" in name else "static" if "static" in name else None
    seed_match = re.search(r"seed(\d+)", name)
    split = "test_in_dist" if "test_in_dist" in name else "validation" if "validation" in name else None
    return {
        "method": method,
        "training_seed": int(seed_match.group(1)) if seed_match else None,
        "split": split,
    }


def load_run(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "run_config.json"
    candidates_path = run_dir / "candidates.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    metrics = json.loads(metrics_path.read_text())
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    inferred = infer_from_name(run_dir)
    method = config.get("method") or metrics.get("method") or inferred["method"]
    if method in {"Stable-RTW", "stable_v06c", "adaptive_stable_v06c"}:
        method = "stable"
    if method in {"static_v06b"}:
        method = "static"
    seed = config.get("training_seed", metrics.get("training_seed", inferred["training_seed"]))
    split = config.get("split", metrics.get("split", inferred["split"]))
    if method is None or seed is None or split is None:
        raise ValueError(f"Could not infer method/seed/split for {run_dir}")
    manifest = None
    intent_exists = (run_dir / "run_intent.json").exists()
    result_exists = (run_dir / "run_result.json").exists()
    if intent_exists or result_exists:
        manifest = verify_completed_run(
            run_dir,
            required_artifact_roles={"candidates", "metrics", "run_config", "summary"},
        )
    return {
        "run_dir": str(run_dir),
        "metrics": metrics,
        "config": config,
        "method": method,
        "training_seed": int(seed),
        "split": split,
        "candidates_path": candidates_path,
        "verified_manifest": manifest,
    }


def metric_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        metrics = run["metrics"]
        for n_text, result in metrics["by_n"].items():
            n = int(n_text)
            practical = result["practical_selected"]
            rows.append(
                {
                    "split": run["split"],
                    "method": run["method"],
                    "training_seed": run["training_seed"],
                    "N": n,
                    "oracle_exact": result["oracle_exact_at_n"],
                    "reranked_exact": result["reranked_exact_at_n"],
                    "selected_valid": practical.get("valid_expression", 0.0),
                    "selected_number_f1": practical.get("number_multiset_f1", 0.0),
                    "reward_hack": practical.get("reward_hacking_candidate", 0.0),
                    "tokens": result.get("tokens_generated", 0),
                    "wall_clock_s": result.get("wall_clock_seconds_estimated", 0.0),
                    "cost_per_oracle_exact": finite_or_inf(result.get("cost_per_oracle_exact", 0.0)),
                    "cost_per_reranked_exact": finite_or_inf(result.get("cost_per_reranked_exact", 0.0)),
                    "total_candidates": metrics.get("total_candidates", 0),
                    "total_tokens_generated": metrics.get("total_tokens_generated", 0),
                }
            )
    return rows


def summarize_mean_std(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric_keys = [
        "oracle_exact",
        "reranked_exact",
        "selected_valid",
        "selected_number_f1",
        "reward_hack",
        "tokens",
        "wall_clock_s",
        "cost_per_reranked_exact",
    ]
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["method"], row["N"])].append(row)
    summary = []
    for (split, method, n), items in sorted(groups.items()):
        out: dict[str, Any] = {"split": split, "method": method, "N": n, "seeds": len(items)}
        for key in numeric_keys:
            vals = [float(item[key]) for item in items if item[key] != "inf"]
            out[f"{key}_mean"] = mean(vals) if vals else math.inf
            out[f"{key}_std"] = stdev(vals) if len(vals) > 1 else 0.0
        summary.append(out)
    return summary


def load_candidates_by_id(candidates_path: Path, n: int) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with candidates_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            groups[row["id"]].append(row)
    for task_id, rows in groups.items():
        rows.sort(key=lambda row: row["candidate_index"])
        if len(rows) < n:
            raise ValueError(f"{candidates_path}: task {task_id} has only {len(rows)} candidates, need {n}")
        candidate_indices = [row["candidate_index"] for row in rows[:n]]
        if candidate_indices != list(range(n)):
            raise ValueError(
                f"{candidates_path}: task {task_id} has duplicate or missing candidate indices "
                f"for N={n}: {candidate_indices}"
            )
        groups[task_id] = rows[:n]
    return dict(groups)


def selected_exact_by_task(candidates_path: Path, n: int, selector: str) -> dict[str, bool]:
    groups = load_candidates_by_id(candidates_path, n)
    out: dict[str, bool] = {}
    for task_id, rows in groups.items():
        if selector == "oracle":
            chosen = max(
                rows,
                key=lambda row: (
                    float(row["metrics"].get("exact_correct", 0.0)),
                    row["practical_score"],
                    -row["candidate_index"],
                ),
            )
        elif selector == "practical":
            chosen = max(rows, key=lambda row: (row["practical_score"], -row["candidate_index"]))
        else:
            raise ValueError(selector)
        out[task_id] = bool(float(chosen["metrics"].get("exact_correct", 0.0)))
    return out


def selected_exact_by_semantic_task(
    candidates_path: Path, n: int, selector: str
) -> dict[str, float]:
    """Reduce one run to one verifier-exact outcome per semantic task."""
    groups = load_candidates_by_id(candidates_path, n)
    by_semantic_task: dict[str, float] = {}
    source_ids: dict[str, str] = {}
    for task_id, rows in groups.items():
        keys = {semantic_task_key(row) for row in rows}
        if len(keys) != 1:
            raise ClusterInferenceError(
                f"{candidates_path}: task ID {task_id!r} changes semantic identity across candidates"
            )
        key = next(iter(keys))
        if key in by_semantic_task:
            raise ClusterInferenceError(
                f"{candidates_path}: duplicate semantic tasks {source_ids[key]!r} and {task_id!r}"
            )
        if selector == "oracle":
            chosen = max(
                rows,
                key=lambda row: (
                    float(row["metrics"].get("exact_correct", 0.0)),
                    row["practical_score"],
                    -row["candidate_index"],
                ),
            )
        elif selector == "practical":
            chosen = max(rows, key=lambda row: (row["practical_score"], -row["candidate_index"]))
        else:
            raise ValueError(selector)
        by_semantic_task[key] = float(bool(float(chosen["metrics"].get("exact_correct", 0.0))))
        source_ids[key] = task_id
    return by_semantic_task


def generation_identity(run: dict[str, Any]) -> dict[str, Any]:
    """Frozen evaluation signature both arms of a comparison must share."""
    manifest = run.get("verified_manifest")
    strict_identity = manifest["intent"]["identity"] if manifest else None
    return evaluation_protocol_signature(
        run.get("config", {}),
        split=run.get("split"),
        strict_identity=strict_identity,
    )


def paired_overlap(runs: list[dict[str, Any]], selector: str = "practical") -> list[dict[str, Any]]:
    by_key = {(run["split"], run["training_seed"], run["method"]): run for run in runs}
    pairs = []
    for split in sorted({run["split"] for run in runs}):
        for seed in sorted({run["training_seed"] for run in runs}):
            static = by_key.get((split, seed, "static"))
            stable = by_key.get((split, seed, "stable"))
            if not static or not stable:
                continue
            try:
                require_matching_evaluation_signatures(
                    {"static": generation_identity(static), "stable": generation_identity(stable)}
                )
            except ClusterInferenceError as exc:
                raise ValueError(
                    f"Evaluation-identity mismatch for split={split} seed={seed}: {exc}"
                ) from exc
            n_values = sorted(int(n) for n in stable["metrics"]["by_n"])
            for n in n_values:
                stable_exact = selected_exact_by_task(stable["candidates_path"], n, selector)
                static_exact = selected_exact_by_task(static["candidates_path"], n, selector)
                if list(stable_exact.keys()) != list(static_exact.keys()):
                    raise ValueError(f"Task ID/order mismatch for split={split} seed={seed} N={n}")
                both = stable_only = static_only = neither = 0
                for task_id in stable_exact:
                    s = stable_exact[task_id]
                    t = static_exact[task_id]
                    if s and t:
                        both += 1
                    elif s and not t:
                        stable_only += 1
                    elif t and not s:
                        static_only += 1
                    else:
                        neither += 1
                stable_rate = (both + stable_only) / max(len(stable_exact), 1)
                static_rate = (both + static_only) / max(len(stable_exact), 1)
                pairs.append(
                    {
                        "split": split,
                        "training_seed": seed,
                        "N": n,
                        "selector": selector,
                        "both": both,
                        "stable_only": stable_only,
                        "static_only": static_only,
                        "neither": neither,
                        "mcnemar_p": exact_mcnemar_p(stable_only, static_only),
                        "inference_scope": "exploratory_within_observed_run_task_panel",
                        "delta_reranked_exact": stable_rate - static_rate,
                    }
                )
    return pairs


def aggregate_paired(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        groups[(pair["split"], pair["N"], pair["selector"])].append(pair)
    out = []
    for (split, n, selector), items in sorted(groups.items()):
        stable_only = sum(item["stable_only"] for item in items)
        static_only = sum(item["static_only"] for item in items)
        both = sum(item["both"] for item in items)
        neither = sum(item["neither"] for item in items)
        deltas = [item["delta_reranked_exact"] for item in items]
        out.append(
            {
                "split": split,
                "N": n,
                "selector": selector,
                "seeds": len(items),
                "both": both,
                "stable_only": stable_only,
                "static_only": static_only,
                "neither": neither,
                "delta_reranked_exact_mean": mean(deltas),
                "delta_reranked_exact_std": stdev(deltas) if len(deltas) > 1 else 0.0,
                "legacy_pseudoreplicated_descriptive_only": {
                    "inference_valid": False,
                    "reason": (
                        "task-by-run cells were pooled as independent pairs; use the "
                        "task-clustered observed-panel contrast instead"
                    ),
                    "pooled_mcnemar_p": exact_mcnemar_p(stable_only, static_only),
                },
            }
        )
    return out


def clustered_paired(
    runs: list[dict[str, Any]],
    *,
    selector: str,
    bootstrap_draws: int = 20_000,
    seed: int = 17,
) -> list[dict[str, Any]]:
    """Compute equal-task inference while retaining each run as repeated measurement."""
    out: list[dict[str, Any]] = []
    for split in sorted({run["split"] for run in runs}):
        split_runs = [run for run in runs if run["split"] == split]
        stable_runs_for_split = [run for run in split_runs if run["method"] == "stable"]
        static_runs_for_split = [run for run in split_runs if run["method"] == "static"]
        stable_by_seed = {
            run["training_seed"]: run for run in stable_runs_for_split
        }
        static_by_seed = {
            run["training_seed"]: run for run in static_runs_for_split
        }
        if len(stable_by_seed) != len(stable_runs_for_split) or len(static_by_seed) != len(
            static_runs_for_split
        ):
            raise ClusterInferenceError(f"split={split}: duplicate method-by-run-label cell")
        if not stable_by_seed and not static_by_seed:
            continue
        if set(stable_by_seed) != set(static_by_seed):
            raise ClusterInferenceError(
                f"split={split}: stable/static observed-run labels do not match"
            )
        run_seeds = sorted(stable_by_seed)
        evaluation_signature = require_matching_evaluation_signatures(
            {
                f"{run['method']}/run{run['training_seed']}": generation_identity(run)
                for run in [*stable_by_seed.values(), *static_by_seed.values()]
            }
        )
        n_sets = [
            {int(value) for value in run["metrics"]["by_n"]}
            for run in [*stable_by_seed.values(), *static_by_seed.values()]
        ]
        if not n_sets or any(values != n_sets[0] for values in n_sets[1:]):
            raise ClusterInferenceError(f"split={split}: N grids do not match across runs")
        for training_seed in run_seeds:
            stable = stable_by_seed[training_seed]
            static = static_by_seed[training_seed]
            require_matching_evaluation_signatures(
                {"stable": generation_identity(stable), "static": generation_identity(static)}
            )
        for n in sorted(n_sets[0]):
            stable_runs = [
                selected_exact_by_semantic_task(
                    stable_by_seed[training_seed]["candidates_path"], n, selector
                )
                for training_seed in run_seeds
            ]
            static_runs = [
                selected_exact_by_semantic_task(
                    static_by_seed[training_seed]["candidates_path"], n, selector
                )
                for training_seed in run_seeds
            ]
            stable_panel = stack_task_runs(stable_runs, label=f"stable/{split}/N={n}")
            static_panel = stack_task_runs(static_runs, label=f"static/{split}/N={n}")
            clustered = task_clustered_difference(
                stable_panel,
                static_panel,
                bootstrap_draws=bootstrap_draws,
                sign_flip_draws=bootstrap_draws,
                seed=seed,
            )
            protocols = {
                run.get("config", {}).get("training_protocol", "countdown-legacy-v1")
                for run in [*stable_by_seed.values(), *static_by_seed.values()]
            }
            if protocols == {TRUE_SEED_PROTOCOL} and len(run_seeds) >= 3:
                for run in [*stable_by_seed.values(), *static_by_seed.values()]:
                    verified = run.get("verified_manifest")
                    if verified is None:
                        raise ClusterInferenceError(
                            f"{run['run_dir']}: true-seed inference requires verified manifests"
                        )
                    seed_roles = verified["intent"]["identity"].get("seed_roles", {})
                    if (
                        seed_roles.get("training_protocol") != TRUE_SEED_PROTOCOL
                        or seed_roles.get("training_seed_label") != run["training_seed"]
                    ):
                        raise ClusterInferenceError(
                            f"{run['run_dir']}: provenance seed roles do not match run labels"
                        )
                seed_generalization = task_seed_product_bootstrap_difference(
                    stable_panel,
                    static_panel,
                    true_seed_protocol=True,
                    bootstrap_draws=bootstrap_draws,
                    seed=seed,
                )
            else:
                seed_generalization = {
                    "available": False,
                    "reason": "requires_at_least_three_countdown_true_seeds_v2_runs",
                    "observed_protocols": sorted(protocols),
                }
            out.append(
                {
                    "split": split,
                    "N": n,
                    "selector": selector,
                    "observed_run_labels": run_seeds,
                    "evaluation_protocol_signature": evaluation_signature,
                    "task_clustered_observed_panel": clustered,
                    "training_seed_generalization": seed_generalization,
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_glob", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--bootstrap_draws", type=int, default=20_000)
    parser.add_argument("--inference_seed", type=int, default=17)
    args = parser.parse_args()

    run_dirs = [Path(path) for path in sorted(glob.glob(args.runs_glob)) if Path(path).is_dir()]
    if not run_dirs:
        raise SystemExit(f"No runs matched {args.runs_glob}")
    runs = [load_run(path) for path in run_dirs]
    for run in runs:
        require_complete_evaluation_signature(
            generation_identity(run), label=str(run["run_dir"])
        )
    rows = metric_rows(runs)
    summary = summarize_mean_std(rows)
    missing_candidate_banks = [
        str(run["candidates_path"]) for run in runs if not run["candidates_path"].exists()
    ]
    if missing_candidate_banks:
        paired = []
        paired_oracle = []
        paired_summary = []
        paired_oracle_summary = []
        unavailable = {
            "available": False,
            "reason": "raw_candidate_banks_unavailable_no_inference_from_aggregates",
            "missing_candidate_banks": missing_candidate_banks,
        }
        clustered = {**unavailable, "analyses": []}
        clustered_oracle = {**unavailable, "analyses": []}
    else:
        paired = paired_overlap(runs, selector="practical")
        paired_oracle = paired_overlap(runs, selector="oracle")
        paired_summary = aggregate_paired(paired)
        paired_oracle_summary = aggregate_paired(paired_oracle)
        clustered = {
            "available": True,
            "analyses": clustered_paired(
                runs,
                selector="practical",
                bootstrap_draws=args.bootstrap_draws,
                seed=args.inference_seed,
            ),
        }
        clustered_oracle = {
            "available": True,
            "analyses": clustered_paired(
                runs,
                selector="oracle",
                bootstrap_draws=args.bootstrap_draws,
                seed=args.inference_seed,
            ),
        }

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(out_csv, index=False)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "rtw-v09-cluster-aware-summary-v2",
        "protocol_correction": {
            "legacy_cross_run_pooled_mcnemar_inference_valid": False,
            "clustered_inference_requires_raw_candidate_banks": True,
        },
        "runs": [
            {"run_dir": run["run_dir"], "split": run["split"], "method": run["method"], "training_seed": run["training_seed"]}
            for run in runs
        ],
        "per_run_rows": rows,
        "mean_std": summary,
        "paired_by_seed": paired,
        "paired_summary": paired_summary,
        "task_clustered_summary": clustered,
        "paired_oracle_by_seed": paired_oracle,
        "paired_oracle_summary": paired_oracle_summary,
        "task_clustered_oracle_summary": clustered_oracle,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"runs": len(runs), "out_csv": str(out_csv), "out_json": str(out_json)}, indent=2))


if __name__ == "__main__":
    main()
