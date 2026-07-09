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
    return {
        "run_dir": str(run_dir),
        "metrics": metrics,
        "config": config,
        "method": method,
        "training_seed": int(seed),
        "split": split,
        "candidates_path": candidates_path,
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


def generation_identity(run: dict[str, Any]) -> dict[str, Any]:
    """Sampling-identity keys both arms of a paired comparison must share.

    Candidate banks are only comparable within one generation path: batched
    RNG consumption depends on batch size, so mode and (for batched) batch
    size are part of the identity. Configs predating hf_gen_mode are loop.
    """
    config = run.get("config", {})
    mode = config.get("hf_gen_mode") or "loop"
    identity: dict[str, Any] = {"hf_gen_mode": mode}
    if mode == "batched":
        identity["batch_size"] = config.get("batch_size")
    return identity


def paired_overlap(runs: list[dict[str, Any]], selector: str = "practical") -> list[dict[str, Any]]:
    by_key = {(run["split"], run["training_seed"], run["method"]): run for run in runs}
    pairs = []
    for split in sorted({run["split"] for run in runs}):
        for seed in sorted({run["training_seed"] for run in runs}):
            static = by_key.get((split, seed, "static"))
            stable = by_key.get((split, seed, "stable"))
            if not static or not stable:
                continue
            if generation_identity(static) != generation_identity(stable):
                raise ValueError(
                    f"Generation-identity mismatch for split={split} seed={seed}: "
                    f"static={generation_identity(static)} stable={generation_identity(stable)}; "
                    "paired comparisons are only valid within one generation mode/batch size"
                )
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
                "mcnemar_p": exact_mcnemar_p(stable_only, static_only),
                "delta_reranked_exact_mean": mean(deltas),
                "delta_reranked_exact_std": stdev(deltas) if len(deltas) > 1 else 0.0,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_glob", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    run_dirs = [Path(path) for path in sorted(glob.glob(args.runs_glob)) if Path(path).is_dir()]
    if not run_dirs:
        raise SystemExit(f"No runs matched {args.runs_glob}")
    runs = [load_run(path) for path in run_dirs]
    rows = metric_rows(runs)
    summary = summarize_mean_std(rows)
    paired = paired_overlap(runs, selector="practical")
    paired_oracle = paired_overlap(runs, selector="oracle")
    paired_summary = aggregate_paired(paired)
    paired_oracle_summary = aggregate_paired(paired_oracle)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(out_csv, index=False)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "runs": [
            {"run_dir": run["run_dir"], "split": run["split"], "method": run["method"], "training_seed": run["training_seed"]}
            for run in runs
        ],
        "per_run_rows": rows,
        "mean_std": summary,
        "paired_by_seed": paired,
        "paired_summary": paired_summary,
        "paired_oracle_by_seed": paired_oracle,
        "paired_oracle_summary": paired_oracle_summary,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"runs": len(runs), "out_csv": str(out_csv), "out_json": str(out_json)}, indent=2))


if __name__ == "__main__":
    main()
