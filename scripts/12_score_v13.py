#!/usr/bin/env python
"""Pre-registered v0.13 scorer (plan A2 + memorization control).

Scores arm banks against a multi-seed baseline distribution on one split:
  1. PRIMARY: candidate-level easy-tier legality rate (valid_expression),
     partitioned into memorization-overlap vs held-out tasks.
  2. PRIMARY: per-candidate P(exact | legal) over the legal subset.
  3. Directional: oracle_exact@N, reranked@N, McNemar counts vs EACH baseline
     seed (never a single seed as the reference).
  4. Guardrail A5: distinct legal expressions per task at N.
  5. Guardrail: cost (tokens/candidate, clip rate at max_new_tokens).
  6. Checks: selection saturation (reranked@N == oracle@N); GRPO-inert
     (batch_group_variance_fraction from an optional run dir).

All comparisons are computed, not estimated; the decision rule lives in
docs/V13_SFT_WARMUP_LEGALITY_PLAN.md.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_bank(bank_dir: Path, n: int) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with (bank_dir / "candidates.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            groups[row["id"]].append(row)
    out = {}
    for task_id, rows in groups.items():
        rows.sort(key=lambda r: r["candidate_index"])
        if len(rows) < n:
            raise ValueError(f"{bank_dir}: task {task_id} has {len(rows)} < {n} candidates")
        out[task_id] = rows[:n]
    return dict(out)


def overlap_task_ids(
    bank: dict[str, list[dict]], train_path: Path
) -> tuple[set[str], dict[str, set[str]]]:
    """Return overlap task ids and, per overlap task, the gold solutions
    (whitespace-normalized) from every train row sharing its (numbers, target).
    Verbatim-gold exact candidates on overlap tasks indicate memorization; a
    different exact expression on the same task is genuine search."""
    train_solutions: dict[tuple, set[str]] = defaultdict(set)
    with train_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            key = (tuple(sorted(row["numbers"])), row["target"])
            train_solutions[key].add(row["solution"].replace(" ", ""))
    hits = set()
    gold_by_task: dict[str, set[str]] = {}
    for task_id, rows in bank.items():
        key = (tuple(sorted(rows[0]["numbers"])), rows[0]["target"])
        if key in train_solutions:
            hits.add(task_id)
            gold_by_task[task_id] = train_solutions[key]
    return hits, gold_by_task


def is_legal(cand: dict) -> bool:
    return float(cand["metrics"].get("valid_expression", 0.0)) >= 1.0


def is_exact(cand: dict) -> bool:
    return float(cand["metrics"].get("exact_correct", 0.0)) >= 1.0


def two_proportion_p(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided two-proportion z-test (pooled)."""
    if min(n1, n2) == 0:
        return 1.0
    p1, p2 = k1 / n1, k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (p1 - p2) / se
    return math.erfc(abs(z) / math.sqrt(2))


def candidate_stats(
    bank: dict[str, list[dict]],
    task_ids: set[str] | None = None,
    tier: str | None = None,
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    cands = [
        c
        for tid, rows in bank.items()
        if task_ids is None or tid in task_ids
        for c in rows
        if tier is None or c.get("difficulty") == tier
    ]
    n = len(cands)
    legal = [c for c in cands if is_legal(c)]
    exact_legal = sum(1 for c in legal if is_exact(c))
    tokens = [c.get("completion_token_count", 0) for c in cands]
    return {
        "n_candidates": n,
        "n_legal": len(legal),
        "legality_rate": len(legal) / n if n else 0.0,
        "n_exact_given_legal": exact_legal,
        "p_exact_given_legal": exact_legal / len(legal) if legal else 0.0,
        "tokens_per_candidate": sum(tokens) / n if n else 0.0,
        "clip_rate": sum(1 for t in tokens if t >= max_new_tokens) / n if n else 0.0,
    }


def selected_exact(bank: dict[str, list[dict]], n: int, selector: str) -> dict[str, bool]:
    out = {}
    for task_id, rows in bank.items():
        rows = rows[:n]
        if selector == "oracle":
            chosen = max(
                rows,
                key=lambda r: (
                    float(r["metrics"].get("exact_correct", 0.0)),
                    r["practical_score"],
                    -r["candidate_index"],
                ),
            )
        else:
            chosen = max(rows, key=lambda r: (r["practical_score"], -r["candidate_index"]))
        out[task_id] = is_exact(chosen)
    return out


def mcnemar_counts(arm: dict[str, bool], base: dict[str, bool]) -> dict[str, int]:
    if set(arm) != set(base):
        raise ValueError("Task-ID mismatch between arm and baseline bank")
    counts = {"both": 0, "arm_only": 0, "base_only": 0, "neither": 0}
    for tid in arm:
        a, b = arm[tid], base[tid]
        key = "both" if a and b else "arm_only" if a else "base_only" if b else "neither"
        counts[key] += 1
    return counts


def distinct_legal_expressions(bank: dict[str, list[dict]], n: int, tier: str | None = None) -> float:
    per_task = []
    for rows in bank.values():
        rows = rows[:n]
        if tier is not None and rows[0].get("difficulty") != tier:
            continue
        exprs = {c["metrics"].get("expression") for c in rows if is_legal(c)}
        per_task.append(len(exprs))
    return sum(per_task) / len(per_task) if per_task else 0.0


def group_variance_fraction(run_dir: Path) -> float | None:
    path = run_dir / "reward_components.jsonl"
    if not path.exists():
        return None
    n = tot = 0
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if "group_has_variance" in row:
                tot += 1
                n += bool(row["group_has_variance"])
    return n / tot if tot else None


def verbatim_gold_split(
    bank: dict[str, list[dict]], overlap: set[str], gold_by_task: dict[str, set[str]], n: int
) -> dict[str, int]:
    """Among exact candidates on overlap tasks: verbatim gold vs novel."""
    verbatim = novel = 0
    for task_id in overlap:
        gold = gold_by_task.get(task_id, set())
        for cand in bank[task_id][:n]:
            if not is_exact(cand):
                continue
            expr = (cand["metrics"].get("expression") or "").replace(" ", "")
            if expr in gold:
                verbatim += 1
            else:
                novel += 1
    return {"verbatim_gold_exact_candidates": verbatim, "novel_exact_candidates": novel}


def score_arm(
    name: str,
    bank: dict[str, list[dict]],
    baselines: dict[str, dict[str, list[dict]]],
    overlap: set[str],
    gold_by_task: dict[str, set[str]],
    n: int,
) -> dict[str, Any]:
    held_out = set(bank) - overlap
    result: dict[str, Any] = {"arm": name, "n_tasks": len(bank), "n_overlap_tasks": len(overlap)}
    result["overlap_exact_split"] = verbatim_gold_split(bank, overlap, gold_by_task, n)

    # Primary surfaces, partitioned. Baseline pooled counts for the p-value.
    for scope, ids in (("all", None), ("held_out", held_out), ("overlap", overlap)):
        arm_easy = candidate_stats(bank, ids, tier="easy")
        base_k = base_n = 0
        base_rates = []
        for b in baselines.values():
            bs = candidate_stats(b, ids, tier="easy")
            base_k += bs["n_legal"]
            base_n += bs["n_candidates"]
            base_rates.append(bs["legality_rate"])
        result[f"easy_legality_{scope}"] = {
            "arm": arm_easy,
            "baseline_rates_per_seed": base_rates,
            "baseline_pooled_rate": base_k / base_n if base_n else 0.0,
            "two_proportion_p_vs_pooled": two_proportion_p(
                arm_easy["n_legal"], arm_easy["n_candidates"], base_k, base_n
            ),
        }
    arm_all = candidate_stats(bank)
    result["all_tier_candidates"] = arm_all
    result["p_exact_given_legal_baseline_per_seed"] = [
        candidate_stats(b)["p_exact_given_legal"] for b in baselines.values()
    ]

    # Directional task-level metrics vs each baseline seed.
    arm_oracle = selected_exact(bank, n, "oracle")
    arm_rerank = selected_exact(bank, n, "practical")
    result["oracle_exact_at_n"] = sum(arm_oracle.values()) / len(arm_oracle)
    result["reranked_exact_at_n"] = sum(arm_rerank.values()) / len(arm_rerank)
    result["selection_saturated"] = arm_oracle == arm_rerank
    result["held_out_oracle_exact"] = (
        sum(arm_oracle[t] for t in held_out) / len(held_out) if held_out else 0.0
    )
    result["overlap_oracle_exact"] = (
        sum(arm_oracle[t] for t in overlap) / len(overlap) if overlap else 0.0
    )
    result["mcnemar_vs_baseline_seeds"] = {}
    for seed_name, b in baselines.items():
        base_rerank = selected_exact(b, n, "practical")
        result["mcnemar_vs_baseline_seeds"][seed_name] = {
            "all": mcnemar_counts(arm_rerank, base_rerank),
            "held_out": mcnemar_counts(
                {t: arm_rerank[t] for t in held_out}, {t: base_rerank[t] for t in held_out}
            ),
        }

    # Guardrails.
    result["distinct_legal_expr_per_task_easy"] = distinct_legal_expressions(bank, n, tier="easy")
    result["distinct_legal_expr_baseline_per_seed"] = [
        distinct_legal_expressions(b, n, tier="easy") for b in baselines.values()
    ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", required=True, help="name=bank_dir", dest="arms")
    parser.add_argument("--baseline_dirs", nargs="+", required=True)
    parser.add_argument("--train_path", default="data/countdown/train.jsonl")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--grpo_run_dir", default=None)
    parser.add_argument("--baseline_run_dir", default=None)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    baselines = {Path(d).name: load_bank(Path(d), args.n) for d in args.baseline_dirs}
    reference = next(iter(baselines.values()))
    overlap, gold_by_task = overlap_task_ids(reference, Path(args.train_path))

    report: dict[str, Any] = {
        "n": args.n,
        "baseline_dirs": args.baseline_dirs,
        "overlap_task_ids": sorted(overlap),
        "arms": [],
    }
    for spec in args.arms:
        name, _, bank_dir = spec.partition("=")
        bank = load_bank(Path(bank_dir), args.n)
        report["arms"].append(score_arm(name, bank, baselines, overlap, gold_by_task, args.n))

    if args.grpo_run_dir:
        report["arm_group_variance_fraction"] = group_variance_fraction(Path(args.grpo_run_dir))
    if args.baseline_run_dir:
        report["baseline_group_variance_fraction"] = group_variance_fraction(
            Path(args.baseline_run_dir)
        )

    Path(args.out_json).write_text(json.dumps(report, indent=2) + "\n")

    for arm in report["arms"]:
        print(f"\n=== {arm['arm']} (N={args.n}) ===")
        for scope in ("all", "held_out", "overlap"):
            e = arm[f"easy_legality_{scope}"]
            a = e["arm"]
            print(
                f"  easy legality [{scope}]: {a['legality_rate']:.3f} "
                f"({a['n_legal']}/{a['n_candidates']}) vs baseline pooled "
                f"{e['baseline_pooled_rate']:.3f} per-seed {['%.3f' % r for r in e['baseline_rates_per_seed']]} "
                f"p={e['two_proportion_p_vs_pooled']:.4f}"
            )
        t = arm["all_tier_candidates"]
        print(
            f"  P(exact|legal): {t['p_exact_given_legal']:.3f} ({t['n_exact_given_legal']}/{t['n_legal']}) "
            f"vs baseline per-seed {['%.3f' % r for r in arm['p_exact_given_legal_baseline_per_seed']]}"
        )
        print(
            f"  oracle@{args.n}={arm['oracle_exact_at_n']:.3f} rerank@{args.n}={arm['reranked_exact_at_n']:.3f} "
            f"(held-out {arm['held_out_oracle_exact']:.3f}, overlap {arm['overlap_oracle_exact']:.3f}) "
            f"selection_saturated={arm['selection_saturated']}"
        )
        ov = arm["overlap_exact_split"]
        print(
            f"  overlap exact candidates: {ov['verbatim_gold_exact_candidates']} verbatim-gold, "
            f"{ov['novel_exact_candidates']} novel"
        )
        for seed_name, mc in arm["mcnemar_vs_baseline_seeds"].items():
            print(f"  vs {seed_name}: all={mc['all']} held_out={mc['held_out']}")
        print(
            f"  diversity easy: {arm['distinct_legal_expr_per_task_easy']:.2f} distinct legal expr/task "
            f"vs baseline {['%.2f' % r for r in arm['distinct_legal_expr_baseline_per_seed']]}"
        )
        print(f"  cost: {t['tokens_per_candidate']:.0f} tok/cand, clip {t['clip_rate']:.3f}")
    if "arm_group_variance_fraction" in report:
        print(
            f"\nGRPO-inert check: arm group-variance {report['arm_group_variance_fraction']} "
            f"vs baseline {report.get('baseline_group_variance_fraction')}"
        )


if __name__ == "__main__":
    main()
