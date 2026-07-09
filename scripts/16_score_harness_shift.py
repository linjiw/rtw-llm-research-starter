#!/usr/bin/env python
"""Score the harness-shift banks per HARNESS_OOD_ANALYSIS_CONTRACT.md (rank-2).

Question: does adaptive shaping (stable) yield policies more robust to a terser
prompt (prompt_mid) than fixed (static), relative to the train-time prompt_high?

PRIMARY (well-powered): candidate-level parseable-span-restricted
number_multiset_f1 + legality rate, per prompt field, per method, per seed.
The degradation prompt_high -> prompt_mid is the robustness signal.

The stable-vs-static INTERACTION is declared at the 3-SEED level (per-seed
degradation deltas + sign consistency) — NOT a candidate-level p-value (the
2400-candidate framing is the recurring trap; the statistical unit is the
3-seed policy comparison, which is underpowered by design). exact@8 is
descriptive only.

Reads harness_{method}_seed{s}_{split}_{field}_limit50_n8 banks. CPU-only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

BANK = "outputs/bestofn/harness_{method}_seed{seed}_{split}_{field}_limit50_n8/candidates.jsonl"
FIELDS = ["prompt_high", "prompt_mid"]
METHODS = ["static", "stable"]
SEEDS = [0, 1, 2]


def load(path: str) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in open(path)]


def field_stats(rows: list[dict]) -> dict:
    """Legality + PARSEABLE-SPAN-restricted number_multiset_f1 (contract primary)."""
    if not rows:
        return {"present": False}
    n = len(rows)
    legal = sum(1 for r in rows if float(r["metrics"].get("valid_expression", 0)) > 0)
    # parseable-span restriction: only candidates that produced an extractable,
    # parseable expression (so number_multiset_f1 reflects assembly, not the
    # extract_answer full-text fallback).
    parseable = [
        r for r in rows
        if float(r["metrics"].get("evaluates_without_exception", 0)) > 0
        or float(r["metrics"].get("valid_expression", 0)) > 0
        or (r.get("extracted_expression") and float(r["metrics"].get("number_multiset_f1", 0)) > 0)
    ]
    f1_vals = [float(r["metrics"].get("number_multiset_f1", 0)) for r in parseable]
    return {
        "present": True,
        "n_cand": n,
        "legal_rate": legal / n,
        "n_parseable": len(parseable),
        "number_f1_parseable_mean": mean(f1_vals) if f1_vals else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["validation", "test_in_dist"])
    ap.add_argument("--out_json", default="outputs/harness_shift_scored.json")
    args = ap.parse_args()

    report: dict = {}
    for split in args.splits:
        # per (method, seed): stats at each field + the high->mid degradation
        per_seed: dict = {}
        any_bank = False
        for method in METHODS:
            for seed in SEEDS:
                fs = {f: field_stats(load(BANK.format(method=method, seed=seed, split=split, field=f)))
                      for f in FIELDS}
                if not (fs["prompt_high"].get("present") and fs["prompt_mid"].get("present")):
                    continue
                any_bank = True
                hi, mid = fs["prompt_high"], fs["prompt_mid"]
                per_seed[f"{method}_s{seed}"] = {
                    "legal_high": hi["legal_rate"], "legal_mid": mid["legal_rate"],
                    "legal_degradation": hi["legal_rate"] - mid["legal_rate"],
                    "f1_high": hi["number_f1_parseable_mean"], "f1_mid": mid["number_f1_parseable_mean"],
                    "f1_degradation": hi["number_f1_parseable_mean"] - mid["number_f1_parseable_mean"],
                }
        if not any_bank:
            report[split] = {"status": "no harness banks yet"}
            continue

        # 3-seed interaction: per-seed degradation deltas + sign consistency.
        # Positive "stable more robust" = stable degrades LESS than static.
        def deg(method, metric):
            return [per_seed[f"{method}_s{s}"][metric] for s in SEEDS if f"{method}_s{s}" in per_seed]

        interaction = {}
        for metric in ["legal_degradation", "f1_degradation"]:
            st = deg("static", metric)
            sb = deg("stable", metric)
            if len(st) == len(sb) == 3:
                per_seed_advantage = [st[i] - sb[i] for i in range(3)]  # >0 => stable more robust
                interaction[metric] = {
                    "static_degradation_by_seed": [round(x, 4) for x in st],
                    "stable_degradation_by_seed": [round(x, 4) for x in sb],
                    "stable_robustness_advantage_by_seed": [round(x, 4) for x in per_seed_advantage],
                    "advantage_mean": round(mean(per_seed_advantage), 4),
                    "advantage_std": round(pstdev(per_seed_advantage), 4),
                    "sign_consistent": all(x > 0 for x in per_seed_advantage) or all(x < 0 for x in per_seed_advantage),
                    "n_seeds_stable_more_robust": sum(1 for x in per_seed_advantage if x > 0),
                }
        report[split] = {"per_seed": per_seed, "interaction_3seed": interaction}

    print(json.dumps(report, indent=2))
    Path(args.out_json).write_text(json.dumps(report, indent=2) + "\n")

    for split, rep in report.items():
        if "interaction_3seed" not in rep:
            print(f"\n{split}: {rep.get('status')}")
            continue
        print(f"\n=== {split} — stable-vs-static robustness (3-seed, underpowered by design) ===")
        for metric, ix in rep["interaction_3seed"].items():
            verdict = ("stable more robust" if ix["sign_consistent"] and ix["advantage_mean"] > 0
                       else "static more robust" if ix["sign_consistent"] and ix["advantage_mean"] < 0
                       else "NEAR-NULL / sign-flipping")
            print(f"  {metric}: advantage {ix['advantage_mean']:+.3f} "
                  f"({ix['n_seeds_stable_more_robust']}/3 seeds favor stable), "
                  f"sign_consistent={ix['sign_consistent']} -> {verdict}")


if __name__ == "__main__":
    main()
