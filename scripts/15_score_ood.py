#!/usr/bin/env python
"""Score the OOD eval banks per HARNESS_OOD_ANALYSIS_CONTRACT.md (rank-3 + 3b).

Distribution-shift read on test_ood_division (5-number, novel '/') and
test_ood_long (6-number). exact is EXPECTED-FLOOR (5-6-number tasks are below
the value-search wall); the informative signals are the LEGALITY panel,
'/'-ADOPTION (division only), and TRUNCATION — all read RELATIVE TO THE BASE
ARM (the base has seen '/'; the question is whether RL narrowed a known
operator). The v13sft arm tests whether SFT-taught legality CAPABILITY
transfers OOD or overfit the 3-5-number/4-op training envelope.

CPU-only, read-only over the ood_* banks. Run after run_ood_eval.sh lands them.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

# method dir prefix -> label; base is the mandatory reference.
BANK_GLOB = "outputs/bestofn/ood_{arm}_{split}_limit50_n8/candidates.jsonl"


def load_bank(path: str) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in open(path)]


def uses_division(expr: str | None) -> bool:
    return bool(expr) and "/" in expr


def arm_stats(rows: list[dict]) -> dict:
    if not rows:
        return {"present": False}
    n = len(rows)
    legal = [r for r in rows if float(r["metrics"].get("valid_expression", 0)) > 0]
    exact = [r for r in rows if float(r["metrics"].get("exact_correct", 0)) > 0]
    # '/'-adoption: fraction of candidates whose extracted expression uses '/'
    div = sum(1 for r in rows if uses_division(r.get("extracted_expression")))
    # truncation proxy: candidates at/over the 256-token cap
    capped = sum(1 for r in rows if r.get("completion_token_count", 0) >= 256)
    by_id: dict[str, list] = defaultdict(list)
    for r in rows:
        by_id[r["id"]].append(r)
    oracle8 = sum(
        1 for cs in by_id.values()
        if any(float(c["metrics"].get("exact_correct", 0)) > 0 for c in sorted(cs, key=lambda x: x["candidate_index"])[:8])
    )
    return {
        "present": True,
        "n_cand": n,
        "n_tasks": len(by_id),
        "legal_rate": len(legal) / n,
        "number_f1_mean": mean(float(r["metrics"].get("number_multiset_f1", 0)) for r in rows),
        "p_exact_given_legal": (len(exact) / len(legal)) if legal else 0.0,
        "exact_cand_rate": len(exact) / n,
        "oracle_at_8": oracle8,
        "div_adoption_rate": div / n,
        "truncation_rate": capped / n,
        "mean_tokens": mean(r.get("completion_token_count", 0) for r in rows),
    }


def discover_arms(split: str) -> dict[str, str]:
    """Find every ood_<arm>_<split> bank on disk. arm includes base/static_seedX/
    stable_seedX/v13sft_seed0."""
    arms = {}
    for path in glob.glob(f"outputs/bestofn/ood_*_{split}_limit50_n8/candidates.jsonl"):
        m = re.search(rf"ood_(.+?)_{re.escape(split)}_limit50_n8", path)
        if m:
            arms[m.group(1)] = path
    return arms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["test_ood_division", "test_ood_long"])
    ap.add_argument("--out_json", default="outputs/ood_scored.json")
    args = ap.parse_args()

    report: dict = {}
    for split in args.splits:
        arms = discover_arms(split)
        if not arms:
            report[split] = {"status": "no OOD banks yet"}
            continue
        stats = {arm: arm_stats(load_bank(path)) for arm, path in sorted(arms.items())}
        base = stats.get("base", {})
        # relative-to-base deltas for the trained arms (the contract's core read)
        rel = {}
        for arm, s in stats.items():
            if arm == "base" or not s.get("present") or not base.get("present"):
                continue
            rel[arm] = {
                "legal_rate_vs_base": s["legal_rate"] - base["legal_rate"],
                "div_adoption_vs_base": s["div_adoption_rate"] - base["div_adoption_rate"],
                "oracle_at_8_vs_base": s["oracle_at_8"] - base["oracle_at_8"],
            }
        report[split] = {"arms": stats, "relative_to_base": rel}

    print(json.dumps(report, indent=2))
    Path(args.out_json).write_text(json.dumps(report, indent=2) + "\n")

    # Human-readable summary of the contract's key reads.
    for split, rep in report.items():
        if "arms" not in rep:
            print(f"\n{split}: {rep.get('status')}")
            continue
        print(f"\n=== {split} ===")
        print(f"{'arm':20} {'legal':>6} {'numF1':>6} {'P(ex|lg)':>8} {'oracle@8':>8} {'/-adopt':>8} {'trunc':>6}")
        for arm, s in rep["arms"].items():
            if not s.get("present"):
                continue
            print(f"{arm:20} {s['legal_rate']:6.2f} {s['number_f1_mean']:6.2f} "
                  f"{s['p_exact_given_legal']:8.2f} {s['oracle_at_8']:8d} "
                  f"{s['div_adoption_rate']:8.2f} {s['truncation_rate']:6.2f}")


if __name__ == "__main__":
    main()
