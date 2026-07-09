#!/usr/bin/env python
"""Score the v0.13 SFT pilot with the memorization control (docs/V13_...PLAN.md).

Partitions the frozen-50 eval set into OVERLAP tasks (share (numbers,target)
with SFT train data => memorization risk) vs HELD-OUT tasks (genuine transfer),
and reports candidate-level legality + P(exact|legal) + oracle/reranked@8 for
each partition, against the stable 3-SEED distribution (not a single seed).

CPU-only, additive, read-only over committed banks. Run after the pilot's
best-of-N banks exist (v13sft_* and v13sftonly_*).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

BANK = "outputs/bestofn/{method}_seed0_{split}_limit50_n8/candidates.jsonl"
STABLE_BANK = "outputs/bestofn/stable_local_seed{s}_{split}_limit50_n8/candidates.jsonl"
TRAIN = "data/countdown/train.jsonl"
DATA = "data/countdown/{split}.jsonl"
SPLITS = ["validation", "test_in_dist"]


def load_bank(path: str) -> dict[str, list[dict]]:
    if not Path(path).exists():
        return {}
    groups: dict[str, list[dict]] = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        groups[r["id"]].append(r)
    for g in groups.values():
        g.sort(key=lambda r: r["candidate_index"])
    return dict(groups)


def overlap_ids(split: str) -> set[str]:
    train_probs = {
        (tuple(sorted(r["numbers"])), r["target"])
        for r in (json.loads(line) for line in open(TRAIN))
    }
    ex = {r["id"]: r for r in (json.loads(line) for line in open(DATA.format(split=split)))}
    return {
        i for i, r in ex.items()
        if (tuple(sorted(r["numbers"])), r["target"]) in train_probs
    }


def cand_stats(groups: dict[str, list[dict]], ids: set[str]) -> dict:
    cands = [c for i in ids if i in groups for c in groups[i]]
    if not cands:
        return {"n_tasks": 0, "n_cand": 0}
    legal = [c for c in cands if float(c["metrics"].get("valid_expression", 0)) > 0]
    exact = [c for c in cands if float(c["metrics"].get("exact_correct", 0)) > 0]
    return {
        "n_tasks": len([i for i in ids if i in groups]),
        "n_cand": len(cands),
        "legal_rate": len(legal) / len(cands),
        "exact_cand_rate": len(exact) / len(cands),
        "p_exact_given_legal": (len(exact) / len(legal)) if legal else 0.0,
        "mean_tokens": mean(c["completion_token_count"] for c in cands),
        "hack_rate": mean(float(c["metrics"].get("reward_hacking_candidate", 0)) for c in cands),
    }


def oracle_at_8(groups: dict[str, list[dict]], ids: set[str]) -> int:
    return sum(
        1 for i in ids if i in groups
        and any(float(c["metrics"].get("exact_correct", 0)) > 0 for c in groups[i][:8])
    )


def distinct_legal_exprs(groups: dict[str, list[dict]], ids: set[str]) -> float:
    counts = []
    for i in ids:
        if i not in groups:
            continue
        legal = {
            c.get("extracted_expression")
            for c in groups[i][:8]
            if float(c["metrics"].get("valid_expression", 0)) > 0 and c.get("extracted_expression")
        }
        counts.append(len(legal))
    return mean(counts) if counts else 0.0


def main() -> None:
    report: dict = {}
    for split in SPLITS:
        ov = overlap_ids(split)
        v13 = load_bank(BANK.format(method="v13sft", split=split))
        v13only = load_bank(BANK.format(method="v13sftonly", split=split))
        stable = {s: load_bank(STABLE_BANK.format(s=s, split=split)) for s in [0, 1, 2]}
        stable = {s: g for s, g in stable.items() if g}
        if not v13:
            report[split] = {"status": "v13 bank not present yet"}
            continue
        all_ids = set(v13)
        held = all_ids - ov
        parts = {"overlap": all_ids & ov, "held_out": held, "all": all_ids}

        split_rep: dict = {"n_overlap": len(all_ids & ov), "n_held_out": len(held)}
        for pname, pids in parts.items():
            v13s = cand_stats(v13, pids)
            v13s["oracle@8"] = oracle_at_8(v13, pids)
            v13s["distinct_legal@8"] = distinct_legal_exprs(v13, pids)
            entry = {"v13": v13s}
            if v13only:
                so = cand_stats(v13only, pids)
                so["oracle@8"] = oracle_at_8(v13only, pids)
                entry["v13_sft_only"] = so
            # stable 3-seed distribution on the same partition
            st_leg = [cand_stats(stable[s], pids)["legal_rate"] for s in stable]
            st_or = [oracle_at_8(stable[s], pids) for s in stable]
            entry["stable_3seed"] = {
                "legal_rate_mean": mean(st_leg), "legal_rate_std": pstdev(st_leg) if len(st_leg) > 1 else 0.0,
                "oracle@8_by_seed": st_or, "oracle@8_mean": mean(st_or),
                "oracle@8_std": pstdev(st_or) if len(st_or) > 1 else 0.0,
            }
            # z-scores of v13 vs stable distribution
            if entry["stable_3seed"]["legal_rate_std"] > 0:
                entry["v13_legal_z"] = (v13s["legal_rate"] - entry["stable_3seed"]["legal_rate_mean"]) / entry["stable_3seed"]["legal_rate_std"]
            if entry["stable_3seed"]["oracle@8_std"] > 0:
                entry["v13_oracle_z"] = (v13s["oracle@8"] - entry["stable_3seed"]["oracle@8_mean"]) / entry["stable_3seed"]["oracle@8_std"]
            split_rep[pname] = entry
        report[split] = split_rep

    print(json.dumps(report, indent=2))
    Path("outputs/v13_sft_scored.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
