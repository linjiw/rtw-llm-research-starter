#!/usr/bin/env python
"""Probe A: tier-scoped paired comparison surface for best-of-N method claims.

Bottleneck diagnosis found exactness is essentially easy-tier-only, and the
tier-balanced eval splits dilute easy-tier gains by ~2/3. This script builds
the honest per-tier comparison so we can see whether restricting to easy tasks
turns any stable-vs-static verdict from "noise" into "signal", or whether it
merely concentrates the effect magnitude while discordant pairs stay too small.

ADDITIVE analysis only. Reuses the verifier-derived exactness and the selector
+ McNemar machinery from scripts/08_summarize_v09_seed_expansion.py; nothing is
reimplemented here. CPU-only (reads committed best-of-N banks, no CUDA).

Outputs:
  - prints all tables to stdout
  - writes outputs/probe_a_tier_scoped.json
"""
from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

# ----------------------------------------------------------------------------
# Reuse selector + McNemar logic from scripts/08 (module name starts with a
# digit, so load it by path rather than a normal import).
# ----------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "summ08", REPO / "scripts" / "08_summarize_v09_seed_expansion.py"
)
_summ08 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_summ08)  # type: ignore[union-attr]

selected_exact_by_task = _summ08.selected_exact_by_task  # (path, n, selector) -> {id: bool}
exact_mcnemar_p = _summ08.exact_mcnemar_p  # (stable_only, static_only) -> two-sided exact p

BANKS = REPO / "outputs" / "bestofn"
N = 8
TIERS = ["easy", "medium", "hard"]
SPLITS = ["validation", "test_in_dist"]
SEEDS = [0, 1, 2]
# Tier subsets for the paired McNemar power analysis.
TIER_SUBSETS = {
    "easy_only": {"easy"},
    "easy_medium": {"easy", "medium"},
    "all_tiers": {"easy", "medium", "hard"},
}


def bank_dir(method: str, seed: int, split: str) -> Path:
    return BANKS / f"{method}_local_seed{seed}_{split}_limit50_n8"


def candidates_path(method: str, seed: int, split: str) -> Path:
    return bank_dir(method, seed, split) / "candidates.jsonl"


def task_difficulty(cand_path: Path) -> dict[str, str]:
    """Map task id -> difficulty tier (one entry per task, from its candidates)."""
    out: dict[str, str] = {}
    with cand_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            out[row["id"]] = row["difficulty"]
    return out


def rate(bools: list[bool]) -> float | None:
    return (sum(1 for b in bools if b) / len(bools)) if bools else None


# ----------------------------------------------------------------------------
# 1. Per-tier reranked_exact@8 and oracle_exact@8, per seed + pooled, both splits.
# ----------------------------------------------------------------------------
def per_tier_rates() -> dict[str, Any]:
    """Return per (method, split, seed, tier) and pooled-across-seeds rates."""
    per_seed: list[dict[str, Any]] = []
    # pooled accumulators keyed by (method, split, tier, selector) -> list[bool]
    pooled_bools: dict[tuple[str, str, str, str], list[bool]] = defaultdict(list)

    for method in ["static", "stable"]:
        for split in SPLITS:
            for seed in SEEDS:
                cp = candidates_path(method, seed, split)
                if not cp.exists():
                    continue
                diff = task_difficulty(cp)
                rer = selected_exact_by_task(cp, N, "practical")
                orc = selected_exact_by_task(cp, N, "oracle")
                for tier in TIERS:
                    ids = [tid for tid in diff if diff[tid] == tier]
                    rer_bools = [rer[t] for t in ids]
                    orc_bools = [orc[t] for t in ids]
                    per_seed.append(
                        {
                            "method": method,
                            "split": split,
                            "seed": seed,
                            "tier": tier,
                            "n_tasks": len(ids),
                            "reranked_exact_at8": rate(rer_bools),
                            "oracle_exact_at8": rate(orc_bools),
                        }
                    )
                    pooled_bools[(method, split, tier, "reranked")].extend(rer_bools)
                    pooled_bools[(method, split, tier, "oracle")].extend(orc_bools)

    pooled: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for (method, split, tier, _sel) in pooled_bools:
        key = (method, split, tier)
        if key in seen:
            continue
        seen.add(key)
        rer_bools = pooled_bools[(method, split, tier, "reranked")]
        orc_bools = pooled_bools[(method, split, tier, "oracle")]
        pooled.append(
            {
                "method": method,
                "split": split,
                "tier": tier,
                "n_task_instances": len(rer_bools),
                "reranked_exact_at8": rate(rer_bools),
                "oracle_exact_at8": rate(orc_bools),
            }
        )
    return {"per_seed": per_seed, "pooled_across_seeds": pooled}


# ----------------------------------------------------------------------------
# 2. Per-tier paired McNemar stable-vs-static at N=8, pooled over seeds AND splits.
# ----------------------------------------------------------------------------
def paired_mcnemar_by_subset(selector: str) -> list[dict[str, Any]]:
    """For each tier subset, accumulate paired stable-vs-static discordants over
    every (split, seed, task-in-subset) instance, then compute exact McNemar p."""
    # Accumulate per subset.
    counts: dict[str, dict[str, int]] = {
        name: {"both": 0, "stable_only": 0, "static_only": 0, "neither": 0, "n_pairs": 0}
        for name in TIER_SUBSETS
    }
    for split in SPLITS:
        for seed in SEEDS:
            static_cp = candidates_path("static", seed, split)
            stable_cp = candidates_path("stable", seed, split)
            if not (static_cp.exists() and stable_cp.exists()):
                continue
            diff = task_difficulty(stable_cp)
            static_exact = selected_exact_by_task(static_cp, N, selector)
            stable_exact = selected_exact_by_task(stable_cp, N, selector)
            if list(stable_exact.keys()) != list(static_exact.keys()):
                raise ValueError(f"Task ID/order mismatch split={split} seed={seed}")
            for tid in stable_exact:
                tier = diff[tid]
                s = stable_exact[tid]
                t = static_exact[tid]
                for name, subset in TIER_SUBSETS.items():
                    if tier not in subset:
                        continue
                    c = counts[name]
                    c["n_pairs"] += 1
                    if s and t:
                        c["both"] += 1
                    elif s and not t:
                        c["stable_only"] += 1
                    elif t and not s:
                        c["static_only"] += 1
                    else:
                        c["neither"] += 1
    out: list[dict[str, Any]] = []
    for name in TIER_SUBSETS:
        c = counts[name]
        discordant = c["stable_only"] + c["static_only"]
        n = c["n_pairs"]
        stable_rate = (c["both"] + c["stable_only"]) / n if n else None
        static_rate = (c["both"] + c["static_only"]) / n if n else None
        out.append(
            {
                "tier_subset": name,
                "selector": selector,
                "n_pairs": n,
                "both": c["both"],
                "stable_only": c["stable_only"],
                "static_only": c["static_only"],
                "neither": c["neither"],
                "discordant": discordant,
                "mcnemar_p": exact_mcnemar_p(c["stable_only"], c["static_only"]),
                "stable_rate": stable_rate,
                "static_rate": static_rate,
                "delta_stable_minus_static": (
                    stable_rate - static_rate if stable_rate is not None else None
                ),
            }
        )
    return out


# ----------------------------------------------------------------------------
# 3. Dilution quantification: easy-only vs all-tier pooled rerank@8 rate + effect.
# 4. Effective sample sizes.
# ----------------------------------------------------------------------------
def dilution_and_sample_sizes(pooled_tier: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool reranked rate over BOTH splits + all seeds for easy-only vs all-tiers."""
    # Re-pool from raw booleans (over seeds AND splits) for a single scalar per method.
    method_bools: dict[tuple[str, str], list[bool]] = defaultdict(list)  # (method, scope)->bools
    tier_task_counts: dict[str, int] = defaultdict(int)  # tier -> total task-instances (one method)
    for method in ["static", "stable"]:
        for split in SPLITS:
            for seed in SEEDS:
                cp = candidates_path(method, seed, split)
                if not cp.exists():
                    continue
                diff = task_difficulty(cp)
                rer = selected_exact_by_task(cp, N, "practical")
                for tid, tier in diff.items():
                    method_bools[(method, "all_tiers")].append(rer[tid])
                    if tier == "easy":
                        method_bools[(method, "easy_only")].append(rer[tid])
                    if tier in {"easy", "medium"}:
                        method_bools[(method, "easy_medium")].append(rer[tid])
                    if method == "static":  # count tiers once
                        tier_task_counts[tier] += 1

    def r(method: str, scope: str) -> float | None:
        return rate(method_bools[(method, scope)])

    scopes = ["easy_only", "easy_medium", "all_tiers"]
    dilution = {}
    for scope in scopes:
        st_stable = r("stable", scope)
        st_static = r("static", scope)
        delta = (st_stable - st_static) if (st_stable is not None and st_static is not None) else None
        dilution[scope] = {
            "stable_rerank8": st_stable,
            "static_rerank8": st_static,
            "delta_stable_minus_static": delta,
            "n_task_instances_per_method": len(method_bools[("static", scope)]),
        }
    # Effect magnitude ratio easy-only vs all-tiers.
    d_easy = dilution["easy_only"]["delta_stable_minus_static"]
    d_all = dilution["all_tiers"]["delta_stable_minus_static"]
    ratio = None
    if d_all not in (None, 0):
        ratio = d_easy / d_all if d_easy is not None else None

    # Sample sizes: easy tasks pooled across seeds+splits (the power ceiling denom).
    easy_val_per_seed = tier_task_counts_from("validation", "easy")
    easy_tid_per_seed = tier_task_counts_from("test_in_dist", "easy")
    sample_sizes = {
        "tier_task_instances_pooled_seeds_and_splits": dict(tier_task_counts),
        "easy_per_split_per_seed": {
            "validation": easy_val_per_seed,
            "test_in_dist": easy_tid_per_seed,
        },
        "n_seeds": len(SEEDS),
        "easy_pooled_total": tier_task_counts["easy"],
        "note": (
            "task instances = tasks x seeds x splits. Task IDs are frozen per split, "
            "so pooled counts multiply the per-split easy count by n_seeds."
        ),
    }
    return {
        "dilution": dilution,
        "easy_vs_all_effect_ratio": ratio,
        "sample_sizes": sample_sizes,
    }


def tier_task_counts_from(split: str, tier: str) -> int:
    """Number of tasks of `tier` in one bank of `split` (seed0 static as reference)."""
    cp = candidates_path("static", 0, split)
    diff = task_difficulty(cp)
    return sum(1 for d in diff.values() if d == tier)


# ----------------------------------------------------------------------------
# Optional context: base + v10c2 seed0 per-tier rerank@8 (seed0-only, not part of
# the stable-vs-static claim, printed as context only).
# ----------------------------------------------------------------------------
def context_seed0() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for method in ["base", "v10c2"]:
        for split in SPLITS:
            cp = candidates_path(method, 0, split)
            if not cp.exists():
                continue
            diff = task_difficulty(cp)
            rer = selected_exact_by_task(cp, N, "practical")
            orc = selected_exact_by_task(cp, N, "oracle")
            for tier in TIERS:
                ids = [t for t in diff if diff[t] == tier]
                out.append(
                    {
                        "method": method,
                        "split": split,
                        "seed": 0,
                        "tier": tier,
                        "n_tasks": len(ids),
                        "reranked_exact_at8": rate([rer[t] for t in ids]),
                        "oracle_exact_at8": rate([orc[t] for t in ids]),
                    }
                )
    return out


# ----------------------------------------------------------------------------
# Printing helpers
# ----------------------------------------------------------------------------
def fmt(x: float | None, pct: bool = True) -> str:
    if x is None:
        return "   -  "
    return f"{x * 100:5.1f}%" if pct else f"{x:.3f}"


def print_per_tier(per_tier: dict[str, Any]) -> None:
    print("\n" + "=" * 84)
    print("1. PER-TIER reranked_exact@8 / oracle_exact@8  (stable vs static, N=8)")
    print("=" * 84)
    for split in SPLITS:
        print(f"\n[{split}]  per-seed then POOLED-across-seeds (0/1/2)")
        print(
            f"{'method':7} {'seed':5} {'tier':7} {'n':>4}  "
            f"{'rerank@8':>9}  {'oracle@8':>9}"
        )
        # per-seed
        rows = [r for r in per_tier["per_seed"] if r["split"] == split]
        for method in ["static", "stable"]:
            for seed in SEEDS:
                for tier in TIERS:
                    r = next(
                        (
                            x
                            for x in rows
                            if x["method"] == method and x["seed"] == seed and x["tier"] == tier
                        ),
                        None,
                    )
                    if r is None:
                        continue
                    print(
                        f"{method:7} {seed:<5} {tier:7} {r['n_tasks']:>4}  "
                        f"{fmt(r['reranked_exact_at8']):>9}  {fmt(r['oracle_exact_at8']):>9}"
                    )
        print("  -- pooled across seeds --")
        prows = [r for r in per_tier["pooled_across_seeds"] if r["split"] == split]
        for method in ["static", "stable"]:
            for tier in TIERS:
                r = next(
                    (x for x in prows if x["method"] == method and x["tier"] == tier), None
                )
                if r is None:
                    continue
                print(
                    f"{method:7} {'POOL':5} {tier:7} {r['n_task_instances']:>4}  "
                    f"{fmt(r['reranked_exact_at8']):>9}  {fmt(r['oracle_exact_at8']):>9}"
                )


def print_mcnemar(mcnemar: list[dict[str, Any]], selector: str) -> None:
    print("\n" + "=" * 84)
    print(
        f"2. PAIRED McNEMAR stable-vs-static, N=8, selector={selector}  "
        "(POOLED over seeds 0/1/2 AND both splits)"
    )
    print("=" * 84)
    print(
        f"{'subset':12} {'pairs':>5} {'both':>5} {'stbl-only':>9} {'stat-only':>9} "
        f"{'neither':>7} {'discord':>7} {'mcnemar_p':>10} {'delta':>8}"
    )
    for r in mcnemar:
        print(
            f"{r['tier_subset']:12} {r['n_pairs']:>5} {r['both']:>5} "
            f"{r['stable_only']:>9} {r['static_only']:>9} {r['neither']:>7} "
            f"{r['discordant']:>7} {r['mcnemar_p']:>10.4f} "
            f"{fmt(r['delta_stable_minus_static']):>8}"
        )


def print_dilution(dz: dict[str, Any]) -> None:
    print("\n" + "=" * 84)
    print("3. DILUTION: easy-only vs all-tier pooled rerank@8 (over seeds+splits)")
    print("=" * 84)
    d = dz["dilution"]
    print(f"{'scope':12} {'n/method':>9} {'stable':>8} {'static':>8} {'delta(st-stat)':>15}")
    for scope in ["easy_only", "easy_medium", "all_tiers"]:
        row = d[scope]
        print(
            f"{scope:12} {row['n_task_instances_per_method']:>9} "
            f"{fmt(row['stable_rerank8']):>8} {fmt(row['static_rerank8']):>8} "
            f"{fmt(row['delta_stable_minus_static']):>15}"
        )
    ratio = dz["easy_vs_all_effect_ratio"]
    print(
        f"\n  easy-only effect / all-tier effect (stable-static delta) = "
        f"{ratio if ratio is None else round(ratio, 2)}"
    )
    print("\n" + "=" * 84)
    print("4. EFFECTIVE SAMPLE SIZES (power ceiling)")
    print("=" * 84)
    ss = dz["sample_sizes"]
    print(f"  tier task-instances pooled (seeds x splits): {ss['tier_task_instances_pooled_seeds_and_splits']}")
    print(f"  easy per split per seed: {ss['easy_per_split_per_seed']}  x {ss['n_seeds']} seeds")
    print(f"  easy pooled total (paired-comparison denom): {ss['easy_pooled_total']}")


def main() -> None:
    per_tier = per_tier_rates()
    mcnemar_rerank = paired_mcnemar_by_subset("practical")
    mcnemar_oracle = paired_mcnemar_by_subset("oracle")
    dz = dilution_and_sample_sizes(per_tier["pooled_across_seeds"])
    ctx = context_seed0()

    print_per_tier(per_tier)
    print_mcnemar(mcnemar_rerank, "practical(reranked)")
    print_mcnemar(mcnemar_oracle, "oracle")
    print_dilution(dz)

    # Verdict (item 5): does easy-scoping create signal?
    easy = next(r for r in mcnemar_rerank if r["tier_subset"] == "easy_only")
    alltier = next(r for r in mcnemar_rerank if r["tier_subset"] == "all_tiers")
    print("\n" + "=" * 84)
    print("5. VERDICT: does easy-scoping turn stable-vs-static from noise into signal?")
    print("=" * 84)
    signal = easy["mcnemar_p"] < 0.05
    print(
        f"  easy-only reranked McNemar: discordant={easy['discordant']} "
        f"(stable_only={easy['stable_only']}, static_only={easy['static_only']}), "
        f"p={easy['mcnemar_p']:.4f}"
    )
    print(
        f"  all-tier reranked McNemar: discordant={alltier['discordant']} "
        f"(stable_only={alltier['stable_only']}, static_only={alltier['static_only']}), "
        f"p={alltier['mcnemar_p']:.4f}"
    )
    print(
        f"  --> easy-scoping creates statistical signal (p<0.05)? "
        f"{'YES' if signal else 'NO'}"
    )

    payload = {
        "config": {
            "N": N,
            "seeds": SEEDS,
            "splits": SPLITS,
            "methods": ["static", "stable"],
            "selector_primary": "practical (reranked_exact@8)",
            "tier_subsets": {k: sorted(v) for k, v in TIER_SUBSETS.items()},
        },
        "per_tier_rates": per_tier,
        "paired_mcnemar_reranked": mcnemar_rerank,
        "paired_mcnemar_oracle": mcnemar_oracle,
        "dilution_and_sample_sizes": dz,
        "context_seed0_base_v10c2": ctx,
        "verdict": {
            "easy_only_reranked_mcnemar_p": easy["mcnemar_p"],
            "easy_only_discordant": easy["discordant"],
            "all_tier_reranked_mcnemar_p": alltier["mcnemar_p"],
            "all_tier_discordant": alltier["discordant"],
            "easy_scoping_creates_signal_p_lt_0p05": signal,
        },
    }
    out_path = REPO / "outputs" / "probe_a_tier_scoped.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
