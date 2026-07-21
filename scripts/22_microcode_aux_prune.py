#!/usr/bin/env python
"""S2/I8: MicroCode aux-channel prune probe (CPU, no GPU, no model).

Decides which of MicroCode's 11 to_components aux channels are LOAD-BEARING
(non-zero within-group variance + not perfectly collinear with another channel)
vs DEAD SCAFFOLD, to finalize the teacher's weighted aux-key set. The spec
(PAPER2_MICROCODE_TESTBED_SPEC.md step 5) mandates this correlation/variance
prune before locking the reward budget.

Method: generate GRPO-like groups of 4 across all tiers (train + ood), each a
realistic mix of correct / partial / visible-hardcode / crash / garbage
candidates a 0.5B might emit; compute per-key within-group std (the GRPO
advantage signal) + a cross-key correlation matrix over all candidate rows.

Finding (2026-07-20): six legality-scaffold channels
(has_extractable_answer_span, format, syntax_parses, defines_target_signature,
imports_safe, valid_expression) are PERFECTLY COLLINEAR (identical std 0.155,
mean 0.898) — keep only valid_expression (the curriculum gate). brevity is DEAD
(std 0.000, always 1.0). Load-bearing dense channels: runs_without_error,
visible_pass_rate (the proxy), no_hardcoding_heuristic, and held_out_pass_rate
(the truth — diagnostic only, never a training wheel). => MICRO_AUX_KEYS =
[valid_expression, runs_without_error, visible_pass_rate, no_hardcoding_heuristic].
"""
from __future__ import annotations

import argparse
import inspect
import json
import random
import statistics

from rtw_llm.microcode import verify_completion
from rtw_llm.microcode_gen import TEMPLATES, difficulty_spec, random_solvable_task

AUX = [
    "has_extractable_answer_span", "format", "syntax_parses",
    "defines_target_signature", "imports_safe", "valid_expression",
    "runs_without_error", "visible_pass_rate", "held_out_pass_rate",
    "no_hardcoding_heuristic", "brevity",
]


def _ref_src(task):
    tmpl = next(t for t in TEMPLATES if t.key == task["template"])
    return inspect.getsource(tmpl.reference).replace(
        f"def {tmpl.reference.__name__}(", f"def {task['fn_name']}(", 1)


def _hack_src(task):
    fn, vis = task["fn_name"], task["visible_tests"]
    br = "\n".join(f"    if list(args)=={list(a)}:\n        return {e!r}" for a, e in vis)
    return f"def {fn}(*args):\n{br}\n    return None"


def _partial_src(task):
    fn = task["fn_name"]
    _, e0 = task["visible_tests"][0]
    return f"def {fn}(*args):\n    return {e0!r}"


def _crash_src(task):
    return f"def {task['fn_name']}(*args):\n    return undefined_name"


def _garbage(task):
    return "not code at all"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups_per_tier", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/microcode_aux_prune.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    per_key_group_stds = {k: [] for k in AUX}
    rows = []
    tiers = ["easy", "medium", "hard", "ood_compose", "ood_transform"]
    builders = [_ref_src, _partial_src, _hack_src, _crash_src, _garbage]
    weights = [0.25, 0.30, 0.15, 0.20, 0.10]
    for tier in tiers:
        spec = difficulty_spec(tier)
        for i in range(args.groups_per_tier):
            task = random_solvable_task(rng, spec, i, tier)
            comps = []
            for b in rng.choices(builders, weights=weights, k=4):
                src = b(task)
                comp = f"<answer>\n{src}\n</answer>" if b is not _garbage else src
                c = verify_completion(comp, task).to_components()
                comps.append(c)
                rows.append(c)
            for k in AUX:
                vals = [c[k] for c in comps]
                per_key_group_stds[k].append(statistics.pstdev(vals))

    summary = {}
    for k in AUX:
        stds = per_key_group_stds[k]
        summary[k] = {
            "mean_within_group_std": sum(stds) / len(stds),
            "overall_mean": sum(c[k] for c in rows) / len(rows),
            "frac_groups_with_variance": sum(1 for s in stds if s > 1e-9) / len(stds),
        }

    # collinearity: which channels are identical to valid_expression on every row
    collinear = [k for k in AUX if k != "valid_expression"
                 and all(r[k] == r["valid_expression"] for r in rows)]

    load_bearing = [k for k in AUX
                    if summary[k]["mean_within_group_std"] > 0.01 and k not in collinear]

    out = {
        "n_rows": len(rows), "tiers": tiers,
        "per_key": summary,
        "collinear_with_valid_expression": collinear,
        "load_bearing_channels": load_bearing,
        "recommended_MICRO_AUX_KEYS": [
            "valid_expression", "runs_without_error",
            "visible_pass_rate", "no_hardcoding_heuristic",
        ],
        "note": "held_out_pass_rate is the source-of-truth/diagnostic; it is the "
                "primary via 'correct' and must NOT be a weighted training wheel.",
    }
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: summary[k] for k in AUX}, indent=2))
    print("collinear with valid_expression:", collinear)
    print("recommended MICRO_AUX_KEYS:", out["recommended_MICRO_AUX_KEYS"])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
