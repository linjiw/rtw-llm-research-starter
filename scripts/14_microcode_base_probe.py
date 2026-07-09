#!/usr/bin/env python
"""GPU base-model pass-rate probe for MicroCode (Paper-2 spec step 6, go/no-go).

The ONE thing the CPU mock-variance gate can't answer: does the untrained base
model (Qwen2.5-0.5B-Instruct) show NON-TRIVIAL held_out_pass_rate and NON-ZERO
WITHIN-GROUP std at the easy rungs (R0-R2) in raw few-shot? If everything
collapses to ~0, MicroCode re-creates Countdown's sparsity and the main GRPO
run is not justified (fall back to 1.5B or an SFT format-warmup).

Samples N completions per task at temperature>0 (so within-group variance is
measurable), scores each through the microcode verifier, and prints the
go/no-go verdict. Small (default 20 tasks x 8 samples) — this is a probe, not a
bank. Needs GPU (generation); nothing else in the pipeline should be running.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict

from rtw_llm.engine import GenerationConfigLite, HFEngine
from rtw_llm.microcode import verify_completion
from rtw_llm.microcode_gen import difficulty_spec, random_solvable_task

FEWSHOT = (
    "Example task:\n"
    "def add_one(n):\n    \"\"\"Return `n` plus 1.\"\"\"\n"
    "Answer:\n<answer>\ndef add_one(n):\n    return n + 1\n</answer>\n\n"
)


def build_prompt(task: dict, field: str) -> str:
    return FEWSHOT + task[field] + "\n\nAnswer:\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter_path", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rungs", default="easy", help="tiers to probe: easy / medium / hard")
    ap.add_argument("--n_tasks", type=int, default=20)
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--prompt_field", default="prompt_high")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_json", default="outputs/microcode_base_probe.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tiers = args.rungs.split(",")
    tasks = []
    for tier in tiers:
        spec = difficulty_spec(tier)
        for i in range(args.n_tasks):
            tasks.append(random_solvable_task(rng, spec, i, f"probe_{tier}"))

    engine = HFEngine(args.model_name, args.adapter_path, device=args.device)
    cfg = GenerationConfigLite(
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_p=args.top_p, do_sample=True,
    )

    per_task: dict[str, list[dict]] = defaultdict(list)
    by_rung_held: dict[int, list[float]] = defaultdict(list)
    within_group_std: list[float] = []
    for task in tasks:
        prompt = build_prompt(task, args.prompt_field)
        comps = engine.generate([prompt] * args.n_samples, cfg)
        held_rates = []
        for c in comps:
            comp = c if "<answer>" in c else f"<answer>\n{c}\n</answer>"
            m = verify_completion(comp, task).to_components()
            per_task[task["id"]].append(m)
            held_rates.append(m["held_out_pass_rate"])
        by_rung_held[task["rung"]].extend(held_rates)
        within_group_std.append(statistics.pstdev(held_rates) if len(held_rates) > 1 else 0.0)

    all_m = [m for ms in per_task.values() for m in ms]
    n = len(all_m)
    summary = {
        "model": args.model_name, "adapter": args.adapter_path, "rungs": tiers,
        "n_tasks": len(tasks), "n_samples": args.n_samples, "n_candidates": n,
        "legal_rate": sum(m["valid_expression"] for m in all_m) / n,
        "runs_rate": sum(m["runs_without_error"] for m in all_m) / n,
        "held_out_pass_rate_mean": sum(m["held_out_pass_rate"] for m in all_m) / n,
        "primary_pass_rate": sum(m["correct"] for m in all_m) / n,
        "oracle_at_n": sum(1 for ms in per_task.values() if any(m["correct"] for m in ms)) / len(per_task),
        "mean_within_group_std_held": statistics.mean(within_group_std),
        "frac_groups_with_variance": sum(1 for s in within_group_std if s > 1e-9) / len(within_group_std),
        "held_by_rung": {str(r): statistics.mean(v) for r, v in sorted(by_rung_held.items())},
    }

    # Go/no-go: non-trivial partial credit AND real within-group variance.
    go = (summary["held_out_pass_rate_mean"] > 0.05
          and summary["frac_groups_with_variance"] > 0.30
          and summary["legal_rate"] > 0.20)
    summary["verdict_go"] = go

    print(json.dumps(summary, indent=2))
    print()
    print(f"VERDICT: {'GO — base shows non-trivial pass-rate + within-group variance; MicroCode is not Countdown-sparse' if go else 'NO-GO — base collapses; fall back to 1.5B or SFT format-warmup before a main GRPO run'}")
    with open(args.out_json, "w") as f:
        f.write(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
