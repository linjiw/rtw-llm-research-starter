#!/usr/bin/env python
"""MicroCode best-of-N eval harness (Paper-2, I10-b).

The MicroCode analogue of scripts/07_best_of_n_rerank.py. 07 is on the FROZEN
list and its selector/metric keys are Countdown-bound, so this script IMPORTS
only 07's provably task-agnostic helpers (parse_n_values, load_task_ids,
select_examples, sampling_identity, is_complete, cost_per_exact,
count_completion_tokens) and re-implements locally the Countdown-global-bound
pieces (selected-metric keys, selectors, summarize/evaluate).

Frozen protocol: docs/PAPER2_FROZEN_PROTOCOL.md — temp 0.7 / top_p 0.95 /
max_new_tokens 256 / sampling seed 0 / N in {1,4,8} / prompt_high; selector =
rtw_llm.microcode.microcode_practical_score (never reads held-out truth);
verifier = the sole source of correctness, rescoring every candidate.

Sandbox: --sandbox worker (default) grades candidates in the hardened spawned
worker (memory wall + crash isolation) — required for GPU runs where model
code executes; --sandbox inprocess is the byte-identical CPU/repro path.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import time
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from rtw_llm.data import read_jsonl, write_jsonl
from rtw_llm.engine import GenerationConfigLite, HFEngine
from rtw_llm.microcode import microcode_practical_score, verify_completion

# Import the task-agnostic helpers from the frozen 07 without editing it.
_SPEC = importlib.util.spec_from_file_location(
    "best_of_n_rerank", Path(__file__).resolve().parent / "07_best_of_n_rerank.py"
)
_BON = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BON)
parse_n_values = _BON.parse_n_values
select_examples = _BON.select_examples
sampling_identity = _BON.sampling_identity
is_complete = _BON.is_complete
cost_per_exact = _BON.cost_per_exact
count_completion_tokens = _BON.count_completion_tokens

# MicroCode metric keys reported for the selected candidate (local, NOT 07's).
MICRO_SELECTED_METRIC_KEYS = [
    "valid_expression",
    "runs_without_error",
    "visible_pass_rate",
    "no_hardcoding_heuristic",
    "held_out_pass_rate",
    "exact_correct",
]

EXAMPLE_FIELDS = ("fn_name", "visible_tests", "held_out_tests")


def microcode_metrics_for_completion(
    completion: str, example: dict[str, Any], sandbox: str
) -> dict[str, Any]:
    result = verify_completion(completion, example, sandbox=sandbox)
    comps = result.to_components()
    return {**comps, "expression": result.expression, "error": result.error}


def choose_practical(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return max(candidates, key=lambda row: (row["practical_score"], -row["candidate_index"]))


def choose_oracle(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    # Oracle = held-out truth (upper bound), tie-broken by the practical score.
    return max(
        candidates,
        key=lambda row: (
            float(row["metrics"].get("exact_correct", 0.0)),
            row["practical_score"],
            -row["candidate_index"],
        ),
    )


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        key: float(sum(float(r["metrics"].get(key, 0.0)) for r in rows) / len(rows))
        for key in MICRO_SELECTED_METRIC_KEYS
    }


def evaluate_candidates(
    rows_by_id: dict[str, list[dict[str, Any]]],
    n_values: list[int],
    *,
    wall_clock_seconds: float = 0.0,
    max_n: int | None = None,
) -> dict[str, Any]:
    max_n = max_n or max(n_values)
    report: dict[str, Any] = {"n_examples": len(rows_by_id), "n_values": n_values, "by_n": {}}
    for n in n_values:
        practical_rows, oracle_rows, tokens_at_n = [], [], 0
        for task_id, candidates in rows_by_id.items():
            if len(candidates) < n:
                raise ValueError(f"Task {task_id} has {len(candidates)} candidates; need {n}")
            prefix = candidates[:n]
            tokens_at_n += sum(int(r.get("completion_token_count", 0)) for r in prefix)
            practical_rows.append(choose_practical(prefix))
            oracle_rows.append(choose_oracle(prefix))
        practical_summary = summarize_rows(practical_rows)
        oracle_summary = summarize_rows(oracle_rows)
        # Headline proxy-primary gap on the PRACTICAL selection (what deployment
        # would ship): visible - held_out; large positive = the selector was
        # gamed by visible-test overfitting.
        proxy_primary_gap = float(
            practical_summary.get("visible_pass_rate", 0.0)
            - practical_summary.get("held_out_pass_rate", 0.0)
        )
        report["by_n"][str(n)] = {
            "samples_per_task": n,
            "tokens_generated": int(tokens_at_n),
            "practical_selected": practical_summary,
            "oracle_selected": oracle_summary,
            "oracle_exact_at_n": oracle_summary.get("exact_correct", 0.0),
            "reranked_exact_at_n": practical_summary.get("exact_correct", 0.0),
            "proxy_primary_gap_practical": proxy_primary_gap,
            "cost_per_oracle_exact": cost_per_exact(n, oracle_summary.get("exact_correct", 0.0)),
            "cost_per_reranked_exact": cost_per_exact(
                n, practical_summary.get("exact_correct", 0.0)
            ),
        }
    report["wall_clock_seconds"] = wall_clock_seconds
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--data_path", default="data/microcode/validation.jsonl")
    parser.add_argument("--task_ids_file", default=None)
    parser.add_argument("--output_dir", default="outputs/microcode_bestofn")
    parser.add_argument("--hf_gen_mode", choices=["loop", "batched"], default="loop")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--prompt_field", default="prompt_high")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_values", nargs="+", default=["1,4,8"])
    parser.add_argument("--max_n", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument(
        "--sandbox",
        choices=["worker", "inprocess"],
        default="worker",
        help="worker = hardened spawned grader (REQUIRED for GPU runs); "
        "inprocess = byte-identical CPU/repro path",
    )
    parser.add_argument("--skip_if_complete", action="store_true")
    args = parser.parse_args()

    n_values = parse_n_values(args.n_values)
    max_n = args.max_n or max(n_values)
    if max(n_values) > max_n:
        raise ValueError(f"max --n_values {max(n_values)} exceeds --max_n {max_n}")

    random.seed(args.seed)
    try:
        import torch
        from transformers import set_seed

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        set_seed(args.seed)
    except Exception:
        pass

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_examples = read_jsonl(args.data_path)
    examples = select_examples(all_examples, limit=args.limit, task_ids_file=args.task_ids_file)
    requested_identity = sampling_identity(
        {
            "hf_gen_mode": args.hf_gen_mode,
            "model_name": args.model_name,
            "model_revision": args.model_revision,
            "adapter_path": args.adapter_path,
            "sampling_seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "prompt_field": args.prompt_field,
        }
    )
    if args.skip_if_complete and is_complete(
        output_dir, max_n=max_n, n_examples=len(examples), requested_identity=requested_identity
    ):
        print(json.dumps({"status": "skipped_complete", "output_dir": str(output_dir)}, indent=2))
        return

    engine = HFEngine(
        args.model_name,
        args.adapter_path,
        model_revision=args.model_revision,
        device=args.device,
        gen_mode=args.hf_gen_mode,
    )
    gen_config = GenerationConfigLite(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.temperature > 0,
    )

    started_at = time.time()
    rows: list[dict[str, Any]] = []
    rows_by_id: dict[str, list[dict[str, Any]]] = {}
    expanded: list[tuple[dict[str, Any], int]] = []
    for ex in examples:
        rows_by_id[ex["id"]] = []
        for candidate_index in range(max_n):
            expanded.append((ex, candidate_index))

    for start in tqdm(range(0, len(expanded), args.batch_size)):
        batch = expanded[start : start + args.batch_size]
        prompts = [ex[args.prompt_field] for ex, _ in batch]
        completions = engine.generate(prompts, gen_config)
        generation_metadata = (
            engine.last_generation_metadata()
            if hasattr(engine, "last_generation_metadata")
            else [{} for _ in completions]
        )
        for (ex, candidate_index), completion, generation_meta in zip(
            batch, completions, generation_metadata
        ):
            metrics = microcode_metrics_for_completion(completion, ex, args.sandbox)
            exact_token_count = generation_meta.get("generated_token_count")
            row = {
                "id": ex["id"],
                "difficulty": ex["difficulty"],
                "rung": ex.get("rung"),
                "template": ex.get("template"),
                "prompt_field": args.prompt_field,
                "candidate_index": candidate_index,
                "completion": completion,
                "completion_token_count": (
                    int(exact_token_count)
                    if exact_token_count is not None
                    else count_completion_tokens(engine, completion)
                ),
                "finish_reason": generation_meta.get("finish_reason"),
                "metrics": metrics,
                "practical_score": microcode_practical_score(metrics),
            }
            rows.append(row)
            rows_by_id[ex["id"]].append(row)

    for candidates in rows_by_id.values():
        candidates.sort(key=lambda row: row["candidate_index"])

    wall_clock_seconds = time.time() - started_at
    report = evaluate_candidates(
        rows_by_id, n_values, wall_clock_seconds=wall_clock_seconds, max_n=max_n
    )
    report.update(
        {
            "task": "microcode",
            "model_name": args.model_name,
            "model_revision": args.model_revision,
            "adapter_path": args.adapter_path,
            "data_path": args.data_path,
            "task_ids_file": args.task_ids_file,
            "prompt_field": args.prompt_field,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "method": args.method,
            "split": args.split,
            "max_n": max_n,
            "hf_gen_mode": args.hf_gen_mode,
            "sandbox": args.sandbox,
            "total_candidates": len(rows),
            "total_tokens_generated": sum(int(r["completion_token_count"]) for r in rows),
        }
    )

    write_jsonl(output_dir / "candidates.jsonl", rows)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    # is_complete() recomputes sampling_identity() from run_config.json's TOP-
    # LEVEL keys, so the identity fields must live at top level (mirrors 07).
    run_config = {
        **requested_identity,
        "batch_size": args.batch_size,
        "n_values": n_values,
        "max_n": max_n,
        "n_examples": len(examples),
        "sandbox": args.sandbox,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    flat_rows = []
    for n, result in report["by_n"].items():
        for selector in ["practical_selected", "oracle_selected"]:
            flat_rows.append({"n": int(n), "selector": selector, **result[selector]})
    pd.DataFrame(flat_rows).to_csv(output_dir / "summary.csv", index=False)
    print(json.dumps({k: report[k] for k in ("by_n", "total_candidates")}, indent=2))


if __name__ == "__main__":
    main()
