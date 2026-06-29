#!/usr/bin/env python
"""Verifier-guided best-of-N candidate selection for Countdown evals."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from rtw_llm.data import read_jsonl, write_jsonl
from rtw_llm.engine import GenerationConfigLite, HFEngine, VLLMEngine
from rtw_llm.rewards import metrics_for_completion


SELECTED_METRIC_KEYS = [
    "valid_expression",
    "exact_correct",
    "reward_hacking_candidate",
    "uses_allowed_numbers",
    "number_multiset_f1",
    "uses_allowed_ops",
    "uses_all_required_numbers",
    "uses_no_extra_numbers",
    "numeric_distance_reward",
]


def practical_score(metrics: dict[str, Any]) -> float:
    """Non-oracle selector score using legality and distance, not exact correctness."""
    score = 0.0
    score += 3.0 * float(metrics.get("valid_expression", 0.0))
    score += 2.0 * float(metrics.get("uses_allowed_numbers", 0.0))
    score += 1.5 * float(metrics.get("number_multiset_f1", 0.0))
    score += 1.0 * float(metrics.get("uses_allowed_ops", 0.0))
    score += 1.0 * float(metrics.get("numeric_distance_reward", 0.0))
    score -= 2.0 * float(metrics.get("reward_hacking_candidate", 0.0))
    return float(score)


def choose_practical(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return max(candidates, key=lambda row: (row["practical_score"], -row["candidate_index"]))


def choose_oracle(candidates: list[dict[str, Any]]) -> dict[str, Any]:
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
    summary: dict[str, float] = {}
    for key in SELECTED_METRIC_KEYS:
        summary[key] = float(sum(float(row["metrics"].get(key, 0.0)) for row in rows) / len(rows))
    return summary


def evaluate_candidates(
    rows_by_id: dict[str, list[dict[str, Any]]],
    n_values: list[int],
    *,
    wall_clock_seconds: float = 0.0,
    max_n: int | None = None,
) -> dict[str, Any]:
    max_n = max_n or max(n_values)
    report: dict[str, Any] = {"n_examples": len(rows_by_id), "n_values": n_values, "by_n": {}}
    total_tokens_generated = sum(
        int(row.get("completion_token_count", 0)) for candidates in rows_by_id.values() for row in candidates
    )
    for n in n_values:
        practical_rows = []
        oracle_rows = []
        tokens_at_n = 0
        for candidates in rows_by_id.values():
            prefix = candidates[:n]
            tokens_at_n += sum(int(row.get("completion_token_count", 0)) for row in prefix)
            practical = choose_practical(prefix)
            oracle = choose_oracle(prefix)
            practical_rows.append(practical)
            oracle_rows.append(oracle)
            practical.setdefault("selected_by_practical_n", []).append(n)
            oracle.setdefault("selected_by_oracle_n", []).append(n)
        practical_summary = summarize_rows(practical_rows)
        oracle_summary = summarize_rows(oracle_rows)
        estimated_wall_clock = float(wall_clock_seconds * (n / max(max_n, 1)))
        report["by_n"][str(n)] = {
            "samples_per_task": n,
            "wall_clock_seconds_estimated": estimated_wall_clock,
            "tokens_generated": int(tokens_at_n),
            "total_tokens_generated_for_max_n_run": int(total_tokens_generated),
            "practical_selected": practical_summary,
            "oracle_selected": oracle_summary,
            "oracle_exact_at_n": oracle_summary.get("exact_correct", 0.0),
            "reranked_exact_at_n": practical_summary.get("exact_correct", 0.0),
            "cost_per_oracle_exact": float(n / max(oracle_summary.get("exact_correct", 0.0), 1e-12)),
            "cost_per_reranked_exact": float(n / max(practical_summary.get("exact_correct", 0.0), 1e-12)),
        }
    return report


def count_completion_tokens(engine: Any, completion: str) -> int:
    tokenizer = getattr(engine, "tokenizer", None)
    if tokenizer is None:
        return len(completion.split())
    encoded = tokenizer(completion, add_special_tokens=False)
    return int(len(encoded.get("input_ids", [])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--data_path", default="data/countdown/validation.jsonl")
    parser.add_argument("--output_dir", default="outputs/best_of_n")
    parser.add_argument("--engine", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--prompt_field", default="prompt")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_values", default="1,4,8,16,32")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

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

    n_values = sorted({int(value) for value in args.n_values.split(",") if value.strip()})
    if not n_values or min(n_values) < 1:
        raise ValueError(f"Invalid --n_values: {args.n_values}")
    max_n = max(n_values)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = read_jsonl(args.data_path, limit=args.limit)
    engine = (
        VLLMEngine(args.model_name)
        if args.engine == "vllm"
        else HFEngine(args.model_name, args.adapter_path, device=args.device)
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
        for (ex, candidate_index), completion in zip(batch, completions):
            metrics = metrics_for_completion(completion, ex)
            row = {
                "id": ex["id"],
                "difficulty": ex["difficulty"],
                "numbers": ex["numbers"],
                "target": ex["target"],
                "allowed_ops": ex["allowed_ops"],
                "prompt_field": args.prompt_field,
                "candidate_index": candidate_index,
                "raw_generation": completion,
                "completion": completion,
                "extracted_expression": metrics.get("expression"),
                "completion_token_count": count_completion_tokens(engine, completion),
                "metrics": metrics,
                "practical_score": practical_score(metrics),
                "selected_by_practical_n": [],
                "selected_by_oracle_n": [],
            }
            rows.append(row)
            rows_by_id[ex["id"]].append(row)

    for candidates in rows_by_id.values():
        candidates.sort(key=lambda row: row["candidate_index"])

    wall_clock_seconds = time.time() - started_at
    report = evaluate_candidates(
        rows_by_id,
        n_values,
        wall_clock_seconds=wall_clock_seconds,
        max_n=max_n,
    )
    report.update(
        {
            "model_name": args.model_name,
            "adapter_path": args.adapter_path,
            "data_path": args.data_path,
            "prompt_field": args.prompt_field,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "wall_clock_seconds": wall_clock_seconds,
            "total_candidates": len(rows),
            "total_tokens_generated": sum(int(row["completion_token_count"]) for row in rows),
        }
    )

    write_jsonl(output_dir / "candidates.jsonl", rows)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2))

    flat_rows = []
    for n, result in report["by_n"].items():
        for selector in ["practical_selected", "oracle_selected"]:
            flat_rows.append({"n": int(n), "selector": selector, **result[selector]})
    pd.DataFrame(flat_rows).to_csv(output_dir / "summary.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
