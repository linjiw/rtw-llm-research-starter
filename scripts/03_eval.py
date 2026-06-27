#!/usr/bin/env python
"""Evaluate a base or adapter model on Countdown JSONL files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from rtw_llm.data import read_jsonl, write_jsonl
from rtw_llm.engine import GenerationConfigLite, HFEngine, VLLMEngine
from rtw_llm.rewards import metrics_for_completion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--data_path", default="data/countdown/test_in_dist.jsonl")
    parser.add_argument("--output_dir", default="outputs/eval")
    parser.add_argument("--engine", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--prompt_field", default="prompt")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = read_jsonl(args.data_path, limit=args.limit)
    engine = VLLMEngine(args.model_name) if args.engine == "vllm" else HFEngine(args.model_name, args.adapter_path)
    gen_config = GenerationConfigLite(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.temperature > 0,
    )

    rows = []
    for start in tqdm(range(0, len(examples), args.batch_size)):
        batch = examples[start : start + args.batch_size]
        prompts = [ex[args.prompt_field] for ex in batch]
        completions = engine.generate(prompts, gen_config)
        for ex, completion in zip(batch, completions):
            metrics = metrics_for_completion(completion, ex)
            rows.append(
                {
                    "id": ex["id"],
                    "difficulty": ex["difficulty"],
                    "numbers": ex["numbers"],
                    "target": ex["target"],
                    "allowed_ops": ex["allowed_ops"],
                    "prompt_field": args.prompt_field,
                    "completion": completion,
                    "metrics": metrics,
                }
            )

    write_jsonl(output_dir / "generations.jsonl", rows)
    df = pd.json_normalize(rows)
    metric_cols = [c for c in df.columns if c.startswith("metrics.") and pd.api.types.is_numeric_dtype(df[c])]
    summary = {c.replace("metrics.", ""): float(df[c].mean()) for c in metric_cols}
    by_diff = df.groupby("difficulty")[metric_cols].mean().reset_index().to_dict(orient="records")
    report = {"n": len(rows), "summary": summary, "by_difficulty": by_diff}
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
