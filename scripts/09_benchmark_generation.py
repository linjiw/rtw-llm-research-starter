#!/usr/bin/env python
"""Equivalence + timing benchmark for HFEngine loop vs batched generation.

Gates the batched path per docs/THROUGHPUT_BATCHED_BESTOFN_PLAN.md:
  --check equivalence  two-tier greedy check (CPU fp32 recommended):
                       tier i  — same-token-length prompts (no padding):
                                 loop vs batched must match token-for-token;
                       tier ii — mixed-length prompts: batched with the safe
                                 pad vs batched with pad=EOS; a divergence
                                 under pad=EOS that disappears with the safe
                                 pad confirms the repetition-penalty bias.
  --check distribution sampled completion-length and EOS-termination stats,
                       loop vs batched, mixed-length batches.
  --check timing       candidates/s per (mode, batch_size); requires an idle
                       GPU on cuda, does a warmup pass, reports the median of
                       --repeats timed runs.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from typing import Any

from rtw_llm.data import read_jsonl
from rtw_llm.engine import GenerationConfigLite, HFEngine


def reseed(seed: int) -> None:
    import torch
    from transformers import set_seed

    torch.manual_seed(seed)
    set_seed(seed)


def assert_gpu_idle() -> None:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if out:
        raise SystemExit(
            f"GPU is not idle; timing numbers would be corrupted. Running:\n{out}"
        )


def load_prompts(data_path: str, prompt_field: str, num_prompts: int) -> list[str]:
    examples = read_jsonl(data_path)
    if len(examples) < num_prompts:
        raise SystemExit(f"{data_path} has only {len(examples)} examples, need {num_prompts}")
    return [ex[prompt_field] for ex in examples[:num_prompts]]


def same_length_prompts(engine: HFEngine, prompts: list[str], count: int) -> list[str]:
    """Prompts sharing one token length, so a batch of them needs no padding."""
    by_len: dict[int, list[str]] = {}
    for prompt in prompts:
        length = len(engine.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        by_len.setdefault(length, []).append(prompt)
    best = max(by_len.values(), key=len)
    if len(best) < count:
        # Repeating one prompt keeps lengths equal without changing the check:
        # tier i isolates batching numerics, not prompt diversity.
        best = best + [best[0]] * (count - len(best))
    return best[:count]


def token_ids(engine: HFEngine, text: str) -> list[int]:
    return engine.tokenizer(text, add_special_tokens=False)["input_ids"]


def check_equivalence(args: argparse.Namespace) -> dict[str, Any]:
    greedy = GenerationConfigLite(
        max_new_tokens=args.max_new_tokens, temperature=0.0, top_p=1.0, do_sample=False
    )
    loop_engine = HFEngine(args.model_name, args.adapter_path, device=args.device, gen_mode="loop")
    batched_engine = HFEngine(
        args.model_name, args.adapter_path, device=args.device, gen_mode="batched"
    )
    prompts = load_prompts(args.data_path, args.prompt_field, args.num_prompts)

    tier1_prompts = same_length_prompts(loop_engine, prompts, min(4, args.num_prompts))
    loop_out = loop_engine.generate(tier1_prompts, greedy)
    batched_out = batched_engine.generate(tier1_prompts, greedy)
    tier1_mismatches = [
        {"index": i, "loop": lo, "batched": ba}
        for i, (lo, ba) in enumerate(zip(loop_out, batched_out))
        if token_ids(loop_engine, lo) != token_ids(loop_engine, ba)
    ]

    mixed = prompts[: min(4, args.num_prompts)]
    safe_out = batched_engine.generate(mixed, greedy)
    eos_pad_engine = HFEngine(
        args.model_name, args.adapter_path, device=args.device, gen_mode="batched"
    )
    eos_id = eos_pad_engine.tokenizer.eos_token_id
    eos_pad_engine.batch_pad_token_id = eos_id
    eos_pad_engine.tokenizer.pad_token_id = eos_id
    eos_pad_engine.tokenizer.pad_token = eos_pad_engine.tokenizer.convert_ids_to_tokens(eos_id)
    eos_out = eos_pad_engine.generate(mixed, greedy)
    tier2_divergences = [
        {"index": i, "safe_pad": sa, "eos_pad": eo}
        for i, (sa, eo) in enumerate(zip(safe_out, eos_out))
        if token_ids(loop_engine, sa) != token_ids(loop_engine, eo)
    ]

    return {
        "tier1_same_length_prompts": len(tier1_prompts),
        "tier1_exact_match": not tier1_mismatches,
        "tier1_mismatches": tier1_mismatches,
        "tier2_mixed_prompts": len(mixed),
        "tier2_safe_vs_eos_pad_divergences": len(tier2_divergences),
        "tier2_divergence_examples": tier2_divergences[:2],
        "batched_pad_token_id": batched_engine.batch_pad_token_id,
        "effective_generation_config": batched_engine.effective_generation_config(),
    }


def completion_stats(
    engine: HFEngine, completions: list[str], max_new_tokens: int
) -> dict[str, float]:
    lengths = [len(token_ids(engine, completion)) for completion in completions]
    n = max(len(completions), 1)
    return {
        "n": len(completions),
        "mean_completion_tokens": statistics.mean(lengths) if lengths else 0.0,
        "median_completion_tokens": statistics.median(lengths) if lengths else 0.0,
        # Decoded length below the cap means generation ended via EOS; this is
        # the statistic the pad=EOS repetition-penalty bias shifts (ADV-1).
        "terminated_before_cap_rate": sum(1 for x in lengths if x < max_new_tokens) / n,
        "capped_rate": sum(1 for x in lengths if x >= max_new_tokens) / n,
    }


def check_distribution(args: argparse.Namespace) -> dict[str, Any]:
    sampled = GenerationConfigLite(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
    )
    prompts = load_prompts(args.data_path, args.prompt_field, args.num_prompts)
    out: dict[str, Any] = {"num_prompts": len(prompts), "config": vars(sampled)}
    for mode in ("loop", "batched"):
        reseed(args.seed)  # each arm independently reproducible
        engine = HFEngine(args.model_name, args.adapter_path, device=args.device, gen_mode=mode)
        completions: list[str] = []
        for start in range(0, len(prompts), args.batch_size):
            completions.extend(engine.generate(prompts[start : start + args.batch_size], sampled))
        out[mode] = completion_stats(engine, completions, args.max_new_tokens)
    return out


def cuda_in_play(device: str) -> bool:
    if device == "cuda":
        return True
    if device != "auto":
        return False
    import torch

    return torch.cuda.is_available()


def check_timing(args: argparse.Namespace) -> dict[str, Any]:
    if cuda_in_play(args.device):
        assert_gpu_idle()
    sampled = GenerationConfigLite(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
    )
    results: dict[str, Any] = {}
    for mode in ("loop", "batched"):
        engine = HFEngine(args.model_name, args.adapter_path, device=args.device, gen_mode=mode)
        for batch_size in args.batch_sizes:
            prompts = load_prompts(args.data_path, args.prompt_field, batch_size)
            engine.generate(prompts, sampled)  # warmup at the timed batch shape
            times = []
            for _ in range(args.repeats):
                started = time.time()
                engine.generate(prompts, sampled)
                times.append(time.time() - started)
            median_s = statistics.median(times)
            results[f"{mode}_batch{batch_size}"] = {
                "median_seconds": median_s,
                "candidates_per_second": batch_size / median_s,
                "all_seconds": times,
            }
    for batch_size in args.batch_sizes:
        loop = results.get(f"loop_batch{batch_size}")
        batched = results.get(f"batched_batch{batch_size}")
        if loop and batched:
            results[f"speedup_batch{batch_size}"] = (
                loop["median_seconds"] / batched["median_seconds"]
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--data_path", default="data/countdown/validation.jsonl")
    parser.add_argument("--prompt_field", default="prompt")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument(
        "--check", choices=["equivalence", "distribution", "timing"], required=True
    )
    parser.add_argument("--num_prompts", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    reseed(args.seed)

    if args.check == "equivalence":
        report = check_equivalence(args)
    elif args.check == "distribution":
        report = check_distribution(args)
    else:
        report = check_timing(args)
    report["check"] = args.check
    text = json.dumps(report, indent=2)
    if args.out_json:
        with open(args.out_json, "w") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
