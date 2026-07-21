#!/usr/bin/env python
"""Verifier-guided best-of-N candidate selection for Countdown evals."""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from rtw_llm.data_access import assert_countdown_data_access
from rtw_llm.data import read_jsonl, write_jsonl
from rtw_llm.engine import GenerationConfigLite, HFEngine, VLLMEngine
from rtw_llm.provenance import (
    build_run_identity,
    verify_completed_run,
    write_intent,
    write_result,
)
from rtw_llm.rewards import metrics_for_completion
from rtw_llm.seed_protocol import LEGACY_SEED_PROTOCOL, SEED_PROTOCOLS, TRUE_SEED_PROTOCOL


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
        for task_id, candidates in rows_by_id.items():
            if len(candidates) < n:
                raise ValueError(f"Task {task_id} has only {len(candidates)} candidates; cannot evaluate N={n}")
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
            "cost_per_oracle_exact": cost_per_exact(n, oracle_summary.get("exact_correct", 0.0)),
            "cost_per_reranked_exact": cost_per_exact(n, practical_summary.get("exact_correct", 0.0)),
        }
    return report


def cost_per_exact(samples_per_task: int, exact_rate: float) -> float:
    return float(samples_per_task / max(float(exact_rate), 1e-12))


def parse_n_values(values: list[str] | str) -> list[int]:
    if isinstance(values, str):
        parts = re.split(r"[\s,]+", values.strip())
    else:
        parts: list[str] = []
        for value in values:
            parts.extend(re.split(r"[\s,]+", value.strip()))
    n_values = sorted({int(value) for value in parts if value})
    if not n_values or min(n_values) < 1:
        raise ValueError(f"Invalid --n_values: {values}")
    return n_values


def load_task_ids(path: str | None) -> list[str] | None:
    if path is None:
        return None
    task_ids = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"Duplicate task IDs in {path}")
    return task_ids


def select_examples(examples: list[dict[str, Any]], *, limit: int | None, task_ids_file: str | None) -> list[dict[str, Any]]:
    task_ids = load_task_ids(task_ids_file)
    if task_ids is None:
        return examples[:limit] if limit is not None else examples
    by_id = {ex["id"]: ex for ex in examples}
    missing = [task_id for task_id in task_ids if task_id not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} task IDs from {task_ids_file} are missing from data: {missing[:5]}")
    selected = [by_id[task_id] for task_id in task_ids]
    if limit is not None and len(selected) != limit:
        raise ValueError(f"--limit {limit} does not match {len(selected)} task IDs from {task_ids_file}")
    return selected


def sampling_identity(config: dict[str, Any]) -> dict[str, Any]:
    """Keys that determine the candidate bank; a --skip_if_complete hit must match all.

    In batched mode the RNG is consumed per step across the whole batch, so
    batch_size is part of the identity; loop mode is chunking-invariant.
    Artifacts predating the hf_gen_mode flag count as loop-mode.
    """
    mode = config.get("hf_gen_mode") or "loop"
    identity = {
        "hf_gen_mode": mode,
        "model_name": config.get("model_name"),
        "model_revision": config.get("model_revision"),
        "adapter_path": config.get("adapter_path"),
        "sampling_seed": config.get("sampling_seed"),
        "temperature": config.get("temperature"),
        "top_p": config.get("top_p"),
        "max_new_tokens": config.get("max_new_tokens"),
        # The prompt field is part of the candidate-bank identity: the
        # harness-shift experiment reuses one checkpoint across prompt_high /
        # prompt_mid, so --skip_if_complete must NOT skip across fields.
        # Legacy artifacts without the key are 'prompt' (== prompt_high here).
        "prompt_field": config.get("prompt_field") or "prompt",
    }
    if mode == "batched":
        identity["batch_size"] = config.get("batch_size")
    return identity


def is_complete(
    output_dir: Path,
    max_n: int,
    n_examples: int | None = None,
    requested_identity: dict[str, Any] | None = None,
) -> bool:
    metrics_path = output_dir / "metrics.json"
    candidates_path = output_dir / "candidates.jsonl"
    summary_path = output_dir / "summary.csv"
    if not (metrics_path.exists() and candidates_path.exists() and summary_path.exists()):
        return False
    try:
        metrics = json.loads(metrics_path.read_text())
    except json.JSONDecodeError:
        return False
    if str(max_n) not in metrics.get("by_n", {}):
        return False
    if n_examples is not None and int(metrics.get("n_examples", -1)) != n_examples:
        return False
    if requested_identity is not None:
        config_path = output_dir / "run_config.json"
        try:
            stored = json.loads(config_path.read_text()) if config_path.exists() else {}
        except json.JSONDecodeError:
            stored = {}
        if sampling_identity(stored) != requested_identity:
            raise ValueError(
                f"{output_dir} holds complete artifacts with a different sampling "
                f"identity; refusing to skip. stored={sampling_identity(stored)} "
                f"requested={requested_identity}"
            )
    expected_candidates = int(metrics.get("n_examples", 0)) * max_n
    actual_candidates = sum(1 for _ in candidates_path.open())
    return actual_candidates == expected_candidates


def is_strict_complete(output_dir: Path, requested_identity: dict[str, Any]) -> bool:
    """A strict skip is allowed only after full manifest/artifact verification."""
    has_manifest = (output_dir / "run_intent.json").exists() or (
        output_dir / "run_result.json"
    ).exists()
    if not has_manifest:
        return False
    verify_completed_run(
        output_dir,
        requested_identity,
        required_artifact_roles={"candidates", "metrics", "run_config", "summary"},
    )
    return True


def count_completion_tokens(engine: Any, completion: str) -> int:
    tokenizer = getattr(engine, "tokenizer", None)
    if tokenizer is None:
        return len(completion.split())
    encoded = tokenizer(completion, add_special_tokens=False)
    return int(len(encoded.get("input_ids", [])))


def write_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    n_values: list[int],
    max_n: int,
    n_examples: int,
    effective_generation_config: dict[str, Any] | None = None,
) -> None:
    config = {
        "method": args.method,
        "training_seed": args.training_seed,
        "training_protocol": args.training_protocol,
        "experiment_protocol": args.experiment_protocol,
        "sampling_seed": args.seed,
        "split": args.split,
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "adapter_path": args.adapter_path,
        "data_path": args.data_path,
        "task_ids_file": args.task_ids_file,
        "limit": args.limit,
        "n_examples": n_examples,
        "n_values": n_values,
        "max_n": max_n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "prompt_field": args.prompt_field,
        "engine": args.engine,
        "device": args.device,
        "hf_gen_mode": args.hf_gen_mode,
        "effective_generation_config": effective_generation_config,
        "final_test_release_record": args.final_test_release_record,
        "test_release_record": args.test_release_record,
        "confirmation_ready_record": args.confirmation_ready_record,
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--data_path", default="data/countdown/validation.jsonl")
    parser.add_argument("--task_ids_file", default=None)
    parser.add_argument("--output_dir", default="outputs/best_of_n")
    parser.add_argument("--engine", choices=["hf", "vllm"], default="hf")
    parser.add_argument(
        "--hf_gen_mode",
        choices=["loop", "batched"],
        default="loop",
        help=(
            "HF generation path. 'loop' (default) is the archival per-prompt path all "
            "v0.9/Gate-0 artifacts use; 'batched' is a throughput path with a different "
            "sampling identity — never mix modes within a paired comparison "
            "(docs/THROUGHPUT_BATCHED_BESTOFN_PLAN.md)."
        ),
    )
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--prompt_field", default="prompt")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_values", nargs="+", default=["1,4,8,16,32"])
    parser.add_argument("--max_n", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", default=None)
    parser.add_argument("--training_seed", type=int, default=None)
    parser.add_argument(
        "--training_protocol",
        choices=SEED_PROTOCOLS,
        default=LEGACY_SEED_PROTOCOL,
    )
    parser.add_argument("--split", default=None)
    parser.add_argument("--skip_if_complete", action="store_true")
    parser.add_argument("--strict_provenance", action="store_true")
    parser.add_argument("--final_test_release_record", default=None)
    parser.add_argument("--test_release_record", default=None)
    parser.add_argument("--experiment_protocol", default=None)
    parser.add_argument("--confirmation_ready_record", default=None)
    args = parser.parse_args()

    if args.final_test_release_record and not args.strict_provenance:
        raise ValueError("Released final-test best-of-N requires --strict_provenance")
    if args.final_test_release_record and (args.limit is not None or args.task_ids_file):
        raise ValueError("Released final-test best-of-N forbids --limit and --task_ids_file")
    if args.test_release_record and (args.limit is not None or args.task_ids_file):
        raise ValueError("Released one-shot test best-of-N forbids --limit and --task_ids_file")
    if args.test_release_record and not args.strict_provenance:
        raise ValueError("Released one-shot test best-of-N requires --strict_provenance")
    if args.training_protocol == TRUE_SEED_PROTOCOL and not args.strict_provenance:
        raise ValueError("countdown-true-seeds-v2 evaluation requires --strict_provenance")
    if args.strict_provenance and args.engine != "hf":
        raise ValueError("Strict provenance currently supports only the hf engine")
    n_values = parse_n_values(args.n_values)
    max_n = args.max_n or max(n_values)
    if max(n_values) > max_n:
        raise ValueError(f"max --n_values {max(n_values)} exceeds --max_n {max_n}")
    if args.experiment_protocol is not None:
        from rtw_llm.v19_protocol import PROTOCOL_ID, validate_v19_eval_args

        if args.experiment_protocol != PROTOCOL_ID:
            raise ValueError(f"Unsupported experiment protocol: {args.experiment_protocol}")
        validate_v19_eval_args({**vars(args), "n_values": n_values, "max_n": max_n})
    repo_root = Path(__file__).resolve().parents[1]
    if args.task_ids_file:
        from rtw_llm.provenance import file_record
        from rtw_llm.v19_protocol import (
            CONFIRMATION_READY_RECORD,
            PROTOCOL_DIR,
            PROTOCOL_ID,
            VIEW_FILES,
            validate_confirmation_ready,
        )

        requested_ids = Path(args.task_ids_file)
        if not requested_ids.is_absolute():
            requested_ids = repo_root / requested_ids
        confirm_ids = repo_root / PROTOCOL_DIR / VIEW_FILES["validation_confirm400"]
        if requested_ids.is_file() and confirm_ids.is_file() and file_record(
            requested_ids
        ) == file_record(confirm_ids):
            if args.experiment_protocol != PROTOCOL_ID:
                raise ValueError("Confirmation task IDs require the registered v0.19 protocol")
            if args.confirmation_ready_record is None:
                raise ValueError("Confirmation task IDs require a readiness record")
            if Path(args.confirmation_ready_record).as_posix() != CONFIRMATION_READY_RECORD.as_posix():
                raise ValueError("Confirmation readiness record path is not registered")
            validate_confirmation_ready(args.confirmation_ready_record, repo_root=repo_root)
    assert_countdown_data_access(
        args.data_path,
        purpose="model_eval",
        runner="07_best_of_n_rerank",
        release_record=args.final_test_release_record,
        test_release_record=args.test_release_record,
        experiment_protocol=args.experiment_protocol,
        ordered_task_ids_file=args.task_ids_file,
        confirmation_ready_record=args.confirmation_ready_record,
        repo_root=repo_root,
    )

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

    if args.engine == "vllm" and args.hf_gen_mode != "loop":
        raise ValueError("--hf_gen_mode applies to the hf engine only")

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
    provenance_identity = None
    if args.strict_provenance:
        input_files = {"data": args.data_path}
        if args.task_ids_file:
            input_files["ordered_task_ids"] = args.task_ids_file
        if args.final_test_release_record:
            input_files["final_test_release"] = args.final_test_release_record
        if args.test_release_record:
            input_files["test_release"] = args.test_release_record
        if args.confirmation_ready_record:
            input_files["confirmation_ready"] = args.confirmation_ready_record
        provenance_identity = build_run_identity(
            run_kind="best_of_n",
            requested_args=vars(args),
            resolved_config={
                **requested_identity,
                "n_values": n_values,
                "max_n": max_n,
                "n_examples": len(examples),
                "engine": args.engine,
                "device": args.device,
            },
            seed_roles={
                "sampling_seed": args.seed,
                "training_seed_label": args.training_seed,
                "training_protocol": args.training_protocol,
            },
            input_files=input_files,
            model_name=args.model_name,
            adapter_path=args.adapter_path,
            repo_root=repo_root,
            model_revision=args.model_revision,
        )
    if args.skip_if_complete:
        if args.strict_provenance:
            if is_strict_complete(output_dir, provenance_identity):
                print(
                    json.dumps(
                        {
                            "status": "skipped_verified",
                            "output_dir": str(output_dir),
                            "max_n": max_n,
                        },
                        indent=2,
                    )
                )
                return
        if not args.strict_provenance and is_complete(
            output_dir, max_n=max_n, n_examples=len(examples), requested_identity=requested_identity
        ):
            print(
                json.dumps(
                    {"status": "skipped_complete", "output_dir": str(output_dir), "max_n": max_n},
                    indent=2,
                )
            )
            return
    if args.strict_provenance:
        write_intent(output_dir, provenance_identity)

    engine = (
        VLLMEngine(args.model_name)
        if args.engine == "vllm"
        else HFEngine(
            args.model_name,
            args.adapter_path,
            model_revision=args.model_revision,
            device=args.device,
            gen_mode=args.hf_gen_mode,
        )
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
        if len(generation_metadata) != len(completions):
            raise RuntimeError("Generation metadata length does not match completions")
        for (ex, candidate_index), completion, generation_meta in zip(
            batch, completions, generation_metadata
        ):
            metrics = metrics_for_completion(completion, ex)
            exact_token_count = generation_meta.get("generated_token_count")
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
                "completion_token_count": (
                    int(exact_token_count)
                    if exact_token_count is not None
                    else count_completion_tokens(engine, completion)
                ),
                "token_count_source": (
                    "generated_token_ids" if exact_token_count is not None else "decoded_retokenized"
                ),
                "finish_reason": generation_meta.get("finish_reason"),
                "completion_hit_cap": generation_meta.get("completion_hit_cap"),
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
    effective_gen_config = None
    if hasattr(engine, "effective_generation_config"):
        effective_gen_config = engine.effective_generation_config()

    report.update(
        {
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
            "training_seed": args.training_seed,
            "training_protocol": args.training_protocol,
            "experiment_protocol": args.experiment_protocol,
            "method": args.method,
            "split": args.split,
            "max_n": max_n,
            "hf_gen_mode": args.hf_gen_mode,
            "effective_generation_config": effective_gen_config,
            "wall_clock_seconds": wall_clock_seconds,
            "total_candidates": len(rows),
            "total_tokens_generated": sum(int(row["completion_token_count"]) for row in rows),
        }
    )

    write_jsonl(output_dir / "candidates.jsonl", rows)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    write_run_config(
        output_dir, args, n_values, max_n, len(examples),
        effective_generation_config=effective_gen_config,
    )

    flat_rows = []
    for n, result in report["by_n"].items():
        for selector in ["practical_selected", "oracle_selected"]:
            flat_rows.append({"n": int(n), "selector": selector, **result[selector]})
    pd.DataFrame(flat_rows).to_csv(output_dir / "summary.csv", index=False)
    if args.strict_provenance:
        write_result(
            output_dir,
            artifact_paths={
                "candidates": output_dir / "candidates.jsonl",
                "metrics": output_dir / "metrics.json",
                "run_config": output_dir / "run_config.json",
                "summary": output_dir / "summary.csv",
            },
            observations={"wall_clock_seconds": wall_clock_seconds},
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
