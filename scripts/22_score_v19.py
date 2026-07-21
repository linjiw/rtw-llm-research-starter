#!/usr/bin/env python
"""Strict, task-clustered scorer for the complete Countdown-v2 v0.19 matrix."""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rtw_llm.cluster_stats import (
    evaluation_protocol_signature,
    require_complete_evaluation_signature,
    require_matching_evaluation_signatures,
    semantic_task_key,
    task_clustered_difference,
)
from rtw_llm.provenance import content_sha256, file_record, verify_completed_run
from rtw_llm.rewards import metrics_for_completion
from rtw_llm.v19_protocol import (
    ARM_SPECS,
    EVAL_CONFIG,
    ENVIRONMENT_LOCK,
    INFERENCE_CONFIG,
    PRIMARY_CONTRASTS,
    PROTOCOL_DIR,
    PROTOCOL_ID,
    SCORE_MANIFEST_SCHEMA,
    TRAINING_SEEDS,
    VIEW_FILES,
    eval_dir,
    require_eligible_protocol,
    state_label,
    training_dir,
    verify_adapter_chain,
    validate_environment_lock_document,
    validate_run_source_identity,
    verify_v19_training_health,
)


class V19ScoreError(RuntimeError):
    """Raised when a candidate matrix is incomplete, inconsistent, or unverifiable."""


def practical_score(metrics: Mapping[str, Any]) -> float:
    score = 0.0
    score += 3.0 * float(metrics.get("valid_expression", 0.0))
    score += 2.0 * float(metrics.get("uses_allowed_numbers", 0.0))
    score += 1.5 * float(metrics.get("number_multiset_f1", 0.0))
    score += 1.0 * float(metrics.get("uses_allowed_ops", 0.0))
    score += 1.0 * float(metrics.get("numeric_distance_reward", 0.0))
    score -= 2.0 * float(metrics.get("reward_hacking_candidate", 0.0))
    return float(score)


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, dict[str, float | bool]]:
    if not p_values:
        return {}
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    running = 0.0
    adjusted: dict[str, dict[str, float | bool]] = {}
    for rank, (name, raw) in enumerate(ordered):
        if not 0.0 <= raw <= 1.0 or not math.isfinite(raw):
            raise V19ScoreError(f"Invalid p-value for Holm adjustment: {name}={raw}")
        running = max(running, min(1.0, (count - rank) * raw))
        adjusted[name] = {
            "raw_p_value_two_sided": float(raw),
            "holm_adjusted_p_value": float(running),
            "reject_familywise_0.05": bool(running < 0.05),
        }
    return adjusted


def _load_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not values or len(values) != len(set(values)):
        raise V19ScoreError(f"Ordered task IDs are empty or duplicated: {path}")
    return values


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise V19ScoreError(f"{path}:{line_number}: candidate must be an object")
        values.append(value)
    return values


def load_frozen_tasks(repo_root: Path, expected_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    rows = _load_jsonl(repo_root / "data/countdown_v2/validation.jsonl")
    by_id = {str(row.get("id")): row for row in rows}
    if len(by_id) != len(rows):
        raise V19ScoreError("Frozen validation source has duplicate IDs")
    missing = [task_id for task_id in expected_ids if task_id not in by_id]
    if missing:
        raise V19ScoreError(f"Frozen validation source is missing IDs: {missing[:5]}")
    return {task_id: by_id[task_id] for task_id in expected_ids}


def _recomputed_candidate(
    row: Mapping[str, Any], *, expected_task: Mapping[str, Any]
) -> dict[str, Any]:
    identity_fields = ("id", "difficulty", "numbers", "target", "allowed_ops")
    mismatches = [key for key in identity_fields if row.get(key) != expected_task.get(key)]
    if mismatches:
        raise V19ScoreError(
            f"Candidate task payload differs from frozen source for {row.get('id')}: {mismatches}"
        )
    task = {
        key: row[key]
        for key in identity_fields
    }
    completion = row.get("raw_generation", row.get("completion"))
    if not isinstance(completion, str):
        raise V19ScoreError(f"Candidate {row.get('id')} has no raw generation")
    metrics = metrics_for_completion(completion, task)
    stored = row.get("metrics")
    if not isinstance(stored, dict):
        raise V19ScoreError(f"Candidate {row.get('id')} has no metric object")
    for key in (
        "exact_correct",
        "valid_expression",
        "uses_allowed_numbers",
        "number_multiset_f1",
        "uses_allowed_ops",
        "reward_hacking_candidate",
    ):
        if float(stored.get(key, -1.0)) != float(metrics.get(key, -2.0)):
            raise V19ScoreError(
                f"Verifier recomputation mismatch for {row.get('id')} candidate "
                f"{row.get('candidate_index')}: {key}"
            )
    if row.get("token_count_source") != "generated_token_ids":
        raise V19ScoreError("V0.19 requires exact generated-token counts")
    token_count = row.get("completion_token_count")
    cap = row.get("completion_hit_cap")
    finish = row.get("finish_reason")
    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0:
        raise V19ScoreError("Malformed exact completion token count")
    if not isinstance(cap, bool) or finish not in {"eos", "length", "other"}:
        raise V19ScoreError("Malformed generation finish metadata")
    if cap != (finish == "length"):
        raise V19ScoreError("Completion-cap flag disagrees with finish reason")
    max_tokens = int(EVAL_CONFIG["max_new_tokens"])
    if token_count > max_tokens:
        raise V19ScoreError("Generated token count exceeds the registered completion cap")
    if finish == "length" and token_count != max_tokens:
        raise V19ScoreError("Length finish reason requires exactly the registered token cap")
    return {**dict(row), "verified_metrics": metrics, "verified_practical_score": practical_score(metrics)}


def load_bank(
    run_dir: Path,
    expected_tasks: Mapping[str, Mapping[str, Any]],
    *,
    max_n: int = 8,
) -> dict[str, Any]:
    verified = verify_completed_run(
        run_dir, required_artifact_roles={"candidates", "metrics", "run_config", "summary"}
    )
    config = json.loads((run_dir / "run_config.json").read_text())
    expected_ids = list(expected_tasks)
    raw_rows = _load_jsonl(run_dir / "candidates.jsonl")
    rows = []
    for row in raw_rows:
        task_id = str(row.get("id"))
        if task_id not in expected_tasks:
            raise V19ScoreError(f"Unexpected candidate task ID: {task_id}")
        rows.append(_recomputed_candidate(row, expected_task=expected_tasks[task_id]))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    semantic_sources: dict[str, str] = {}
    for row in rows:
        task_id = str(row.get("id"))
        grouped[task_id].append(row)
        key = semantic_task_key(row)
        previous = semantic_sources.setdefault(key, task_id)
        if previous != task_id:
            raise V19ScoreError(f"Duplicate semantic tasks in bank: {previous}, {task_id}")
    if list(grouped) != list(expected_ids) or set(grouped) != set(expected_ids):
        raise V19ScoreError(f"Candidate bank task order/grid mismatch: {run_dir}")
    for task_id, candidates in grouped.items():
        candidates.sort(key=lambda row: int(row["candidate_index"]))
        if [row["candidate_index"] for row in candidates] != list(range(max_n)):
            raise V19ScoreError(f"{task_id}: candidate indices must be exactly 0..{max_n - 1}")
        if len({semantic_task_key(row) for row in candidates}) != 1:
            raise V19ScoreError(f"{task_id}: semantic identity changes across candidates")
    signature = evaluation_protocol_signature(
        config,
        split=config.get("split"),
        strict_identity=verified["intent"]["identity"],
    )
    require_complete_evaluation_signature(signature, label=str(run_dir))
    return {
        "verified": verified,
        "config": config,
        "rows": rows,
        "grouped": dict(grouped),
        "signature": signature,
    }


def task_outcomes(bank: Mapping[str, Any], *, n: int = 8) -> dict[str, Any]:
    practical: dict[str, float] = {}
    oracle: dict[str, float] = {}
    tiers: dict[str, str] = {}
    generated_tokens = 0
    cap_hits = 0
    candidates_total = 0
    legality_sum = 0.0
    for task_id, candidates in bank["grouped"].items():
        prefix = candidates[:n]
        selected = max(
            prefix,
            key=lambda row: (row["verified_practical_score"], -int(row["candidate_index"])),
        )
        oracle_selected = max(
            prefix,
            key=lambda row: (
                float(row["verified_metrics"]["exact_correct"]),
                row["verified_practical_score"],
                -int(row["candidate_index"]),
            ),
        )
        practical[task_id] = float(selected["verified_metrics"]["exact_correct"])
        oracle[task_id] = float(oracle_selected["verified_metrics"]["exact_correct"])
        tiers[task_id] = str(selected["difficulty"])
        generated_tokens += sum(int(row["completion_token_count"]) for row in prefix)
        cap_hits += sum(int(bool(row["completion_hit_cap"])) for row in prefix)
        candidates_total += len(prefix)
        legality_sum += sum(float(row["verified_metrics"]["valid_expression"]) for row in prefix)
    practical_exact = int(sum(practical.values()))
    return {
        "practical": practical,
        "oracle": oracle,
        "tiers": tiers,
        "generated_tokens": generated_tokens,
        "candidate_count": candidates_total,
        "completion_cap_hit_fraction": cap_hits / candidates_total,
        "candidate_legality": legality_sum / candidates_total,
        "tokens_per_practical_exact_task": (
            generated_tokens / practical_exact if practical_exact else None
        ),
        "mean_generated_tokens_per_task": generated_tokens / len(practical),
    }


def _summary_for_runs(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_seed = []
    for run in runs:
        practical = run["outcomes"]["practical"]
        oracle = run["outcomes"]["oracle"]
        tiers = run["outcomes"]["tiers"]
        by_tier = {}
        for tier in ("easy", "medium", "hard"):
            ids = [task_id for task_id, value in tiers.items() if value == tier]
            by_tier[tier] = {
                "n": len(ids),
                "practical_exact_at_8": sum(practical[value] for value in ids) / len(ids),
                "oracle_exact_at_8": sum(oracle[value] for value in ids) / len(ids),
            }
        per_seed.append(
            {
                "state": run["state"],
                "training_seed": run["seed"],
                "n_tasks": len(practical),
                "practical_exact_at_8": sum(practical.values()) / len(practical),
                "oracle_exact_at_8": sum(oracle.values()) / len(oracle),
                "macro_practical_exact_at_8": sum(
                    value["practical_exact_at_8"] for value in by_tier.values()
                )
                / 3,
                "by_tier": by_tier,
                "candidate_legality": run["outcomes"]["candidate_legality"],
                "generated_tokens": run["outcomes"]["generated_tokens"],
                "mean_generated_tokens_per_task": run["outcomes"][
                    "mean_generated_tokens_per_task"
                ],
                "tokens_per_practical_exact_task": run["outcomes"][
                    "tokens_per_practical_exact_task"
                ],
                "completion_cap_hit_fraction": run["outcomes"][
                    "completion_cap_hit_fraction"
                ],
                "training_wall_clock_seconds": run.get("training_wall_clock_seconds", 0.0),
                "single_gpu_hours": run.get("training_wall_clock_seconds", 0.0) / 3600.0,
            }
        )
    return {"observed_states": per_seed}


def _panel(runs: Sequence[Mapping[str, Any]], endpoint: str = "practical") -> dict[str, tuple[float, ...]]:
    if not runs:
        raise V19ScoreError("Cannot construct an empty arm panel")
    ids = list(runs[0]["outcomes"][endpoint])
    for run in runs[1:]:
        if list(run["outcomes"][endpoint]) != ids:
            raise V19ScoreError("Arm runs do not share an ordered task panel")
    return {
        task_id: tuple(float(run["outcomes"][endpoint][task_id]) for run in runs)
        for task_id in ids
    }


def _normalized_config(identity: Mapping[str, Any], *, family: str) -> dict[str, Any]:
    config = json.loads(json.dumps(identity["resolved_config"]))
    config.pop("seed", None)
    if family == "sft":
        config.pop("data_seed", None)
    return config


def score_matrix(
    repo_root: Path,
    runs_root: Path,
    *,
    view: str,
    seeds: Sequence[int] = TRAINING_SEEDS,
) -> dict[str, Any]:
    if view not in {"validation_dev100", "validation_confirm400"}:
        raise V19ScoreError(f"Unsupported score view: {view}")
    require_eligible_protocol(repo_root)
    expected_ids = _load_ids(repo_root / PROTOCOL_DIR / VIEW_FILES[view])
    expected_tasks = load_frozen_tasks(repo_root, expected_ids)
    arm_runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    signatures: dict[str, Mapping[str, Any]] = {}
    runtimes: dict[str, Mapping[str, Any]] = {}
    training_runtimes: dict[str, Mapping[str, Any]] = {}
    training_configs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unique_training_seconds: dict[str, float] = {}
    state_evidence: dict[str, dict[str, Any]] = {}

    bank_dir = eval_dir(runs_root, view, "base", None)
    bank = load_bank(bank_dir, expected_tasks)
    if bank["config"].get("experiment_protocol") != PROTOCOL_ID:
        raise V19ScoreError("Base bank is not a v0.19 artifact")
    validate_run_source_identity(bank["verified"]["intent"]["identity"], repo_root)
    base_run = {
        "state": "base",
        "seed": None,
        "bank": bank,
        "outcomes": task_outcomes(bank),
        "training_wall_clock_seconds": 0.0,
    }
    arm_runs["base"].append(base_run)
    signatures["base"] = bank["signature"]
    runtimes["base"] = bank["verified"]["intent"]["identity"]["runtime"]
    state_evidence["base"] = {
        "eval_experiment_id": bank["verified"]["intent"]["experiment_id"],
        "eval_result_manifest": file_record(bank_dir / "run_result.json"),
    }

    for arm, spec in ARM_SPECS.items():
        if arm == "base":
            continue
        for seed in seeds:
            label = state_label(arm, seed)
            current_eval = eval_dir(runs_root, view, arm, seed)
            bank = load_bank(current_eval, expected_tasks)
            config = bank["config"]
            if (
                config.get("experiment_protocol") != PROTOCOL_ID
                or config.get("method") != arm
                or config.get("training_seed") != seed
            ):
                raise V19ScoreError(f"Evaluation label mismatch: {label}")
            current_train = training_dir(runs_root, arm, seed)
            parent = training_dir(runs_root, "sft_only", seed) if spec.get("sft_parent") else None
            verify_adapter_chain(
                arm=arm,
                seed=seed,
                training_run=current_train,
                eval_run=current_eval,
                sft_parent=parent,
                repo_root=repo_root,
            )
            train_verified = verify_completed_run(current_train)
            health = verify_v19_training_health(
                current_train,
                run_kind="sft" if arm == "sft_only" else "grpo",
                expected_steps=313 if arm == "sft_only" else 300,
                expected_strategy=spec.get("reward_strategy"),
                require_group_variance=arm != "sft_only",
                repo_root=repo_root,
            )
            state_training_seconds = float(health["wall_clock_seconds"])
            unique_training_seconds[str(current_train)] = state_training_seconds
            if parent is not None:
                parent_health = verify_v19_training_health(
                    parent,
                    run_kind="sft",
                    expected_steps=313,
                    require_group_variance=False,
                    repo_root=repo_root,
                )
                parent_seconds = float(parent_health["wall_clock_seconds"])
                unique_training_seconds[str(parent)] = parent_seconds
                state_training_seconds += parent_seconds
            identity = train_verified["intent"]["identity"]
            requested = identity["requested_args"]
            expected_requested = {
                "experiment_protocol": PROTOCOL_ID,
                "seed": seed,
                "seed_protocol": "countdown-true-seeds-v2",
            }
            if arm != "sft_only":
                expected_requested.update(
                    {
                        "method_arm": arm,
                        "reward_strategy": spec["reward_strategy"],
                        "trainer_seed": seed,
                    }
                )
            mismatches = [
                key for key, value in expected_requested.items() if requested.get(key) != value
            ]
            if mismatches:
                raise V19ScoreError(f"Training manifest label mismatch for {label}: {mismatches}")
            family = "sft" if arm == "sft_only" else "grpo"
            training_configs[family].append(_normalized_config(identity, family=family))
            training_runtimes[label] = identity["runtime"]
            arm_runs[arm].append(
                {
                    "state": label,
                    "seed": seed,
                    "bank": bank,
                    "outcomes": task_outcomes(bank),
                    "training_wall_clock_seconds": state_training_seconds,
                }
            )
            signatures[label] = bank["signature"]
            runtimes[label] = bank["verified"]["intent"]["identity"]["runtime"]
            evidence = {
                "eval_experiment_id": bank["verified"]["intent"]["experiment_id"],
                "eval_result_manifest": file_record(current_eval / "run_result.json"),
                "training_experiment_id": train_verified["intent"]["experiment_id"],
                "training_result_manifest": file_record(current_train / "run_result.json"),
            }
            if parent is not None:
                parent_verified = verify_completed_run(parent)
                evidence.update(
                    {
                        "sft_parent_experiment_id": parent_verified["intent"]["experiment_id"],
                        "sft_parent_result_manifest": file_record(parent / "run_result.json"),
                    }
                )
            state_evidence[label] = evidence

    require_matching_evaluation_signatures(signatures)
    all_runtimes = {**runtimes, **training_runtimes}
    if len({json.dumps(value, sort_keys=True) for value in all_runtimes.values()}) != 1:
        raise V19ScoreError("Runtime drift across v0.19 training/evaluation states")
    lock_path = repo_root / ENVIRONMENT_LOCK
    if not lock_path.is_file():
        raise V19ScoreError("Production environment lock is missing")
    locked_runtime = validate_environment_lock_document(
        json.loads(lock_path.read_text()), repo_root
    )
    if next(iter(all_runtimes.values())) != locked_runtime:
        raise V19ScoreError("Run runtime does not match the production environment lock")
    for family, configs in training_configs.items():
        if len({json.dumps(value, sort_keys=True) for value in configs}) != 1:
            raise V19ScoreError(f"Resolved {family} config drift across v0.19 states")

    contrasts: dict[str, Any] = {}
    secondary_p: dict[str, float] = {}
    for index, (name, arm, baseline) in enumerate(PRIMARY_CONTRASTS):
        result = task_clustered_difference(
            _panel(arm_runs[arm]),
            _panel(arm_runs[baseline]),
            bootstrap_draws=INFERENCE_CONFIG["bootstrap_draws"],
            sign_flip_draws=INFERENCE_CONFIG["sign_flip_draws"],
            seed=INFERENCE_CONFIG["random_seed"],
            confidence=INFERENCE_CONFIG["confidence"],
        )
        if index == 0:
            interval = result["confidence_interval"]
            result["positive_claim_criteria_met"] = bool(
                result["estimate"] > 0
                and result["sign_flip"]["p_value_two_sided"] < 0.05
                and interval["lower"] > 0
            )
        else:
            secondary_p[name] = float(result["sign_flip"]["p_value_two_sided"])
        contrasts[name] = result
    adjusted = holm_adjust(secondary_p)
    for name, result in adjusted.items():
        contrasts[name]["multiplicity"] = result

    return {
        "schema_version": "countdown-v19-score-v1",
        "protocol_id": PROTOCOL_ID,
        "view": view,
        "ordered_task_count": len(expected_ids),
        "observed_training_seeds": list(seeds),
        "sampling_uncertainty_estimated": False,
        "arm_summaries": {arm: _summary_for_runs(runs) for arm, runs in arm_runs.items()},
        "matrix_unique_training_wall_clock_seconds": sum(unique_training_seconds.values()),
        "matrix_unique_single_gpu_hours": sum(unique_training_seconds.values()) / 3600.0,
        "evidence": {
            "protocol_manifest": file_record(repo_root / PROTOCOL_DIR / "manifest.json"),
            "states": state_evidence,
        },
        "contrasts": contrasts,
        "claim_scope": "finite_task_panel_conditional_on_observed_seeds_and_sampling_stream",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--runs_root", type=Path, default=Path("outputs/v19/production"))
    parser.add_argument(
        "--view", choices=["validation_dev100", "validation_confirm400"], required=True
    )
    parser.add_argument("--development_seed0", action="store_true")
    parser.add_argument("--out_json", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    runs_root = args.runs_root if args.runs_root.is_absolute() else root / args.runs_root
    seeds = (0,) if args.development_seed0 else TRAINING_SEEDS
    if args.development_seed0 and args.view != "validation_dev100":
        raise V19ScoreError("Seed-0-only scoring is permitted only on validation_dev100")
    report = score_matrix(root, runs_root, view=args.view, seeds=seeds)
    output = args.out_json if args.out_json.is_absolute() else root / args.out_json
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
    manifest_core = {
        "schema_version": SCORE_MANIFEST_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "view": args.view,
        "observed_training_seeds": list(seeds),
        "report": file_record(output),
        "scorer": file_record(Path(__file__).resolve()),
        "protocol_manifest": file_record(root / PROTOCOL_DIR / "manifest.json"),
        "evidence_sha256": content_sha256(report["evidence"]),
    }
    score_manifest = {
        **manifest_core,
        "manifest_sha256": content_sha256(manifest_core),
    }
    manifest_path = output.with_name(output.name + ".manifest.json")
    descriptor = os.open(manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(score_manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
