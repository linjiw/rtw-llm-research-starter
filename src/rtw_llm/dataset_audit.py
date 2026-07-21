"""Deterministic integrity, leakage, and reproducibility audit for Countdown data."""
from __future__ import annotations

import importlib.util
import itertools
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from .countdown import difficulty_spec, verify_completion, verify_expression
from .prompts import make_prompt, make_sft_completion
from .provenance import file_record

AUDIT_SCHEMA = "countdown-dataset-audit-v1"
SPLIT_FILES = {
    "train": "data/countdown/train.jsonl",
    "validation": "data/countdown/validation.jsonl",
    "test_in_dist": "data/countdown/test_in_dist.jsonl",
    "test_ood_division": "data/countdown/test_ood_division.jsonl",
    "test_ood_long": "data/countdown/test_ood_long.jsonl",
}
FROZEN_FILES = {
    "validation": "outputs/v09_task_ids_validation_limit50.txt",
    "test_in_dist": "outputs/v09_task_ids_test_in_dist_limit50.txt",
    "test_ood_division": "outputs/v09_task_ids_test_ood_division_limit50.txt",
    "test_ood_long": "outputs/v09_task_ids_test_ood_long_limit50.txt",
}
IN_DIST_DIFFICULTIES = {"easy", "medium", "hard"}
REQUIRED_FIELDS = {
    "id",
    "split",
    "difficulty",
    "numbers",
    "target",
    "allowed_ops",
    "solution",
    "prompt_low",
    "prompt_mid",
    "prompt_high",
    "prompt",
    "completion",
    "metadata",
}
NON_PROMPT_REPLAY_FIELDS = (
    "id",
    "split",
    "difficulty",
    "numbers",
    "target",
    "allowed_ops",
    "solution",
    "metadata",
)


def semantic_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(sorted(int(number) for number in record["numbers"])),
        int(record["target"]),
        tuple(sorted(set(str(op) for op in record["allowed_ops"]))),
    )


def loose_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (tuple(sorted(int(number) for number in record["numbers"])), int(record["target"]))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: malformed JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be a JSON object")
            records.append(value)
    return records


def audit_records(
    split_name: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    schema_errors: list[str] = []
    solution_failures: list[str] = []
    completion_failures: list[str] = []
    prompt_drift = {"prompt_low": [], "prompt_mid": [], "prompt_high": []}
    completion_template_drift: list[str] = []
    default_prompt_mismatch: list[str] = []
    operator_order_warnings: list[str] = []
    ids = []
    exact_keys: list[tuple[Any, ...]] = []

    for index, record in enumerate(records):
        row_label = str(record.get("id", f"row-{index}"))
        missing = sorted(REQUIRED_FIELDS - set(record))
        if missing:
            schema_errors.append(f"{row_label}: missing fields {missing}")
            continue
        if not _valid_record_types(record):
            schema_errors.append(f"{row_label}: invalid field types")
            continue
        ids.append(record["id"])
        if record["split"] != split_name:
            schema_errors.append(
                f"{row_label}: split={record['split']!r}, expected {split_name!r}"
            )
        if not _difficulty_allowed(split_name, record["difficulty"]):
            schema_errors.append(
                f"{row_label}: difficulty={record['difficulty']!r} is invalid for {split_name}"
            )
        try:
            spec = difficulty_spec(record["difficulty"])
            if len(record["numbers"]) != int(spec["n_numbers"]):
                schema_errors.append(f"{row_label}: wrong number count for difficulty")
            if set(record["allowed_ops"]) != set(spec["allowed_ops"]):
                schema_errors.append(f"{row_label}: wrong operator set for difficulty")
            if list(record["allowed_ops"]) != list(spec["allowed_ops"]):
                operator_order_warnings.append(row_label)
        except (KeyError, ValueError):
            schema_errors.append(f"{row_label}: unknown difficulty specification")

        exact_keys.append(semantic_key(record))
        solution = verify_expression(
            record["solution"],
            record["numbers"],
            record["target"],
            record["allowed_ops"],
        )
        if not solution.correct:
            solution_failures.append(row_label)
        if not verify_completion(record["completion"], record).correct:
            completion_failures.append(row_label)

        for level in ("low", "mid", "high"):
            expected_prompt = make_prompt(
                record["numbers"], record["target"], record["allowed_ops"], level
            )
            field = f"prompt_{level}"
            if record[field] != expected_prompt:
                prompt_drift[field].append(row_label)
        if record["prompt"] != record["prompt_high"]:
            default_prompt_mismatch.append(row_label)
        if record["completion"] != make_sft_completion(record["solution"], record["target"]):
            completion_template_drift.append(row_label)

    duplicate_ids = sorted(key for key, count in Counter(ids).items() if count > 1)
    duplicate_key_groups = sum(1 for count in Counter(exact_keys).values() if count > 1)
    return {
        "count": len(records),
        "unique_ids": len(set(ids)),
        "schema_errors": sorted(schema_errors),
        "duplicate_ids": duplicate_ids,
        "within_split_semantic_duplicate_groups": duplicate_key_groups,
        "solution_failure_ids": sorted(solution_failures),
        "completion_failure_ids": sorted(completion_failures),
        "prompt_drift": {
            field: {"count": len(values), "ids": sorted(values)}
            for field, values in prompt_drift.items()
        },
        "default_prompt_mismatch_ids": sorted(default_prompt_mismatch),
        "completion_template_drift_ids": sorted(completion_template_drift),
        "operator_order_warning_ids": sorted(operator_order_warnings),
    }


def structurally_valid_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows safe for semantic indexing after schema errors have been recorded."""
    return [
        record
        for record in records
        if REQUIRED_FIELDS.issubset(record) and _valid_record_types(record)
    ]


def overlap_stats(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], tuple[Any, ...]] = semantic_key,
) -> dict[str, Any]:
    left_groups = _group_ids(left, key_fn)
    right_groups = _group_ids(right, key_fn)
    shared = sorted(set(left_groups) & set(right_groups), key=repr)
    left_occurrences = sum(len(left_groups[key]) for key in shared)
    right_occurrences = sum(len(right_groups[key]) for key in shared)
    left_ids = sorted({item for key in shared for item in left_groups[key]})
    right_ids = sorted({item for key in shared for item in right_groups[key]})
    combinations = sum(len(left_groups[key]) * len(right_groups[key]) for key in shared)
    return {
        "shared_key_groups": len(shared),
        "affected_left_records": left_occurrences,
        "affected_right_records": right_occurrences,
        "record_pair_combinations": combinations,
        "left_rate": left_occurrences / len(left) if left else 0.0,
        "right_rate": right_occurrences / len(right) if right else 0.0,
        "left_ids": left_ids,
        "right_ids": right_ids,
    }


def audit_frozen_ids(
    split_name: str,
    task_ids: list[str],
    split_records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    intended = {record["id"]: record for record in split_records[split_name]}
    global_owner = {
        record["id"]: name for name, records in split_records.items() for record in records
    }
    duplicates = sorted(key for key, count in Counter(task_ids).items() if count > 1)
    missing = sorted(task_id for task_id in task_ids if task_id not in global_owner)
    wrong_split = sorted(
        task_id
        for task_id in task_ids
        if task_id in global_owner and global_owner[task_id] != split_name
    )
    selected = [intended[task_id] for task_id in task_ids if task_id in intended]
    train_overlap = overlap_stats(split_records["train"], selected)
    return {
        "count": len(task_ids),
        "unique_count": len(set(task_ids)),
        "duplicate_ids": duplicates,
        "missing_ids": missing,
        "wrong_split_ids": wrong_split,
        "ordered_ids_sha256": _text_sha256("\n".join(task_ids) + ("\n" if task_ids else "")),
        "train_exposure": train_overlap,
        "selected_records": selected,
    }


def audit_repository(repo_root: str | Path, replay_generator: bool = True) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    split_records = {
        name: load_jsonl(root / relative_path) for name, relative_path in SPLIT_FILES.items()
    }
    safe_split_records = {
        name: structurally_valid_records(records) for name, records in split_records.items()
    }
    split_reports = {
        name: {
            "path": SPLIT_FILES[name],
            "raw": file_record(root / SPLIT_FILES[name]),
            **audit_records(name, records),
        }
        for name, records in split_records.items()
    }

    all_ids = [record["id"] for records in safe_split_records.values() for record in records]
    global_duplicate_ids = sorted(key for key, count in Counter(all_ids).items() if count > 1)
    exact_overlaps = {}
    loose_overlaps = {}
    for left, right in itertools.combinations(SPLIT_FILES, 2):
        label = f"{left}__{right}"
        exact_overlaps[label] = overlap_stats(
            safe_split_records[left], safe_split_records[right]
        )
        loose_overlaps[label] = overlap_stats(
            safe_split_records[left], safe_split_records[right], key_fn=loose_key
        )

    frozen_reports = {}
    frozen_selected = {}
    for split_name, relative_path in FROZEN_FILES.items():
        path = root / relative_path
        ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        report = audit_frozen_ids(split_name, ids, safe_split_records)
        frozen_selected[split_name] = report.pop("selected_records")
        frozen_reports[split_name] = {
            "path": relative_path,
            "raw": file_record(path),
            **report,
        }
    frozen_pair_overlaps = {
        f"{left}__{right}": overlap_stats(frozen_selected[left], frozen_selected[right])
        for left, right in itertools.combinations(FROZEN_FILES, 2)
    }

    integrity_failures = []
    warnings = []
    for name, report in split_reports.items():
        if report["schema_errors"]:
            integrity_failures.append(f"{name}: schema errors")
        if report["duplicate_ids"] or report["within_split_semantic_duplicate_groups"]:
            integrity_failures.append(f"{name}: within-split duplicates")
        if report["solution_failure_ids"] or report["completion_failure_ids"]:
            integrity_failures.append(f"{name}: verifier failures")
        if any(value["count"] for value in report["prompt_drift"].values()):
            warnings.append(f"{name}: stored prompts differ from current generator")
        if report["default_prompt_mismatch_ids"]:
            warnings.append(f"{name}: default prompt differs from stored prompt_high")
        if report["completion_template_drift_ids"]:
            warnings.append(f"{name}: completion differs from current completion template")
        if report["operator_order_warning_ids"]:
            warnings.append(f"{name}: noncanonical operator presentation order")
    if global_duplicate_ids:
        integrity_failures.append("global duplicate IDs")
    for name, report in frozen_reports.items():
        if report["duplicate_ids"] or report["missing_ids"] or report["wrong_split_ids"]:
            integrity_failures.append(f"{name}: invalid frozen task IDs")

    cross_split_exact_groups = sum(
        report["shared_key_groups"] for report in exact_overlaps.values()
    )
    for label, loose in loose_overlaps.items():
        if loose["shared_key_groups"] > exact_overlaps[label]["shared_key_groups"]:
            warnings.append(f"{label}: loose-key-only overlap exists")
    corrected_eligible = not integrity_failures and cross_split_exact_groups == 0
    source_paths = {
        "generator": "scripts/00_generate_countdown_dataset.py",
        "prompts": "src/rtw_llm/prompts.py",
        "verifier": "src/rtw_llm/countdown.py",
        "audit": "src/rtw_llm/dataset_audit.py",
    }
    documentation = _documentation_recipe(root, split_records)
    has_structural_errors = any(
        len(safe_split_records[name]) != len(split_records[name]) for name in split_records
    )
    if replay_generator and has_structural_errors:
        replay = {
            "skipped_due_to_structural_errors": True,
            "all_non_prompt_fields_match": False,
            "by_split": {},
        }
    else:
        replay = _generator_replay(root, split_records) if replay_generator else None
    if any(
        item is not None and not item["matches_committed_counts"]
        for item in documentation.values()
    ):
        warnings.append("documented generation counts differ from committed datasets")
    if any(item is None for item in documentation.values()):
        warnings.append("one or more documentation files omit the generation recipe")
    if replay is not None and not replay["all_non_prompt_fields_match"]:
        warnings.append("inferred generator replay is skipped or does not match non-prompt fields")
    report: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA,
        "protocol_id": "countdown-legacy-v1",
        "git_head": _git_head(root),
        "sources": {
            role: {"path": path, "raw": file_record(root / path)}
            for role, path in source_paths.items()
        },
        "splits": split_reports,
        "global_duplicate_ids": global_duplicate_ids,
        "exact_overlap_matrix": exact_overlaps,
        "loose_overlap_matrix": loose_overlaps,
        "frozen": frozen_reports,
        "frozen_pair_overlap_matrix": frozen_pair_overlaps,
        "documentation": documentation,
        "generator_replay": replay,
        "verdict": {
            "legacy_integrity_pass": not integrity_failures,
            "corrected_v2_eligible": corrected_eligible,
            "status": (
                "INTEGRITY_FAIL"
                if integrity_failures
                else "ELIGIBLE"
                if corrected_eligible
                else "BLOCK_CORRECTED_V2"
            ),
            "integrity_failures": sorted(integrity_failures),
            "warnings": sorted(warnings),
            "cross_split_exact_shared_key_groups": cross_split_exact_groups,
        },
    }
    return report


def write_report(path: str | Path, report: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def assert_safe_report_path(repo_root: str | Path, output_path: str | Path) -> Path:
    """Restrict audit writes to dedicated report directories, never research inputs."""
    root = Path(repo_root).resolve()
    output = Path(output_path)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    allowed_roots = [(root / "docs/artifacts").resolve(), (root / "outputs/audits").resolve()]
    if not any(output == allowed or allowed in output.parents for allowed in allowed_roots):
        raise ValueError(
            f"Audit output must be under docs/artifacts or outputs/audits; got {output}"
        )
    return output


def _valid_record_types(record: dict[str, Any]) -> bool:
    return (
        isinstance(record["id"], str)
        and isinstance(record["split"], str)
        and isinstance(record["difficulty"], str)
        and isinstance(record["numbers"], list)
        and all(isinstance(value, int) and not isinstance(value, bool) for value in record["numbers"])
        and isinstance(record["target"], int)
        and not isinstance(record["target"], bool)
        and isinstance(record["allowed_ops"], list)
        and all(isinstance(value, str) for value in record["allowed_ops"])
        and all(isinstance(record[field], str) for field in REQUIRED_FIELDS - {"metadata", "numbers", "target", "allowed_ops"})
        and isinstance(record["metadata"], dict)
    )


def _difficulty_allowed(split_name: str, difficulty: str) -> bool:
    if split_name in {"train", "validation", "test_in_dist"}:
        return difficulty in IN_DIST_DIFFICULTIES
    return difficulty == split_name.removeprefix("test_")


def _group_ids(
    records: list[dict[str, Any]], key_fn: Callable[[dict[str, Any]], tuple[Any, ...]]
) -> dict[tuple[Any, ...], list[str]]:
    groups: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for record in records:
        groups[key_fn(record)].append(record["id"])
    return {key: sorted(ids) for key, ids in groups.items()}


def _generator_replay(
    repo_root: Path, split_records: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    path = repo_root / "scripts/00_generate_countdown_dataset.py"
    spec = importlib.util.spec_from_file_location("countdown_generator_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load dataset generator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recipe = {
        "train": (["easy", "medium", "hard"], 42),
        "validation": (["easy", "medium", "hard"], 43),
        "test_in_dist": (["easy", "medium", "hard"], 44),
        "test_ood_long": (["ood_long"], 45),
        "test_ood_division": (["ood_division"], 46),
    }
    by_split = {}
    for split_name, records in split_records.items():
        difficulties, seed = recipe[split_name]
        replayed = module.build_records(len(records), split_name, difficulties, seed)
        mismatches = []
        for index, (stored, candidate) in enumerate(zip(records, replayed)):
            if any(stored[field] != candidate[field] for field in NON_PROMPT_REPLAY_FIELDS):
                mismatches.append(index)
        by_split[split_name] = {
            "count": len(records),
            "seed": seed,
            "difficulties": difficulties,
            "non_prompt_fields_match": not mismatches,
            "first_mismatch_indices": mismatches[:10],
        }
    return {
        "base_seed": 42,
        "all_non_prompt_fields_match": all(
            item["non_prompt_fields_match"] for item in by_split.values()
        ),
        "by_split": by_split,
    }


def _documentation_recipe(
    repo_root: Path, split_records: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    pattern = re.compile(
        r"--train\s+(\d+).*?--valid\s+(\d+).*?--test\s+(\d+).*?--ood\s+(\d+).*?--seed\s+(\d+)",
        re.DOTALL,
    )
    out = {}
    for name in ("README.md", "Makefile"):
        text = (repo_root / name).read_text()
        match = pattern.search(text)
        if match:
            recipe = {
                "train": int(match.group(1)),
                "validation": int(match.group(2)),
                "test": int(match.group(3)),
                "ood_each": int(match.group(4)),
                "base_seed": int(match.group(5)),
            }
            recipe["matches_committed_counts"] = bool(
                recipe["train"] == len(split_records["train"])
                and recipe["validation"] == len(split_records["validation"])
                and recipe["test"] == len(split_records["test_in_dist"])
                and recipe["ood_each"] == len(split_records["test_ood_long"])
                and recipe["ood_each"] == len(split_records["test_ood_division"])
            )
            out[name] = recipe
        else:
            out[name] = None
    return out


def _git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _text_sha256(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()
