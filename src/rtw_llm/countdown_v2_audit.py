"""Independent fail-closed audit for the frozen Countdown-v2 protocol."""
from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .countdown import verify_completion, verify_expression
from .countdown_v2 import build_artifact_bytes, build_manifest, canonical_json_bytes
from .prompts import make_prompt, make_sft_completion
from .provenance import content_sha256, file_record

AUDIT_SCHEMA = "countdown-v2-audit-v1"
EXPECTED_PROTOCOL = "countdown-dataset-v2"
EXPECTED_GENERATOR_VERSION = "countdown-v2-generator-v1"
EXPECTED_MANIFEST_SCHEMA = "countdown-v2-manifest-v1"
EXPECTED_BASE_SEED = 20260710
EXPECTED_FINAL_POLICY = (
    "NO_MODEL_EVALUATION_UNTIL_METHODS_CLAIMS_ANALYSIS_AND_STOPPING_RULES_ARE_FROZEN"
)
EXPECTED_GENERATION_ORDER = ("easy", "medium", "hard", "ood_long", "ood_division")
EXPECTED_SLICE_ORDER = ("train", "validation", "test_in_dist", "final_test_in_dist")
EXPECTED_SPLITS = (
    "train",
    "validation",
    "test_in_dist",
    "final_test_in_dist",
    "test_ood_long",
    "test_ood_division",
)
EXPECTED_PROPOSAL_OFFSETS = {
    "easy": 0,
    "medium": 1,
    "hard": 2,
    "ood_long": 3,
    "ood_division": 4,
}
EXPECTED_ALLOCATION_OFFSETS = {"easy": 100, "medium": 101, "hard": 102}
EXPECTED_SPLIT_ORDER_OFFSETS = {
    "train": 200,
    "validation": 201,
    "test_in_dist": 202,
    "final_test_in_dist": 203,
    "test_ood_long": 204,
    "test_ood_division": 205,
}
EXPECTED_MAX_PROPOSALS = 250_000
EXPECTED_EASY_CAPACITY = 1_264
EXPECTED_QUOTAS: dict[str, dict[str, int]] = {
    "train": {"easy": 900, "medium": 2_050, "hard": 2_050},
    "validation": {"easy": 50, "medium": 225, "hard": 225},
    "test_in_dist": {"easy": 50, "medium": 225, "hard": 225},
    "final_test_in_dist": {"easy": 50, "medium": 225, "hard": 225},
    "test_ood_long": {"ood_long": 500},
    "test_ood_division": {"ood_division": 500},
}
EXPECTED_SPECS = {
    "easy": {"n_numbers": 3, "allowed_ops": ["+", "-"]},
    "medium": {"n_numbers": 4, "allowed_ops": ["+", "-", "*"]},
    "hard": {"n_numbers": 5, "allowed_ops": ["+", "-", "*"]},
    "ood_long": {"n_numbers": 6, "allowed_ops": ["+", "-", "*"]},
    "ood_division": {"n_numbers": 5, "allowed_ops": ["+", "-", "*", "/"]},
}
EXPECTED_SOURCE_PATHS = (
    "scripts/18_generate_countdown_v2.py",
    "src/rtw_llm/countdown_v2.py",
    "src/rtw_llm/countdown.py",
    "src/rtw_llm/prompts.py",
)
EXPECTED_FILES = tuple(
    [f"{split}.jsonl" for split in EXPECTED_SPLITS]
    + [f"task_ids/{split}.txt" for split in EXPECTED_SPLITS]
    + ["manifest.json"]
)
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
PINNED_LEGACY_HASHES = {
    "data/countdown/train.jsonl": "1324319ee6294e2b25a4220f7970715bfa63c8b41a704d298e0cb0a3cb708a2b",
    "data/countdown/validation.jsonl": "bd5fa14e31ab5a3b664e51b62bc54b2040afd75685a606279c739f3800d2d81d",
    "data/countdown/test_in_dist.jsonl": "ef07376962cba6c50d2042d08ac24d23484b6dbd9deb16dd71ed6100bfb60ba2",
    "data/countdown/test_ood_division.jsonl": "8fe562970f10f1003432d67f5e6e50036e89567d24b642b4658afa0d4608e170",
    "data/countdown/test_ood_long.jsonl": "601d86bf2b90b0fa5b297540fa16b9a82c60b9017036bbaf44b885e233c6113d",
    "outputs/v09_task_ids_validation_limit50.txt": "6fc3f3ca2b67b43bcde1b67a23f270ce49419e2988147bafb671425c30202ad6",
    "outputs/v09_task_ids_test_in_dist_limit50.txt": "81421f6c0f0e1f10b14b9234e3e0bf3c85880d556476d27ff65df7baa991fa4d",
    "outputs/v09_task_ids_test_ood_division_limit50.txt": "fd8d3f41862de7ba0d527224286a40ef9772214c33baf71d2ce98378b3d5a7bd",
    "outputs/v09_task_ids_test_ood_long_limit50.txt": "2fe61c942499a96eb51dc4df246507688dc192647697332b92673de034944715",
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def _semantic_key(record: dict[str, Any]) -> tuple[tuple[int, ...], int, tuple[str, ...]]:
    return (
        tuple(sorted(record["numbers"])),
        record["target"],
        tuple(record["allowed_ops"]),
    )


def _loose_key(record: dict[str, Any]) -> tuple[tuple[int, ...], int]:
    return tuple(sorted(record["numbers"])), record["target"]


def _set_digest(values: list[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(values)).encode()
    return hashlib.sha256(payload).hexdigest()


def _row_digest(record: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(record)).hexdigest()


def _identity_safe(record: dict[str, Any]) -> bool:
    return (
        isinstance(record.get("id"), str)
        and isinstance(record.get("numbers"), list)
        and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in record["numbers"]
        )
        and isinstance(record.get("target"), int)
        and not isinstance(record.get("target"), bool)
        and isinstance(record.get("allowed_ops"), list)
        and all(isinstance(value, str) for value in record["allowed_ops"])
    )


def _record_errors(split: str, records: list[dict[str, Any]]) -> list[str]:
    errors = []
    difficulty_counts = Counter()
    exact_keys = []
    loose_keys = []
    for index, record in enumerate(records):
        label = str(record.get("id", f"row-{index}"))
        missing = sorted(REQUIRED_FIELDS - set(record))
        if missing:
            errors.append(f"{label}: missing fields {missing}")
            continue
        if (
            not isinstance(record["id"], str)
            or not isinstance(record["split"], str)
            or record["split"] != split
            or not isinstance(record["difficulty"], str)
            or not isinstance(record["numbers"], list)
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in record["numbers"])
            or not isinstance(record["target"], int)
            or isinstance(record["target"], bool)
            or not isinstance(record["allowed_ops"], list)
            or not all(isinstance(value, str) for value in record["allowed_ops"])
            or not all(
                isinstance(record[field], str)
                for field in (
                    "solution",
                    "prompt_low",
                    "prompt_mid",
                    "prompt_high",
                    "prompt",
                    "completion",
                )
            )
            or not isinstance(record["metadata"], dict)
        ):
            errors.append(f"{label}: invalid field types or split")
            continue
        difficulty = record["difficulty"]
        spec = EXPECTED_SPECS.get(difficulty)
        if spec is None or difficulty not in EXPECTED_QUOTAS[split]:
            errors.append(f"{label}: invalid difficulty {difficulty!r} for {split}")
            continue
        difficulty_counts[difficulty] += 1
        if len(record["numbers"]) != spec["n_numbers"]:
            errors.append(f"{label}: wrong operand count")
        if record["allowed_ops"] != spec["allowed_ops"]:
            errors.append(f"{label}: wrong ordered operator list")
        if record["metadata"].get("dataset_protocol") != EXPECTED_PROTOCOL:
            errors.append(f"{label}: wrong metadata protocol")
        if record["metadata"].get("generator_version") != EXPECTED_GENERATOR_VERSION:
            errors.append(f"{label}: wrong metadata generator version")
        if record["metadata"].get("n_numbers") != spec["n_numbers"]:
            errors.append(f"{label}: wrong metadata operand count")
        if not verify_expression(
            record["solution"], record["numbers"], record["target"], record["allowed_ops"]
        ).correct:
            errors.append(f"{label}: stored solution fails verifier")
        if not verify_completion(record["completion"], record).correct:
            errors.append(f"{label}: stored completion fails verifier")
        for level in ("low", "mid", "high"):
            if record[f"prompt_{level}"] != make_prompt(
                record["numbers"], record["target"], record["allowed_ops"], level
            ):
                errors.append(f"{label}: prompt_{level} drift")
        if record["prompt"] != record["prompt_high"]:
            errors.append(f"{label}: default prompt mismatch")
        if record["completion"] != make_sft_completion(
            record["solution"], record["target"]
        ):
            errors.append(f"{label}: completion template drift")
        exact_keys.append(_semantic_key(record))
        loose_keys.append(_loose_key(record))
    if difficulty_counts != Counter(EXPECTED_QUOTAS[split]):
        errors.append(
            f"{split}: quota mismatch actual={dict(difficulty_counts)} "
            f"expected={EXPECTED_QUOTAS[split]}"
        )
    if len(exact_keys) != len(set(exact_keys)):
        errors.append(f"{split}: within-split exact semantic duplicates")
    if len(loose_keys) != len(set(loose_keys)):
        errors.append(f"{split}: within-split loose semantic duplicates")
    return errors


def _git_blob_record(repo_root: Path, commit: str, relative_path: str) -> dict[str, Any]:
    payload = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
        "line_count": payload.count(b"\n") + int(bool(payload) and not payload.endswith(b"\n")),
    }


def _audit_countdown_v2_impl(repo_root: str | Path, *, replay: bool = True) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    dataset_root = root / "data/countdown_v2"
    failures: list[str] = []
    actual_files = tuple(
        sorted(path.relative_to(dataset_root).as_posix() for path in dataset_root.rglob("*") if path.is_file())
    )
    if set(actual_files) != set(EXPECTED_FILES):
        failures.append(
            f"file set mismatch missing={sorted(set(EXPECTED_FILES) - set(actual_files))} "
            f"extra={sorted(set(actual_files) - set(EXPECTED_FILES))}"
        )
    manifest_path = dataset_root / "manifest.json"
    manifest = _load_json(manifest_path)
    supplied_digest = manifest.get("manifest_core_sha256")
    manifest_core = {key: value for key, value in manifest.items() if key != "manifest_core_sha256"}
    if supplied_digest != content_sha256(manifest_core):
        failures.append("manifest core digest mismatch")
    expected_manifest_fields = {
        "schema_version": EXPECTED_MANIFEST_SCHEMA,
        "protocol_id": EXPECTED_PROTOCOL,
        "generator_version": EXPECTED_GENERATOR_VERSION,
        "base_seed": EXPECTED_BASE_SEED,
        "generation_order": list(EXPECTED_GENERATION_ORDER),
        "in_dist_slice_order": list(EXPECTED_SLICE_ORDER),
        "split_order": list(EXPECTED_SPLITS),
        "proposal_seed_offsets": EXPECTED_PROPOSAL_OFFSETS,
        "allocation_seed_offsets": EXPECTED_ALLOCATION_OFFSETS,
        "split_order_seed_offsets": EXPECTED_SPLIT_ORDER_OFFSETS,
        "max_proposals_per_pool": EXPECTED_MAX_PROPOSALS,
        "easy_loose_key_capacity": EXPECTED_EASY_CAPACITY,
        "split_quotas": EXPECTED_QUOTAS,
        "final_test_policy": EXPECTED_FINAL_POLICY,
        "artifacts_exclude_manifest": True,
    }
    field_mismatches = [
        key for key, value in expected_manifest_fields.items() if manifest.get(key) != value
    ]
    if field_mismatches:
        failures.append(f"manifest frozen field mismatch: {field_mismatches}")
    expected_runtime = {
        "python_implementation": "cpython",
        "python_major_minor": "3.11",
        "rng": "random.Random",
        "serialization": "utf8_sorted_keys_compact_json_one_lf_per_record",
        "ordering": "sha256_seed_domain_semantic_key_then_semantic_key",
    }
    if manifest.get("runtime_contract") != expected_runtime:
        failures.append("manifest runtime contract mismatch")
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 11):
        failures.append("audit runtime is not pinned CPython 3.11")
    expected_pool_targets = {"easy": 1_050, "medium": 2_725, "hard": 2_725, "ood_long": 500, "ood_division": 500}
    pool_stats = manifest.get("generation_stats", {}).get("pool_stats", {})
    for difficulty in EXPECTED_GENERATION_ORDER:
        stats = pool_stats.get(difficulty, {})
        if stats.get("target") != expected_pool_targets[difficulty]:
            failures.append(f"manifest pool target mismatch: {difficulty}")
        if stats.get("proposal_seed") != EXPECTED_BASE_SEED + EXPECTED_PROPOSAL_OFFSETS[difficulty]:
            failures.append(f"manifest proposal seed mismatch: {difficulty}")
        proposals = stats.get("proposals")
        if not isinstance(proposals, int) or not 0 < proposals <= EXPECTED_MAX_PROPOSALS:
            failures.append(f"manifest proposal count invalid: {difficulty}")
        elif stats.get("duplicate_rejections") != proposals - expected_pool_targets[difficulty]:
            failures.append(f"manifest rejection accounting mismatch: {difficulty}")
    if manifest.get("generation_stats", {}).get("global_loose_keys") != 7_500:
        failures.append("manifest global loose-key count mismatch")
    if manifest.get("generation_stats", {}).get("max_proposals_per_pool") != EXPECTED_MAX_PROPOSALS:
        failures.append("manifest generation-stat proposal budget mismatch")

    artifacts = manifest.get("artifacts", {})
    expected_artifact_files = set(EXPECTED_FILES) - {"manifest.json"}
    if set(artifacts) != expected_artifact_files:
        failures.append("manifest artifact file set mismatch")
    for relative in sorted(expected_artifact_files & set(artifacts)):
        if file_record(dataset_root / relative) != artifacts[relative]:
            failures.append(f"artifact digest mismatch: {relative}")

    source_commit = manifest.get("source_commit")
    source_records = manifest.get("source_records", {})
    if set(source_records) != set(EXPECTED_SOURCE_PATHS):
        failures.append("manifest source file set mismatch")
    elif (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        failures.append("manifest source commit missing")
    else:
        for relative in EXPECTED_SOURCE_PATHS:
            actual = file_record(root / relative)
            if actual != source_records[relative]:
                failures.append(f"current source hash mismatch: {relative}")
            try:
                at_commit = _git_blob_record(root, source_commit, relative)
            except subprocess.CalledProcessError:
                failures.append(f"source commit cannot resolve {relative}")
            else:
                if at_commit != source_records[relative]:
                    failures.append(f"source commit hash mismatch: {relative}")

    split_records = {
        split: _load_jsonl(dataset_root / f"{split}.jsonl") for split in EXPECTED_SPLITS
    }
    safe_split_records = {
        split: [record for record in records if _identity_safe(record)]
        for split, records in split_records.items()
    }
    split_reports = {}
    all_ids = []
    for split, records in split_records.items():
        errors = _record_errors(split, records)
        failures.extend(errors)
        ids = [record.get("id") for record in records]
        all_ids.extend(ids)
        ordered_ids = [
            line.strip()
            for line in (dataset_root / f"task_ids/{split}.txt").read_text().splitlines()
            if line.strip()
        ]
        if ordered_ids != ids:
            failures.append(f"{split}: ordered ID file mismatch")
        split_reports[split] = {
            "count": len(records),
            "difficulty_counts": dict(Counter(record.get("difficulty") for record in records)),
            "raw": file_record(dataset_root / f"{split}.jsonl"),
            "ordered_ids_raw": file_record(dataset_root / f"task_ids/{split}.txt"),
            "error_count": len(errors),
            "first_errors": errors[:10],
        }
    if len(all_ids) != len(set(all_ids)):
        failures.append("global duplicate IDs")

    overlap_matrix = {}
    for left, right in itertools.combinations(EXPECTED_SPLITS, 2):
        left_exact = {_semantic_key(record) for record in safe_split_records[left]}
        right_exact = {_semantic_key(record) for record in safe_split_records[right]}
        left_loose = {_loose_key(record) for record in safe_split_records[left]}
        right_loose = {_loose_key(record) for record in safe_split_records[right]}
        exact_count = len(left_exact & right_exact)
        loose_count = len(left_loose & right_loose)
        overlap_matrix[f"{left}__{right}"] = {
            "exact_shared_groups": exact_count,
            "loose_shared_groups": loose_count,
        }
        if exact_count or loose_count:
            failures.append(
                f"{left}__{right}: cross-split overlap exact={exact_count} loose={loose_count}"
            )

    final_records = safe_split_records["final_test_in_dist"]
    final_ids = [record["id"] for record in final_records]
    final_semantics = [json.dumps(_semantic_key(record), separators=(",", ":")) for record in final_records]
    final_loose_keys = [json.dumps(_loose_key(record), separators=(",", ":")) for record in final_records]
    final_digests = [_row_digest(record) for record in final_records]
    expected_protection = {
        "row_count": 500,
        "jsonl_sha256": file_record(dataset_root / "final_test_in_dist.jsonl")["sha256"],
        "ordered_ids_sha256": file_record(
            dataset_root / "task_ids/final_test_in_dist.txt"
        )["sha256"],
        "id_set_sha256": _set_digest(final_ids),
        "semantic_key_set_sha256": _set_digest(final_semantics),
        "loose_key_set_sha256": _set_digest(final_loose_keys),
        "canonical_row_digest_set_sha256": _set_digest(final_digests),
    }
    if manifest.get("final_test_protection") != expected_protection:
        failures.append("final-test protection metadata mismatch")

    legacy_report = {}
    for relative, expected_hash in PINNED_LEGACY_HASHES.items():
        actual_hash = file_record(root / relative)["sha256"]
        legacy_report[relative] = {"expected_sha256": expected_hash, "actual_sha256": actual_hash}
        if actual_hash != expected_hash:
            failures.append(f"legacy artifact changed: {relative}")

    replay_report: dict[str, Any] | None = None
    if replay:
        replay_artifacts, replay_stats, replay_records = build_artifact_bytes(
            base_seed=EXPECTED_BASE_SEED,
            split_quotas=EXPECTED_QUOTAS,
            max_proposals=EXPECTED_MAX_PROPOSALS,
        )
        mismatched_artifacts = [
            relative
            for relative, payload in replay_artifacts.items()
            if payload != (dataset_root / relative).read_bytes()
        ]
        replay_manifest = build_manifest(
            source_commit=source_commit,
            source_records={relative: file_record(root / relative) for relative in EXPECTED_SOURCE_PATHS},
            artifacts=replay_artifacts,
            stats=replay_stats,
            records=replay_records,
            base_seed=EXPECTED_BASE_SEED,
            split_quotas=EXPECTED_QUOTAS,
        )
        manifest_matches = replay_manifest == manifest
        if mismatched_artifacts:
            failures.append(f"deterministic replay artifact mismatch: {mismatched_artifacts}")
        if not manifest_matches:
            failures.append("deterministic replay manifest mismatch")
        replay_report = {
            "artifact_bytes_match": not mismatched_artifacts,
            "manifest_payload_matches": manifest_matches,
            "mismatched_artifacts": mismatched_artifacts,
        }

    report = {
        "schema_version": AUDIT_SCHEMA,
        "protocol_id": EXPECTED_PROTOCOL,
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "manifest_raw": file_record(manifest_path),
        "manifest_source_commit": source_commit,
        "audit_sources": {
            relative: file_record(root / relative)
            for relative in (
                "scripts/19_audit_countdown_v2.py",
                "src/rtw_llm/countdown_v2_audit.py",
                "src/rtw_llm/countdown.py",
                "src/rtw_llm/prompts.py",
            )
        },
        "splits": split_reports,
        "overlap_matrix": overlap_matrix,
        "legacy_artifacts": legacy_report,
        "deterministic_replay": replay_report,
        "verdict": {
            "status": "ELIGIBLE" if not failures else "INTEGRITY_FAIL",
            "eligible_for_corrected_v2": not failures,
            "failure_count": len(failures),
            "failures": sorted(set(failures)),
            "final_test_released": False,
        },
    }
    return report


def audit_countdown_v2(repo_root: str | Path, *, replay: bool = True) -> dict[str, Any]:
    """Return an integrity verdict even when repository inputs are malformed."""
    root = Path(repo_root).resolve()
    try:
        return _audit_countdown_v2_impl(root, replay=replay)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}".replace(str(root), "<repo>")
        try:
            git_head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            git_head = None
        manifest_path = root / "data/countdown_v2/manifest.json"
        manifest_raw = None
        if manifest_path.is_file():
            try:
                manifest_raw = file_record(manifest_path)
            except Exception:
                manifest_raw = None
        return {
            "schema_version": AUDIT_SCHEMA,
            "protocol_id": EXPECTED_PROTOCOL,
            "git_head": git_head,
            "manifest_raw": manifest_raw,
            "manifest_source_commit": None,
            "splits": {},
            "overlap_matrix": {},
            "legacy_artifacts": {},
            "deterministic_replay": None,
            "verdict": {
                "status": "INTEGRITY_FAIL",
                "eligible_for_corrected_v2": False,
                "failure_count": 1,
                "failures": [f"audit input failure: {message}"],
                "final_test_released": False,
            },
        }


def write_audit_report(path: str | Path, report: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
