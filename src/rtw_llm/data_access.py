"""Fail-closed access controls for the sealed Countdown-v2 final test."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Literal

from .cluster_stats import semantic_task_key
from .countdown_v2 import FINAL_TEST_POLICY, canonical_json_bytes
from .provenance import content_sha256, file_record

FINAL_RELEASE_SCHEMA = "countdown-v2-final-release-v1"
FINAL_SPLIT_NAME = "final_test_in_dist"
FINAL_JSONL = f"{FINAL_SPLIT_NAME}.jsonl"
FINAL_IDS = f"task_ids/{FINAL_SPLIT_NAME}.txt"
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
FINAL_RELEASE_CAPABLE_RUNNERS = {"07_best_of_n_rerank"}


class DataAccessError(RuntimeError):
    """Raised when an official runner attempts unauthorized final-test access."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DataAccessError(f"Cannot read protected-data metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DataAccessError(f"Protected-data metadata must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise DataAccessError(f"{path}:{line_number}: row must be an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise DataAccessError(f"Cannot inspect data access for {path}: {exc}") from exc
    return rows


def _row_digest(row: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(row)).hexdigest()


def _loose_task_key(row: dict[str, Any]) -> str:
    semantic = json.loads(semantic_task_key(row))
    return json.dumps(
        {"numbers": semantic["numbers"], "target": semantic["target"]},
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_manifest(manifest: dict[str, Any]) -> None:
    supplied = manifest.get("manifest_core_sha256")
    core = {key: value for key, value in manifest.items() if key != "manifest_core_sha256"}
    if supplied != content_sha256(core):
        raise DataAccessError("Countdown-v2 manifest core digest mismatch")
    if manifest.get("final_test_policy") != FINAL_TEST_POLICY:
        raise DataAccessError("Countdown-v2 final-test policy mismatch")


def _final_reference(
    dataset_root: Path, manifest: dict[str, Any]
) -> tuple[list[dict[str, Any]], set[str], set[str], set[str], set[str]]:
    final_path = dataset_root / FINAL_JSONL
    ids_path = dataset_root / FINAL_IDS
    artifacts = manifest.get("artifacts", {})
    for relative, path in ((FINAL_JSONL, final_path), (FINAL_IDS, ids_path)):
        expected = artifacts.get(relative)
        if not isinstance(expected, dict) or file_record(path) != expected:
            raise DataAccessError(f"Protected final artifact does not match manifest: {relative}")
    rows = _read_jsonl(final_path)
    ids = [line.strip() for line in ids_path.read_text().splitlines() if line.strip()]
    if ids != [str(row.get("id")) for row in rows]:
        raise DataAccessError("Protected final ordered IDs do not match final JSONL")
    return (
        rows,
        set(ids),
        {semantic_task_key(row) for row in rows},
        {_loose_task_key(row) for row in rows},
        {_row_digest(row) for row in rows},
    )


def _git_output(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DataAccessError(f"Cannot verify final-test release Git state: {exc}") from exc


def _validate_release(
    release_path: Path,
    *,
    runner: str,
    repo_root: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> None:
    if runner not in FINAL_RELEASE_CAPABLE_RUNNERS:
        raise DataAccessError(f"Runner {runner!r} is never final-release authorized")
    release = _read_json(release_path)
    protection = manifest["final_test_protection"]
    expected = {
        "schema_version": FINAL_RELEASE_SCHEMA,
        "dataset_manifest_sha256": file_record(manifest_path)["sha256"],
        "final_jsonl_sha256": protection["jsonl_sha256"],
        "final_ordered_ids_sha256": protection["ordered_ids_sha256"],
        "final_test_policy": FINAL_TEST_POLICY,
    }
    mismatches = [key for key, value in expected.items() if release.get(key) != value]
    if mismatches:
        raise DataAccessError(f"Final-test release record mismatch: {sorted(mismatches)}")
    if release.get("human_approval") is not True:
        raise DataAccessError("Final-test release requires explicit human_approval=true")
    authorized_runners = release.get("authorized_runners")
    if (
        not isinstance(authorized_runners, list)
        or not all(isinstance(value, str) for value in authorized_runners)
        or runner not in authorized_runners
    ):
        raise DataAccessError(f"Final-test release does not authorize runner {runner!r}")
    frozen_commit = release.get("frozen_commit")
    if not isinstance(frozen_commit, str) or not COMMIT_RE.fullmatch(frozen_commit):
        raise DataAccessError("Final-test release has invalid frozen_commit")
    if _git_output(repo_root, "rev-parse", "HEAD") != frozen_commit:
        raise DataAccessError("Current HEAD does not equal final-test frozen_commit")
    if _git_output(repo_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise DataAccessError("Final-test evaluation requires a completely clean worktree")


def assert_countdown_data_access(
    data_path: str | Path,
    *,
    purpose: Literal["training", "training_eval", "model_eval"],
    runner: str,
    release_record: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> None:
    """Reject any unreleased row-level intersection with the v2 final test."""
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    dataset_root = root / "data/countdown_v2"
    manifest_path = dataset_root / "manifest.json"
    path = Path(data_path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not manifest_path.exists():
        if dataset_root.exists():
            raise DataAccessError(
                "Countdown-v2 directory exists without its required manifest; "
                "all official data access is blocked"
            )
        if path.name == FINAL_JSONL:
            raise DataAccessError("Final-test path is blocked before a verified v2 manifest exists")
        return
    manifest = _read_json(manifest_path)
    _validate_manifest(manifest)
    final_rows, final_ids, final_semantics, final_loose_keys, final_digests = _final_reference(
        dataset_root, manifest
    )
    input_rows = _read_jsonl(path)
    input_ids = {str(row.get("id")) for row in input_rows}
    try:
        input_semantics = {semantic_task_key(row) for row in input_rows}
        input_loose_keys = {_loose_task_key(row) for row in input_rows}
    except Exception as exc:
        raise DataAccessError(f"Cannot canonicalize input task semantics: {exc}") from exc
    input_digests = {_row_digest(row) for row in input_rows}
    intersections = {
        "ids": len(input_ids & final_ids),
        "semantic_keys": len(input_semantics & final_semantics),
        "loose_keys": len(input_loose_keys & final_loose_keys),
        "row_digests": len(input_digests & final_digests),
    }
    if not any(intersections.values()):
        return
    if purpose != "model_eval":
        raise DataAccessError(
            f"Sealed final-test rows are forbidden for {purpose}: intersections={intersections}"
        )
    if file_record(path)["sha256"] != manifest["final_test_protection"]["jsonl_sha256"]:
        raise DataAccessError("Released final evaluation requires the exact complete final JSONL")
    if len(input_rows) != len(final_rows):
        raise DataAccessError("Released final evaluation requires every final-test row")
    if release_record is None:
        raise DataAccessError("Final-test model evaluation requires an explicit release record")
    release_path = Path(release_record)
    if not release_path.is_absolute():
        release_path = root / release_path
    _validate_release(
        release_path,
        runner=runner,
        repo_root=root,
        manifest_path=manifest_path,
        manifest=manifest,
    )
