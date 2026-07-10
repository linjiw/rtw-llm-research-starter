#!/usr/bin/env python
"""Generate and atomically publish the frozen Countdown-v2 dataset."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from rtw_llm.countdown_v2 import (
    BASE_SEED,
    SOURCE_PATHS,
    SPLIT_ORDER,
    build_artifact_bytes,
    build_manifest,
    write_dataset_atomic,
)
from rtw_llm.provenance import file_record


def clean_source_commit(repo_root: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("Countdown-v2 generation requires a clean committed worktree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def require_runtime() -> None:
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            "Countdown-v2 byte replay is pinned to CPython 3.11; "
            f"found {sys.implementation.name} {sys.version_info.major}.{sys.version_info.minor}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out_dir", type=Path, default=Path("data/countdown_v2"))
    args = parser.parse_args()

    root = args.repo_root.resolve()
    output = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    if output.resolve() != (root / "data/countdown_v2").resolve():
        raise ValueError("Protocol generation output is frozen to data/countdown_v2")
    require_runtime()
    source_commit = clean_source_commit(root)
    source_records = {path: file_record(root / path) for path in SOURCE_PATHS}
    artifacts, stats, records = build_artifact_bytes(base_seed=BASE_SEED)
    manifest = build_manifest(
        source_commit=source_commit,
        source_records=source_records,
        artifacts=artifacts,
        stats=stats,
        records=records,
        base_seed=BASE_SEED,
    )
    write_dataset_atomic(
        output,
        artifacts=artifacts,
        manifest=manifest,
        legacy_dir=root / "data/countdown",
    )
    print(
        json.dumps(
            {
                "status": "generated",
                "protocol_id": manifest["protocol_id"],
                "source_commit": source_commit,
                "manifest_core_sha256": manifest["manifest_core_sha256"],
                "output_dir": str(output),
                "splits": {split: len(records[split]) for split in SPLIT_ORDER},
                "pool_stats": stats["pool_stats"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
