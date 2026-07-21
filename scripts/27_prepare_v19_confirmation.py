#!/usr/bin/env python
"""Create the commit-before-access v0.19 confirmation-readiness record."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rtw_llm.v19_protocol import (
    CONFIRMATION_READY_RECORD,
    DEV_SCORE_MANIFEST,
    DEV_SCORE_REPORT,
    build_confirmation_ready_record,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--runs_root", type=Path, default=Path("outputs/v19/production"))
    parser.add_argument("--dev_score_report", type=Path, default=DEV_SCORE_REPORT)
    parser.add_argument("--dev_score_manifest", type=Path, default=DEV_SCORE_MANIFEST)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    payload = build_confirmation_ready_record(
        root,
        runs_root=args.runs_root,
        dev_score_report=args.dev_score_report,
        dev_score_manifest=args.dev_score_manifest,
    )
    output = root / CONFIRMATION_READY_RECORD
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": "confirmation_ready", "path": str(output)}, indent=2))


if __name__ == "__main__":
    main()
