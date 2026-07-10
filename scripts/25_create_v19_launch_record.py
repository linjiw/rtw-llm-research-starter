#!/usr/bin/env python
"""Create the commit-before-compute v0.19 launch authorization record."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rtw_llm.v19_protocol import LAUNCH_RECORD, build_launch_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--approved_host_label", required=True)
    parser.add_argument("--usd_per_gpu_hour", type=float, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = root / LAUNCH_RECORD
    payload = build_launch_record(
        root,
        approved_host_label=args.approved_host_label,
        usd_per_gpu_hour=args.usd_per_gpu_hour,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": "created", "path": str(output), **payload}, indent=2))


if __name__ == "__main__":
    main()
