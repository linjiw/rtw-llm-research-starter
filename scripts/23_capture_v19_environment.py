#!/usr/bin/env python
"""Capture, but never overwrite, the v0.19 production CUDA environment lock."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rtw_llm.v19_protocol import ENVIRONMENT_LOCK, capture_environment_lock


def write_new(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--container_image_digest", default=None)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    payload = capture_environment_lock(
        root, container_image_digest=args.container_image_digest
    )
    output = root / ENVIRONMENT_LOCK
    write_new(output, payload)
    print(json.dumps({"status": "captured", "path": str(output), **payload}, indent=2))


if __name__ == "__main__":
    main()
