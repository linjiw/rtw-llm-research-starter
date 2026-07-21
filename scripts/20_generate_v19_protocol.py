#!/usr/bin/env python
"""Generate the frozen v0.19 validation views and protocol manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rtw_llm.v19_protocol import (
    PROTOCOL_DIR,
    build_protocol_artifacts,
    write_protocol_atomic,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.repo_root.resolve()
    artifacts, manifest = build_protocol_artifacts(root)
    write_protocol_atomic(root / PROTOCOL_DIR, artifacts=artifacts, manifest=manifest)
    print(
        json.dumps(
            {
                "status": "generated",
                "protocol_id": manifest["protocol_id"],
                "manifest_core_sha256": manifest["manifest_core_sha256"],
                "artifacts": manifest["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
