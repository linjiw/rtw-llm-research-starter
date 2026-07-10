#!/usr/bin/env python
"""Replay-audit the frozen v0.19 protocol without reading confirmation outcomes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rtw_llm.v19_protocol import audit_protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report = audit_protocol(args.repo_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["eligible"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
