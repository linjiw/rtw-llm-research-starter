#!/usr/bin/env python
"""Audit Countdown-v2 without evaluating any model on the sealed final test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rtw_llm.countdown_v2_audit import audit_countdown_v2, write_audit_report
from rtw_llm.dataset_audit import assert_safe_report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--out_json",
        type=Path,
        default=Path("docs/artifacts/countdown_v2_audit.json"),
    )
    parser.add_argument("--skip_replay", action="store_true")
    args = parser.parse_args()

    report = audit_countdown_v2(args.repo_root, replay=not args.skip_replay)
    output = assert_safe_report_path(args.repo_root, args.out_json)
    write_audit_report(output, report)
    print(json.dumps(report["verdict"], indent=2, sort_keys=True))
    print(f"Wrote {output}")
    if not report["verdict"]["eligible_for_corrected_v2"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
