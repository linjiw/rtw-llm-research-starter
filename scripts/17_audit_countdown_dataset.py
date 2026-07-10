#!/usr/bin/env python
"""Write the deterministic Countdown legacy integrity/leakage audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rtw_llm.dataset_audit import assert_safe_report_path, audit_repository, write_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--out_json",
        type=Path,
        default=Path("docs/artifacts/countdown_legacy_v1_audit.json"),
    )
    parser.add_argument("--skip_generator_replay", action="store_true")
    parser.add_argument("--require_corrected_eligible", action="store_true")
    args = parser.parse_args()

    report = audit_repository(args.repo_root, replay_generator=not args.skip_generator_replay)
    output = assert_safe_report_path(args.repo_root, args.out_json)
    write_report(output, report)
    print(json.dumps(report["verdict"], indent=2, sort_keys=True))
    print(f"Wrote {output}")
    if args.require_corrected_eligible and not report["verdict"]["corrected_v2_eligible"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
