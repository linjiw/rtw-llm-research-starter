"""JSONL helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    records = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
