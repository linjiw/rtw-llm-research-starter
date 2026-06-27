"""Analysis helpers for metrics and reward-weight logs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_jsonl_df(path: str | Path) -> pd.DataFrame:
    rows = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def flatten_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if "metrics" not in df.columns:
        return df
    metrics_df = pd.json_normalize(df["metrics"])
    return pd.concat([df.drop(columns=["metrics"]), metrics_df], axis=1)
