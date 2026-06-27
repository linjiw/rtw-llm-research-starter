#!/usr/bin/env python
"""Create simple plots from eval and teacher logs."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from rtw_llm.analysis import load_jsonl_df


def plot_teacher_weights(path: Path, output_dir: Path) -> None:
    df = load_jsonl_df(path)
    if df.empty:
        return
    weights = pd.json_normalize(df["weights"])
    weights["step"] = df["step"]
    for col in [c for c in weights.columns if c != "step"]:
        plt.figure()
        plt.plot(weights["step"], weights[col])
        plt.xlabel("Teacher update")
        plt.ylabel("Auxiliary reward weight")
        plt.title(f"RTW weight evolution: {col}")
        plt.tight_layout()
        plt.savefig(output_dir / f"teacher_weight_{col}.png", dpi=180)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default="outputs/grpo_rtw_qwen05b")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    figs = run_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    teacher_log = run_dir / "teacher_weights.jsonl"
    if teacher_log.exists():
        plot_teacher_weights(teacher_log, figs)
        print(f"Wrote figures to {figs}")
    else:
        print(f"No teacher log found: {teacher_log}")


if __name__ == "__main__":
    main()
