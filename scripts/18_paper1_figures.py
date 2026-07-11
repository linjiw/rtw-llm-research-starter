#!/usr/bin/env python
"""Paper-1 FIGURE generator (companion to scripts/17_paper1_assets.py).

scripts/17 emits the authoritative claim TABLES (docs/PAPER1_ASSETS.md +
outputs/paper1_assets.json). This script renders the paper FIGURES for the same
shaping-vs-capability characterization (claims C1-C5), reading every shared
scalar straight from outputs/paper1_assets.json so there is exactly ONE source
of truth and the figures can never drift from the tables. Figure-only diagnostic
panels (tier-collapse, loss decomposition) additionally read the committed probe
artifacts. C6 (harness-shift/OOD) is skipped-with-log until its artifacts land.

Run scripts/17 FIRST (it writes paper1_assets.json). This script is additive,
read-only, CPU-only, and never touches frozen components. Re-running after
v13 seeds 1/2 + OOD land refreshes both tables (via 17) and figures with no
code edits.

Usage:
  uv run python scripts/17_paper1_assets.py            # tables + JSON (run first)
  uv run python scripts/18_paper1_figures.py           # figures from that JSON
  uv run python scripts/18_paper1_figures.py --self_check
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs"
ASSETS_JSON = OUT / "paper1_assets.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)
log = logging.getLogger("paper1_figures")

PALETTE = {
    "base": "#8c8c8c", "static": "#4c78a8", "stable": "#f58518",
    "sft_only": "#54a24b", "sft_grpo": "#b279a2", "accent": "#e45756", "muted": "#bab0ac",
}
plt.rcParams.update({
    "savefig.bbox": "tight", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _save(fig, name: str, out_dir: Path, formats: list[str], dpi: int, dry: bool) -> list[str]:
    if dry:
        plt.close(fig)
        return [f"{name}.{ext}" for ext in formats]
    written = []
    for ext in formats:
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=dpi if ext == "png" else None)
        written.append(str(p))
    plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# C1 -- selection saturates: per-bank oracle vs reranked scatter (from 17's JSON)
#       + loss decomposition inset (from probe_b).
# ---------------------------------------------------------------------------
def fig_c1(assets, out_dir, formats, dpi, dry):
    c1 = assets.get("C1_selection_saturation")
    if not c1:
        log.warning("[C1] skipped -- no C1_selection_saturation in %s", ASSETS_JSON.name)
        return None
    per_bank = c1["per_bank"]
    xs = [b["oracle@8"] for b in per_bank]
    ys = [b["reranked@8"] for b in per_bank]
    # invariant: reranked <= oracle in every bank
    assert all(b["reranked@8"] <= b["oracle@8"] for b in per_bank), "C1: reranked>oracle (selector bug)"

    probe_b = _load(OUT / "probe_b_generation_headroom.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), width_ratios=[1.3, 1])
    ax = axes[0]
    hi = max(xs + [1]) * 1.1
    ax.plot([0, hi], [0, hi], "--", color=PALETTE["muted"], lw=1, label="y = x (perfect selection)")
    ax.scatter(xs, ys, color=PALETTE["stable"], s=45, zorder=3, edgecolor="white", lw=0.5)
    ax.set_xlabel("oracle_exact@8 (exact candidate exists, task count)")
    ax.set_ylabel("reranked_exact@8 (selector recovers it)")
    ax.set_title(c1["verdict"], fontsize=9)
    ax.legend(loc="upper left", fontsize=8)

    ax2 = axes[1]
    if probe_b:
        marg = probe_b["pooled"]["marginal"]
        n_units = marg["n_units"]
        ever = marg["n_units_ever_solved_by_n8"]
        no_exact = probe_b["pooled"]["clip_recovery"]["n_no_exact_units"]
        missel = n_units - ever - no_exact
        labels = ["oracle-\nrecoverable", "misselected", "no exact\ncandidate"]
        shares = [ever / n_units, missel / n_units, no_exact / n_units]
        ax2.bar(labels, [s * 100 for s in shares],
                color=[PALETTE["sft_only"], PALETTE["accent"], PALETTE["static"]])
        for i, s in enumerate(shares):
            ax2.text(i, s * 100 + 1, f"{s*100:.2f}%", ha="center", fontsize=8)
        ax2.set_ylabel(f"% of {n_units} task-instances")
        ax2.set_ylim(0, 100)
        ax2.set_title("Where exact solutions are lost", fontsize=9)
    else:
        ax2.text(0.5, 0.5, "probe_b absent", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_axis_off()
    fig.suptitle("C1 -- The bottleneck is generation, not selection", fontweight="bold")
    return _save(fig, "fig_c1_selection_saturation", out_dir, formats, dpi, dry)


# ---------------------------------------------------------------------------
# C3 -- capability ladder (base -> RL-only -> SFT-only -> SFT+GRPO), test@8.
#       Reads bank-derived oracle@8 seed counts from 17's JSON for the stamp.
# ---------------------------------------------------------------------------
def fig_c3(assets, out_dir, formats, dpi, dry):
    c3 = assets.get("C3_sft_capability")
    if not c3:
        log.warning("[C3] skipped -- no C3_sft_capability in %s", ASSETS_JSON.name)
        return None
    # Ladder held constant at (split=test_in_dist, N=8). Base + RL-only from the
    # gate0 CSV (single/3-seed); SFT rungs from 17's JSON bank-derived oracle@8.
    base_row = rl_row = None
    csv_path = OUT / "gate0_local_ladder_seeds012_summary.csv"
    base_csv = OUT / "gate0_local_ladder_summary.csv"
    import csv as _csv

    def _read(path, method):
        if not path.exists():
            return None
        for row in _csv.DictReader(path.open()):
            if row["split"] == "test_in_dist" and row["method"] == method and int(row["N"]) == 8:
                return row
        return None

    base_row = _read(base_csv, "base")
    rl_row = _read(csv_path, "stable")
    ti = c3.get("test_in_dist", {})
    seed_oracle = ti.get("v13_oracle@8_by_seed", {})
    # SFT+GRPO bank-derived exact@8 rate = oracle count / n_tasks, seed 0 present.
    seeds_present = c3.get("seeds_present", [])
    n_seeds = len(seeds_present)
    if not seed_oracle:
        log.warning("[C3] skipped -- no v13 bank oracle@8 in JSON yet")
        return None
    grpo_rate = mean_rate(seed_oracle)

    # SFT-only rung: prefer the pre-scored JSON (authoritative), fall back to note.
    sftonly = _load(OUT / "v13_score_sftonly_test.json")
    sfto_rate = None
    if sftonly:
        arm = next((a for a in sftonly.get("arms", []) if a.get("arm") == "v13sftonly"), None)
        if arm:
            sfto_rate = arm["oracle_exact_at_n"]

    # (label, value, std, color) per rung; only rungs with data are appended.
    ladder = []
    if base_row:
        ladder.append(("base", float(base_row["reranked_exact_mean"]), 0.0, PALETTE["base"]))
    if rl_row:
        ladder.append(("RL-only\n(stable)", float(rl_row["reranked_exact_mean"]),
                       float(rl_row["oracle_exact_std"]), PALETTE["stable"]))
    if sfto_rate is not None:
        ladder.append(("SFT-only", sfto_rate, 0.0, PALETTE["sft_only"]))
    ladder.append(("SFT+GRPO", grpo_rate, 0.0, PALETTE["sft_grpo"]))
    rungs = [r[0] for r in ladder]
    vals = [r[1] for r in ladder]
    stds = [r[2] for r in ladder]
    colors = [r[3] for r in ladder]

    # invariant: ladder monotone non-decreasing
    assert all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1)), f"ladder not monotone: {vals}"

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = range(len(rungs))
    ax.bar(list(x), vals, color=colors, width=0.6, zorder=2, yerr=stds, capsize=4)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(rungs)
    ax.set_ylabel("test_in_dist exact@8")
    ax.set_ylim(0, max(vals) * 1.25)
    if rl_row and vals[-1]:
        ratio = grpo_rate / float(rl_row["reranked_exact_mean"])
        ax.annotate(f"~{ratio:.1f}x", xy=(len(rungs) - 1, grpo_rate),
                    xytext=(len(rungs) / 2, grpo_rate), fontsize=11, color=PALETTE["accent"],
                    ha="center", arrowprops=dict(arrowstyle="<->", color=PALETTE["accent"]))
    seed_note = "seed 0 only, confirmatory seeds pending" if n_seeds <= 1 else f"{n_seeds} seeds"
    ax.set_title(f"C3 -- capability lever, not shaping, moves the wall ({seed_note})")
    return _save(fig, "fig_c3_capability_ladder", out_dir, formats, dpi, dry)


def mean_rate(seed_oracle: dict) -> float:
    """SFT+GRPO exact@8 rate averaged over the v13 seeds present (bank-derived)."""
    rates = []
    for _s, v in seed_oracle.items():
        n = v.get("n_tasks") or 50
        rates.append(v["oracle@8"] / n)
    return sum(rates) / len(rates) if rates else 0.0


# ---------------------------------------------------------------------------
# C4 -- tier collapse + variance saturation (figure-only; from probe artifacts).
# ---------------------------------------------------------------------------
def fig_c4(assets, out_dir, formats, dpi, dry):
    probe_b = _load(OUT / "probe_b_generation_headroom.json")
    probe_a = _load(OUT / "probe_a_tier_scoped.json")
    if not (probe_b and probe_a):
        log.warning("[C4] skipped -- probe_a/probe_b absent")
        return None
    tiers = ["easy", "medium", "hard"]
    rates, solved, totals = [], [], []
    for t in tiers:
        m = probe_b["per_tier"][t]["marginal"]
        rates.append(m["oracle_exact_at_n"][-1])
        solved.append(m["n_units_ever_solved_by_n8"])
        totals.append(m["n_units"])
    assert rates[0] >= rates[1] >= rates[2], f"tier ordering violated: {rates}"
    effect = probe_a["dilution_and_sample_sizes"]["easy_vs_all_effect_ratio"]

    vj = _load(OUT / "v13_score_validation.json")
    var_base = vj.get("baseline_group_variance_fraction") if vj else None
    var_arm = vj.get("arm_group_variance_fraction") if vj else None

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    ax.bar(tiers, [r * 100 for r in rates],
           color=[PALETTE["sft_only"], PALETTE["stable"], PALETTE["accent"]], zorder=2)
    for i, (r, s, tt) in enumerate(zip(rates, solved, totals)):
        ax.text(i, r * 100 + 0.6, f"{r*100:.1f}%\n({s}/{tt})", ha="center", fontsize=8)
    ax.set_ylabel("% task-instances with an exact candidate @8")
    ax.set_title(f"Tier collapse (easy-vs-all dilution {effect:.1f}x)", fontsize=9)

    ax2 = axes[1]
    if var_base is not None:
        ax2.bar(["baseline\n(static/stable)", "SFT+GRPO arm"],
                [var_base * 100, (var_arm or 0) * 100],
                color=[PALETTE["static"], PALETTE["sft_grpo"]], zorder=2)
        ax2.axhline(100, ls="--", lw=1, color=PALETTE["muted"])
        ax2.set_ylim(0, 105)
        ax2.set_ylabel("% GRPO groups variance-saturated")
        for i, v in enumerate([var_base, var_arm or 0]):
            ax2.text(i, v * 100 + 1, f"{v*100:.1f}%", ha="center", fontsize=8)
        ax2.set_title("Reward variance saturation (validation)", fontsize=9)
    else:
        ax2.text(0.5, 0.5, "variance fraction n/a\n(validation JSON only)",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=9)
        ax2.set_axis_off()
    fig.suptitle("C4 -- why the adaptive controller is inert here", fontweight="bold")
    return _save(fig, "fig_c4_tier_collapse", out_dir, formats, dpi, dry)


# ---------------------------------------------------------------------------
# C5 -- token cost at matched exact (ratio bars straight from 17's JSON).
# ---------------------------------------------------------------------------
def fig_c5(assets, out_dir, formats, dpi, dry):
    c5 = assets.get("C5_cost")
    if not c5:
        log.warning("[C5] skipped -- no C5_cost in %s", ASSETS_JSON.name)
        return None
    splits = [s for s in ("validation", "test_in_dist") if s in c5]
    labels = [s for s in splits]
    ratios = [c5[s]["ratio_stable_static"] for s in splits]
    gaps = [c5[s].get("gap_over_noise") for s in splits]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.bar(labels, ratios, color=PALETTE["stable"], width=0.5, zorder=2)
    ax.axhline(1.0, ls="--", lw=1, color=PALETTE["muted"], label="parity (static)")
    for i, (r, g) in enumerate(zip(ratios, gaps)):
        tag = f"{r:.2f}x" + (f"\ngap/noise {g:.1f}x" if g else "")
        ax.text(i, r + 0.02, tag, ha="center", fontsize=8)
    ax.set_ylabel("stable tokens / static tokens (@N=8)")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8)
    lo, hi = min(ratios), max(ratios)
    ax.set_title(
        f"C5 -- adaptive-stable spends {lo:.2f}-{hi:.2f}x tokens at matched exact@8\n"
        "(observed; near-uniform learned weights -- not a causal adaptivity claim)", fontsize=9)
    return _save(fig, "fig_c5_cost_at_equal_exact", out_dir, formats, dpi, dry)


FIGS = {"C1": fig_c1, "C3": fig_c3, "C4": fig_c4, "C5": fig_c5}


def self_check() -> int:
    assets = _load(ASSETS_JSON)
    if not assets:
        log.error("self_check FAIL: run scripts/17_paper1_assets.py first (no %s)", ASSETS_JSON.name)
        return 1
    failures = []

    def check(name, cond):
        log.info("[self_check] %s: %s", "PASS" if cond else "FAIL", name)
        if not cond:
            failures.append(name)

    c1 = assets.get("C1_selection_saturation", {})
    check("C1 reranked<=oracle every bank",
          all(b["reranked@8"] <= b["oracle@8"] for b in c1.get("per_bank", [])))
    check("C1 all banks saturated", c1.get("n_banks") == c1.get("n_banks_zero_gap"))
    check("global: matplotlib Agg backend", matplotlib.get_backend().lower() == "agg")
    import importlib.util as _u
    check("global: scipy absent", _u.find_spec("scipy") is None)
    if failures:
        log.error("self_check FAILED: %s", failures)
        return 1
    log.info("self_check PASSED")
    return 0


def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Paper-1 figure generator (companion to scripts/17)")
    ap.add_argument("--out_dir", default=str(OUT / "paper_assets"))
    ap.add_argument("--claims", default="all", help="subset e.g. 'c1,c3' (case-insensitive)")
    ap.add_argument("--formats", default="png,pdf")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--self_check", action="store_true")
    args = ap.parse_args(argv)

    if args.self_check:
        return self_check()

    assets = _load(ASSETS_JSON)
    if not assets:
        log.error("no %s -- run scripts/17_paper1_assets.py first", ASSETS_JSON.name)
        return 1

    want = list(FIGS) if args.claims.strip().lower() == "all" else \
        [c.strip().upper() for c in args.claims.split(",") if c.strip()]
    unknown = [c for c in want if c not in FIGS]
    if unknown:
        ap.error(f"unknown claims {unknown}; valid {list(FIGS)}")

    out_dir = Path(args.out_dir)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    formats = [f.strip() for f in args.formats.split(",")]

    written = {}
    for c in want:
        try:
            paths = FIGS[c](assets, out_dir, formats, args.dpi, args.dry_run)
        except Exception as exc:  # one figure's failure never aborts the rest
            log.exception("[%s] errored", c)
            written[c] = {"status": "error", "reason": repr(exc)}
            continue
        if paths is None:
            written[c] = {"status": "skipped"}
        else:
            written[c] = {"status": "ok", "assets": paths}
            log.info("[%s] ok (%d files)", c, len(paths))

    manifest = {"out_dir": str(out_dir), "dry_run": args.dry_run,
                "source": str(ASSETS_JSON), "figures": written}
    if not args.dry_run:
        (out_dir / "figures_manifest.json").write_text(json.dumps(manifest, indent=2))
    else:
        print(json.dumps(manifest, indent=2))
    n_ok = sum(v["status"] == "ok" for v in written.values())
    log.info("done: %d ok, %d skipped, %d error -> %s", n_ok,
             sum(v["status"] == "skipped" for v in written.values()),
             sum(v["status"] == "error" for v in written.values()), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
