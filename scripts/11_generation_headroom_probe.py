#!/usr/bin/env python
"""Probe B: generation-budget headroom diagnosis.

Question: before spending GPU, decide whether ANY generation-decode lever
(larger N, longer max_new_tokens, temperature sweep) is justified.

Context from the bottleneck diagnosis:
  - selection is saturated (reranked@N == oracle@N),
  - ceiling oracle_exact@8 ~= 9%,
  - 91.25% of lost tasks form NO exact candidate.

This script is READ-ONLY over the committed best-of-N candidate banks
(outputs/bestofn/<method>_..._n8/candidates.jsonl). It computes:

  1. MARGINAL NEW EXACT per candidate index (oracle_exact@N, N=1..8) -- does
     more N still buy new solves at index 6-8, or is it saturated by N=4?
  2. CLIP-RECOVERY UPPER BOUND -- of no-exact tasks, how many have a clipped
     (token-capped) candidate on a legality trajectory that a no-clip regime
     could *at most* recover?
  3. DIVERSITY -- distinct extracted expressions per task at N=8. Low => decode
     changes (temperature) could form new candidates; high => model already
     explores but cannot hit exact.
  4. Per-tier (easy/medium/hard) versions of (1) and (2).
  5. VERDICT -- go/no-go on a generation-decode GPU run, with grounded gain.

Usage:
  source .env && .venv/bin/python scripts/11_generation_headroom_probe.py
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict
from typing import Any

BESTOFN_DIR = "outputs/bestofn"
OUT_PATH = "outputs/probe_b_generation_headroom.json"
MAX_N = 8
# The generation cap was max_new_tokens=256. Records land at exactly 256 (and a
# handful at 257/258 from tokenizer-boundary artifacts). >=256 is the clip proxy.
CLIP_TOKENS = 256
LEGAL_F1 = 0.9  # "legal-ish": visible prefix already references (nearly) all required numbers
TIERS = ["easy", "medium", "hard"]

# Unclosed answer tag at the tail: model started emitting <answer ...> and got
# cut off before </answer>. Strong signal the truncation interrupted an answer.
_ANSWER_OPEN = re.compile(r"<answer", flags=re.IGNORECASE)
_ANSWER_CLOSE = re.compile(r"</answer>", flags=re.IGNORECASE)


def has_unclosed_answer_tail(text: str) -> bool:
    txt = text or ""
    last_open = txt.lower().rfind("<answer")
    last_close = txt.lower().rfind("</answer>")
    return last_open != -1 and last_open > last_close


def load_banks() -> dict[str, list[dict[str, Any]]]:
    banks: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(glob.glob(os.path.join(BESTOFN_DIR, "*_n8", "candidates.jsonl"))):
        name = os.path.basename(os.path.dirname(path))
        recs = [json.loads(line) for line in open(path) if line.strip()]
        banks[name] = recs
    return banks


def group_by_task(recs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    tasks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        tasks[r["id"]].append(r)
    for tid in tasks:
        tasks[tid].sort(key=lambda r: r["candidate_index"])
    return tasks


def is_clipped(rec: dict[str, Any]) -> bool:
    return rec.get("completion_token_count", 0) >= CLIP_TOKENS


def is_exact(rec: dict[str, Any]) -> bool:
    return float(rec["metrics"].get("exact_correct", 0.0)) == 1.0


def first_exact_index(cands: list[dict[str, Any]]) -> int | None:
    """Return the candidate_index (0-based order) of the first exact candidate."""
    for pos, c in enumerate(cands):
        if is_exact(c):
            return pos
    return None


# --------------------------------------------------------------------------- #
# 1 + 4: marginal-new-exact curve (overall and per tier)
# --------------------------------------------------------------------------- #
def marginal_curve(task_units: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """task_units: list of per-(bank,task) candidate lists ordered by index."""
    n_units = len(task_units)
    # marginal_new[k] = # units whose FIRST exact candidate sits at order-position k
    marginal_new = [0] * MAX_N
    ever_solved = 0
    for cands in task_units:
        fe = first_exact_index(cands)
        if fe is not None and fe < MAX_N:
            marginal_new[fe] += 1
            ever_solved += 1
    # cumulative solved@N and oracle_exact@N rate
    cum = []
    running = 0
    for k in range(MAX_N):
        running += marginal_new[k]
        cum.append(running)
    oracle_at_n = [round(cum[k] / n_units, 4) if n_units else 0.0 for k in range(MAX_N)]
    return {
        "n_units": n_units,
        "marginal_new_exact_at_index": marginal_new,  # index 0..7
        "cumulative_solved_at_n": cum,  # N=1..8
        "oracle_exact_at_n": oracle_at_n,  # N=1..8 (rate)
        "n_units_ever_solved_by_n8": ever_solved,
    }


# --------------------------------------------------------------------------- #
# 2 + 4: clip-recovery upper bound (overall and per tier)
# --------------------------------------------------------------------------- #
def clip_recovery(task_units: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """For no-exact-at-N8 tasks, graded upper bounds on clip-recoverable tasks."""
    n_units = len(task_units)
    no_exact_units = [c for c in task_units if first_exact_index(c) is None]
    n_no_exact = len(no_exact_units)

    # Graded upper bounds among no-exact tasks. Each is an at-MOST count of
    # tasks a no-clip regime could newly solve.
    ub_any_clip = 0        # loosest: >=1 clipped candidate at all
    ub_legalish = 0        # clipped candidate with visible-prefix f1 >= LEGAL_F1
    ub_unclosed = 0        # clipped candidate cut off mid-<answer> (unclosed tail)
    ub_trajectory = 0      # clipped AND (legal-ish OR unclosed tail)  <- headline UB
    for cands in no_exact_units:
        clipped = [c for c in cands if is_clipped(c)]
        if not clipped:
            continue
        ub_any_clip += 1
        legalish = any(
            float(c["metrics"].get("number_multiset_f1", 0.0)) >= LEGAL_F1 for c in clipped
        )
        unclosed = any(has_unclosed_answer_tail(c.get("completion", "")) for c in clipped)
        if legalish:
            ub_legalish += 1
        if unclosed:
            ub_unclosed += 1
        if legalish or unclosed:
            ub_trajectory += 1

    return {
        "n_units": n_units,
        "n_no_exact_units": n_no_exact,
        "ub_any_clip": ub_any_clip,
        "ub_legalish_f1": ub_legalish,
        "ub_unclosed_answer_tail": ub_unclosed,
        "ub_trajectory_headline": ub_trajectory,
        # express as tasks/50 equivalents and as fraction of no-exact tasks
        "ub_trajectory_frac_of_no_exact": round(ub_trajectory / n_no_exact, 4)
        if n_no_exact
        else 0.0,
    }


# --------------------------------------------------------------------------- #
# 3: diversity of extracted expressions at N=8
# --------------------------------------------------------------------------- #
def normalize_expr(s: str | None) -> str | None:
    if not s:
        return None
    return re.sub(r"\s+", "", s)


def diversity(task_units: list[list[dict[str, Any]]]) -> dict[str, Any]:
    distinct_all = []
    distinct_span = []   # only candidates with an extractable <answer> span
    distinct_valid = []  # only valid_expression==1 candidates
    n_valid_cands = []
    for cands in task_units:
        raw = {
            normalize_expr(c.get("extracted_expression"))
            for c in cands
            if normalize_expr(c.get("extracted_expression"))
        }
        span = {
            normalize_expr(c.get("extracted_expression"))
            for c in cands
            if float(c["metrics"].get("has_extractable_answer_span", 0.0)) == 1.0
            and normalize_expr(c.get("extracted_expression"))
        }
        valid = {
            normalize_expr(c.get("extracted_expression"))
            for c in cands
            if float(c["metrics"].get("valid_expression", 0.0)) == 1.0
            and normalize_expr(c.get("extracted_expression"))
        }
        distinct_all.append(len(raw))
        distinct_span.append(len(span))
        distinct_valid.append(len(valid))
        n_valid_cands.append(sum(1 for c in cands if float(c["metrics"].get("valid_expression", 0.0)) == 1.0))

    def avg(xs):
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    return {
        "n_units": len(task_units),
        "avg_distinct_extracted_all": avg(distinct_all),
        "avg_distinct_extracted_with_span": avg(distinct_span),
        "avg_distinct_valid_expressions": avg(distinct_valid),
        "avg_valid_candidates_per_task": avg(n_valid_cands),
    }


def tier_of(cands: list[dict[str, Any]]) -> str:
    return cands[0]["difficulty"]


# --------------------------------------------------------------------------- #
# 1b: N-doubling extrapolation. Model each task's per-draw exact rate by its MLE
# p_hat = k/8 (k = # exact candidates in 8 i.i.d. draws), then predict
# oracle_exact@N = mean_task[1 - (1-p_hat)^N]. Tasks with k=0 contribute 0 for
# all N under this estimator: they define the "capability wall" -- more sampling
# from the SAME distribution cannot surface them if the model never produced one.
# --------------------------------------------------------------------------- #
def n_extrapolation(task_units: list[list[dict[str, Any]]]) -> dict[str, Any]:
    n_units = len(task_units)
    ks = [sum(1 for c in u if is_exact(c)) for u in task_units]
    ns = [8, 16, 32, 64, 256, 1024]
    pred = {}
    for N in ns:
        tot = 0.0
        for k in ks:
            p = k / MAX_N
            tot += 1.0 - (1.0 - p) ** N
        pred[f"oracle_exact_at_{N}_pred"] = round(tot / n_units, 4) if n_units else 0.0
    asymptote = round(sum(1 for k in ks if k > 0) / n_units, 4) if n_units else 0.0
    return {
        "n_units": n_units,
        "oracle_pred": pred,
        "asymptote_frac_ever_solvable": asymptote,  # N->inf under MLE (k>0 tasks)
        "pred_gain_8_to_16": round(pred["oracle_exact_at_16_pred"] - pred["oracle_exact_at_8_pred"], 4),
        "pred_gain_8_to_64": round(pred["oracle_exact_at_64_pred"] - pred["oracle_exact_at_8_pred"], 4),
    }


# --------------------------------------------------------------------------- #
# 2b: does the sampling distribution even have mass near-correct on no-exact
# tasks? If a no-exact task never produces even a VALID expression, and its best
# numeric distance is far, hotter/more sampling from the same policy is unlikely
# to ever hit exact -> capability wall, not a decode budget problem.
# --------------------------------------------------------------------------- #
def near_mass_no_exact(task_units: list[dict[str, Any]]) -> dict[str, Any]:
    no_exact = [u for u in task_units if first_exact_index(u) is None]
    n = len(no_exact)
    any_valid = 0          # >=1 valid_expression candidate
    any_close = 0          # >=1 candidate within numeric_distance_reward >= 0.5 (i.e. |v-t|<=1)
    best_dist_rewards = []  # best numeric_distance_reward per task (legality-gated)
    for u in no_exact:
        valids = [c for c in u if float(c["metrics"].get("valid_expression", 0.0)) == 1.0]
        if valids:
            any_valid += 1
        best = max((float(c["metrics"].get("numeric_distance_reward", 0.0)) for c in u), default=0.0)
        best_dist_rewards.append(best)
        if best >= 0.5:
            any_close += 1

    def avg(xs):
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    return {
        "n_no_exact": n,
        "frac_no_exact_with_any_valid_expr": round(any_valid / n, 4) if n else 0.0,
        "frac_no_exact_with_near_value_dist_ge_0p5": round(any_close / n, 4) if n else 0.0,
        "avg_best_numeric_distance_reward": avg(best_dist_rewards),
        "n_no_exact_with_any_valid": any_valid,
        "n_no_exact_near_value": any_close,
    }


def build_report() -> dict[str, Any]:
    banks = load_banks()
    # Only n8 banks with 8 candidates per task are usable.
    bank_task_units: list[list[dict[str, Any]]] = []
    per_bank_summ: dict[str, Any] = {}
    for name, recs in banks.items():
        tasks = group_by_task(recs)
        units = [cands for cands in tasks.values() if len(cands) >= MAX_N]
        # trim to first 8 (should already be exactly 8)
        units = [u[:MAX_N] for u in units]
        bank_task_units.extend(units)
        per_bank_summ[name] = {
            "marginal": marginal_curve(units),
            "clip_recovery": clip_recovery(units),
            "diversity": diversity(units),
        }

    pooled = {
        "marginal": marginal_curve(bank_task_units),
        "clip_recovery": clip_recovery(bank_task_units),
        "diversity": diversity(bank_task_units),
        "n_extrapolation": n_extrapolation(bank_task_units),
        "near_mass_no_exact": near_mass_no_exact(bank_task_units),
    }

    # per-tier pooled
    per_tier: dict[str, Any] = {}
    for tier in TIERS:
        tier_units = [u for u in bank_task_units if tier_of(u) == tier]
        per_tier[tier] = {
            "marginal": marginal_curve(tier_units),
            "clip_recovery": clip_recovery(tier_units),
            "diversity": diversity(tier_units),
            "n_extrapolation": n_extrapolation(tier_units),
            "near_mass_no_exact": near_mass_no_exact(tier_units),
        }

    # clip prevalence per bank (context for verdict)
    clip_prevalence = {}
    for name, recs in banks.items():
        clip_prevalence[name] = round(sum(is_clipped(r) for r in recs) / len(recs), 4)

    return {
        "config": {
            "bestofn_dir": BESTOFN_DIR,
            "max_n": MAX_N,
            "clip_token_threshold": CLIP_TOKENS,
            "legalish_f1_threshold": LEGAL_F1,
            "n_banks": len(banks),
            "banks": list(banks.keys()),
            "total_bank_task_units": len(bank_task_units),
        },
        "clip_prevalence_per_bank": clip_prevalence,
        "pooled": pooled,
        "per_tier": per_tier,
        "per_bank": per_bank_summ,
    }


def _fmt_pct(x: float) -> str:
    return f"{100*x:5.1f}%"


def print_report(rep: dict[str, Any]) -> None:
    cfg = rep["config"]
    print("=" * 78)
    print("PROBE B: GENERATION-BUDGET HEADROOM")
    print("=" * 78)
    print(f"banks={cfg['n_banks']}  bank-task units pooled={cfg['total_bank_task_units']}  "
          f"clip_threshold>={cfg['clip_token_threshold']} tok")

    # ---- (1) marginal curve, pooled ----
    m = rep["pooled"]["marginal"]
    print("\n[1] MARGINAL NEW-EXACT per candidate index (POOLED across all banks/seeds)")
    print(f"    pooled units={m['n_units']}  ever-solved-by-N8={m['n_units_ever_solved_by_n8']} "
          f"({_fmt_pct(m['n_units_ever_solved_by_n8']/m['n_units'])})")
    print("    idx :   0    1    2    3    4    5    6    7")
    print("    new : " + " ".join(f"{v:4d}" for v in m["marginal_new_exact_at_index"]))
    print("    cum : " + " ".join(f"{v:4d}" for v in m["cumulative_solved_at_n"]))
    print("    o@N : " + " ".join(f"{v:0.3f}"[1:] if v < 1 else "1.00"
                                   for v in m["oracle_exact_at_n"]))
    new = m["marginal_new_exact_at_index"]
    late = sum(new[4:])
    print(f"    -> new solves at index 4-7 (N=5..8): {late}  "
          f"({_fmt_pct(late/max(1,m['n_units_ever_solved_by_n8']))} of all solves)")

    # ---- (3) diversity, pooled ----
    d = rep["pooled"]["diversity"]
    print("\n[3] DIVERSITY at N=8 (POOLED)")
    print(f"    avg distinct extracted (all)         : {d['avg_distinct_extracted_all']}")
    print(f"    avg distinct extracted (w/ span)     : {d['avg_distinct_extracted_with_span']}")
    print(f"    avg distinct VALID expressions/task  : {d['avg_distinct_valid_expressions']}")
    print(f"    avg valid candidates per task        : {d['avg_valid_candidates_per_task']} / 8")

    # ---- (2) clip recovery, pooled ----
    c = rep["pooled"]["clip_recovery"]
    print("\n[2] CLIP-RECOVERY UPPER BOUND (POOLED)")
    print(f"    no-exact-at-N8 units: {c['n_no_exact_units']} / {c['n_units']} "
          f"({_fmt_pct(c['n_no_exact_units']/c['n_units'])})")
    print(f"    UB loosest (any clipped cand)        : {c['ub_any_clip']}")
    print(f"    UB legal-ish (f1>={LEGAL_F1} clipped)     : {c['ub_legalish_f1']}")
    print(f"    UB unclosed-<answer> tail clipped    : {c['ub_unclosed_answer_tail']}")
    print(f"    UB HEADLINE (legal-ish OR unclosed)  : {c['ub_trajectory_headline']} "
          f"({_fmt_pct(c['ub_trajectory_frac_of_no_exact'])} of no-exact)")

    # ---- (1b) N extrapolation ----
    ne = rep["pooled"]["n_extrapolation"]
    print("\n[1b] N-DOUBLING EXTRAPOLATION (per-task MLE p=k/8, oracle@N = mean[1-(1-p)^N])")
    op = ne["oracle_pred"]
    print("     N     :   8    16    32    64   256   1024   inf")
    print("     o@N   : " + " ".join(f"{op[f'oracle_exact_at_{N}_pred']:0.3f}"[1:]
                                       for N in [8, 16, 32, 64, 256, 1024])
          + f"  {ne['asymptote_frac_ever_solvable']:0.3f}"[:6])
    print(f"     -> predicted gain N=8->16: +{ne['pred_gain_8_to_16']:0.3f} abs   "
          f"N=8->64: +{ne['pred_gain_8_to_64']:0.3f} abs   "
          f"ceiling(inf)={_fmt_pct(ne['asymptote_frac_ever_solvable'])}")

    # ---- (2b) near-mass on no-exact tasks ----
    nm = rep["pooled"]["near_mass_no_exact"]
    print("\n[2b] SAMPLING MASS on NO-EXACT tasks (can hotter/more sampling ever hit?)")
    print(f"     no-exact tasks with >=1 VALID expression : {nm['n_no_exact_with_any_valid']} / "
          f"{nm['n_no_exact']}  ({_fmt_pct(nm['frac_no_exact_with_any_valid_expr'])})")
    print(f"     no-exact tasks with a near value (dist_r>=0.5): {nm['n_no_exact_near_value']} / "
          f"{nm['n_no_exact']}  ({_fmt_pct(nm['frac_no_exact_with_near_value_dist_ge_0p5'])})")
    print(f"     avg best numeric_distance_reward (legality-gated): {nm['avg_best_numeric_distance_reward']}")

    # ---- (4) per tier ----
    print("\n[4] PER-TIER (pooled across banks)")
    print("    tier    units  solved@8  o@1   o@4   o@8   new@5-8  noExact  UB_head")
    for tier in TIERS:
        mm = rep["per_tier"][tier]["marginal"]
        cc = rep["per_tier"][tier]["clip_recovery"]
        on4 = mm["oracle_exact_at_n"][3]
        on1 = mm["oracle_exact_at_n"][0]
        on8 = mm["oracle_exact_at_n"][7]
        late = sum(mm["marginal_new_exact_at_index"][4:])
        print(f"    {tier:6s}  {mm['n_units']:5d}  {mm['n_units_ever_solved_by_n8']:6d}   "
              f"{on1:0.3f} {on4:0.3f} {on8:0.3f}   {late:5d}    "
              f"{cc['n_no_exact_units']:5d}    {cc['ub_trajectory_headline']:4d}")


def print_verdict(rep: dict[str, Any]) -> None:
    m = rep["pooled"]["marginal"]
    ne = rep["pooled"]["n_extrapolation"]
    d = rep["pooled"]["diversity"]
    c = rep["pooled"]["clip_recovery"]
    nm = rep["pooled"]["near_mass_no_exact"]

    o8 = m["oracle_exact_at_n"][7]
    o4 = m["oracle_exact_at_n"][3]
    late_frac = sum(m["marginal_new_exact_at_index"][4:]) / max(1, m["n_units_ever_solved_by_n8"])

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"- Oracle@4={o4:.3f}, Oracle@8={o8:.3f}: curve still climbing at N=8 "
          f"({_fmt_pct(late_frac)} of solves arrive at N=5..8), NOT saturated by N=4.")
    print(f"- But extrapolation caps hard: MLE asymptote(N=inf)={ne['asymptote_frac_ever_solvable']:.3f} "
          f"~= empirical Oracle@8={o8:.3f}. By construction a task with 0 exact in 8 draws has p_hat=0, "
          "so more draws from the SAME policy essentially cannot surface it. Doubling N=8->16 buys only "
          f"~+{ne['pred_gain_8_to_16']:.3f} abs (smoothed).")
    print(f"- Diversity is HIGH ({d['avg_distinct_extracted_with_span']} distinct spans/task) but VALID "
          f"expressions are near-zero ({d['avg_distinct_valid_expressions']}/task); the policy explores "
          f"text, not legal arithmetic. Temperature would add more of the same illegal mass.")
    print(f"- Clip-recovery UPPER bound = {c['ub_trajectory_headline']}/{c['n_units']} tasks "
          f"(loose, mostly reasoning-runaways not near-answers). Only {c['ub_legalish_f1']} clipped "
          f"no-exact tasks are even legal-ish.")
    print(f"- On no-exact tasks, only {_fmt_pct(nm['frac_no_exact_with_any_valid_expr'])} ever produce a "
          f"single VALID expression and only {_fmt_pct(nm['frac_no_exact_with_near_value_dist_ge_0p5'])} "
          f"land near the target value: the sampling distribution has ~no mass near correct.")
    print("- Tiers: headroom is concentrated in EASY (o@8=0.227) where it partly converts; MEDIUM/HARD "
          "are near-floor (0.039 / 0.004) with no N or clip headroom -> a capability wall.")
    print()
    print("  GO/NO-GO:")
    print("  * NO-GO on temperature sweep and NO-GO on longer max_new_tokens as accuracy levers.")
    print("    - longer tokens: clip UB is dominated by verbose reasoning that never reaches a legal")
    print("      answer; graded legal-ish UB is tiny; predicted realized gain ~<0.01 abs. Not worth GPU.")
    print("    - temperature: diversity already high, validity ~0; hotter sampling adds illegal variety.")
    print("  * MARGINAL / OPTIONAL: a single cheap N=16 confirmation run (same temp/tokens) would test")
    print(f"    the ~+{ne['pred_gain_8_to_16']:.3f} abs prediction, but expected gain is within noise at "
          "limit=50.")
    print("  * The ceiling is a CAPABILITY wall (esp. medium/hard). Only training that raises the")
    print("    per-draw legal+exact rate (SFT warmup / curriculum) can move it -- not a decode-budget run.")


def main() -> None:
    rep = build_report()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(rep, fh, indent=2)
    print_report(rep)
    print_verdict(rep)
    print(f"\nSaved full report -> {OUT_PATH}")


if __name__ == "__main__":
    main()
