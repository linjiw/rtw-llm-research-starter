#!/usr/bin/env python
"""Regenerate every Paper-1 table from committed evidence (claims C1-C6).

Paper 1 = the shaping-vs-capability characterization (RESEARCH_GOAL_AND_PLANS
§2.1). This script reads ONLY committed artifacts — the score JSONs under
outputs/ and the candidate banks — and emits one consolidated
`docs/PAPER1_ASSETS.md` (markdown tables) + `outputs/paper1_assets.json`
(machine-readable). Additive, read-only, no frozen-component edits. Robust to
missing artifacts (e.g. v13 seeds 1/2 still running) — reports what's present.

Claims (RESEARCH_GOAL_AND_PLANS §2.1):
  C1 selection saturates (reranked@N == oracle@N)
  C2 shaping moves intermediates not success (v0.10, v0.12 strikes)
  C3 SFT capability lever moves both walls ~5x (v0.13, novel/held-out)
  C4 mechanism: adaptivity preconditions unmet (documented; pointer)
  C5 cost: stable ~0.58x tokens at equal exact
  C6 robustness: harness-shift / OOD (pre-registered; may be pending)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

OUT = Path("outputs")
BANKS = OUT / "bestofn"


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _bank(name: str):
    p = BANKS / name / "candidates.jsonl"
    if not p.exists():
        return None
    return [json.loads(line) for line in p.open()]


def _by_task(rows):
    g = defaultdict(list)
    for r in rows:
        g[r["id"]].append(r)
    for v in g.values():
        v.sort(key=lambda r: r["candidate_index"])
    return g


def _oracle_at(rows, n=8):
    return sum(
        1 for cs in _by_task(rows).values()
        if any(float(c["metrics"].get("exact_correct", 0)) > 0 for c in cs[:n])
    )


def _reranked_at(rows, n=8):
    # practical selector = max practical_score, tie -> lower candidate_index
    total = 0
    for cs in _by_task(rows).values():
        pre = cs[:n]
        chosen = max(pre, key=lambda r: (r.get("practical_score", 0.0), -r["candidate_index"]))
        total += int(float(chosen["metrics"].get("exact_correct", 0)) > 0)
    return total


# ---------- C1: selection saturation ----------
def claim_c1():
    rows = []
    for name in sorted(p.parent.name for p in BANKS.glob("*/candidates.jsonl")):
        rr = _bank(name)
        if not rr:
            continue
        n_tasks = len(_by_task(rr))
        if n_tasks == 0:
            continue
        rer, ora = _reranked_at(rr, 8), _oracle_at(rr, 8)
        rows.append({"bank": name, "n_tasks": n_tasks, "reranked@8": rer, "oracle@8": ora,
                     "gap": ora - rer})
    n_banks = len(rows)
    n_zero_gap = sum(1 for r in rows if r["gap"] == 0)
    return {"n_banks": n_banks, "n_banks_zero_gap": n_zero_gap, "per_bank": rows,
            "verdict": f"reranked@8 == oracle@8 in {n_zero_gap}/{n_banks} banks (selection saturated)"}


# ---------- C2: shaping strikes ----------
def claim_c2():
    # Pull the recorded strike results from the score-adjacent banks if present.
    out = {}
    for method, split in [("v10c2", "validation"), ("v12legality", "validation")]:
        rr = _bank(f"{method}_seed0_{split}_limit50_n8")
        stable = _bank(f"stable_local_seed0_{split}_limit50_n8")
        if rr and stable:
            out[method] = {"oracle@8": _oracle_at(rr), "stable_oracle@8": _oracle_at(stable),
                           "n_tasks": len(_by_task(rr))}
    out["note"] = "Two pre-registered reward-shaping strikes; details in V10/V12 plan docs + ledger."
    return out


# ---------- C3: SFT capability lever ----------
def claim_c3():
    out = {"seeds_present": []}
    for split in ["validation", "test_in_dist"]:
        score = _load(OUT / f"v13_score_{split}.json")
        entry = {}
        if score:
            arm = next((a for a in score.get("arms", []) if a.get("arm") == "v13sft"), None)
            if arm:
                entry["easy_legality_arm"] = arm.get("easy_legality_all", {}).get("arm", {}).get("legality_rate")
                entry["easy_legality_baseline_pooled"] = arm.get("easy_legality_all", {}).get("baseline_pooled_rate")
                entry["all_tier_p_exact_given_legal"] = arm.get("all_tier_candidates", {}).get("p_exact_given_legal")
        # bank-derived oracle@8 (seed 0) and multi-seed if present
        seed_oracle = {}
        for s in [0, 1, 2]:
            rr = _bank(f"v13sft_seed{s}_{split}_limit50_n8")
            if rr:
                seed_oracle[s] = {"oracle@8": _oracle_at(rr), "reranked@8": _reranked_at(rr),
                                  "n_tasks": len(_by_task(rr))}
                if s not in out["seeds_present"]:
                    out["seeds_present"].append(s)
        stable_oracle = [_oracle_at(_bank(f"stable_local_seed{s}_{split}_limit50_n8"))
                         for s in [0, 1, 2] if _bank(f"stable_local_seed{s}_{split}_limit50_n8")]
        entry["v13_oracle@8_by_seed"] = seed_oracle
        entry["stable_oracle@8_by_seed"] = stable_oracle
        if stable_oracle:
            entry["stable_oracle@8_mean"] = round(mean(stable_oracle), 2)
        out[split] = entry
    # 3-seed score JSONs (land when E0 finishes)
    for split in ["validation", "test_in_dist"]:
        s3 = _load(OUT / f"v13_score_seeds012_{split}.json")
        if s3:
            out[f"seeds012_{split}_present"] = True
    return out


# ---------- C5: cost ----------
def claim_c5():
    out = {}
    for split in ["validation", "test_in_dist"]:
        st = [mean(r["completion_token_count"] for r in _bank(f"static_local_seed{s}_{split}_limit50_n8"))
              for s in [0, 1, 2] if _bank(f"static_local_seed{s}_{split}_limit50_n8")]
        sb = [mean(r["completion_token_count"] for r in _bank(f"stable_local_seed{s}_{split}_limit50_n8"))
              for s in [0, 1, 2] if _bank(f"stable_local_seed{s}_{split}_limit50_n8")]
        if st and sb:
            gap = mean(st) - mean(sb)
            noise = (pstdev(st) ** 2 + pstdev(sb) ** 2) ** 0.5 if len(st) > 1 else 0.0
            out[split] = {"static_tok_mean": round(mean(st), 1), "stable_tok_mean": round(mean(sb), 1),
                          "ratio_stable_static": round(mean(sb) / mean(st), 3),
                          "gap": round(gap, 1), "cross_seed_noise": round(noise, 1),
                          "gap_over_noise": round(gap / noise, 1) if noise else None}
    return out


# ---------- C6: robustness (harness-shift + OOD) ----------
def claim_c6():
    out = {}
    # Harness-shift: surface the 3-seed interaction summary (near-null closure).
    hs = _load(OUT / "harness_shift_scored.json")
    if hs:
        summ = {}
        for split, rep in hs.items():
            ix = rep.get("interaction_3seed") if isinstance(rep, dict) else None
            if ix:
                summ[split] = {m: {"advantage_mean": v.get("advantage_mean"),
                                   "sign_consistent": v.get("sign_consistent"),
                                   "n_seeds_favor_stable": v.get("n_seeds_stable_more_robust")}
                               for m, v in ix.items()}
        out["harness_shift"] = summ or "present (no interaction rows)"
    else:
        out["harness_shift"] = "pending (E1)"
    # OOD: surface per-arm legality + /-adoption + P(exact|legal) per split.
    od = _load(OUT / "ood_scored.json")
    if od:
        summ = {}
        for split, rep in od.items():
            arms = rep.get("arms") if isinstance(rep, dict) else None
            if arms:
                summ[split] = {a: {"legality": round(s.get("legal_rate", 0), 3),
                                   "div_adoption": round(s.get("div_adoption_rate", 0), 3),
                                   "p_exact_given_legal": round(s.get("p_exact_given_legal", 0), 3)}
                               for a, s in arms.items() if s.get("present")}
        out["ood"] = summ or "present (no arms)"
    else:
        out["ood"] = "pending (E0 OOD stage)"
    return out


def md_table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def main():
    c1, c2, c3, c5, c6 = claim_c1(), claim_c2(), claim_c3(), claim_c5(), claim_c6()
    payload = {"C1_selection_saturation": c1, "C2_shaping_strikes": c2,
               "C3_sft_capability": c3, "C5_cost": c5, "C6_robustness": c6}
    (OUT / "paper1_assets.json").write_text(json.dumps(payload, indent=2) + "\n")

    md = ["# Paper 1 assets (auto-generated from committed evidence)",
          "",
          "Regenerate: `.venv/bin/python scripts/17_paper1_assets.py`. Reads only",
          "committed score JSONs + candidate banks. Claims per",
          "`RESEARCH_GOAL_AND_PLANS_20260709.md` §2.1.",
          "",
          "## C1 — Inference-time selection saturates",
          "",
          c1["verdict"],
          "",
          md_table(c1["per_bank"], ["bank", "n_tasks", "reranked@8", "oracle@8", "gap"]),
          "",
          "## C2 — Shaping moves intermediates, not success (two strikes)",
          "",
          "```", json.dumps(c2, indent=2), "```",
          "",
          "## C3 — SFT capability lever moves both walls (~5×)",
          f"\nseeds present in banks: {c3['seeds_present']}",
          "",
          "```", json.dumps({k: v for k, v in c3.items() if k != "seeds_present"}, indent=2), "```",
          "",
          "## C5 — Cost: stable ~0.58× tokens at equal exact",
          "",
          md_table([{"split": k, **v} for k, v in c5.items()],
                   ["split", "static_tok_mean", "stable_tok_mean", "ratio_stable_static", "gap_over_noise"]),
          "",
          "## C6 — Robustness (pre-registered): harness-shift + OOD",
          "",
          "harness-shift = stable-vs-static 3-seed interaction (near-null closes",
          "pillar 3); OOD = per-arm legality / div-adoption / P(exact|legal).",
          "",
          "```", json.dumps(c6, indent=2), "```",
          ""]
    Path("docs/PAPER1_ASSETS.md").write_text("\n".join(md) + "\n")
    print("wrote docs/PAPER1_ASSETS.md + outputs/paper1_assets.json")
    print(f"C1: {c1['verdict']}")
    print(f"C3 seeds present: {c3['seeds_present']}")
    print(f"C5: {c5}")


if __name__ == "__main__":
    main()
