#!/usr/bin/env python
"""S5 — stronger-teacher replay: precondition-failure vs weak-controller (Q3).

Pre-registered design: docs/S5_BANDIT_TEACHER_REPLAY_PLAN.md (advisor-amended).

Answers the top reviewer threat: is Countdown's adaptive-controller inertness a
property of PRECONDITION FAILURE (~97% variance-saturated GRPO groups,
primary-dominated non-hackable reward) or of the WEAK heuristic teacher?

CPU-only, reads committed reward_components.jsonl streams. No GPU, no frozen
component touched.

PRIMARY evidence = an assumption-free per-group ORACLE CEILING on how much ANY
auxiliary-weight vector can change the within-group structure the GRPO optimizer
acts on (top-1 identity, positive-advantage set, and — the sharp test — whether
a *correct* completion's preference can be manufactured/rescued). Because reward
= correct + Σ_k w_k·aux_k with primary_weight=1, a correct completion's +1.0
primary term dominates the aux budget (~1.2 split over six [0,1] components), so
the oracle ceiling on "rescue a correct completion by reweighting" is a
mechanism result, not a tuning artifact.

SECONDARY = a self-contained DynaOpt-faithful EXP3 bandit (realism check):
does a plausible learned controller, fed the *available* per-step ranking
leverage (not the already-exhausted EMA improvement), issue a materially
different weight trajectory than adaptive_stable on the same stream?

Also runs the adaptive_stable offline-replay-matches-logged-trajectory sanity
check (a correctness guard on the replay harness itself).

Decision rule (pre-registered, see plan):
- Outcome A: oracle top-1-flip ceiling low AND correct-rescue ceiling
  negligible -> precondition failure dominates (at the observed operating
  point); Q3 defused without GPU.
- Outcome B: oracle ceiling high AND bandit realizes L1 > 2x adaptive_stable ->
  escalate to a single pre-registered GPU arm.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
from typing import Iterable

from rtw_llm.teacher import AUX_KEYS, RTWTeacher, TeacherConfig

GROUP_SIZE = 4  # num_generations; groups are consecutive slices within a batch
UNIFORM = {k: 0.2 for k in AUX_KEYS}


# ------------------------------- reward model -------------------------------

def total_reward(components: dict, weights: dict, primary_weight: float = 1.0) -> float:
    """Mirror countdown.reward_breakdown: correct + sum_k w_k * aux_k."""
    r = primary_weight * float(components.get("correct", 0.0))
    for k in AUX_KEYS:
        r += float(weights.get(k, 0.0)) * float(components.get(k, 0.0))
    return r


def _pop_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


# --------------------------- candidate weight set ---------------------------

def candidate_weight_vectors(n_random: int = 200, seed: int = 0) -> list[dict]:
    """The oracle's search set over the aux-weight space.

    Includes the extreme vertices (uniform, each pure-key direction, and
    primary-only=all-aux-zero) PLUS a seeded random simplex sample so the
    ceiling is not understated by testing vertices alone (understating the
    ceiling would bias toward Outcome A). Ranking of a linear reward depends
    only on the aux-weight direction given the fixed primary term, so exploring
    directions is what matters; magnitudes use the stable weight budget scale.
    """
    vecs: list[dict] = [dict(UNIFORM)]
    # primary-only: aux all zero
    vecs.append({k: 0.0 for k in AUX_KEYS})
    # pure-key directions
    for key in AUX_KEYS:
        vecs.append({k: (1.0 if k == key else 0.0) for k in AUX_KEYS})
    rng = random.Random(seed)
    for _ in range(n_random):
        raw = {k: rng.random() for k in AUX_KEYS}
        s = sum(raw.values()) or 1.0
        # scale to the stable target weight-sum budget (1.2) for realism
        vecs.append({k: 1.2 * raw[k] / s for k in AUX_KEYS})
    return vecs


# ----------------------------- group structure ------------------------------

def load_groups(path: str) -> list[dict]:
    """Reconstruct GRPO groups from a reward_components.jsonl stream.

    Groups = GROUP_SIZE consecutive rows within one reward_batch_index. Asserts
    (a) group size divides rows-per-batch and (b) all rows in a reconstructed
    group share one id (guards a std-based grouping shortcut that would merge
    distinct prompts that coincidentally share group_reward_std).
    """
    rows = [json.loads(line) for line in open(path)]
    # bucket by batch index
    batches: dict[int, list[dict]] = {}
    for r in rows:
        batches.setdefault(int(r["reward_batch_index"]), []).append(r)
    groups: list[dict] = []
    for bi in sorted(batches):
        brows = batches[bi]
        if len(brows) % GROUP_SIZE != 0:
            raise AssertionError(f"batch {bi}: {len(brows)} rows not divisible by {GROUP_SIZE}")
        for i in range(0, len(brows), GROUP_SIZE):
            grp = brows[i : i + GROUP_SIZE]
            ids = {g.get("id") for g in grp}
            if len(ids) != 1:
                raise AssertionError(f"batch {bi} group at {i}: mixed ids {ids}")
            groups.append(
                {
                    "batch_index": bi,
                    "id": grp[0].get("id"),
                    "difficulty": grp[0].get("difficulty"),
                    "components": [g["components"] for g in grp],
                }
            )
    return groups


# ------------------------- oracle ceiling per group -------------------------

def _structure(comps: list[dict], w: dict) -> tuple[int, frozenset, list[float]]:
    """Return (top1 index, positive-advantage set, per-completion rewards)."""
    rewards = [total_reward(c, w) for c in comps]
    top1 = max(range(len(rewards)), key=lambda i: rewards[i])
    mean = sum(rewards) / len(rewards)
    pos = frozenset(i for i, r in enumerate(rewards) if r > mean + 1e-12)
    return top1, pos, rewards


def oracle_group_metrics(group: dict, cand: list[dict]) -> dict:
    comps = group["components"]
    correct_idx = [i for i, c in enumerate(comps) if float(c.get("correct", 0.0)) >= 0.5]
    has_correct = bool(correct_idx)

    u_top1, u_pos, u_rewards = _structure(comps, UNIFORM)
    u_std = _pop_std(u_rewards)

    top1_flip = False
    posadv_change = False
    correct_rescuable = False   # correct NOT top1 under uniform, but some w makes it top1
    correct_demotable = False   # correct IS top1 under uniform, but some w demotes it
    dead_revivable = False      # uniform std==0, some w gives std>0

    uniform_correct_on_top = has_correct and (u_top1 in correct_idx)

    for w in cand:
        top1, pos, rewards = _structure(comps, w)
        if top1 != u_top1:
            top1_flip = True
        if pos != u_pos:
            posadv_change = True
        if has_correct:
            if uniform_correct_on_top:
                if top1 not in correct_idx:
                    correct_demotable = True
            else:
                if top1 in correct_idx:
                    correct_rescuable = True
        if u_std <= 1e-9 and _pop_std(rewards) > 1e-9:
            dead_revivable = True

    return {
        "has_correct": has_correct,
        "uniform_dead": u_std <= 1e-9,
        "uniform_correct_on_top": uniform_correct_on_top,
        "top1_flip": top1_flip,
        "posadv_change": posadv_change,
        "correct_rescuable": correct_rescuable,
        "correct_demotable": correct_demotable,
        "dead_revivable": dead_revivable,
    }


def aggregate_oracle(groups: list[dict], cand: list[dict]) -> dict:
    n = len(groups)
    gm = [oracle_group_metrics(g, cand) for g in groups]
    n_correct = sum(1 for m in gm if m["has_correct"])
    n_dead = sum(1 for m in gm if m["uniform_dead"])
    n_correct_on_top = sum(1 for m in gm if m["uniform_correct_on_top"])
    n_rescue_candidates = sum(1 for m in gm if m["has_correct"] and not m["uniform_correct_on_top"])
    return {
        "n_groups": n,
        "n_groups_with_correct": n_correct,
        "n_dead_groups": n_dead,
        "frac_top1_flip_ceiling": _safe(sum(m["top1_flip"] for m in gm), n),
        "frac_posadv_change_ceiling": _safe(sum(m["posadv_change"] for m in gm), n),
        "frac_dead_revivable": _safe(sum(m["dead_revivable"] for m in gm), n),
        # sharp test: of groups WITH a correct completion, is it already on top,
        # and can reweighting ever rescue/demote it?
        "correct_already_on_top": _safe(n_correct_on_top, n_correct),
        "n_rescue_candidate_groups": n_rescue_candidates,
        "frac_correct_rescuable": _safe(
            sum(m["correct_rescuable"] for m in gm), n_rescue_candidates
        ),
        "frac_correct_demotable": _safe(
            sum(m["correct_demotable"] for m in gm if m["uniform_correct_on_top"]),
            n_correct_on_top,
        ),
    }


def _safe(num: int, den: int) -> float:
    return float(num) / den if den else 0.0


# ----------------------- DynaOpt-faithful EXP3 bandit -----------------------

class Exp3Bandit:
    """Non-contextual EXP3 over N+1 arms (six aux keys + 'do nothing').

    Faithful to Min et al. 2024 (DynaOpt): EXP3 arm-weight update, N+1 arms,
    chosen arm increments that reward's weight, periodic update. The dev-set
    improvement feedback is replaced (disclosed) by the *available within-group
    ranking leverage* the arm would create on the observed groups this round —
    NOT already-exhausted EMA improvement, which is ~0 at stable convergence and
    would bake in inertness (advisor amendment 3).
    """

    def __init__(self, gamma: float = 0.1, step: float = 0.05, seed: int = 0):
        self.arms = list(AUX_KEYS) + ["__none__"]
        self.n = len(self.arms)
        self.gamma = gamma
        self.step = step
        self.arm_w = [1.0] * self.n
        self.rng = random.Random(seed)
        self.weights = dict(UNIFORM)

    def _probs(self) -> list[float]:
        s = sum(self.arm_w)
        return [(1 - self.gamma) * (w / s) + self.gamma / self.n for w in self.arm_w]

    def _apply_arm(self, weights: dict, arm: str) -> dict:
        # Increment the chosen key, then RENORMALIZE to the stable weight budget
        # (0.2*6 = 1.2). Renormalization is what makes the realized trajectory
        # SIGNAL-DRIVEN rather than a monotone ratchet-to-cap: under a flat/zero
        # arm-reward EXP3 draws arms ~uniformly, so every key is incremented
        # about equally and renorm pulls the vector back toward uniform (L1~0);
        # only when the signal concentrates draws on a few keys does the
        # renormalized vector move materially off uniform. (Advisor fix: the
        # increment-only version moved L1=0.80 even on a zero signal.)
        w = dict(weights)
        if arm != "__none__":
            w[arm] = min(0.35, w[arm] + self.step)
            budget = sum(UNIFORM.values())
            s = sum(w.values()) or 1.0
            w = {k: budget * v / s for k, v in w.items()}
        return w

    def round(self, round_groups: list[dict]) -> None:
        probs = self._probs()
        a = self.rng.choices(range(self.n), weights=probs, k=1)[0]
        arm = self.arms[a]
        candidate = self._apply_arm(self.weights, arm)
        # arm reward = available ranking leverage: how many groups this round
        # change positive-advantage set (the GRPO learning signal) vs current
        # weights, normalized to [0,1]
        reward = self._leverage(round_groups, self.weights, candidate)
        # EXP3 update on the drawn arm
        est = reward / probs[a]
        self.arm_w[a] *= math.exp(self.gamma * est / self.n)
        # renormalize to avoid overflow
        s = sum(self.arm_w)
        self.arm_w = [w / s * self.n for w in self.arm_w]
        self.weights = candidate

    @staticmethod
    def _leverage(groups: list[dict], w_old: dict, w_new: dict) -> float:
        if not groups:
            return 0.0
        changed = 0
        for g in groups:
            _, pos_old, _ = _structure(g["components"], w_old)
            _, pos_new, _ = _structure(g["components"], w_new)
            if pos_old != pos_new:
                changed += 1
        return changed / len(groups)


def run_bandit(groups: list[dict], round_bandit: int = 10, seed: int = 0) -> dict:
    """Replay the observed group stream through the EXP3 bandit, grouping the
    stream into rounds of `round_bandit` batches (matching DynaOpt's periodic
    update). Report the realized weight trajectory movement."""
    bandit = Exp3Bandit(seed=seed)
    init_w = dict(bandit.weights)
    # rounds are contiguous blocks of batches
    by_batch: dict[int, list[dict]] = {}
    for g in groups:
        by_batch.setdefault(g["batch_index"], []).append(g)
    batch_ids = sorted(by_batch)
    max_step_l1 = 0.0
    prev = dict(init_w)
    for start in range(0, len(batch_ids), round_bandit):
        block = batch_ids[start : start + round_bandit]
        rgroups = [g for b in block for g in by_batch[b]]
        bandit.round(rgroups)
        step_l1 = sum(abs(bandit.weights[k] - prev[k]) for k in AUX_KEYS)
        max_step_l1 = max(max_step_l1, step_l1)
        prev = dict(bandit.weights)
    final_l1 = sum(abs(bandit.weights[k] - init_w[k]) for k in AUX_KEYS)
    return {
        "final_weights": bandit.weights,
        "final_vs_init_l1": final_l1,
        "max_round_l1": max_step_l1,
    }


# ---------------- adaptive_stable trajectory-match sanity check --------------

def adaptive_stable_final_l1(groups: list[dict]) -> dict:
    """Re-run the REAL adaptive_stable teacher offline on the batch-averaged
    component stream and report its final-vs-init L1 (the incumbent's known
    near-uniform movement, ~0.16). Also a harness correctness guard: the teacher
    depends only on batch-averaged components, so this is exactly reproducible.
    """
    teacher = RTWTeacher(TeacherConfig(strategy="adaptive_stable", seed=0))
    init_w = dict(teacher.get_weights())
    by_batch: dict[int, list[dict]] = {}
    for g in groups:
        by_batch.setdefault(g["batch_index"], []).append(g)
    for bi in sorted(by_batch):
        batch_comps = [c for g in by_batch[bi] for c in g["components"]]
        teacher.update(batch_comps)
    final_w = teacher.get_weights()
    return {
        "final_weights": final_w,
        "final_vs_init_l1": sum(abs(final_w[k] - init_w[k]) for k in AUX_KEYS),
    }


# ----------------------------------- main -----------------------------------

def discover_streams(pattern: str) -> list[str]:
    paths = sorted(glob.glob(pattern))
    return [p for p in paths if os.path.getsize(p) > 0]


def _zero_signal_control(groups: list[dict], round_bandit: int, seed: int) -> dict:
    """Bandit movement when fed a FLAT signal (all completions in every group
    replaced by an identical one → zero ranking leverage on every arm). The
    realized-vs-control gap is what shows the bandit responds to signal rather
    than ratcheting; a near-zero control L1 confirms the renormalization fix."""
    flat = []
    for g in groups:
        c0 = dict(g["components"][0])
        flat.append({"batch_index": g["batch_index"], "components": [dict(c0) for _ in g["components"]]})
    return run_bandit(flat, round_bandit=round_bandit, seed=seed)


def analyze_stream(path: str, cand: list[dict], round_bandit: int, seed: int) -> dict:
    groups = load_groups(path)
    return {
        "path": path,
        "n_groups": len(groups),
        "oracle": aggregate_oracle(groups, cand),
        "bandit": run_bandit(groups, round_bandit=round_bandit, seed=seed),
        "bandit_zero_signal_control": _zero_signal_control(groups, round_bandit, seed),
        "adaptive_stable": adaptive_stable_final_l1(groups),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--streams",
        nargs="*",
        default=None,
        help="explicit reward_components.jsonl paths; default = auto-discover stable/static/v10c2 seeds",
    )
    ap.add_argument("--n_random", type=int, default=200)
    ap.add_argument("--round_bandit", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/bandit_replay.json")
    args = ap.parse_args()

    if args.streams:
        streams = args.streams
    else:
        pats = [
            "outputs/checkpoints/grpo_stable_seed*_300/reward_components.jsonl",
            "outputs/checkpoints/grpo_static_seed*_300/reward_components.jsonl",
            "outputs/checkpoints/grpo_v10_c2_adaptive_curr_seed0_300/reward_components.jsonl",
        ]
        streams = []
        for p in pats:
            streams.extend(discover_streams(p))

    cand = candidate_weight_vectors(n_random=args.n_random, seed=args.seed)
    print(f"candidate weight vectors: {len(cand)} (uniform + primary-only + 6 pure + {args.n_random} random)")

    results = []
    for path in streams:
        print(f"analyzing {path} ...")
        results.append(analyze_stream(path, cand, args.round_bandit, args.seed))

    # cross-stream summary, split by arm family
    def _family(p: str) -> str:
        if "stable" in p:
            return "stable"
        if "static" in p:
            return "static"
        if "v10_c2" in p:
            return "v10c2"
        return "other"

    summary: dict = {"by_family": {}}
    for fam in ("stable", "static", "v10c2"):
        fam_res = [r for r in results if _family(r["path"]) == fam]
        if not fam_res:
            continue
        summary["by_family"][fam] = {
            "n_streams": len(fam_res),
            "mean_frac_top1_flip_ceiling": _mean(r["oracle"]["frac_top1_flip_ceiling"] for r in fam_res),
            "mean_frac_posadv_change_ceiling": _mean(r["oracle"]["frac_posadv_change_ceiling"] for r in fam_res),
            "mean_frac_correct_rescuable": _mean(r["oracle"]["frac_correct_rescuable"] for r in fam_res),
            "mean_correct_already_on_top": _mean(r["oracle"]["correct_already_on_top"] for r in fam_res),
            "mean_frac_dead_revivable": _mean(r["oracle"]["frac_dead_revivable"] for r in fam_res),
            "mean_bandit_final_l1": _mean(r["bandit"]["final_vs_init_l1"] for r in fam_res),
            "mean_bandit_zero_signal_l1": _mean(r["bandit_zero_signal_control"]["final_vs_init_l1"] for r in fam_res),
            "mean_adaptive_stable_final_l1": _mean(r["adaptive_stable"]["final_vs_init_l1"] for r in fam_res),
        }

    out = {"streams": results, "summary": summary, "config": vars(args)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")


def _mean(it: Iterable[float]) -> float:
    vals = list(it)
    return sum(vals) / len(vals) if vals else 0.0


if __name__ == "__main__":
    main()
