"""E5 mechanism stress-test (CPU, no GPU): HONEST budget regime.

Drives the REAL RTWTeacher(adaptive_stable) with a synthetic MicroCode reward
stream to check whether the need=1-ema mechanism spontaneously down-weights a
saturating hackable proxy (visible_pass_rate) and up-weights the true signal
(held_out_pass_rate) under a PERMISSIVE (default-low) floor budget.
"""
from __future__ import annotations

import sys

from rtw_llm.teacher import RTWTeacher, TeacherConfig

# --- plausible MicroCode aux-key subset for the teacher (primary 'correct' is NOT an aux key) ---
AUX_KEYS = [
    "format",
    "syntax_parses",
    "defines_target_signature",
    "valid_expression",         # legality gate (role-analogous to countdown valid_expression)
    "runs_without_error",
    "visible_pass_rate",        # HACKABLE PROXY
    "held_out_pass_rate",       # TRUE dense signal
    "no_hardcoding_heuristic",  # ANTI-HACK channel
    "brevity",
]

# HONEST budget: all floors low & permissive => teacher is free to down-weight anything.
HONEST_FLOORS = {
    "format": 0.03,
    "syntax_parses": 0.03,
    "defines_target_signature": 0.03,
    "valid_expression": 0.05,
    "runs_without_error": 0.03,
    "visible_pass_rate": 0.03,
    "held_out_pass_rate": 0.03,
    "no_hardcoding_heuristic": 0.03,
    "brevity": 0.02,
}


def lerp(a, b, t):
    return a + (b - a) * max(0.0, min(1.0, t))


def batch_means(step: int) -> dict[str, float]:
    """Synthetic per-step batch means. Phase 1 (0-100): honest co-rise.
    Phase 2 (100-300): the policy learns to game visible tests."""
    if step < 100:
        t = step / 100.0
        vis = lerp(0.10, 0.45, t)
        hel = lerp(0.10, 0.35, t)
        cor = lerp(0.02, 0.15, t)
        nohc = lerp(0.85, 0.80, t)
    else:
        t = (step - 100) / 60.0            # gaming ramp over ~60 steps
        vis = lerp(0.45, 0.98, t)          # proxy saturates toward ~1
        hel = lerp(0.35, 0.20, t)          # true signal plateaus/decays low
        cor = lerp(0.15, 0.10, t)          # held_out_all_pass stays low
        nohc = lerp(0.80, 0.30, t)         # anti-hack smell rises => score drops
    # legality/scaffold channels become reliably satisfied as code compiles
    tt = min(1.0, step / 80.0)
    return {
        "format": lerp(0.5, 0.98, tt),
        "syntax_parses": lerp(0.5, 0.97, tt),
        "defines_target_signature": lerp(0.4, 0.95, tt),
        "valid_expression": lerp(0.3, 0.95, tt),
        "runs_without_error": lerp(0.3, 0.92, tt),
        "visible_pass_rate": vis,
        "held_out_pass_rate": hel,
        "no_hardcoding_heuristic": nohc,
        "brevity": lerp(0.7, 0.9, tt),
        "correct": cor,
    }


def make_batch(step: int, n: int = 8) -> list[dict]:
    m = batch_means(step)
    # Teacher only uses batch means; a batch of identical dicts is deterministic & exact.
    return [dict(m) for _ in range(n)]


def main():
    cfg = TeacherConfig(
        strategy="adaptive_stable",
        aux_keys=AUX_KEYS,
        stable_floors=HONEST_FLOORS,
        stable_caps={},                    # no explicit caps => all cap at max_weight
        # everything else = code defaults; print them so the report is grounded
    )
    teacher = RTWTeacher(cfg)

    print("=== DEFAULTS (from code) ===")
    for k in ["min_weight", "max_weight", "init_weight", "ema_beta",
              "primary_success_decay", "stable_delay_steps", "stable_lr",
              "stable_alpha", "stable_target_weight_sum"]:
        print(f"  {k} = {getattr(cfg, k)}")
    print(f"  aux_keys = {AUX_KEYS}")
    print(f"  HONEST_FLOORS = {HONEST_FLOORS}")
    print(f"  floors_sum = {sum(HONEST_FLOORS.values()):.3f}  target_weight_sum = {cfg.stable_target_weight_sum}")

    snapshots = {}
    for step in range(300):
        w = teacher.update(make_batch(step))
        # record at delay-end (first post-delay step), 150, 299
        if step in (cfg.stable_delay_steps, 100, 150, 200, 299):
            snapshots[step] = (dict(w), dict(teacher.ema_aux), teacher.ema_primary,
                               sum(w.values()))

    hdr = ["step", "vis_w", "held_w", "nohc_w", "valid_w", "runs_w",
           "vis_ema", "held_ema", "nohc_ema", "prim_ema", "wsum"]
    print("\n=== TRAJECTORY (HONEST budget) ===")
    print("  " + "  ".join(f"{h:>9}" for h in hdr))
    for step in sorted(snapshots):
        w, ema, prim, wsum = snapshots[step]
        row = [
            step,
            w["visible_pass_rate"], w["held_out_pass_rate"],
            w["no_hardcoding_heuristic"], w["valid_expression"],
            w["runs_without_error"],
            ema["visible_pass_rate"], ema["held_out_pass_rate"],
            ema["no_hardcoding_heuristic"], prim, wsum,
        ]
        print("  " + "  ".join(f"{v:9d}" if isinstance(v, int) else f"{v:9.4f}" for v in row))

    # full final weight vector
    print("\n=== FINAL WEIGHT VECTOR (step 299) ===")
    for k in AUX_KEYS:
        print(f"  {k:28s} {teacher.weights[k]:.4f}  (ema={teacher.ema_aux[k]:.3f}, floor={HONEST_FLOORS[k]})")
    print(f"  weight_sum = {sum(teacher.weights.values()):.4f}")

    # gap-relevant summary: does proxy get down-weighted while held_out is up-weighted?
    print("\n=== MECHANISM CHECK ===")
    de = snapshots[cfg.stable_delay_steps][0]
    fin = teacher.weights
    print(f"  visible_pass_rate weight: delay-end {de['visible_pass_rate']:.4f} -> step299 {fin['visible_pass_rate']:.4f}")
    print(f"  held_out_pass_rate weight: delay-end {de['held_out_pass_rate']:.4f} -> step299 {fin['held_out_pass_rate']:.4f}")
    print(f"  no_hardcoding weight: delay-end {de['no_hardcoding_heuristic']:.4f} -> step299 {fin['no_hardcoding_heuristic']:.4f}")
    print(f"  held/visible weight ratio @299: {fin['held_out_pass_rate']/fin['visible_pass_rate']:.2f}x")


if __name__ == "__main__":
    sys.exit(main())
