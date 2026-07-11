"""ADVERSARIAL re-drive of the REAL RTWTeacher for Paper-2 E5 temptation-floor-sweep.

Imports the real teacher; never reimplements its math. Tests each load-bearing
finding independently and checks the known failure modes:
  - wrong aux_key set / floors
  - projection (floor/cap/redistribution) misread
  - EMA + double-smoothing lag making 300 steps too few
  - floor high enough that projection can't move the proxy
"""
from __future__ import annotations

from rtw_llm.teacher import RTWTeacher, TeacherConfig

# ---- E5 aux-key set implied by the finding numbers (vis, hel, nohc, syntax, runs, brevity) ----
AUX_KEYS = [
    "visible_pass_rate",        # HACKABLE PROXY
    "held_out_pass_rate",       # TRUE dense signal
    "no_hardcoding_heuristic",  # anti-hack
    "syntax_parses",            # scaffold
    "runs_without_error",       # scaffold
    "brevity",                  # scaffold
]
# HONEST budget: all floors low so teacher is free to move the proxy.
HONEST_FLOORS = {
    "visible_pass_rate": 0.05,
    "held_out_pass_rate": 0.05,
    "no_hardcoding_heuristic": 0.05,
    "syntax_parses": 0.03,
    "runs_without_error": 0.03,
    "brevity": 0.02,
}
TARGET_SUM = 1.20  # init 0.20 * 6 = 1.20 -> flat start


def make(strategy, floors, caps=None):
    return RTWTeacher(TeacherConfig(
        strategy=strategy, aux_keys=list(AUX_KEYS),
        stable_floors=dict(floors), stable_caps=dict(caps or {}),
        stable_target_weight_sum=TARGET_SUM,
    ))


def lerp(a, b, t):
    return a + (b - a) * max(0.0, min(1.0, t))


def stream(step):
    """Honest co-rise then policy games the visible tests (the E5 fingerprint)."""
    if step < 100:
        t = step / 100.0
        vis, hel, cor, nohc = lerp(0.10, 0.45, t), lerp(0.10, 0.30, t), lerp(0.02, 0.12, t), lerp(0.85, 0.80, t)
    else:
        t = (step - 100) / 60.0
        vis, hel, cor, nohc = lerp(0.45, 0.985, t), lerp(0.30, 0.145, t), lerp(0.12, 0.08, t), lerp(0.80, 0.30, t)
    tt = min(1.0, step / 80.0)
    return {
        "visible_pass_rate": vis, "held_out_pass_rate": hel,
        "no_hardcoding_heuristic": nohc,
        "syntax_parses": lerp(0.5, 0.97, tt), "runs_without_error": lerp(0.3, 0.92, tt),
        "brevity": lerp(0.7, 0.9, tt), "correct": cor,
    }


def batch(step, n=8):
    m = stream(step)
    return [dict(m) for _ in range(n)]


def gap(w):
    return w["visible_pass_rate"] - w["held_out_pass_rate"]


def run(strategy, floors, nsteps=300, snap_at=()):
    t = make(strategy, floors)
    snaps = {}
    for s in range(nsteps):
        w = t.update(batch(s))
        if (s + 1) in snap_at:
            snaps[s + 1] = dict(w)
    return t, dict(t.weights), snaps


print("=== DEFAULTS ===")
c = TeacherConfig()
for k in ("min_weight", "max_weight", "init_weight", "ema_beta", "stable_lr",
          "stable_alpha", "stable_delay_steps", "stable_target_weight_sum", "primary_success_decay"):
    print(f"  {k} = {getattr(c, k)}")

print("\n=== F1/F2: HONEST budget, adaptive_stable, 300 steps ===")
t, w, snaps = run("adaptive_stable", HONEST_FLOORS, 300, snap_at=(50, 60, 150, 200, 300))
for s in (50, 60, 150, 300):
    ws = snaps[s]
    print(f"  step {s:3d}: vis={ws['visible_pass_rate']:.4f} hel={ws['held_out_pass_rate']:.4f} "
          f"nohc={ws['no_hardcoding_heuristic']:.4f} gap={gap(ws):+.4f} sum={sum(ws.values()):.3f}")
print(f"  ema: vis={t.ema_aux['visible_pass_rate']:.3f} hel={t.ema_aux['held_out_pass_rate']:.3f} "
      f"nohc={t.ema_aux['no_hardcoding_heuristic']:.3f} primary={t.ema_primary:.3f}")
print(f"  need: vis={1-t.ema_aux['visible_pass_rate']:.3f} hel={1-t.ema_aux['held_out_pass_rate']:.3f}")

print("\n=== CONVERGENCE CHECK: is 300 steps enough? (frozen ema, run long) ===")
# Freeze the late-stage ema by feeding the phase-2 endpoint constantly, compare 300 vs 3000.
def run_const(nsteps):
    t = make("adaptive_stable", HONEST_FLOORS)
    endpoint = stream(300)
    for _ in range(nsteps):
        t.update([dict(endpoint) for _ in range(8)])
    return dict(t.weights)
for n in (100, 300, 1000, 3000):
    w = run_const(n)
    print(f"  n={n:5d}: vis={w['visible_pass_rate']:.4f} hel={w['held_out_pass_rate']:.4f} gap={gap(w):+.4f}")

print("\n=== F3/F5: proxy FLOOR sweep (adaptive_stable, 300 steps) ===")
print(f"  {'floor':>6} {'vis_w':>7} {'hel_w':>7} {'nohc_w':>7} {'sum':>6} {'gap':>8}")
cross = None
for floor in (0.05, 0.10, 0.15, 0.20, 0.25, 0.285, 0.290, 0.295, 0.30, 0.35, 0.40, 0.50):
    _, w, _ = run("adaptive_stable", {**HONEST_FLOORS, "visible_pass_rate": floor}, 300)
    print(f"  {floor:>6.3f} {w['visible_pass_rate']:>7.4f} {w['held_out_pass_rate']:>7.4f} "
          f"{w['no_hardcoding_heuristic']:>7.4f} {sum(w.values()):>6.3f} {gap(w):>+8.4f}")

print("\n=== F4: where does withheld proxy mass go? floor 0.05 vs 0.30 (all keys) ===")
_, w05, _ = run("adaptive_stable", {**HONEST_FLOORS, "visible_pass_rate": 0.05}, 300)
_, w30, _ = run("adaptive_stable", {**HONEST_FLOORS, "visible_pass_rate": 0.30}, 300)
for k in AUX_KEYS:
    print(f"  {k:26s} {w05[k]:.4f} -> {w30[k]:.4f}  (delta {w30[k]-w05[k]:+.4f})")

print("\n=== F6: STATIC ignores floors entirely ===")
for floors, lab in ((HONEST_FLOORS, "HONEST"), ({**HONEST_FLOORS, "visible_pass_rate": 0.30}, "TEMPT floor .30")):
    _, w, _ = run("static", floors, 300)
    print(f"  static/{lab}: vis={w['visible_pass_rate']:.4f} hel={w['held_out_pass_rate']:.4f} gap={gap(w):+.4f}")
print("  (init_weight is a SCALAR in TeacherConfig; static cannot encode vis=.30/hel=.10)")

print("\n=== FLOOR-PINS-PROXY 1:1 (does vis_w == floor above natural target?) ===")
for floor in (0.15, 0.20, 0.25, 0.30):
    _, w, _ = run("adaptive_stable", {**HONEST_FLOORS, "visible_pass_rate": floor}, 300)
    print(f"  floor {floor:.2f} -> vis_w {w['visible_pass_rate']:.4f} (diff {w['visible_pass_rate']-floor:+.4f})")
