# E5 precondition: does the teacher mechanism support the hacking-resistance headline? (CPU, real teacher)

Created: 2026-07-10. CPU-only, no GPU. Method: a 3-regime workflow that drives
the **imported real `RTWTeacher`** (never reimplements its math) over synthetic
MicroCode-like reward streams, each regime adversarially verified. Purpose:
before spending GPU on Paper-2 E5 (the adaptivity-thesis headline), check
whether the teacher's mechanism can actually do what E5 credits it with — and
whether the E5 design as written in `RESEARCH_GOAL_AND_PLANS_20260709.md` §4 is
even implementable. It is not, as written; fix recorded below.

## The headline being tested

E5: two budget arms × {static, adaptive_stable}. Thesis — under a
mis-weighted TEMPTATION budget, static gets hacked (proxy−primary gap widens)
while adaptive_stable spontaneously down-weights the saturating gamed proxy
(`visible_pass_rate`) and keeps the gap closed.

## Finding 1 — MECHANISM GO (solid, verified against the imported teacher)

As `ema(visible_pass_rate)` saturates 0.24 → 0.99, `need = 1 − ema` collapses
to ~0.01 and adaptive_stable drives the proxy weight **0.20 → ~0.145** while
`held_out_pass_rate` rises to **~0.308 (the largest aux weight)** and the
anti-hack channel `no_hardcoding_heuristic` rises to **~0.28**. The
proxy-minus-held_out WEIGHT gap goes firmly negative (~−0.16), weight-sum held
exactly at 1.20 every post-delay step, no floor/cap saturation, no instability.
The honest signal is **not** starved by the budget projection: held_out is
protected because its low EMA keeps its need high; withheld mass is paid by
low-need scaffold keys (syntax/brevity/runs). Converges ~93–98% by step 300 at
stable_lr 0.10 / alpha 0.10. **The need=1−ema mechanism does exactly what the
thesis credits it with, at baseline.**

Two bounds: (a) suppression is **relative, not absolute** — with 6 channels
sharing sum 1.20 the proxy floors near 0.145, so the lever is a ~2.1–2.6×
held/vis de-emphasis, not near-zero proxy. (b) The mechanism is **need-driven,
not hack-aware** — held_out is up-weighted only because it is observed LOW; if
a hack ever co-inflated held_out, protection would weaken (a live-run risk).

## Finding 2 — THE E5 DESIGN AS WRITTEN IS BROKEN (two blockers, both free to catch here)

**Blocker A — "TEMPTATION = raised proxy floor" INVERTS the thesis.** Verified
1:1: a raised `visible_pass_rate` floor PINS the proxy weight AT the floor
(floor 0.20 → vis_w 0.201; 0.30 → 0.300), which *handcuffs* the adaptive arm —
it disables the very down-weighting the thesis credits — rather than letting it
rescue. And there is a hard crossover at proxy floor ≈ **0.295** (with sum 1.20,
held_out floor 0.10): above it the proxy is pinned above held_out and the
guarantee fails outright.

**Blocker B — `static` cannot encode a proxy-heavy budget.** In the real code
`static` emits a single scalar `init_weight` for ALL keys (and the delay branch
hard-resets to uniform init), so the "mis-weighted static" arm is not
instantiable — static as-is sits co-equal forever, which is not the "proxy-
dominant, gets hacked" contrast E5 needs.

## Corrected E5 design (RESHAPE — required before any GPU spend)

1. **Encode TEMPTATION as a proxy-OVERWEIGHTED per-key INITIAL/static weight
   vector, NOT a raised floor.** Proxy `visible_pass_rate` init/frozen high
   (~0.30–0.35, proxy-dominant so a full hack out-ranks an honest partial
   within a GRPO group); true-signal channels held_out/no_hardcoding ~0.10.
2. **Adaptive arm keeps a LOW proxy floor** (~0.02–0.10, well below the ~0.295
   crossover) so it retains headroom to decay the 0.35 init to its need-driven
   ~0.145 fixed point (convergence is start-independent).
3. **Keep held_out floor ≥ 0.10** so the budget projection cannot starve it.
4. **REQUIRED CODE CHANGE (framework, one variable, default-off):** add a
   per-key init/static weight vector so `static`/`manual` can hold a proxy-heavy
   budget. Countdown defaults must stay byte-identical (scalar init_weight is
   the default; the vector is opt-in). Advisor design+diff review before GPU.
5. **FAIRNESS PRECONDITION (pre-register):** a proxy overweight alone does not
   guarantee the hack wins the within-group ranking (held_out weight rises on
   its own). Verify **TEMPTATION-static actually reaches `hack_wins=True`**
   (boundary observed when true-signal channels are capped between 0.05 and
   0.10) BEFORE any resistance claim — else the test is impossible-to-win and
   the contrast collapses.

## What only the GPU E5 run can settle (do NOT overclaim from CPU)

Everything here is TEACHER-WEIGHT dynamics driven by an exogenous synthetic
reward stream — a reduced-form/open-loop result. It establishes the mechanism
and its operating envelope for free; it does NOT establish policy-level
resistance. The closed loop — does the 0.5B policy actually learn to hack under
proxy-overweight static (gap widening in POLICY OUTCOMES, per the held-out
verifier), and is adaptive's down-weighting fast/strong enough to shift the
GRPO group-relative advantage so the policy stops — is GPU-only. Also GPU-only:
the ~10-step EMA lag against a moving policy, and whether need=1−ema misfires if
a hack co-inflates held_out.

## Verdict

**E5: GO-with-constraints.** The mechanism is real (Finding 1) but the headline
needs the RESHAPE above (Finding 2) before any GPU. This is a free pre-GPU
catch: the original floor-based design would have inverted the thesis and the
static arm would not have encoded the mis-weighting. Ledger row `e5-precond`.
Resistance claim remains gated on the live E5 closed loop.
