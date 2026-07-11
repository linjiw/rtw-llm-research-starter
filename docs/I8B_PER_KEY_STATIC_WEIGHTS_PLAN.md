# I8b: per-key init/static weight vector (design → advisor → implement)

Created: 2026-07-11. Status: design (pre-advisor-review). Unblocked by E2 GO.
Grounding: `E5_PRECONDITION_TEACHER_MECHANISM_20260710.md` — the reshaped E5
needs `static`/`manual` (and the delay-period init) to hold a **per-key**
weight vector so the TEMPTATION arm can be a proxy-overweight budget. Today
`TeacherConfig.init_weight` is a single scalar applied to all aux_keys, so a
proxy-heavy budget is not expressible. This is the ONLY code change gating E5.

## Requirement

- `static` must return a per-key weight vector (the TEMPTATION budget:
  visible_pass_rate ~0.30–0.35, held_out/no_hardcoding ~0.10, others low).
- The `adaptive_stable` delay-period reset and the `__init__` seed should honor
  the per-key vector too (so the adaptive arm can START from the same
  proxy-overweight init and then decay it — the E5 precondition showed
  convergence is start-independent, but starting matched keeps the arms
  comparable).
- **Default-off / byte-identical:** when the new field is unset, every site
  behaves exactly as today (scalar `init_weight`). Countdown runs must be
  bit-identical.

## Design (one variable, minimal surface)

Add one optional field to `TeacherConfig`:

```python
init_weights: dict[str, float] | None = None  # per-key init/static override
```

Add one private helper on `RTWTeacher`:

```python
def _init_weight(self, key: str) -> float:
    if self.config.init_weights is not None and key in self.config.init_weights:
        return float(self.config.init_weights[key])
    return float(self.config.init_weight)
```

Replace the 4 scalar-init sites to route through it:
- `__init__` (line ~103): `{k: self._init_weight(k) for k in aux_keys}`
- `update` static branch (~158): same
- `_adaptive_stable_update` delay reset (~215): same
- `_adaptive_phased_update` delay reset (~232): same
- `get_weights` static path returns `dict(self.weights)` already (weights were
  set from `_init_weight` in the static branch) — verify `static` never falls
  through to the scalar; the update() static branch sets self.weights each step
  so get_weights returns them. **Check:** does `static` call update() every
  step? If a run reads get_weights BEFORE the first update, __init__ must have
  seeded per-key (it will, via site 1). OK.
- `_project_stable_weights` fallback (~317, `candidate.get(key, init_weight)`):
  route through `_init_weight(key)` for consistency (only hit if a key is
  missing from the candidate — defensive; keep aligned).

`manual` (linear decay max→min) is left scalar for now — the TEMPTATION budget
is a `static` proxy-overweight, and `manual`'s decay semantics don't cleanly
take a per-key start without more design; out of scope unless E5 needs it.

## Validation (VALIDATE checkpoint)

- Unit tests: (a) with `init_weights=None`, weights == scalar path for
  static/adaptive_stable delay (byte-identical — assert exact dict equality
  against current behavior); (b) with `init_weights={visible:0.35, held_out:0.10,...}`,
  static returns exactly that vector and the adaptive_stable delay seeds it;
  (c) a key absent from `init_weights` falls back to the scalar; (d) adaptive
  arm with a low proxy floor decays a 0.35 proxy init toward its need-driven
  fixed point (reuses the E5 sim assertion).
- Full suite + ruff; Countdown adaptive_stable/static training config unchanged
  → existing teacher tests must pass untouched.
- Advisor diff review before this is used in any GPU run.

## Non-goals / guardrails

- No change to floors/caps/budget projection math, EMA, or the need=1−ema rule.
- No new strategy name (this is a config knob on existing static/adaptive_stable).
- Countdown defaults: `init_weights` unset everywhere → zero behavior change.
- Ledger row `i8b` (infra); then E5 pre-registration doc can specify the exact
  TEMPTATION `init_weights` vector + the hack_wins precondition check.
