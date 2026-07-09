# Bottleneck diagnosis: what caps exact-solution rate (2026-07-09)

Method: 5-lens adversarial workflow over the full local artifact corpus
(8 checkpoints × reward_components/teacher_weights, 16 best-of-N banks =
6400 candidate records). Each lens independently recomputed by a verifier
agent; all five returned **solid**. Numbers below are computed, not
estimated. This doc reframes the research program.

## Headline

Exact-solution rate at Qwen2.5-0.5B on Countdown is capped by **generation,
not selection**, and generation fails as a **two-layer wall (legality →
value-search) that collapses with difficulty tier**.

## The five load-bearing findings

### 1. Selection has ZERO headroom — retire the reranker lever

- `reranked_exact@N == oracle_exact@N` in **all 16 banks at N=1/4/8**.
  0 misselections across 800 task-instances.
- Loss decomposition at N=8 (800 task-instances): oracle-selected **70
  (8.75%)**, misselected **0**, **no-exact-candidate-formed 730 (91.25%)**.
  Every lost task is a generation failure.
- The practical selector separates cleanly by construction: all 111 exact
  candidates score `practical_score = 8.500`; all 6289 non-exact score
  ≤ 8.000 (mean 0.603). Hard 0.5 margin, zero ties.
- **Implication:** no selector/reranker change and no N>8 (as a *selection*
  win) can raise exactness. Retire selector engineering.

### 2. Generation ceiling is a LEGALITY wall first

- 81–87% of trained candidates fail legality. Single largest bucket:
  wrong-number-multiset ≈ **52% of all candidates** (missing-required
  dominates). Direct number-multiset failure is 66–80% of *parseable*
  candidates — the prior "53–64%" figure was an underestimate.
- True model ceiling under best-of-8: **oracle_exact@8 ≈ 9%**
  (per-candidate pass@1 ≈ 1.7–1.9%).

### 3. Behind legality sits a VALUE-SEARCH wall

- When fully legal, the model is still far: **P(exact | legal) = 0.14**
  pooled; median |value−target| ≈ 16; only ~12% within ±5.
- number_multiset_f1 is already saturating near the top (52–62% of
  non-exact parseable ≥ 0.75; ~20% at exactly 1.0) — an F1 aux reward would
  push mass to 1.0 without discriminating the last-number gap.

### 4. Tier collapse — exactness is essentially easy-tier-only

- Tasks with ≥1 exact candidate: **easy 25.0%, medium 4.1%, hard 0.4%**
  (1 exact in 1848 hard candidates; only 1 of 33 unique hard eval-tasks
  ever solved by any run/seed/split).
- Hard legality is 4.8% (vs ~22% easy/medium); even the rare legal hard
  candidates are numerically hopeless (median numeric_distance 0.006, best
  ever off-by-1). Hard (5-number) is a **compositional-search capability
  floor at 0.5B**, not a method gap.
- Eval splits are tier-balanced (val easy17/med15/hard18; test
  easy15/med20/hard15), so a strong easy-tier method gain is **diluted ~2/3**
  by medium+hard where the model is at the floor. This is why every method
  comparison looks like noise on the pooled 50-task metric.
- v10c2 (curriculum, up-weighted hard exposure) tripled hard *legality*
  (9.8% vs stable 3.4%) with **zero exactness gain** (1 hard exact vs 0) —
  confirms more hard exposure moves the legality wall a little but does not
  touch the search wall.

### 5. Reward hacking is NOT a driver — stop worrying about it

- A `reward_hacking_candidate` is format-compliant-but-unsolved, not a
  high-reward exploit. Training total_reward cleanly separates correctness:
  correct mean 2.20 vs **max incorrect 1.10** (< the 2.2 floor any correct
  answer gets); primary reward never leaks to wrong answers.
- For incorrect completions, aux reward is dominated by format+brevity
  (~50%), numeric_distance ~1.5%. The ~0.74 eval hack rate is an artifact of
  the base model not emitting answers at all (base emits `<answer>` 30% of
  the time, trained 90%), not shaping teaching an exploit.
- The reported metric's integrity holds: selector loses zero exact
  solutions; its −2.0 hack penalty roughly halves selected-answer hack rate.

## Paper-claim implication (feeds the escalated stable-vs-static question)

The **only** defensible stable-vs-static claim is **cost/efficiency**:
stable reaches the same exactness at **~0.58–0.60× the tokens** (paired sign
test p≈1e-14, 80–81% of tasks shorter) and **~half the 256-token clip rate**
(val 16.3% vs 32.9%). Accuracy: statistically indistinguishable (rerank@8
val 7-vs-3 discordants p=0.34, test 10-vs-7 p=0.63; training trajectories
near-identical). Robustness/variance: does NOT survive (cross-seed std flips
direction between splits, n=3 too small). Honest caveat: the cost saving is
not yet mechanistically pinned — the adaptive controller drifts only L1 0.16
off uniform 0.2 and even ends with *lower* brevity weight (0.159 vs 0.200)
despite shorter outputs, so the savings likely come from delay/EMA/low-LR
training dynamics, not the endpoint weight vector. **Scope all method claims
to easy-tier (3-number) Countdown.**

## Discarded directions (data says do NOT pursue)

- Selector/reranker engineering (0 headroom).
- Reward-hacking / anti-wireheading fixes (not the bottleneck).
- More hard-tier-exposure curricula (v10c2 disproved it).
- F1-shaped aux reward (already saturating near top).
- Adaptive-weight tuning from uniform init on this task (controller barely
  leaves uniform; GRPO groups only 2–3% zero-variance so signal exists but
  moves are too small to matter).
- Lower-seed-variance/robustness claims for stable (unsupported).
- N>8 as a selection win (only helps by forming new candidates).

## Ranked next experiments (full detail in NEXT_STEPS)

1. **SFT warmup on verifier-generated legal solutions** (data/capability
   lever — does NOT consume the reward-shaping strike). Attacks the legality
   wall directly. Predicted pooled oracle_exact@8 ~9%→~12–15%,
   concentrated on easy. Bounded by exact = legality × P(exact|legal).
2. **CPU probe:** freeze an easy-tier (and easy+medium) sub-metric + per-tier
   paired McNemar on existing banks — the honest comparison surface.
3. **CPU probe:** marginal-new-exact-per-candidate-index + clip-recovery
   upper bound — gates whether any generation-budget GPU run is justified.
4. Generation-decode (longer max_new_tokens / temperature) — low leverage,
   run only if probe 3 shows headroom.
5. Adversarial non-uniform init (static-bad vs stable-bad) — the one regime
   where RTW adaptivity could demonstrably beat a fixed schedule; defer.

## v0.12 scoring prediction (pre-registered)

Strong prior: v0.12 legality-envelope **moves legality but not exact**
(exact = legality × P(exact|legal≈0.14); shaping doesn't touch the search
wall — exactly what v10c2 showed). If legality rises but oracle_exact@8
stays within the ~9% ceiling / 3–5-per-50 noise band → **strike two for
reward-shaping-for-legality → retire that lever, pivot GPU to SFT warmup.**
Only exact clearly beyond noise (oracle@8 well above 7/50 with a lopsided
McNemar split) keeps reward shaping alive.
