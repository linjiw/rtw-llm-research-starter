# Does the post-SFT regime revive adaptivity on Countdown? (precondition check)

Created: 2026-07-09. CPU analysis, no GPU. Question: v0.13 SFT-warmup lifted the
model off the generation floor (oracle@8 ~0.10→~0.47). The bottleneck diagnosis
said adaptive curricula (RTW/GACL) were inert on Countdown *because* the regime
was variance-saturated and bimodal-floored. Did SFT remove those preconditions,
making "adaptive arms ON TOP OF SFT" a live thesis-direct experiment — or is it
another v0.10/v0.12 trap? **Check the precondition before spending GPU.**

## What SFT changed (last-third of training, seed 0, vs cold-start C0)

| signal | C0 cold-start | v13 post-SFT | verdict for adaptivity |
|---|---|---|---|
| valid (legality) easy/med/hard | 0.53 / 0.43 / 0.16 | 0.99 / 0.88 / 0.73 | **smooth gradient now exists** ✓ |
| exact easy/med/hard | 0.11 / 0.01 / 0.04 | 0.49 / 0.07 / 0.07 | still **bimodal** (value-search wall) ✗ |
| mean group_reward_std | 0.245 | 0.205 | **did NOT densify** (slightly lower) ✗ |

## Read (honest, mixed)

- SFT moved the **legality** wall into a smooth easy>med>hard gradient — a
  legality-phase curriculum would, for the first time, have something to
  climb. This is genuinely new vs the diagnosis regime.
- But the **value-search** wall is intact: exact is still 0.49 vs 0.07/0.07
  (a cliff, not a gradient), so an exact-phase task-curriculum still starves
  above easy.
- Critically, **reward variance did not increase** — the single thing the
  diagnosis identified as most limiting for adaptivity (GRPO advantage needs
  within-group variance) is no better post-SFT.

## Decision: do NOT run the full adaptive stack on post-SFT Countdown

Two of three preconditions are still unmet (bimodal exact, unchanged variance).
Running RTW/GACL on top of SFT would most likely reproduce the v0.10/v0.12
outcome — adaptivity moving an intermediate quantity (here, legality, which is
already near-ceiling post-SFT) without moving exact. That would burn ~2×3.5h
GPU to re-learn the lesson. **The north-star home for the adaptivity thesis
remains Paper 2 (MicroCode)**, which was *designed* to supply the dense
within-group variance (CPU gate confirmed std 0.393 on held-out pass-rate) and
a live hacking surface — precisely the preconditions Countdown lacks even
after SFT.

## The one narrow post-SFT adaptivity probe that WOULD be defensible (deferred)

If, after Paper 2, we want a Countdown adaptivity data point: a **legality-phase
task-curriculum on the post-SFT checkpoint targeting medium/hard legality**
(where the gradient now exists 0.88/0.73) — NOT exact. It would test whether a
curriculum can push medium/hard legality toward easy's 0.99, a well-powered
candidate-level signal. But it does not test the exact-success thesis and is
lower priority than Paper 2. Recorded, not queued.

## Ledger

id `postsft-precondition`, insight (CPU). Closes the tempting "adaptivity on
top of SFT" direction on precondition grounds before any GPU spend — the same
discipline (measure saturation first) that the diagnosis established.
