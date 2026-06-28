# Paper Outline: Stability-Constrained Adaptive Reward Training Wheels for Verifier-Based LLM Post-Training

## Working title

**Stability-Constrained Reward Training Wheels for Verifier-Based LLM Harness Acquisition**

Alternative title:

**Adaptive Auxiliary Rewards Need Stability: A Countdown Wind-Tunnel Study for LLM Post-Training**

## Abstract draft

Verifier-based LLM post-training often relies on sparse task success signals, while practical agent harnesses impose additional structural requirements such as parseable outputs, legal tool calls, and validator-compatible formatting. We study whether Reward Training Wheels (RTW)—adaptive auxiliary reward weighting originally motivated by robotics reinforcement learning—can improve harness acquisition for language models. Using Countdown arithmetic as a controlled verifier-backed wind tunnel, we decompose harness success into exact verifier correctness and auxiliary legality signals: answer tags, expression parseability, number multiset use, allowed operators, numeric distance, and brevity.

A naive adaptive RTW teacher underperforms balanced static shaping because it overreacts to dense proxy signals and suppresses legality components. We introduce a stability-constrained adaptive teacher with delayed adaptation, smoothed updates, legality floors, numeric-distance caps, and preserved auxiliary reward budget. Across three seeds, the stabilized teacher improves in-distribution legality robustness and reduces seed sensitivity relative to static shaping: validation valid-expression rate improves from 0.252±0.151 to 0.393±0.033, and validation reward-hacking candidates fall from 0.737±0.136 to 0.607±0.033. However, exact correctness remains low, OOD-long remains unsolved, and OOD-division does not favor the adaptive teacher. These results suggest that adaptive reward weighting can help LLM harness acquisition, but only with explicit stability constraints and scoped claims.

## 1. Introduction

### Motivation

LLM agents are model-plus-harness systems. A deployed agent is not only model weights; it also includes prompts, tool schemas, validators, execution loops, output parsers, logging, and environment feedback. Post-training should therefore optimize not only final task success, but also acquisition of verifier-compatible harness behavior.

### Problem

Sparse verifier correctness is often too weak early in training. Auxiliary rewards can provide shaping, but static auxiliary weights may be brittle and naive adaptive weighting may overfit dense proxy signals.

### Thesis

Adaptive auxiliary rewards for LLM verifier harnesses are promising, but the adaptive controller itself needs stability constraints. Without them, it can reward-hack proxy components; with them, it can improve in-distribution legality robustness and reduce seed sensitivity.

### Contributions

1. A verifier-backed Countdown wind tunnel for studying LLM harness acquisition.
2. A decomposed reward trace separating primary verifier correctness from auxiliary legality rewards.
3. A naive adaptive RTW failure-mode analysis.
4. A stability-constrained adaptive RTW teacher.
5. A 3-seed static vs adaptive_stable comparison with scoped claims and limitations.

## 2. Related work

### Reward Training Wheels

Discuss RTW-style adaptive auxiliary rewards and how this work ports the idea from robotics/control to verifier-backed LLM post-training.

### RLVR and verifier-based training

Position Countdown as a small RLVR-style task where correctness is objective and verifier-owned.

### Reward shaping and process supervision

Auxiliary rewards help create gradient signal but can introduce reward hacking if not constrained.

### Curriculum and harness learning

GACL and adaptive curriculum methods are relevant future directions but are not the current main experiment.

## 3. Countdown harness wind tunnel

### Task

Given a multiset of numbers, a target, and allowed operators, produce an arithmetic expression inside answer tags.

### Verifier

The verifier is the source of truth. A completion is correct only if it passes exact expression validation:

- extract answer span;
- parse expression;
- use each required number exactly once;
- introduce no new numbers;
- use only allowed operators;
- evaluate exactly to target.

### Splits

- validation
- test_in_dist
- test_ood_long
- test_ood_division

## 4. Reward decomposition and RTW teacher

### Primary reward

Exact verifier correctness only.

### Auxiliary rewards

- format / answer tags
- valid expression
- number multiset F1
- allowed operators
- numeric distance reward
- brevity

### Strategies

1. Static v0.6b: balanced fixed auxiliary weights.
2. Naive adaptive v0.6b: adaptive weights without sufficient stability constraints.
3. Adaptive_stable v0.6c/v0.6d: delayed, smoothed, floor/cap constrained adaptive weights.

### Stabilization mechanisms

- 50-step delayed adaptation
- lower teacher learning rate
- EMA smoothing
- fixed auxiliary budget
- legality floors
- numeric-distance cap
- teacher mechanics diagnostics

## 5. Experiments

### Experiment 1: Dense legality acquisition

Show base model has near-zero usable harness behavior, while dense legality rewards produce parseable and partially legal expressions.

### Experiment 2: Naive adaptive failure mode

Compare naive adaptive vs static seed 0. Static generally wins. Diagnose adaptive collapse into numeric-distance reward and away from legality components.

### Experiment 3: Stability-constrained adaptive teacher

Compare static vs adaptive_stable over seeds 0, 1, 2.

### Experiment 4: Prompt/length diagnostic

Evaluate whether inference harness settings explain remaining low exact correctness. This is diagnostic, not a new trained method.

## 6. Results to include

### Table 1: Base model to dense reward acquisition

Columns:

```text
format, parse_ok, allowed_numbers, valid_expression, exact_correct
```

### Table 2: Naive adaptive vs static seed 0

Show naive adaptive underperforms static on held-out legality and reward-hacking rate.

### Table 3: Static vs adaptive_stable 3-seed main table

Use the v0.6d 3-seed aggregate:

| split | key result |
|---|---|
| validation | adaptive_stable improves valid_expression and reward_hacking_candidate |
| test_in_dist | adaptive_stable improves legality but exact_correct is lower |
| OOD-long | both weak; adaptive_stable not a win |
| OOD-division | static favored on valid_expression/reward_hacking |

### Table 4: Teacher mechanics

Show fixed budget, high constraint mass, bounded numeric-distance ratio, and smooth update sizes.

### Table 5: Prompt/length diagnostic

Include only if v0.7 produces a clear actionable result.

## 7. Failure analysis

### Reward hacking candidates

Many completions are parseable but wrong or use incomplete/extra numbers.

### Exact correctness bottleneck

Legality acquisition improves more than final arithmetic correctness.

### OOD behavior

OOD-long remains largely unsolved. OOD-division favors static on some legality metrics.

### Clipping / over-generation

Training completions reached max length, suggesting answer discipline remains a bottleneck.

## 8. Limitations

- Countdown is a wind-tunnel task, not a full coding-agent harness.
- Small model: Qwen2.5-0.5B-Instruct.
- Exact correctness is low across all methods.
- OOD-long remains unsolved.
- OOD-division does not support a global adaptive win.
- Adaptive_stable improves legality robustness more than arithmetic correctness.
- Three seeds are enough for a pilot paper claim, not a definitive scaling law.
- Prompt/length ablations are inference diagnostics unless retrained.

## 9. Future work

1. Conditional teacher state based on component improvement rates.
2. Two-phase legality-then-correctness teacher.
3. Curriculum over task difficulty / GACL-style task scheduling.
4. Larger models and longer training.
5. Coding-agent harnesses with tool-call validators.
6. Inference harness optimization separated from training reward design.

## 10. Current paper claim

Recommended final claim for the current evidence:

> In a verifier-backed Countdown harness, stability-constrained adaptive auxiliary reward weighting improves in-distribution legality robustness and reduces seed sensitivity relative to static shaping, while naive adaptive weighting can underperform and while exact correctness and OOD transfer remain open problems.

## 11. Claims to avoid

Do not claim:

- adaptive_stable globally beats static;
- exact correctness is solved;
- Countdown proves coding-agent harness transfer;
- prompt/length diagnostics are trained-method improvements;
- OOD generalization is improved overall.
