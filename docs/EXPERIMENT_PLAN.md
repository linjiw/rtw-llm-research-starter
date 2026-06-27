# Full experiment plan: RTW for harness-aware LLM post-training

## 1. Research question

Can adaptive auxiliary reward weights improve LLM post-training on verifiable reasoning tasks, compared with fixed expert rewards, manual reward schedules, and random reward weights?

The core transfer from robotics RTW is:

```text
student reward = primary correctness reward + weighted auxiliary rewards
teacher state  = history / EMA of primary reward, auxiliary scores, and previous weights
teacher action = next auxiliary reward weight vector
teacher reward = improvement in primary task success
```

The core transfer from GACL is:

```text
curriculum should track task state, student performance, and domain grounding
```

In the first version, we focus on RTW reward adaptation. The dataset also includes harness levels and OOD splits so the next iteration can add harness and task curriculum.

## 2. Task

Countdown-style arithmetic reasoning.

Each example contains:

- `numbers`: a multiset of integers.
- `target`: target integer.
- `allowed_ops`: available arithmetic operators.
- `solution`: one known solution expression.
- `prompt_low`, `prompt_mid`, `prompt_high`: three harness informativeness levels.
- `completion`: synthetic SFT-style completion.

The verifier checks:

1. The model uses `<answer>...</answer>`.
2. The expression parses as safe arithmetic.
3. It uses every provided number exactly once.
4. It uses only allowed operations.
5. It evaluates exactly to the target using rational arithmetic.

## 3. Reward design

Primary reward:

```text
R_primary = 1 if verifier says expression is correct else 0
```

Auxiliary rewards:

```text
R_format           = 1 if answer tag exists else 0
R_parse_ok         = 1 if expression parses else 0
R_uses_numbers     = 1 if expression uses exactly the number multiset else 0
R_allowed_ops      = 1 if expression only uses allowed ops else 0
R_valid_expression = 1 if parse + numbers + ops + numeric evaluation are valid else 0
R_brevity          = 1 if answer is below length threshold else 0
```

Total reward:

```text
R = R_primary + Σ_k w_k(t) R_aux,k
```

The first experiments use the auxiliary keys:

```text
format, valid_expression, uses_numbers, allowed_ops, brevity
```

## 4. RTW controller

The default teacher is a stable controller, not a PPO outer-loop teacher. This is intentional: the first result should isolate the post-training phenomenon without a noisy second RL process.

State:

```text
EMA(primary correctness)
EMA(auxiliary component scores)
previous weights
```

Adaptive update intuition:

```text
if the student fails an auxiliary behavior -> increase that auxiliary weight
as primary correctness rises -> decay all training wheels
```

Formula in the scaffold:

```text
competence = EMA(R_primary)
global_decay = 1 - primary_success_decay * competence
need_k = 1 - EMA(R_aux,k)
target_w_k = min_w + (max_w - min_w) * need_k * global_decay
w_k <- (1 - lr) * w_k + lr * target_w_k
```

## 5. Baselines

Run the same model, data, and GRPO settings across:

1. Base model, no post-training.
2. GRPO + static expert auxiliary weights.
3. GRPO + manual linear reward decay.
4. GRPO + random reward weights.
5. GRPO + adaptive RTW reward weights.

Optional later baselines:

- SFT only.
- SFT + GRPO.
- DPO from correct/incorrect pairs.
- No auxiliary rewards, primary-only GRPO.
- Task curriculum without reward curriculum.
- Reward curriculum without task curriculum.

## 6. Metrics

Main:

- Exact success / pass@1.
- Format validity.
- Expression validity.
- Uses-all-numbers rate.
- Allowed-ops rate.
- OOD exact success.

Training-efficiency:

- Success vs. optimizer steps.
- Success vs. rollout tokens.
- Steps/tokens to reach 20%, 40%, 60% success.

Reward-hacking diagnostics:

- `<answer>` present but expression invalid.
- Expression uses target as invented constant.
- Expression ignores one or more required numbers.
- Expression uses an operator outside `allowed_ops`.
- Very long answer with no valid final expression.

Harness-shift diagnostics:

- Train on `prompt_high`; evaluate on `prompt_high`, `prompt_mid`, `prompt_low`.
- Strong performance only under `prompt_high` implies harness overfitting.
- Stable performance across levels implies the model learned the task more than the prompt surface.

## 7. Dataset splits

Default generated splits:

- `train`: easy/medium/hard mixture; no division.
- `validation`: same distribution.
- `test_in_dist`: same distribution.
- `test_ood_long`: six-number tasks.
- `test_ood_division`: division-enabled tasks.

This creates two useful OOD axes:

1. longer horizon;
2. new operator/tool affordance.

## 8. Hardware plan

Recommended starter:

- One 24 GB GPU for Qwen2.5-0.5B/1.5B LoRA GRPO smoke runs.
- One 48 GB or 80 GB GPU for larger batch, longer rollouts, or 3B/7B experiments.
- CPU-only is acceptable for dataset generation and verifier tests.

Expected bottleneck:

- GRPO rollout generation, not the verifier.
- Keep `num_generations`, prompt length, completion length, and dataset size small until curves are stable.

## 9. Minimum viable study

Phase A: smoke test

```text
N_train = 5k tasks
model = Qwen2.5-0.5B-Instruct
max_steps = 300
num_generations = 4
single seed
```

Phase B: main small study

```text
3 seeds x 4 baselines + RTW
same train/eval data
report mean ± std
```

Phase C: harness-shift study

```text
train prompt_high
evaluate prompt_low/prompt_mid/prompt_high
```

Phase D: generalization study

```text
evaluate test_in_dist, test_ood_long, test_ood_division
```

## 10. Paper-shaped claims to test

Strong claim:

> Adaptive auxiliary reward weights improve sample efficiency and reduce verifier/harness failure modes while preserving or improving final correctness.

More cautious claim:

> RTW changes the learning trajectory: early rewards emphasize format and validity, later rewards decay toward sparse correctness, producing a measurable training-wheel pattern.

The second claim is easier to prove and still valuable.

## 11. Failure modes to watch

1. Auxiliary rewards dominate primary correctness.
2. Model learns answer tags but not arithmetic.
3. Teacher weights oscillate due noisy batch rewards.
4. Static baseline is already too strong on easy tasks.
5. OOD division is too hard for a small model.
6. GRPO instability hides curriculum effects.

Mitigations:

- Keep primary reward weight fixed and dominant.
- Evaluate primary-only success separately.
- Smooth teacher state with EMA.
- Increase task difficulty only after base smoke runs.
- Use multiple seeds before making strong claims.
