# Stable-RTW Paper Draft Direction

## Preferred title

**Stable-RTW: Stability-Constrained Adaptive Reward Weighting for Verifier-Based LLM Harness Acquisition**

Alternative shorter title:

**Learning the Legal Action Space: Stable Adaptive Reward Weighting for Verifier-Based LLM Harnesses**

## One-sentence claim

Dense legality rewards can teach an LLM to enter a constrained verifier-defined action space, but reward weighting must be stability-constrained; otherwise static or naive adaptive shaping can be seed-sensitive, unstable, or prone to reward-hacking behavior.

This is **not** a Countdown-solving paper and should not claim adaptive weighting always improves exact correctness. The defensible claim is:

> Stable adaptive reward weighting improves legal-expression acquisition and reduces seed sensitivity in verifier-based harness training.

## Abstract draft

Verifier-based reinforcement learning offers an appealing path for training language models on structured reasoning and coding tasks, where correctness can be checked by execution rather than judged only by text similarity. However, final correctness rewards are often sparse: before a model can solve a task, it must first learn to produce outputs that are parseable, executable, and legal under the task's constraints. We study this problem through a controlled Countdown expression-generation harness, where a model must construct an arithmetic expression that uses exactly the provided numbers and allowed operators to reach a target value.

We introduce a legality-focused verifier harness that preserves strict correctness semantics while exposing dense auxiliary signals for expression format, number multiset matching, operator legality, executable evaluation, numeric distance to the target, and brevity. We then compare static reward shaping, naive adaptive reward weighting, and a stability-constrained adaptive reward teacher, Stable-RTW. Stable-RTW dynamically reallocates auxiliary reward weight toward current legality bottlenecks while enforcing constraints on total weight mass, update size, numeric-distance dominance, and constraint-focused reward allocation.

Across multi-seed experiments, dense legality rewards substantially improve legal-expression acquisition, but reward weighting strategy strongly affects robustness. Static shaping can achieve useful runs but exhibits seed sensitivity and held-out legality collapse. Naive adaptive weighting is unstable. In contrast, Stable-RTW improves in-distribution legality robustness, reduces reward-hacking candidates, and preserves nonzero strict verifier correctness. At the same time, exact correctness remains low, OOD-long generalization remains weak, and training traces reveal persistent over-generation and max-length clipping. These results suggest that adaptive reward teachers are useful not as a complete solution to symbolic reasoning, but as a stabilizing mechanism for verifier-based harness acquisition. We conclude that future progress requires combining stable legality acquisition with better termination control, curriculum learning, and verifier-guided search over legal expressions.

## Core framing: model × harness

Modern LLM coding/task performance should be understood as **model × harness configuration**. Context construction, tools, execution, validation, tracing, governance, and reward logic directly shape outcomes, not just the base model. This paper studies a compact, controlled instance of that broader claim: before moving to a full coding agent, we test whether a verifier-based training harness can reliably move an LLM into a legal action space.

Countdown is therefore a microscope for verifier-harness dynamics, not the final application domain.

## Contributions

### Contribution 1: Verifier-based harness acquisition

We define **verifier-based harness acquisition** as training an LLM not merely to maximize final correctness, but to acquire the intermediate behavioral constraints required by a verifier-controlled task environment.

In this setting, the model must learn to produce outputs that are:

- parseable;
- expression-like;
- operator-legal;
- number-legal;
- executable;
- not overlong;
- close enough to be useful;
- eventually exactly correct.

Final correctness is too sparse early in training. A model that never enters legal expression space cannot benefit much from an exact verifier. The harness must expose dense legality structure without weakening the final correctness definition.

### Contribution 2: A verifier-preserving legality reward harness

We introduce a Countdown harness where strict correctness remains unchanged:

```text
exact_correct = 1 only if the expression passes the strict verifier.
```

The model is never counted as correct merely because it receives dense auxiliary reward.

Dense auxiliary metrics are training wheels:

- format;
- valid_expression;
- number_precision;
- number_recall;
- number_multiset_f1;
- uses_no_extra_numbers;
- uses_all_required_numbers;
- allowed_ops;
- operator_precision;
- operator_recall;
- evaluates_without_exception;
- numeric_distance_reward;
- brevity.

Design principle:

> Dense rewards can shape learning, but they must not redefine correctness.

### Contribution 3: Stable-RTW

Static reward shaping fixes auxiliary weights by hand. It can work, but it may overfit one seed or collapse on held-out legality.

Naive adaptive reward weighting moves weights dynamically, but can become unstable: it may chase noisy metrics, overemphasize numeric distance, or reward partial hacks that do not improve legality.

Stable-RTW adds constraints to the adaptive teacher:

- fixed total auxiliary weight mass;
- minimum constraint-focused reward mass;
- cap on numeric-distance dominance;
- small teacher updates;
- reward-variance health checking;
- separation between primary and auxiliary reward;
- strict verifier preserved.

Algorithmic identity:

> Stable-RTW adapts toward current legality bottlenecks, but refuses to let the reward teacher abandon verifier-relevant constraints.

### Contribution 4: Stability-centered evaluation

Instead of reporting only exact correctness, evaluate:

1. legal action-space acquisition;
2. strict verifier correctness;
3. reward-hacking risk;
4. seed stability;
5. teacher mechanics.

Key metrics:

- valid_expression;
- allowed_numbers;
- number_multiset_f1;
- allowed_ops;
- exact_correct;
- reward_hacking_candidate;
- reward_variance_nonzero_fraction;
- teacher_update_l1_mean;
- constraint_weight_mass_final;
- numeric_distance_ratio_final.

This supports a precise claim:

> Stable-RTW improves legality robustness and reduces seed sensitivity, but it does not yet solve exact arithmetic reasoning.

## Method: Stable-RTW

### Problem setup

Each task is:

```text
x = (N, O, y)
```

where:

- `N` is the required number multiset;
- `O` is the allowed operator set;
- `y` is the target value.

The model generates a candidate expression:

```text
e ~ πθ(. | x)
```

The strict verifier is:

```text
V(e, x) = 1[
  parseable(e)
  AND nums(e) = N
  AND ops(e) subseteq O
  AND eval(e) = y
]
```

So:

```text
R_primary(e, x) = V(e, x)
```

Auxiliary metrics are:

```text
m(e, x) = [
  m_format,
  m_valid,
  m_number_f1,
  m_ops,
  m_distance,
  m_brevity
]
```

The shaped training reward is:

```text
R(e, x) = alpha * R_primary(e, x) + beta * sum_j w_j * m_j(e, x)
```

Evaluation correctness remains:

```text
exact_correct = R_primary
```

not the shaped reward.

### Stable-RTW teacher update

The adaptive teacher maintains auxiliary weights:

```text
w_t = [w_1,t, ..., w_J,t]
```

At each teacher update, compute moving averages:

```text
mu_j,t = EMA(m_j)
```

Define metric deficits:

```text
d_j,t = max(0, tau_j - mu_j,t)
```

where `tau_j` is the desired target level for that legality signal.

The teacher proposes new weights from deficits, then projects the proposal into a safe constraint set:

```text
w_{t+1} = Project_C(proposed_w_{t+1})
```

The constraint set enforces:

- `sum(w) = W`;
- `w_j >= 0`;
- `constraint_weight_mass >= rho`;
- `numeric_distance_weight <= kappa * W`;
- `||w_{t+1} - w_t||_1 <= epsilon`;
- reward variance remains nonzero.

The projection is the important part: it prevents the teacher from becoming too reactive.

### Algorithm block

```text
Algorithm 1: Stable-RTW

Input:
  policy model πθ
  reference model πref
  task distribution D
  strict verifier V
  auxiliary metrics M
  auxiliary weight budget W
  teacher update rate η
  stability constraints C
  GRPO group size K

Initialize:
  auxiliary weights w0
  teacher statistics S0
  policy parameters θ

For training step t = 1 ... T:

  1. Sample a batch of tasks:
       x1, ..., xb ~ D

  2. For each task xi, sample K completions:
       ei,1, ..., ei,K ~ πθ(. | xi)

  3. For each completion, compute:
       primary_reward = V(ei,k, xi)
       auxiliary_metrics =
         format,
         valid_expression,
         number_multiset_f1,
         allowed_ops,
         numeric_distance_reward,
         brevity
       shaped_reward =
         α * primary_reward
         + β * Σj wj * auxiliary_metric_j

  4. Within each task group, compute group-relative advantages:
       Ai,k = normalize_group(shaped_reward_i,k)

  5. Update πθ with GRPO using:
       group-relative advantage,
       KL regularization to πref,
       verifier-harness reward

  6. Every U steps, update the teacher:
       a. Compute EMA metric means μj
       b. Compute metric deficits dj = max(0, τj - μj)
       c. Propose adaptive weights from deficits
       d. Project weights into stability constraints:
            fixed total weight mass
            minimum constraint mass
            numeric-distance cap
            small L1 update
            nonzero reward-variance health

  7. Log:
       strict correctness,
       dense legality metrics,
       reward variance,
       teacher weight movement,
       reward-hacking candidates,
       completion length statistics

Return:
  trained policy πθ
  teacher trace
  verifier-based evaluation table
```

## Proposed paper structure

### 1. Introduction

Large language models are increasingly evaluated inside executable task environments rather than static text benchmarks. In coding, software repair, and tool-use settings, success depends not only on model parameters but also on the surrounding harness: the prompt protocol, context builder, tool interface, execution sandbox, verifier, reward function, and observability layer. Recent harness-engineering work argues that performance should be reported as a property of the model-harness pair, since changing the harness can alter both success rates and failure modes.

In this work, we study a controlled instance of harness acquisition: training an LLM to produce legal arithmetic expressions under a strict verifier. The task is intentionally small, but it exposes a general problem in verifier-based learning. Final correctness is sparse. A candidate expression is correct only if it is parseable, executable, uses exactly the provided number multiset, uses only allowed operators, and evaluates to the target. Most early model outputs fail before reaching the final arithmetic objective. Therefore, the harness must teach the model to enter the legal action space before exact correctness can become a useful learning signal.

We ask: **how should dense auxiliary rewards be weighted so that they help a model acquire verifier-relevant legality without encouraging reward hacking or seed-specific collapse?**

We compare static reward shaping, naive adaptive reward weighting, and a stability-constrained adaptive reward teacher, Stable-RTW. Dense legality rewards improve legal expression acquisition, but reward weighting strategy determines whether this improvement is stable across seeds. Stable-RTW reduces seed sensitivity and improves in-distribution legality robustness, while exact correctness and OOD-long generalization remain open challenges.

### 2. Related work

Organize related work into four buckets:

1. **Harness engineering for LLM coding and task agents** — evaluation harnesses, execution harnesses, agent harnesses, meta-harnesses; this paper studies reward harness design as one component of a broader executable task loop.
2. **Verifier-based learning and RL for code/reasoning** — RLVR, GRPO, test-based reward, executable verification.
3. **Reward shaping and reward hacking** — dense legality rewards are useful but can be gamed if they redefine or dominate correctness.
4. **Curriculum and legal action-space acquisition** — current method handles legality acquisition; OOD-long suggests curriculum/search is needed next.

### 3. Task and harness

Define Countdown formally:

- numbers: `[n1, n2, ..., nk]`;
- allowed operators: `O`;
- target: `y`;
- candidate expression string: `e`.

Strict correctness requires:

- parseable expression;
- exactly the provided number multiset;
- only allowed operators;
- evaluates without exception;
- equals target.

Dense diagnostics include:

- number_multiset_f1;
- operator_precision;
- operator_recall;
- numeric_distance_reward;
- valid_expression;
- brevity.

Repeat the central semantic guardrail:

> The strict verifier is never relaxed. Dense rewards only affect training, not final correctness.

### 4. Method: Stable-RTW

Subsections:

1. Reward decomposition.
2. Why static shaping is insufficient.
3. Why naive adaptation is dangerous.
4. Stable-RTW projection and health constraints.
5. Logging and teacher mechanics.

### 5. Experiments

Research questions:

- **RQ1:** Do dense legality rewards move the model into legal expression space?
- **RQ2:** Does adaptive reward weighting improve over static shaping?
- **RQ3:** Does stability-constrained adaptation reduce seed sensitivity?
- **RQ4:** Does improved legality translate into exact correctness?
- **RQ5:** Where does the trained harness still fail under OOD splits?

Methods:

- static_v0.6b;
- naive adaptive RTW;
- Stable-RTW / adaptive_stable_v0.6c-v0.6d.

Splits:

- validation;
- test_in_dist;
- test_ood_long;
- test_ood_division.

Metrics:

- valid_expression;
- allowed_numbers;
- number_multiset_f1;
- allowed_ops;
- exact_correct;
- reward_hacking_candidate;
- reward_variance_nonzero_fraction;
- teacher_update_l1_mean;
- numeric_distance_ratio_final;
- constraint_weight_mass_final.

### 6. Results narrative

Write the result section as a controlled diagnosis, not as a global win.

1. **Legality improves substantially.** Dense legality rewards move the model into legal expression space.
2. **Stable-RTW improves robustness.** Stable-RTW has lower seed variance than static.
3. **Static can collapse.** Seed 1 shows static held-out legality collapse.
4. **Exact correctness remains sparse.** Legal expression acquisition is not equivalent to exact solving.
5. **OOD-long remains difficult.** Longer compositions require curriculum or search.
6. **Termination remains a harness bottleneck.** Training logs show widespread max-length clipping.

### 7. Failure analysis

Required taxonomy:

- parse failure;
- illegal number multiset;
- illegal operator;
- legal but wrong value;
- overlong clipped output;
- reward-hacking candidate;
- OOD-long composition failure.

### 8. Limitations

- Exact correctness remains low.
- OOD-long remains poor.
- OOD-division does not support a global Stable-RTW win.
- Over-generation persists.
- Countdown is controlled and narrow: a microscope for verifier-harness dynamics, not a full coding benchmark.
- v0.7 prompt-length diagnostics are limitation diagnostics unless retrained.

### 9. Future work

- Termination control / answer discipline.
- Curriculum for longer compositions.
- Verifier-guided search over legal expressions.
- More expressive teacher state based on improvement rates.
- Larger models and longer training.
- Coding-agent harnesses with tool-call validators.

## Main paper tables

### Main 3-seed table

| Method | Split | Valid Expr ↑ | Exact ↑ | Reward Hack ↓ | Allowed Numbers ↑ | Number F1 ↑ | Allowed Ops ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| static | validation | 0.252±0.151 | 0.028±0.017 | 0.737±0.136 | 0.272±0.158 | 0.661±0.272 | 0.693±0.278 |
| Stable-RTW | validation | 0.393±0.033 | 0.047±0.006 | 0.607±0.033 | 0.412±0.030 | 0.840±0.015 | 0.900±0.015 |
| static | test_in_dist | 0.260±0.157 | 0.030±0.023 | 0.717±0.125 | 0.287±0.175 | 0.675±0.271 | 0.705±0.265 |
| Stable-RTW | test_in_dist | 0.400±0.028 | 0.020±0.007 | 0.600±0.028 | 0.417±0.026 | 0.859±0.007 | 0.917±0.006 |
| static | test_ood_long | 0.053±0.008 | 0.013±0.002 | 0.947±0.008 | 0.058±0.008 | 0.693±0.048 | 0.915±0.046 |
| Stable-RTW | test_ood_long | 0.045±0.007 | 0.015±0.000 | 0.955±0.007 | 0.047±0.008 | 0.719±0.008 | 0.963±0.009 |
| static | test_ood_division | 0.303±0.055 | 0.027±0.002 | 0.695±0.053 | 0.303±0.055 | 0.828±0.047 | 0.950±0.044 |
| Stable-RTW | test_ood_division | 0.265±0.022 | 0.028±0.002 | 0.735±0.022 | 0.265±0.022 | 0.847±0.000 | 0.975±0.004 |

### Paired-delta table

| Split | Δ Valid Expr | Δ Exact | Δ Reward Hack | Δ Number F1 | Interpretation |
|---|---:|---:|---:|---:|---|
| validation | +0.142 | +0.018 | -0.130 | +0.178 | robustness gain |
| test_in_dist | +0.140 | -0.010 | -0.117 | +0.184 | in-distribution legality gain, exact mixed |
| test_ood_long | -0.008 | +0.002 | +0.008 | +0.026 | still weak |
| test_ood_division | -0.038 | +0.002 | +0.040 | +0.019 | mixed; static favored on legality/reward-hacking |

For reward-hacking, negative deltas are improvements.

## Strongest claim

> Stable-RTW does not solve Countdown, but it solves an earlier harness-acquisition bottleneck: it makes legal expression generation substantially more stable across seeds than static shaping.

## Final clean framing

> We study how an LLM acquires a legal action space under a strict verifier. Dense legality rewards help, but the reward teacher itself must be stable. Stable-RTW improves in-distribution legality robustness and reduces seed sensitivity compared with static shaping, while exact solving and OOD-long generalization remain open.

## Claims to avoid

Do not claim:

- Stable-RTW globally beats static;
- exact correctness is solved;
- Countdown proves coding-agent harness transfer;
- v0.7 prompt/length diagnostics are trained-method improvements;
- OOD generalization is improved overall.
