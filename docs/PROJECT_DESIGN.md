# RTW-LLM: Adaptive Reward Training Wheels for Harness-Aware LLM Post-Training

## 1. Core thesis

Modern LLM agents are not only defined by model weights. They are defined by the
system around the model: the context builder, tool interface, execution loop,
validators, guardrails, memory/state, and trace logging. In other words,
model plus harness is the actual agent.

This project studies a specific question inside that broader
harness-engineering frame:

> Can adaptive auxiliary rewards and adaptive curricula help LLM agents acquire
> harness competence more sample-efficiently, with less reward hacking, and
> better transfer across task and harness shifts?

The key idea is to transfer Reward Training Wheels (RTW) and Grounded Adaptive
Curriculum Learning (GACL) from robotics into LLM post-training. RTW gives us
adaptive auxiliary reward weighting: preserve the primary objective, but
dynamically emphasize or de-emphasize auxiliary reward components as the student
improves. GACL gives us adaptive task generation with performance monitoring and
domain grounding, so the curriculum challenges the learner without drifting away
from the target distribution.

The short version of the contribution is:

> We treat the LLM post-training environment as a harnessed RL system and study
> how a teacher should adapt reward, task difficulty, feedback strength, and
> harness complexity over training.

## 2. Why we start with Countdown instead of SWE-bench

Coding-agent harness engineering eventually needs Docker sandboxes, repository
state, patch application, unit tests, trace logging, verifier reranking, and
strong baselines. That is the right long-term direction.

But for the first experiment, jumping directly into SWE-bench would mix too many
variables:

```text
model ability
repo retrieval
tool interface
test quality
patch application
dependency setup
execution timeout
reward sparsity
harness design
teacher adaptation
```

Countdown gives us a clean wind tunnel. It is small enough that we can isolate
the RTW mechanism, but still harness-relevant because the model must obey a
strict output contract, produce a parseable artifact, satisfy constraints, and
be verified by an external checker.

So the current project has two layers:

```text
Layer 1: Countdown = controlled verifier-based harness for method validation.
Layer 2: Coding agents = realistic software-engineering harness for scaling.
```

This is the right scientific progression. Countdown is not the final
destination; it is the controlled unit test for the research idea.

## 3. Formal problem framing

We define a harness as:

```text
H = (C, T, S, L, V, G, O)
```

where:

```text
C = context builder
T = tools / APIs
S = state / memory
L = loop / controller
V = validators / verifiers
G = guardrails / permissions
O = observability / logging
```

For a harnessed LLM policy:

```text
a_t ~ pi_theta(a_t | h_t, H)
```

where `h_t` is the interaction history: prompts, model actions, tool outputs,
verifier feedback, and previous failures.

The post-training reward is decomposed as:

```text
R(x, y; w_t) = R_primary(x, y) + sum_k w_{t,k} R_aux,k(x, y)
```

This is the direct LLM analogue of RTW's decomposed student reward. In the
robotics version, the student reward consists of a primary reward plus weighted
auxiliary reward components, and the teacher adapts those weights from histories
of previous weights, primary rewards, and auxiliary reward values.

For the current Countdown harness:

```text
primary reward:
  exact_correct

auxiliary rewards:
  contains_open_answer_tag
  contains_close_answer_tag
  has_extractable_answer_span
  expression_parseable
  uses_allowed_numbers
  uses_allowed_ops
```

The teacher state is:

```text
s_teacher_t = {
  w_{t-H:t-1},
  p_{t-H:t-1},
  r_aux_{t-H:t-1},
  d_{t-H:t-1},
  h_{t-H:t-1}
}
```

where `w` is reward weight history, `p` is primary performance, `r_aux` is
auxiliary reward history, `d` is task difficulty, and `h` is harness level or
feedback condition.

The teacher action eventually becomes:

```text
a_teacher_t = (w_t, P_t(d), level_t, epsilon_t)
```

where `w_t` are reward weights, `P_t(d)` is the task difficulty distribution,
`level_t` is harness/feedback level, and `epsilon_t` is grounding probability.
The grounding term comes from GACL: instead of letting generated tasks drift
freely, the teacher alternates between reference tasks and synthetic/adaptive
tasks to preserve relevance to the target distribution.

## 4. Main research questions

### RQ1: Reward adaptation

Does adaptive auxiliary reward weighting improve harness acquisition and sample
efficiency compared with static, manual, and random reward schedules?

This is the current v0.5/v0.6 question.

The core hypothesis is:

> Early in training, the teacher should emphasize format and parseability. As
> those skills saturate, it should reduce wrapper rewards and shift weight
> toward constraint satisfaction and exact correctness.

The result we want is not merely higher final accuracy. We want to see an
interpretable progression:

```text
tag compliance
-> extractable answer
-> parseable expression
-> valid number/operator use
-> exact correctness
```

That progression would mirror the RTW training-wheels pattern, where auxiliary
components help early learning but become less central once the student has
internalized the subskill.

### RQ2: Harness-aware training

Does training inside a harness improve robustness compared with applying a
richer harness only at inference time?

This comes directly from the distinction between the agent harness and the
evaluation harness. The harness is not just a prompt wrapper; it is the system
that exposes tools, manages state, verifies outputs, controls stopping, and logs
behavior.

For Countdown, this means training under `prompt_high`, `prompt_mid`, and
`prompt_low`, then evaluating cross-harness transfer.

For coding, this later becomes:

```text
train with visible tests -> evaluate with hidden tests
train with structured edit tool -> evaluate with patch-only interface
train with rich tool docs -> evaluate with compressed tool docs
```

### RQ3: Task curriculum

Does GACL-style adaptive task selection improve OOD robustness compared with
uniform sampling and manual schedules?

This is the natural second phase. GACL's three important ideas are task
representation, active performance monitoring, and grounding against reference
tasks.

For Countdown, difficulty can be controlled by:

```text
number of operands
allowed operators
target range
division required or not
solution depth
number of distractor solutions
```

For coding, difficulty can later be controlled by:

```text
number of files touched
number of failing tests
amount of context needed
API/library complexity
patch size
dependency complexity
historical model success rate
```

### RQ4: Coding-agent harness scaling

When we move to coding tasks, which harness components actually matter?

We must not compare a complex agent only against weak direct prompting. We need
strong baselines such as direct generation, RAG-direct, Agentless-style
localization/repair/validation, ReAct/SWE-agent loops, CodeAct-style action
spaces, and verifier reranking.

This protects the work from the usual agent-paper weakness: confusing
complexity with capability.

## 5. Current experiment: v0.5 Countdown CUDA bootstrap

The current v0.5 experiment should remain narrow.

### Objective

Determine whether adaptive RTW-GRPO has a usable learning signal before SFT
warmup.

### Model

```text
Qwen/Qwen2.5-0.5B-Instruct
```

### Task

Countdown arithmetic with strict answer extraction:

```xml
<answer>expression</answer>
```

### Primary metric

```text
exact_correct
```

### Auxiliary metrics

```text
contains_open_answer_tag
contains_close_answer_tag
has_extractable_answer_span
expression_parseable
uses_allowed_numbers
uses_allowed_ops
```

### Reward logging guardrail

Every run must separately log:

```text
primary_reward
primary_reward_weighted
aux_reward_weighted
total_reward
reward_batch_reward_std
reward_batch_has_variance
```

This is important because the main reviewer critique will be:

> Is the method improving task correctness, or merely optimizing easy formatting
> rewards?

Our answer must be visible in the logs. Auxiliary reward is allowed to rise
first, but primary correctness must remain the real endpoint.

### v0.5 acceptance criteria

A clean pass does not require correctness yet. It requires evidence that the
learning landscape is not flat:

```text
training completes
reward logs are populated
teacher logs are populated
reward_variance_nonzero_fraction > 0
tag/extractable rates are nonzero
teacher weights move
primary and auxiliary rewards remain separated
```

A strong pass means:

```text
expression_parseable_rate > 0
allowed_numbers_rate > 0
allowed_ops_rate > 0
tag_only_rate does not dominate everything
correct_given_parseable becomes measurable
```

If the run shows only tag learning but no parseability, insert a 100-step SFT
harness warmup and then continue GRPO.

## 6. v0.6 experiment: adaptive vs static reward schedules

Once the CUDA bootstrap is healthy, the first real comparison should be:

```text
Base model eval
SFT warmup eval, if needed
GRPO + static reward
GRPO + manual schedule
GRPO + random reward
GRPO + adaptive RTW reward
```

The main comparison should use the same:

```text
model
dataset
number of steps
number of generations
seed
prompt template
max completion length
```

The first run should use seed 0 only. Once the curves are sane, expand to seeds
0, 1, and 2.

### Main metrics

```text
exact_correct_rate
format_rate
extractable_span_rate
expression_parseable_rate
allowed_numbers_rate
allowed_ops_rate
tag_only_rate
parseable_but_wrong_rate
correct_given_parseable
reward_variance_nonzero_fraction
time/steps/tokens to first nonzero correctness
best validation exact_correct
OOD-long exact_correct
OOD-division exact_correct
```

### Main expected result

The most convincing first result would be:

```text
adaptive RTW reaches parseability and exact correctness faster than
static/manual/random, while avoiding persistent tag-only reward hacking.
```

A secondary but very important result would be interpretable teacher-weight
evolution:

```text
format weights high early
parseability/constraint weights rise next
format weights decay after saturation
correctness dominates later
```

This is where the method becomes more than another reward schedule. It becomes
an adaptive diagnostic of what the model currently lacks.

## 7. v0.7 experiment: harness-shift robustness

After adaptive vs static is established, test whether the model learned the task
or merely learned one wrapper.

### Harness levels

```text
low:
  terse instruction, minimal schema

mid:
  clear schema, no examples

high:
  schema + examples + common mistakes + verifier rules
```

### Train/eval matrix

```text
train high -> eval high
train high -> eval mid
train high -> eval low
train mid -> eval high
train mid -> eval mid
train mid -> eval low
```

The key metric is performance drop under harness shift.

A good result would show that adaptive RTW improves not only in-distribution
exact correctness, but also transfer to leaner or modified harnesses.

This connects directly to the broader point: harness design is itself an
experimental variable, not just infrastructure.

## 8. v0.8 experiment: GACL-style task curriculum

Once reward adaptation works, add task adaptation.

### Teacher action

```text
a_teacher_t = (w_t, P_t(d), epsilon_t)
```

where `P_t(d)` controls task difficulty and `epsilon_t` controls grounding
probability.

### Reference vs synthetic tasks

Reference tasks are sampled from a fixed target distribution.

Synthetic/adaptive tasks are generated by controlling difficulty:

```text
operand count
operator set
solution depth
division requirement
target range
number repetition
ambiguity/multiple-solution rate
```

### Baselines

```text
uniform task sampling
manual easy-to-hard curriculum
random difficulty curriculum
GACL-style adaptive task curriculum
RTW-only reward curriculum
joint RTW + GACL curriculum
```

### Main hypothesis

Reward curriculum and task curriculum are complementary:

```text
RTW answers: what should the learner be rewarded for right now?
GACL answers: what kind of task should the learner face right now?
```

The joint method should be especially helpful when the primary reward is sparse
and difficulty varies widely.

## 9. v1.0 expansion: coding-agent harnesses

Only after the controlled Countdown results are stable should we move into
coding.

The dataset ladder should start with function-level tasks, then API/cross-file
tasks, then repo-level issue repair, then executable SWE training environments.

### Coding task ladder

```text
Stage 1: HumanEval / MBPP / LiveCodeBench
Stage 2: BigCodeBench / API-heavy coding
Stage 3: RepoBench / CrossCodeEval
Stage 4: SWE-bench Lite / Verified
Stage 5: SWE-Gym / R2E-Gym for training
```

### Coding reward decomposition

For code, the primary reward is:

```text
resolved task / hidden tests pass / exact functional correctness
```

Auxiliary rewards can include:

```text
patch applies cleanly
build succeeds
public tests improve
no regression tests fail
lint/typecheck passes
diff is minimal
edit is localized
tool call is valid
test failure is correctly interpreted
stop decision is correct
```

This is where the RTW idea becomes especially interesting. In coding, early
training may need rewards for patch application, build success, and test
execution. Later, those should become less important than hidden-test
correctness and no-regression behavior.

### Coding baselines

```text
direct generation
RAG + direct patch
Agentless-style localization -> repair -> validation
ReAct shell loop
SWE-agent / ACI-style agent
CodeAct-style code-as-action
best-of-n verifier reranking
RTW adaptive harness reward
RTW + GACL task curriculum
```

This makes the future paper much stronger, because we will not claim that a
complex agent is useful merely because it beats a weak prompt baseline.

## 10. Failure modes and guardrails

The design should explicitly track the ways it can fool itself.

### Failure mode 1: tag farming

```text
open_tag_rate rises
extractable_span_rate rises
expression_parseable_rate stays low
exact_correct stays zero
```

Interpretation: the model learned the wrapper but not the task.

Guardrail: decay format reward after saturation; track `tag_only_rate`.

### Failure mode 2: parseable nonsense

```text
expression_parseable_rate rises
allowed_numbers_rate stays low
exact_correct stays zero
```

Interpretation: the model learned arithmetic-looking syntax but not task
constraints.

Guardrail: increase allowed-number and allowed-op weights only after
parseability appears.

### Failure mode 3: auxiliary dominance

```text
aux_reward_weighted_mean rises
primary_reward_mean stays zero
teacher keeps emphasizing easy aux rewards
```

Interpretation: RTW is optimizing the scaffold rather than the objective.

Guardrail: always plot primary and auxiliary separately; report exact
correctness as the primary metric.

### Failure mode 4: harness overfitting

```text
train-harness performance high
cross-harness performance low
```

Interpretation: the model learned a surface protocol, not the underlying task.

Guardrail: run harness-shift evaluations.

### Failure mode 5: coding public-test overfitting

In the coding phase, a model may learn to satisfy visible tests without solving
the actual issue.

Guardrail: hidden tests, generated tests, no-regression checks, patch review,
and verifier-reranking ablations.

## 11. Paper-shaped contribution plan

A clean first paper should not try to solve all of coding-agent harness
engineering. It should make one precise claim:

> Adaptive auxiliary reward weighting improves verifier-based harness
> acquisition in LLM post-training compared with fixed, manual, and random
> reward schedules.

The paper structure could be:

```text
1. Introduction
   LLM post-training increasingly happens inside harnesses.
   Fixed reward shaping is brittle.
   We propose Reward Training Wheels for harness-aware LLM post-training.

2. Background
   Harnessed LLM agents
   Verifier-based RL / GRPO
   RTW and curriculum learning

3. Method
   Harness formalization
   Primary/auxiliary reward decomposition
   RTW teacher state/action
   Dense reward diagnostics
   Optional task curriculum extension

4. Experiments
   Countdown verifier harness
   Adaptive/static/manual/random comparison
   Harness-shift evaluation
   OOD difficulty evaluation
   Ablations

5. Results
   Sample efficiency
   Primary vs auxiliary reward
   Teacher weight evolution
   Failure taxonomy

6. Discussion
   What transfers to coding agents
   Why coding requires stronger eval harnesses
   Limitations and next step toward SWE tasks
```

The second paper or extended project can then be:

> Adaptive reward and task curricula for execution-aware coding-agent harnesses.

That second project would use the full harness-engineering stack: Docker
sandboxing, repository APIs, execution APIs, static analysis, verifier APIs,
trace stores, and strong baseline gradients.

## 12. Revised roadmap

### v0.5: CUDA bootstrap evidence

```text
Goal:
  prove adaptive RTW-GRPO has usable reward variance and teacher movement

Artifacts:
  outputs/grpo_rtw_cuda_smoke_50/health.txt
  docs/v05_cuda_bootstrap_report.md

Decision:
  direct 300-step GRPO or 100-step SFT warmup first
```

### v0.6: First adaptive-vs-static result

```text
Goal:
  compare adaptive, static, manual, random reward schedules

Artifacts:
  learning curves
  reward decomposition plots
  teacher weight evolution
  sample generations
  failure taxonomy
```

### v0.7: Harness shift

```text
Goal:
  test whether the model learned the task or the wrapper

Artifacts:
  train/eval harness matrix
  robustness drop table
```

### v0.8: GACL task curriculum

```text
Goal:
  add adaptive difficulty and grounding probability

Artifacts:
  difficulty progression curve
  RTW-only vs GACL-only vs joint comparison
```

### v1.0: Coding harness pilot

```text
Goal:
  transfer method from Countdown to executable code tasks

Start with:
  HumanEval/MBPP/LiveCodeBench or BigCodeBench

Do not start with:
  full SWE-bench Verified training
```

### v1.5: Repo-level coding agents

```text
Goal:
  evaluate RTW/GACL in real software-engineering harnesses

Datasets:
  SWE-bench Lite/Verified
  SWE-Gym
  R2E-Gym

Baselines:
  direct
  RAG-direct
  Agentless-style
  ReAct/SWE-agent-style
  verifier reranking
  RTW/GACL adaptive harness
```

## 13. Positioning sentence

> This project studies harness-aware LLM post-training: instead of treating
> prompts, tools, validators, and reward functions as fixed engineering details,
> we treat them as adaptive training signals. Building on Reward Training Wheels
> and Grounded Adaptive Curriculum Learning, we develop teacher-guided methods
> that dynamically adjust auxiliary reward weights, task difficulty, feedback
> strength, and grounding to help LLM agents acquire verifiable skills
> efficiently and robustly.
