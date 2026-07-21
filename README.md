# RTW-LLM: Reward Training Wheels for Harness-Aware LLM Post-Training

This repository is a starter research scaffold for adapting **Reward Training Wheels (RTW)** and **Grounded Adaptive Curriculum Learning (GACL)** ideas to LLM post-training.

The first experiment is deliberately small and verifiable:

> Can adaptive auxiliary reward weights improve GRPO post-training on a symbolic reasoning task, compared with fixed rewards, manual reward schedules, and random reward weights?

The default task is a Countdown-style arithmetic environment with a strict harness:

- A prompt gives `numbers`, `target`, and `allowed_ops`.
- The model must produce `<answer>...</answer>`.
- A verifier checks whether the expression uses each number exactly once, uses only allowed operators, and evaluates to the target.
- RTW dynamically adjusts auxiliary reward weights for format validity, expression validity, number usage, and brevity while preserving final correctness as the primary objective.

This is intended as a fast entry point. Once the pipeline is stable, the same harness/reward/controller pattern can be moved to code-generation tasks with unit tests.

## Current status

The current strongest result is the **v0.9B verifier-guided best-of-N seed expansion**:

- Stable-RTW shows a robust validation advantage under fixed best-of-N reranking: at `N=8`, reranked exact is `0.133 ± 0.012` for Stable-RTW vs `0.067 ± 0.050` for static shaping over seeds 0/1/2.
- Test-in-distribution confirms best-of-N as a useful general harness mechanism, but the Stable-specific edge is smaller/mixed: `0.133 ± 0.042` vs `0.120 ± 0.020` at `N=8`.
- The practical reranker does **not** use exact correctness as an input; exactness is measured afterward with the strict verifier.

See `docs/CURRENT_PROJECT_STATUS_AND_PAPER_ASSESSMENT.md` and `docs/V09_BEST_OF_N_RERANKING_PLAN.md` for the paper-safe interpretation, paired overlap tables, and cost notes.

Autonomous research iterations follow `docs/AUTORESEARCH_PROGRAM.md` (goal, fixed experiment budget, keep/discard rules, prioritized queue) with results recorded in `docs/EXPERIMENT_LEDGER.md`. The current execution plan is `docs/NEXT_STEPS.md`; `docs/README.md` indexes all documentation.

## Repo layout

```text
configs/                     Experiment configs
scripts/                     CLI entrypoints for data, SFT, GRPO, eval, analysis
src/rtw_llm/                 Core task, verifier, rewards, teacher, engines
data/countdown/              Small generated starter dataset
docs/                        Detailed research plan and experiment protocol
slurm/                       Example cluster job scripts
tests/                       Pytest smoke tests
```

## Quickstart

### 1. Create environment

```bash
conda create -n rtw-llm python=3.11 -y
conda activate rtw-llm
pip install -e .
```

For GPU training, install a CUDA-compatible PyTorch build first, then run `pip install -e .`.

### 2. Generate the versioned corrected dataset

```bash
python scripts/18_generate_countdown_v2.py
python scripts/19_audit_countdown_v2.py
```

`data/countdown/` is frozen legacy-v1 evidence and must not be refreshed.
Script 00 exists only to replay that historical generator into a separate path.
Countdown-v2 is write-once, globally disjoint, verifier-audited, and includes a
technically guarded final test. See
[`docs/COUNTDOWN_V2_DATASET_CARD.md`](docs/COUNTDOWN_V2_DATASET_CARD.md).

### 3. Run verifier tests

```bash
pytest -q
```

### 4. Evaluate a base model on a tiny subset

```bash
python scripts/03_eval.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --data_path data/countdown/test_in_dist.jsonl \
  --limit 32 \
  --output_dir outputs/eval_base
```

### 5. GRPO with adaptive RTW reward

```bash
python scripts/02_grpo_train.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --train_path data/countdown/train.jsonl \
  --eval_path data/countdown/validation.jsonl \
  --output_dir outputs/grpo_rtw_qwen05b \
  --reward_strategy adaptive \
  --max_steps 300 \
  --num_generations 4
```

### 6. Baseline runs

```bash
# Fixed expert reward
python scripts/02_grpo_train.py --reward_strategy static --output_dir outputs/grpo_static

# Manual schedule reward
python scripts/02_grpo_train.py --reward_strategy manual --output_dir outputs/grpo_manual

# Random reward weights
python scripts/02_grpo_train.py --reward_strategy random --output_dir outputs/grpo_random
```

## Main research comparison

Compare:

1. Base model, no post-training.
2. GRPO + fixed expert auxiliary weights.
3. GRPO + manual reward schedule.
4. GRPO + random reward weights.
5. GRPO + RTW adaptive reward weights.

Primary metrics:

- Exact success / pass@1.
- Format validity.
- Expression validity.
- Uses-all-numbers rate.
- OOD robustness on longer and division-enabled tasks.
- Reward-hacking candidates.
- Rollout tokens to reach success thresholds.
- Auxiliary reward weight evolution.

See `docs/EXPERIMENT_PLAN.md` for the full design.
