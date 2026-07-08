# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

RTW-LLM is a research project studying **adaptive auxiliary reward weighting (Reward Training Wheels, RTW) and adaptive curricula (GACL)** for harness-aware LLM post-training. The current testbed is a Countdown-style arithmetic task with a strict verifier; the long-term target is agentic coding tasks. The deliverable is a paper, not a product — experimental hygiene (frozen task IDs, paired comparisons, separate reward-component logging) matters more than code elegance.

Read these before planning any experiment:
- `docs/CURRENT_PROJECT_STATUS_AND_PAPER_ASSESSMENT.md` — current empirical arc (v0.6b→v0.9B) and the paper-safe claim wording
- `docs/BASELINE_INVENTORY_AND_NEXT_EXPERIMENTS_20260708.md` — what baselines exist, what's missing locally, next experiment gate
- `docs/PROJECT_DESIGN.md` — research questions, roadmap, failure modes
- `AGENTS.md` — repo invariants (summarized below)

## Commands

Always `source .env` first — it sets `HF_HOME`, `UV_CACHE_DIR`, output dirs, and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` (a host ROS pytest plugin breaks otherwise). The environment is managed with `uv`; run everything through `uv run` or `.venv/bin/python`.

```bash
source .env
uv run pytest -q                      # all tests (fast, <1s)
uv run pytest -q tests/test_teacher.py            # single test file
uv run pytest -q tests/test_teacher.py -k name    # single test
uv run ruff check .                   # lint (line-length 100)
```

Data generation (a starter dataset is already committed under `data/countdown/`):

```bash
uv run python scripts/00_generate_countdown_dataset.py --out_dir data/countdown \
  --train 5000 --valid 500 --test 500 --ood 500 --seed 42
```

Training (GRPO + LoRA, needs the CUDA host — single A10G, 23 GB):

```bash
WANDB_DISABLED=true uv run python scripts/02_grpo_train.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --output_dir outputs/checkpoints/<run_name> \
  --reward_strategy adaptive_stable \      # or: static | manual | random | adaptive | adaptive_phased
  --seed 0 --max_steps 300 --batch_size 2 --grad_accum 8 \
  --num_generations 4 --report_to none
```

Post-run health check and analysis (run these after every training run):

```bash
uv run python scripts/05_check_run_health.py --run_dir outputs/checkpoints/<run_name>
uv run python scripts/04_analyze_results.py --run_dir outputs/checkpoints/<run_name>
```

Evaluation and best-of-N reranking (the main inference-time harness):

```bash
uv run python scripts/03_eval.py --model_name <base> --adapter_path <ckpt> \
  --data_path data/countdown/<split>.jsonl --output_dir outputs/evals/<name> --limit 50

uv run python scripts/07_best_of_n_rerank.py --model_name <base> --adapter_path <ckpt> \
  --engine hf --device cuda --limit 50 --max_n 8 --n_values 1 4 8 \
  --temperature 0.7 --top_p 0.95 --seed 0 \
  --task_ids_file outputs/v09_task_ids_validation_limit50.txt \
  --data_path data/countdown/validation.jsonl --output_dir outputs/bestofn/<name>
```

See `docs/BASELINE_INVENTORY_AND_NEXT_EXPERIMENTS_20260708.md` for full canonical command templates and `scripts/run_v09b_seed_expansion.sh` for the reference multi-run protocol.

## Architecture

The core loop: TRL's `GRPOTrainer` calls `RTWRewardManager` (`src/rtw_llm/rewards.py`) as its reward function → the manager scores each completion with the verifier (`src/rtw_llm/countdown.py`) → feeds per-component scores to `RTWTeacher` (`src/rtw_llm/teacher.py`) → the teacher adapts the auxiliary weight vector used for the next batch. Reward = `primary (exact_correct) + Σ w_k · aux_k` where aux components are format, valid_expression, number_multiset_f1, allowed_ops, numeric_distance_reward, brevity.

- `src/rtw_llm/countdown.py` — task generator, expression parser (AST-based, `Fraction` arithmetic), and **the verifier that is the single source of truth for correctness**. Everything downstream (rewards, eval metrics, best-of-N exactness, reranking features) derives from `verify_completion`/`score_completion` here.
- `src/rtw_llm/teacher.py` — the six reward strategies. `adaptive_stable` (Stable-RTW, the current best method) adds a delay period, low LR, EMA smoothing, per-component floors/caps, and a weight-sum budget projected via iterative redistribution. `adaptive_phased` adds hysteresis-based phase switching on top. Diagnostics (floor/cap hits, weight movement, phase state) are logged per step to `teacher_weights.jsonl`.
- `src/rtw_llm/trl_compat.py` — shims for TRL kwarg renames across versions; use `set_first_supported_kwarg` when passing newer `GRPOConfig` options.
- `src/rtw_llm/engine.py` — thin HF/vLLM generation wrappers used by eval and best-of-N scripts.
- `scripts/07_best_of_n_rerank.py` — the v0.9 inference-time harness. `practical_score` selects candidates from legality/distance features only; **it must never use exact correctness as an input** (exactness is measured afterward by the verifier). Oracle selection exists separately as an upper bound.
- Scripts are numbered by pipeline stage: 00 data → 01 SFT warmup → 02 GRPO → 03 eval → 04 analysis → 05 health → 06 failure taxonomy → 07 best-of-N → 08 seed-expansion summary.

Training runs write three artifact streams into the output dir: `reward_components.jsonl` (per-completion component scores), `teacher_weights.jsonl` (per-step weights + diagnostics), and the LoRA adapter. Analysis and health scripts consume these.

## Repo invariants (from AGENTS.md — do not break)

1. The verifier in `src/rtw_llm/countdown.py` is the source of truth. Never count an output as correct unless it passes the verifier.
2. Log reward components separately — never only total reward. Keep `primary_reward`, `aux_reward_weighted`, and `total_reward` distinguishable in every run and report.
3. The primary reward must remain final task success; auxiliary rewards are training wheels, not the objective.
4. New tasks need a deterministic generator, a verifier, a dataset card, and tests.
5. Keep `prompt_low` / `prompt_mid` / `prompt_high` fields in datasets so harness-shift experiments remain possible.

## Experimental hygiene

- Comparisons against v0.9B results must reuse the frozen task IDs (`outputs/v09_task_ids_*_limit50.txt`), the same selector, sampling config (temp 0.7, top-p 0.95, max_new_tokens 256, seed 0), and cost accounting (tokens + wall-clock).
- Commit before launching CUDA runs so each run has an archival code state.
- Document each experiment round in `docs/` (see the `V0*` files for the pattern: plan → result → safe conclusion → overclaims to avoid).
- Method claims require paired per-task comparisons (McNemar-style overlap counts), not just mean ± std across seeds.
- HF generation is slow (~6 s/example at 256 new tokens on the A10G); keep diagnostics at `--limit 50` or improve throughput before large sweeps.

## Infrastructure notes

- Single NVIDIA A10G (23 GB), 1 TiB root EBS. `outputs/` is git-ignored; durable artifacts go under `outputs/checkpoints/`, `outputs/evals/`, `outputs/bestofn/`, `outputs/logs/` (see `outputs/STORAGE_LAYOUT.md`). `/workspace` is ephemeral instance store — scratch only.
- Long runs: use `nohup`/background with `tee` into a log file inside the run dir, then check health after.
