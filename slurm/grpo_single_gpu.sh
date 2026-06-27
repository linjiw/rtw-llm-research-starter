#!/bin/bash
#SBATCH --job-name=rtw-grpo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source ~/.bashrc
conda activate rtw-llm

python scripts/02_grpo_train.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --train_path data/countdown/train.jsonl \
  --eval_path data/countdown/validation.jsonl \
  --output_dir outputs/grpo_rtw_${SLURM_JOB_ID} \
  --reward_strategy adaptive \
  --max_steps 1000 \
  --num_generations 4
