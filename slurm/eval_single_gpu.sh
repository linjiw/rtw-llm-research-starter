#!/bin/bash
#SBATCH --job-name=rtw-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source ~/.bashrc
conda activate rtw-llm

RUN_DIR=${1:-outputs/grpo_rtw}
MODEL=${2:-Qwen/Qwen2.5-0.5B-Instruct}

for SPLIT in test_in_dist test_ood_long test_ood_division; do
  for PROMPT_FIELD in prompt_high prompt_mid prompt_low; do
    python scripts/03_eval.py \
      --model_name ${MODEL} \
      --adapter_path ${RUN_DIR} \
      --data_path data/countdown/${SPLIT}.jsonl \
      --prompt_field ${PROMPT_FIELD} \
      --output_dir ${RUN_DIR}/eval_${SPLIT}_${PROMPT_FIELD}
  done
done
