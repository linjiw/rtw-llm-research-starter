#!/usr/bin/env bash
# v0.13 seed expansion (seeds 1/2): same protocol as run_v13_sft_pilot.sh
# minus the smoke stage (path validated at seed 0) and minus the SFT-only
# arms (decomposition established at seed 0). Program standard (G0-seeds12
# lesson): 3-seed distribution before any paper number.
# Seed semantics match the stable baseline seeds: --seed N varies the SFT
# seed and the teacher seed; GRPOConfig seed stays at its default like C0
# (per plan A7 — do not change that plumbing mid-experiment).
set -euo pipefail
cd "$(dirname "$0")/.."
source .env
export WANDB_DISABLED=true

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
PY=.venv/bin/python

for seed in 1 2; do
  SFT=outputs/checkpoints/sft_warmup_legal_seed${seed}
  PILOT=outputs/checkpoints/grpo_v13_sft_seed${seed}_300

  if [ ! -f "$SFT/adapter_model.safetensors" ]; then
    echo "=== $(date -Is) v0.13 SFT warmup seed=${seed} -> ${SFT}"
    mkdir -p "$SFT"
    $PY scripts/01_sft_warmup.py \
      --model_name "$MODEL" \
      --train_path data/countdown/train.jsonl \
      --output_dir "$SFT" \
      --max_steps 125 \
      --batch_size 2 \
      --grad_accum 8 \
      --learning_rate 5e-5 \
      --seed "$seed" \
      --completion_only_loss \
      --report_to none 2>&1 | tee "$SFT/train.log"
  else
    echo "=== skip SFT (exists): $SFT"
  fi

  if [ ! -f "$PILOT/adapter_model.safetensors" ]; then
    echo "=== $(date -Is) v0.13 GRPO(from SFT) seed=${seed} -> ${PILOT}"
    mkdir -p "$PILOT"
    $PY scripts/02_grpo_train.py \
      --model_name "$MODEL" \
      --train_path data/countdown/train.jsonl \
      --eval_path data/countdown/validation.jsonl \
      --output_dir "$PILOT" \
      --reward_strategy adaptive_stable \
      --init_adapter_path "$SFT" \
      --seed "$seed" \
      --max_steps 300 \
      --batch_size 2 \
      --grad_accum 8 \
      --num_generations 4 \
      --max_prompt_length 768 \
      --max_completion_length 256 \
      --report_to none 2>&1 | tee "$PILOT/train.log"
    $PY scripts/05_check_run_health.py --run_dir "$PILOT" | tee "$PILOT/health_final.txt"
  else
    echo "=== skip GRPO (exists): $PILOT"
  fi

  for split in validation test_in_dist; do
    out="outputs/bestofn/v13sft_seed${seed}_${split}_limit50_n8"
    mkdir -p "$out"
    echo "=== $(date -Is) best-of-N v13sft seed=${seed} split=${split} -> ${out}"
    $PY scripts/07_best_of_n_rerank.py \
      --model_name "$MODEL" \
      --adapter_path "$PILOT" \
      --engine hf \
      --device cuda \
      --prompt_field prompt \
      --limit 50 \
      --batch_size 8 \
      --max_n 8 \
      --n_values 1 4 8 \
      --max_new_tokens 256 \
      --temperature 0.7 \
      --top_p 0.95 \
      --seed 0 \
      --method v13sft \
      --training_seed "$seed" \
      --split "$split" \
      --data_path "data/countdown/${split}.jsonl" \
      --task_ids_file "outputs/v09_task_ids_${split}_limit50.txt" \
      --skip_if_complete \
      --output_dir "$out" 2>&1 | tee "$out/run.log"
  done
done

# 3-seed scoring: each v13 seed vs the stable 3-seed baseline, both splits.
for split in validation test_in_dist; do
  $PY scripts/12_score_v13.py \
    --arm v13sft_s0=outputs/bestofn/v13sft_seed0_${split}_limit50_n8 \
    --arm v13sft_s1=outputs/bestofn/v13sft_seed1_${split}_limit50_n8 \
    --arm v13sft_s2=outputs/bestofn/v13sft_seed2_${split}_limit50_n8 \
    --combine_arms_as v13sft_observed_panel \
    --baseline_dirs \
      outputs/bestofn/stable_local_seed0_${split}_limit50_n8 \
      outputs/bestofn/stable_local_seed1_${split}_limit50_n8 \
      outputs/bestofn/stable_local_seed2_${split}_limit50_n8 \
    --out_json "outputs/v13_score_seeds012_${split}.json"
done

echo "=== $(date -Is) v0.13 seeds 1/2 complete"
