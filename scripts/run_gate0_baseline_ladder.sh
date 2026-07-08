#!/usr/bin/env bash
# Gate 0 of docs/AUTORESEARCH_PROGRAM.md: rebuild the local baseline ladder
# (base -> static seed0 -> Stable-RTW seed0) under the frozen v0.9B protocol.
set -euo pipefail
cd "$(dirname "$0")/.."
source .env
export WANDB_DISABLED=true

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
PY=.venv/bin/python
LOGDIR=outputs/logs
mkdir -p "$LOGDIR" outputs/checkpoints outputs/bestofn

train_one() {
  local strategy="$1" out="$2"
  if [ -f "$out/adapter_model.safetensors" ]; then
    echo "=== skip train (exists): $out"
    return 0
  fi
  mkdir -p "$out"
  echo "=== $(date -Is) train strategy=${strategy} -> ${out}"
  $PY scripts/02_grpo_train.py \
    --model_name "$MODEL" \
    --train_path data/countdown/train.jsonl \
    --eval_path data/countdown/validation.jsonl \
    --output_dir "$out" \
    --reward_strategy "$strategy" \
    --seed 0 \
    --max_steps 300 \
    --batch_size 2 \
    --grad_accum 8 \
    --num_generations 4 \
    --max_prompt_length 768 \
    --max_completion_length 256 \
    --report_to none 2>&1 | tee "$out/train.log"
  $PY scripts/05_check_run_health.py --run_dir "$out" | tee "$out/health_final.txt"
}

bestofn_one() {
  local method="$1" seed="$2" split="$3" out="$4" adapter="$5"
  local task_ids="outputs/v09_task_ids_${split}_limit50.txt"
  local adapter_args=()
  if [ -n "$adapter" ]; then adapter_args=(--adapter_path "$adapter"); fi
  echo "=== $(date -Is) bestofn method=${method} split=${split} -> ${out}"
  mkdir -p "$out"
  $PY scripts/07_best_of_n_rerank.py \
    --model_name "$MODEL" \
    "${adapter_args[@]}" \
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
    --method "$method" \
    --training_seed "$seed" \
    --split "$split" \
    --data_path "data/countdown/${split}.jsonl" \
    --task_ids_file "$task_ids" \
    --skip_if_complete \
    --output_dir "$out" 2>&1 | tee "$out/run.log"
}

STATIC_CKPT=outputs/checkpoints/grpo_static_seed0_300
STABLE_CKPT=outputs/checkpoints/grpo_stable_seed0_300

train_one static "$STATIC_CKPT"
train_one adaptive_stable "$STABLE_CKPT"

# Method names must be exactly base/static/stable: the summarizer's paired
# overlap only pairs methods named "static" and "stable".
for split in validation test_in_dist; do
  bestofn_one base 0 "$split" "outputs/bestofn/base_local_seed0_${split}_limit50_n8" ""
  bestofn_one static 0 "$split" "outputs/bestofn/static_local_seed0_${split}_limit50_n8" "$STATIC_CKPT"
  bestofn_one stable 0 "$split" "outputs/bestofn/stable_local_seed0_${split}_limit50_n8" "$STABLE_CKPT"
done

$PY scripts/08_summarize_v09_seed_expansion.py \
  --runs_glob "outputs/bestofn/*_local_seed0_*_limit50_n8" \
  --out_csv outputs/gate0_local_ladder_summary.csv \
  --out_json outputs/gate0_local_ladder_paired.json

echo "=== $(date -Is) gate0 complete"
