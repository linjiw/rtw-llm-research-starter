#!/usr/bin/env bash
# Rank-2 harness-shift eval (docs/STRATEGIC_REPIVOT_20260709.md).
# Thesis pillar 3, never tested: does adaptive shaping (stable) yield policies
# more robust to how the task is PRESENTED (terse prompt_mid vs train-time
# prompt_high) than fixed (static)?
#
# ONLY prompt_mid vs prompt_high — prompt_low is CONFOUNDED (extract_answer
# falls back to full text when there is no <answer> tag, contaminating both
# format and assembly metrics). Both mid and high retain the <answer> cue.
#
# Eval-only: reuses the 6 EXISTING checkpoints (static/stable seeds 0/1/2), no
# retraining. prompt_high banks already exist from Gate 0 (dir name has no
# prompt-field suffix, so we re-point to suffixed dirs for both fields to keep
# the pairing unambiguous). Stage validation first, then test_in_dist.
set -euo pipefail
cd "$(dirname "$0")/.."
source .env
export WANDB_DISABLED=true

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
PY=.venv/bin/python
STAGE="${1:-validation}"   # pass "test_in_dist" for the second stage

bestofn() {
  local method="$1" seed="$2" split="$3" field="$4" adapter="$5"
  local out="outputs/bestofn/harness_${method}_seed${seed}_${split}_${field}_limit50_n8"
  mkdir -p "$out"
  echo "=== $(date -Is) harness method=${method} seed=${seed} split=${split} field=${field} -> ${out}"
  $PY scripts/07_best_of_n_rerank.py \
    --model_name "$MODEL" \
    --adapter_path "$adapter" \
    --engine hf --device cuda \
    --prompt_field "$field" \
    --limit 50 --batch_size 8 --max_n 8 --n_values 1 4 8 \
    --max_new_tokens 256 --temperature 0.7 --top_p 0.95 --seed 0 \
    --method "$method" --training_seed "$seed" --split "$split" \
    --data_path "data/countdown/${split}.jsonl" \
    --task_ids_file "outputs/v09_task_ids_${split}_limit50.txt" \
    --skip_if_complete \
    --output_dir "$out" 2>&1 | tee "$out/run.log"
}

for seed in 0 1 2; do
  for method in static stable; do
    ckpt="outputs/checkpoints/grpo_${method}_seed${seed}_300"
    for field in prompt_high prompt_mid; do
      bestofn "$method" "$seed" "$STAGE" "$field" "$ckpt"
    done
  done
done

echo "=== $(date -Is) harness-shift eval (${STAGE}) complete"
