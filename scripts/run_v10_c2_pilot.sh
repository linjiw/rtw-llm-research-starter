#!/usr/bin/env bash
# v0.10 C2 arm (docs/V10_TASK_CURRICULUM_PLAN.md): adaptive task curriculum on
# top of Stable-RTW rewards. Runs a 60-step smoke first; aborts before the
# 300-step pilot if the smoke health check or curriculum log looks wrong.
set -euo pipefail
cd "$(dirname "$0")/.."
source .env
export WANDB_DISABLED=true

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
PY=.venv/bin/python

train_c2() {
  local steps="$1" out="$2"
  mkdir -p "$out"
  echo "=== $(date -Is) v0.10 C2 train steps=${steps} -> ${out}"
  $PY scripts/02_grpo_train.py \
    --model_name "$MODEL" \
    --train_path data/countdown/train.jsonl \
    --eval_path data/countdown/validation.jsonl \
    --output_dir "$out" \
    --reward_strategy adaptive_stable \
    --task_curriculum adaptive \
    --seed 0 \
    --max_steps "$steps" \
    --batch_size 2 \
    --grad_accum 8 \
    --num_generations 4 \
    --max_prompt_length 768 \
    --max_completion_length 256 \
    --report_to none 2>&1 | tee "$out/train.log"
  $PY scripts/05_check_run_health.py --run_dir "$out" | tee "$out/health_final.txt"
}

check_curriculum_log() {
  local out="$1" min_updates="$2"
  $PY - "$out" "$min_updates" <<'PY'
import json, sys
run, min_updates = sys.argv[1], int(sys.argv[2])
rows = [json.loads(l) for l in open(f"{run}/curriculum_state.jsonl")]
assert len(rows) >= min_updates, f"only {len(rows)} controller updates, expected >= {min_updates}"
last = rows[-1]
probs = last["tier_probs"]
assert abs(sum(probs.values()) - 1.0) < 1e-6, f"probs do not sum to 1: {probs}"
assert all(p >= 0.10 - 1e-9 for p in probs.values()), f"tier floor violated: {probs}"
draws = last["cumulative_draws"]
assert all(v > 0 for v in draws.values()), f"tier starved: {draws}"
moved = any(
    abs(r["tier_probs"][t] - 1/3) > 0.01
    for r in rows[30:] for t in r["tier_probs"]
)
print(f"curriculum log OK: {len(rows)} updates, draws={draws}, probs_moved_after_delay={moved}")
PY
}

SMOKE=outputs/checkpoints/grpo_v10_c2_adaptive_curr_smoke_60_seed0
PILOT=outputs/checkpoints/grpo_v10_c2_adaptive_curr_seed0_300

train_c2 60 "$SMOKE"
check_curriculum_log "$SMOKE" 55

train_c2 300 "$PILOT"
check_curriculum_log "$PILOT" 290

for split in validation test_in_dist; do
  out="outputs/bestofn/v10c2_local_seed0_${split}_limit50_n8"
  mkdir -p "$out"
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
    --method v10c2 \
    --training_seed 0 \
    --split "$split" \
    --data_path "data/countdown/${split}.jsonl" \
    --task_ids_file "outputs/v09_task_ids_${split}_limit50.txt" \
    --skip_if_complete \
    --output_dir "$out" 2>&1 | tee "$out/run.log"
done

echo "=== $(date -Is) v0.10 C2 complete"
