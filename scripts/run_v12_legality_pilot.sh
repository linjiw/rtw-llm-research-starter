#!/usr/bin/env bash
# v0.12 legality-weight envelope (docs/V12_NUMBER_LEGALITY_REWARD_PLAN.md):
# adaptive_stable_v12 = Stable-RTW with valid_expression floor 0.30 / cap 0.45.
# 60-step smoke (health + envelope + clipping gates) -> 300-step pilot ->
# frozen best-of-N. Comparator: C0 = grpo_stable_seed0_300 banks.
set -euo pipefail
cd "$(dirname "$0")/.."
source .env
export WANDB_DISABLED=true

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
PY=.venv/bin/python

train_v12() {
  local steps="$1" out="$2"
  mkdir -p "$out"
  echo "=== $(date -Is) v0.12 train steps=${steps} -> ${out}"
  $PY scripts/02_grpo_train.py \
    --model_name "$MODEL" \
    --train_path data/countdown/train.jsonl \
    --eval_path data/countdown/validation.jsonl \
    --output_dir "$out" \
    --reward_strategy adaptive_stable_v12 \
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

check_v12_run() {
  local out="$1" min_updates="$2"
  $PY - "$out" "$min_updates" <<'PY'
import json, sys
run, min_updates = sys.argv[1], int(sys.argv[2])
rows = [json.loads(l) for l in open(f"{run}/teacher_weights.jsonl")]
assert len(rows) >= min_updates, f"only {len(rows)} teacher updates, expected >= {min_updates}"
post = [r for r in rows if not r["diagnostics"]["delay_active"]]
assert post, "no post-delay updates"
ve = [r["weights"]["valid_expression"] for r in post]
assert min(ve) >= 0.30 - 1e-9, f"valid_expression floor violated: min={min(ve):.4f}"
assert max(ve) <= 0.45 + 1e-9, f"valid_expression cap violated: max={max(ve):.4f}"
assert abs(sum(rows[-1]["weights"].values()) - 1.20) < 1e-6, "weight budget broken"

# Clipping guardrail (advisor: this design cuts format/brevity mass).
# C0 (grpo_stable_seed0_300) train-time reference: brevity component ~0.99
# late in training; a large drop means longer/capped generations.
comp_rows = [json.loads(l) for l in open(f"{run}/reward_components.jsonl")]
last = [r for r in comp_rows if r["reward_batch_index"] >= comp_rows[-1]["reward_batch_index"] - 10]
brevity = sum(r["components"].get("brevity", 0.0) for r in last) / max(len(last), 1)
print(f"v12 run OK: {len(rows)} updates, valid_expr range [{min(ve):.3f},{max(ve):.3f}], late brevity={brevity:.3f}")
if brevity < 0.85:
    print(f"WARNING: late brevity {brevity:.3f} < 0.85 — clipping risk, inspect before eval")
PY
}

SMOKE=outputs/checkpoints/grpo_v12_legality_smoke_60_seed0
PILOT=outputs/checkpoints/grpo_v12_legality_seed0_300

train_v12 60 "$SMOKE"
check_v12_run "$SMOKE" 55

train_v12 300 "$PILOT"
check_v12_run "$PILOT" 290

# Dir name starts with v12 (not 'stable') and method is explicit: script 08's
# name-inference maps any dir containing 'stable' to method stable, which
# would silently pool these runs with C0 in a glob summary.
for split in validation test_in_dist; do
  out="outputs/bestofn/v12legality_seed0_${split}_limit50_n8"
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
    --method v12legality \
    --training_seed 0 \
    --split "$split" \
    --data_path "data/countdown/${split}.jsonl" \
    --task_ids_file "outputs/v09_task_ids_${split}_limit50.txt" \
    --skip_if_complete \
    --output_dir "$out" 2>&1 | tee "$out/run.log"
done

echo "=== $(date -Is) v0.12 pilot complete"
