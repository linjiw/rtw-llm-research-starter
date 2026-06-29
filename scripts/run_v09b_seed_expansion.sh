#!/usr/bin/env bash
set -euo pipefail

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
COMMON=(
  --model_name "$MODEL"
  --engine hf
  --device cuda
  --prompt_field prompt
  --limit 50
  --batch_size 8
  --max_n 8
  --n_values 1 4 8
  --max_new_tokens 256
  --temperature 0.7
  --top_p 0.95
  --seed 0
  --skip_if_complete
)

run_one() {
  local method="$1"
  local adapter="$2"
  local seed="$3"
  local split="$4"
  local out="$5"
  local task_ids="outputs/v09_task_ids_${split}_limit50.txt"
  local log="${out}.log"
  mkdir -p "$out"
  echo "=== $(date -Is) v0.9B method=${method} training_seed=${seed} split=${split} ===" | tee "$log"
  .venv/bin/python scripts/07_best_of_n_rerank.py \
    "${COMMON[@]}" \
    --method "$method" \
    --training_seed "$seed" \
    --split "$split" \
    --adapter_path "$adapter" \
    --data_path "data/countdown/${split}.jsonl" \
    --task_ids_file "$task_ids" \
    --output_dir "$out" 2>&1 | tee -a "$log"
  echo "=== $(date -Is) done method=${method} training_seed=${seed} split=${split} ===" | tee -a "$log"
}

for seed in 1 2; do
  run_one static "outputs/grpo_static_v06b_dense_numbers_cuda_pilot_300_seed${seed}" "$seed" validation "outputs/v09b_bestofn_static_v06b_seed${seed}_validation_limit50_n8"
  run_one static "outputs/grpo_static_v06b_dense_numbers_cuda_pilot_300_seed${seed}" "$seed" test_in_dist "outputs/v09b_bestofn_static_v06b_seed${seed}_test_in_dist_limit50_n8"
  run_one stable "outputs/grpo_rtw_v06c_adaptive_stable_cuda_pilot_300_seed${seed}" "$seed" validation "outputs/v09b_bestofn_stable_v06c_seed${seed}_validation_limit50_n8"
  run_one stable "outputs/grpo_rtw_v06c_adaptive_stable_cuda_pilot_300_seed${seed}" "$seed" test_in_dist "outputs/v09b_bestofn_stable_v06c_seed${seed}_test_in_dist_limit50_n8"
done

.venv/bin/python scripts/08_summarize_v09_seed_expansion.py \
  --runs_glob "outputs/v09b_bestofn_*_limit50_n8" \
  --out_csv outputs/v09_seed_expansion_summary.csv \
  --out_json outputs/v09_seed_expansion_paired.json

uv run pytest -q
uv run ruff check .
git status --short
