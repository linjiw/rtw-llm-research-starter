#!/usr/bin/env bash
# Rank-3 OOD eval (docs/STRATEGIC_REPIVOT_20260709.md).
# Distribution shift: eval on the never-touched OOD splits (frozen IDs already
# committed: test_ood_division = 5-number tasks introducing '/', test_ood_long
# = 6-number tasks). '/' NEVER appears in the 2000 fine-tune examples but the
# PRETRAINED BASE has seen it — so the BASE-MODEL ARM IS MANDATORY: the real
# question is whether RL narrowed an operator the base knows.
#
# Scope statement, not a rescue: exact is expected ~floor (5-6-number tasks are
# below the 0.5B capability floor); the informative signals are the legality
# panel, '/'-adoption (division), and truncation. Eval-only, no retraining.
# seed0-first to de-risk before spending on seeds 1/2.
set -euo pipefail
cd "$(dirname "$0")/.."
source .env
export WANDB_DISABLED=true

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
PY=.venv/bin/python
SPLIT="${1:-test_ood_division}"   # or test_ood_long
SEEDS="${2:-0}"                   # space-separated, e.g. "0 1 2"

bestofn() {
  local method="$1" seed="$2" split="$3" adapter_args=("${@:4}")
  local tag="${method}"; [ "$seed" != "base" ] && tag="${method}_seed${seed}"
  local out="outputs/bestofn/ood_${tag}_${split}_limit50_n8"
  mkdir -p "$out"
  echo "=== $(date -Is) OOD method=${method} seed=${seed} split=${split} -> ${out}"
  $PY scripts/07_best_of_n_rerank.py \
    --model_name "$MODEL" \
    "${adapter_args[@]}" \
    --engine hf --device cuda \
    --prompt_field prompt \
    --limit 50 --batch_size 8 --max_n 8 --n_values 1 4 8 \
    --max_new_tokens 256 --temperature 0.7 --top_p 0.95 --seed 0 \
    --method "$method" --split "$split" \
    --data_path "data/countdown/${split}.jsonl" \
    --task_ids_file "outputs/v09_task_ids_${split}_limit50.txt" \
    --skip_if_complete \
    --output_dir "$out" 2>&1 | tee "$out/run.log"
}

# Base-model arm (mandatory reference; no adapter, no training_seed).
bestofn base base "$SPLIT"

for seed in $SEEDS; do
  for method in static stable; do
    bestofn "$method" "$seed" "$SPLIT" \
      --adapter_path "outputs/checkpoints/grpo_${method}_seed${seed}_300" \
      --training_seed "$seed"
  done
done

# v0.13 SFT-warmup arm (NEXT_STEPS item 3): does the SFT-taught legality
# CAPABILITY transfer OOD, or did it overfit the 3-5-number / 4-op training
# envelope? Load-bearing for the "capability lever" claim. seed0 only until
# v13 seeds 1/2 exist; add them here when they do.
if [ -d "outputs/checkpoints/grpo_v13_sft_seed0_300" ]; then
  bestofn v13sft 0 "$SPLIT" \
    --adapter_path "outputs/checkpoints/grpo_v13_sft_seed0_300" \
    --training_seed 0
fi

echo "=== $(date -Is) OOD eval (${SPLIT}) complete"
