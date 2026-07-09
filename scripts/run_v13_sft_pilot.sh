#!/usr/bin/env bash
# v0.13 SFT warmup -> GRPO stable (docs/V13_SFT_WARMUP_LEGALITY_PLAN.md):
# light completion-only SFT on the 2000 verifier-exact gold solutions, then
# GRPO stable CONTINUING that adapter (--init_adapter_path). Comparator:
# C0 = grpo_stable_seed0_300 (fresh-LoRA, no SFT); score vs the stable 3-SEED
# distribution, never a single seed.
# Stages: SFT (light) -> SFT health -> GRPO smoke 60 (health + variance gate)
# -> GRPO 300 -> frozen best-of-N (GRPO) + SFT-only eval arm (A6).
set -euo pipefail
cd "$(dirname "$0")/.."
source .env
export WANDB_DISABLED=true

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
PY=.venv/bin/python

SFT=outputs/checkpoints/sft_warmup_legal_seed0
SMOKE=outputs/checkpoints/grpo_v13_sft_smoke_60_seed0
PILOT=outputs/checkpoints/grpo_v13_sft_seed0_300

# --- Stage 1: light completion-only SFT on gold solutions (all tiers) ---
# Light schedule (advisor A4): ~1 epoch (2000 ex / eff-batch 16 = 125 steps),
# lr 5e-5, completion-only loss so the budget teaches expression construction.
echo "=== $(date -Is) v0.13 SFT warmup -> ${SFT}"
mkdir -p "$SFT"
$PY scripts/01_sft_warmup.py \
  --model_name "$MODEL" \
  --train_path data/countdown/train.jsonl \
  --output_dir "$SFT" \
  --max_steps 125 \
  --batch_size 2 \
  --grad_accum 8 \
  --learning_rate 5e-5 \
  --seed 0 \
  --completion_only_loss \
  --report_to none 2>&1 | tee "$SFT/train.log"

# --- GRPO trainer (continues the SFT adapter) ---
train_v13() {
  local steps="$1" out="$2"
  mkdir -p "$out"
  echo "=== $(date -Is) v0.13 GRPO(from SFT) steps=${steps} -> ${out}"
  $PY scripts/02_grpo_train.py \
    --model_name "$MODEL" \
    --train_path data/countdown/train.jsonl \
    --eval_path data/countdown/validation.jsonl \
    --output_dir "$out" \
    --reward_strategy adaptive_stable \
    --init_adapter_path "$SFT" \
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

# GRPO-not-inert gate (advisor A6): if SFT sharpened the policy so groups go
# zero-variance, GRPO does nothing and the result is SFT-only, misattributed.
check_v13_variance() {
  local out="$1"
  $PY - "$out" <<'PY'
import json, sys
run = sys.argv[1]
rows = [json.loads(l) for l in open(f"{run}/reward_components.jsonl")]
frac = [float(r.get("batch_group_variance_fraction", 0.0)) for r in rows
        if r.get("batch_group_variance_fraction") is not None]
mean_frac = sum(frac) / max(len(frac), 1)
print(f"v13 GRPO group-variance fraction (mean over run) = {mean_frac:.3f} "
      f"(C0 stable ~0.97; a large drop => SFT collapsed exploration, GRPO ~inert)")
if mean_frac < 0.50:
    print(f"WARNING: group-variance fraction {mean_frac:.3f} < 0.50 — GRPO may be "
          f"inert; interpret as SFT-dominated. Inspect before over-crediting GRPO.")
PY
}

train_v13 60 "$SMOKE"
check_v13_variance "$SMOKE"

train_v13 300 "$PILOT"
check_v13_variance "$PILOT"

# --- best-of-N eval: GRPO(from-SFT) arm + SFT-only arm (A6) on frozen IDs ---
bestofn() {
  local adapter="$1" method="$2" split="$3" out="$4"
  mkdir -p "$out"
  echo "=== $(date -Is) best-of-N method=${method} split=${split} -> ${out}"
  $PY scripts/07_best_of_n_rerank.py \
    --model_name "$MODEL" \
    --adapter_path "$adapter" \
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
    --training_seed 0 \
    --split "$split" \
    --data_path "data/countdown/${split}.jsonl" \
    --task_ids_file "outputs/v09_task_ids_${split}_limit50.txt" \
    --skip_if_complete \
    --output_dir "$out" 2>&1 | tee "$out/run.log"
}

for split in validation test_in_dist; do
  bestofn "$PILOT" v13sft "$split" "outputs/bestofn/v13sft_seed0_${split}_limit50_n8"
  # SFT-only arm: eval the warmup adapter directly (no GRPO) to decompose
  # SFT-alone vs SFT+GRPO and detect the GRPO-inert case.
  bestofn "$SFT" v13sftonly "$split" "outputs/bestofn/v13sftonly_seed0_${split}_limit50_n8"
done

echo "=== $(date -Is) v0.13 SFT pilot complete"
