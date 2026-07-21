#!/bin/bash
# v0.19 production seed-0 chain (runbook steps 3): waits for the shared A10G
# to free up, then train_seed 0 -> eval_dev -> score_dev, fail-closed.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=outputs/v19/logs/seed0_chain.log

echo "[chain] waiting for GPU to free (other tenant's sweep)..." | tee -a "$LOG"
while true; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$USED" -lt 2000 ]; then break; fi
  sleep 120
done
echo "[chain] GPU free (used=${USED}MiB) at $(date -u +%FT%TZ); starting train_seed 0" | tee -a "$LOG"

$PY scripts/24_run_v19.py --stage train_seed --seed 0 --execute 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then echo "[chain] train_seed FAILED rc=$rc — stopping" | tee -a "$LOG"; exit $rc; fi

$PY scripts/24_run_v19.py --stage eval_dev --execute 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then echo "[chain] eval_dev FAILED rc=$rc — stopping" | tee -a "$LOG"; exit $rc; fi

$PY scripts/24_run_v19.py --stage score_dev --execute 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then echo "[chain] score_dev FAILED rc=$rc — stopping" | tee -a "$LOG"; exit $rc; fi

echo "[chain] SEED-0 DEV GATE COMPLETE at $(date -u +%FT%TZ)" | tee -a "$LOG"
