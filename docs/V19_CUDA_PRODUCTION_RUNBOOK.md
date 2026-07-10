# v0.19 CUDA production runbook

Status: operator handoff for the frozen `countdown-v19-within-v2-reset-v1`
protocol. This runbook does not authorize a protocol change.

## Preconditions

- Use a clean checkout of the registered commit and exactly one visible CUDA
  GPU. Do not substitute MPS, CPU, multiple GPUs, or a different model revision.
- Install the checkout as a package (`python -m pip install -e .`) before using
  bare `python`. For audit-only commands, `PYTHONPATH=src python ...` is also
  sufficient.
- Keep the final test sealed. Do not inspect or evaluate the confirmation view
  before the repository-created readiness record permits it.
- Production is capped at 60 single-GPU hours and USD 150. The launch record
  additionally binds the actual hourly GPU price.

## 1. Verify the checkout and protocol

```bash
git pull --ff-only origin main
git status --short --branch
python -m pip install -e .
python scripts/21_audit_v19_protocol.py
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
python -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
```

Stop unless the worktree is clean, the audit reports `ELIGIBLE`, and exactly
one GPU is visible.

## 2. Lock the environment before compute

The environment and launch files are write-once. Replace the two shell values
with the real host label and provider price.

```bash
python scripts/23_capture_v19_environment.py
python scripts/25_create_v19_launch_record.py \
  --approved_host_label '<provider-host-id>' \
  --usd_per_gpu_hour '<actual-hourly-rate>'
git add protocols/countdown_v2/v19/environment_lock.json \
  protocols/countdown_v2/v19/launch_record.json
git commit -m 'Lock v0.19 CUDA production environment'
git push origin main
```

Do not train from an uncommitted lock. If capture fails, preserve the error and
fix the host; never hand-edit either record.

## 3. Seed-0 development gate

Dry-run first, then execute. Each stage is fail-closed and writes logs beneath
`outputs/v19/logs/production/`.

```bash
python scripts/24_run_v19.py --stage train_seed --seed 0
python scripts/24_run_v19.py --stage train_seed --seed 0 --execute
python scripts/24_run_v19.py --stage eval_dev
python scripts/24_run_v19.py --stage eval_dev --execute
python scripts/24_run_v19.py --stage score_dev
python scripts/24_run_v19.py --stage score_dev --execute
```

Review the registered dev score and health artifacts. The development view may
guide the already-preregistered continuation decision; it may not alter the
primary endpoint, split, sampling stream, verifier, or arm definitions.

## 4. Complete the true-seed panel

If the registered development rule continues the experiment:

```bash
python scripts/24_run_v19.py --stage train_seed --seed 1 --execute
python scripts/24_run_v19.py --stage train_seed --seed 2 --execute
python scripts/24_run_v19.py --stage prepare_confirm --execute
git add protocols/countdown_v2/v19/confirmation_ready.json
git commit -m 'Record v0.19 confirmation readiness'
git push origin main
```

The readiness command verifies all 15 trained states, adapter ancestry,
runtime/config/source identity, health, and the six seed-0 development banks.
Never create the record manually.

## 5. One confirmation pass

Only after the committed readiness record exists:

```bash
python scripts/24_run_v19.py --stage eval_confirm
python scripts/24_run_v19.py --stage eval_confirm --execute
python scripts/24_run_v19.py --stage score_confirm
python scripts/24_run_v19.py --stage score_confirm --execute
```

Commit the content-addressed score report and manifest together with the
experiment-ledger interpretation. Confirmation is the primary analysis;
candidate rows are clustered by task and are never treated as independent.

## Recovery and stop rules

- A completed strict run is reusable only when its manifest verifies. The eval
  path uses `--skip_if_complete`; training output directories and log files are
  intentionally non-overwriting.
- On failure, do not delete or overwrite evidence. Record the failing command,
  log tail, commit, GPU identity, and consumed hours; diagnose before deciding
  whether a new output root is scientifically equivalent.
- Abort on missing reward components, failed verifier recomputation, unhealthy
  training, no required GRPO group variance, provenance mismatch, or budget
  exhaustion.
- Do not access the one-shot in-distribution test until the complete confirm
  score passes its release gate. The untouched final test remains sealed.

