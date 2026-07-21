# E4 trainer design — `scripts/33_microcode_grpo_train.py`

Design note (2026-07-21) for the one missing execution piece the E5 advisor
review flagged: **no MicroCode GRPO trainer exists** — `02_grpo_train.py` is
Countdown-bound (its `assert_countdown_data_access`, default scorer, and
`TeacherConfig` without the MicroCode tables). Per the one-variable / additive
discipline, 02 is NOT edited; a new numbered script adapts it.

## What script 33 is

A copy-adaptation of `02_grpo_train.py` with exactly these deltas (everything
else — GRPOConfig pinning (dapo/group/beta-0), seed plan, LoRA handling,
provenance, TRL-compat — is carried over verbatim):

1. **Data**: `--train_path data/microcode/train.jsonl`; no
   `assert_countdown_data_access` (MicroCode has no sealed splits yet; the
   frozen IDs are eval-side). The prompt field default is `prompt` (== the
   committed records' `prompt_high` content, per generator).
2. **Scorer dispatch**: `RTWRewardManager(..., scorer=microcode_score_worker,
   example_fields=("fn_name", "visible_tests", "held_out_tests"))` where
   `microcode_score_worker` is `microcode.score_completion` with
   `sandbox="worker"` bound (model code executes during training → the
   hardened sandbox is mandatory, per S3 gate).
3. **Teacher tables (the E5-review landmine)**: the `TeacherConfig` is built
   with `aux_keys=MICRO_AUX_KEYS, stable_floors=MICRO_STABLE_FLOORS,
   stable_caps=MICRO_STABLE_CAPS, stable_target_weight_sum=MICRO_TARGET_WEIGHT_SUM`
   — never the Countdown defaults (the silent 1.20 budget default would inflate
   adaptive aux mass 50%). `--init_weights_json` optional arg for the E5 TEMPT
   vector (default None = HONEST).
4. **Strategy restriction**: choices = static / adaptive_stable / manual /
   random (adaptive_phased is Countdown-coupled — excluded).
5. **Curriculum**: `--task_curriculum` restricted to `uniform` for E4/E5 (the
   adaptive curriculum's MicroCode wiring is a later, separate variable).
6. **Health asserts** (fail-closed, per E5 v2 pre-registration): after training,
   verify every post-delay `teacher_weights.jsonl` record has
   `weight_sum == stable_target_weight_sum ± 1e-6` (adaptive) or weights ==
   init vector (static); abort/flag the run otherwise.
7. **Sandbox worker lifecycle**: the reward manager's scorer uses the
   module-singleton worker; it is spawned lazily on first batch (safe: spawn-
   after-CUDA). `backstop_firings` is written into the run summary — >0
   qualifies the bit-stability claim (S3).

## E4 arms (frozen protocol)

```
33_microcode_grpo_train.py --reward_strategy static          --seed 0 ...  # E4-static
33_microcode_grpo_train.py --reward_strategy adaptive_stable --seed 0 ...  # E4-adaptive
```
300 steps, batch 2, grad_accum 8, num_generations 4, lr 5e-6, uniform
curriculum, true-seed semantics (trainer seed = teacher seed = 0), commit
before CUDA, 60-step smoke first, then `05_check_run_health.py`-style gate +
the live `group_reward_std` precondition check (must stay unsaturated DURING
training — the E4 primary health metric).

## Eval

`scripts/32_microcode_best_of_n.py` on the frozen validation IDs
(`data/microcode/frozen_microcode_task_ids_validation_limit50.txt`), N=1/4/8,
loop mode, sandbox=worker; paired held_out_all_pass@8 McNemar vs the other arm.

## Order of operations (respects the GPU queue)

v19 seed-0 chain → v19 seeds 1/2 + confirm (priority) → **E4 smoke (60 steps)
→ E4 full (2 arms)** → E5 (per the v2 pre-registration, only if E4 healthy).
