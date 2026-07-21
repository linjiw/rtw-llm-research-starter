#!/usr/bin/env python
"""MicroCode GRPO training (Paper-2 E4/E5) — adapted from the frozen 02.

Design: docs/E4_TRAINER_DESIGN.md. 02_grpo_train.py stays untouched
(Countdown-bound: data-access guard, default scorer, Countdown TeacherConfig).
Deltas here, everything else carried over:
  1. data = data/microcode/*.jsonl; scorer = microcode.score_completion with
     the HARDENED sandbox worker bound (model code executes during training).
  2. TeacherConfig built with the MicroCode tables (MICRO_AUX_KEYS/FLOORS/CAPS/
     TARGET_WEIGHT_SUM=0.80 — never the silent Countdown 1.20 default).
  3. --init_weights_json for the E5 TEMPT vector (default None = HONEST).
  4. strategies restricted to static/adaptive_stable/manual/random.
  5. curriculum restricted to uniform (E4/E5 pre-registration).
  6. post-run health asserts: post-delay weight_sum == 0.80 (adaptive) or
     weights == init vector (static); backstop_firings recorded.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from rtw_llm import microcode
from rtw_llm.rewards import RTWRewardManager
from rtw_llm.seed_protocol import (
    SEED_PROTOCOLS,
    TRUE_SEED_PROTOCOL,
    apply_pre_model_seed,
    resolve_grpo_seed_plan,
)
from rtw_llm.teacher import (
    MICRO_AUX_KEYS,
    MICRO_STABLE_CAPS,
    MICRO_STABLE_FLOORS,
    MICRO_TARGET_WEIGHT_SUM,
    RTWTeacher,
    TeacherConfig,
)
from rtw_llm.trl_compat import set_first_supported_kwarg, supported_config_kwargs

MICRO_EXAMPLE_FIELDS = ("fn_name", "visible_tests", "held_out_tests")


def microcode_score_worker(completion, example, aux_weights=None, primary_weight=1.0):
    """score_completion with the hardened sandbox bound (S3 gate: worker is
    mandatory whenever model code executes during a GPU run)."""
    result = microcode.verify_completion(completion, example, sandbox="worker")
    components = result.to_components()
    total = primary_weight * components["correct"]
    if aux_weights:
        total += sum(w * components.get(k, 0.0) for k, w in aux_weights.items())
    return total, components, result


def check_teacher_health(output_dir: Path, strategy: str, init_vector: dict | None,
                         delay_steps: int, target_sum: float) -> dict:
    """Post-run fail-closed asserts from the E5 v2 pre-registration."""
    path = output_dir / "teacher_weights.jsonl"
    records = [json.loads(line) for line in path.open()]
    problems: list[str] = []
    expected_static = init_vector or {k: 0.20 for k in MICRO_AUX_KEYS}
    for rec in records:
        w = rec["weights"]
        if strategy == "static":
            if any(abs(w[k] - expected_static[k]) > 1e-9 for k in w):
                problems.append(f"step {rec['step']}: static weights drifted")
                break
        elif strategy == "adaptive_stable" and rec["step"] > delay_steps:
            if abs(sum(w.values()) - target_sum) > 1e-6:
                problems.append(
                    f"step {rec['step']}: weight_sum {sum(w.values()):.6f} != {target_sum}"
                )
                break
    health = {"teacher_records": len(records), "problems": problems}
    if problems:
        raise RuntimeError(f"Teacher health check FAILED: {problems}")
    return health


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--train_path", default="data/microcode/train.jsonl")
    parser.add_argument("--output_dir", default="outputs/microcode_grpo")
    parser.add_argument(
        "--reward_strategy",
        choices=["static", "adaptive_stable", "manual", "random"],  # phased excluded
        default="adaptive_stable",
    )
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=768)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trainer_seed", type=int, default=None,
                        help="defaults to --seed (true-seed semantics)")
    parser.add_argument("--seed_protocol", choices=SEED_PROTOCOLS, default=TRUE_SEED_PROTOCOL)
    parser.add_argument("--prompt_field", default="prompt")
    parser.add_argument("--report_to", default="none")
    parser.add_argument(
        "--init_weights_json",
        default=None,
        help="JSON dict of per-key init weights (the E5 TEMPT vector); None = HONEST",
    )
    parser.add_argument("--stable_delay_steps", type=int, default=50)
    args = parser.parse_args()

    trainer_seed = args.seed if args.trainer_seed is None else args.trainer_seed
    seed_plan = resolve_grpo_seed_plan(
        teacher_seed=args.seed, trainer_seed=trainer_seed, protocol_id=args.seed_protocol
    )
    apply_pre_model_seed(seed_plan)
    print(f"Resolved seed plan: {seed_plan}")

    init_weights = json.loads(Path(args.init_weights_json).read_text()) if (
        args.init_weights_json and Path(args.init_weights_json).is_file()
    ) else (json.loads(args.init_weights_json) if args.init_weights_json else None)
    if init_weights is not None:
        unknown = set(init_weights) - set(MICRO_AUX_KEYS)
        if unknown:
            raise ValueError(f"init_weights has non-MicroCode keys: {sorted(unknown)}")
        total = sum(init_weights.values())
        if abs(total - MICRO_TARGET_WEIGHT_SUM) > 1e-9:
            raise ValueError(
                f"init_weights must sum to the budget {MICRO_TARGET_WEIGHT_SUM} "
                f"(equal aux mass across arms — E5 v2 amendment 1); got {total}"
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_ds = load_dataset("json", data_files=args.train_path, split="train")
    if args.prompt_field != "prompt":
        train_ds = train_ds.map(lambda x: {"prompt": x[args.prompt_field]})

    use_cuda = torch.cuda.is_available()
    config_kwargs = {
        # identical GRPO loss pinning to 02 (dapo / group scaling / beta 0)
        "loss_type": "dapo",
        "scale_rewards": "group",
        "beta": 0.0,
        "output_dir": str(output_dir),
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "logging_steps": 10,
        "save_steps": 100,
        "bf16": use_cuda,
        "fp16": False,
        "optim": "adamw_torch_fused" if use_cuda else "adamw_torch",
        "report_to": args.report_to,
        "run_name": output_dir.name,
        "trust_remote_code": True,
        "seed": int(seed_plan["trainer_seed"]),
        "lr_scheduler_type": "linear",
        "warmup_steps": 0,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "gradient_checkpointing": True,
        "dataloader_drop_last": False,
        "save_strategy": "steps",
        "eval_strategy": "no",
        "num_iterations": 1,
    }
    if args.model_revision:
        config_kwargs["model_init_kwargs"] = {
            "revision": args.model_revision,
            "trust_remote_code": True,
        }
    set_first_supported_kwarg(
        GRPOConfig, config_kwargs, ["max_prompt_length", "max_length"], args.max_prompt_length
    )
    train_args = GRPOConfig(**supported_config_kwargs(GRPOConfig, config_kwargs))
    if int(train_args.seed) != int(seed_plan["trainer_seed"]):
        raise RuntimeError("Resolved GRPOConfig seed != requested trainer seed")

    teacher = RTWTeacher(
        TeacherConfig(
            strategy=args.reward_strategy,
            seed=int(seed_plan["teacher_seed"]),
            log_path=str(output_dir / "teacher_weights.jsonl"),
            aux_keys=list(MICRO_AUX_KEYS),
            stable_floors=dict(MICRO_STABLE_FLOORS),
            stable_caps=dict(MICRO_STABLE_CAPS),
            stable_target_weight_sum=MICRO_TARGET_WEIGHT_SUM,
            stable_delay_steps=args.stable_delay_steps,
            init_weights=init_weights,
        )
    )
    reward_manager = RTWRewardManager(
        teacher=teacher,
        primary_weight=1.0,
        log_path=str(output_dir / "reward_components.jsonl"),
        curriculum=None,  # uniform only for E4/E5 (pre-registered)
        group_size=args.num_generations,
        scorer=microcode_score_worker,
        example_fields=MICRO_EXAMPLE_FIELDS,
    )

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear",
    )
    processing_class = None
    if args.model_revision:
        from transformers import AutoTokenizer

        processing_class = AutoTokenizer.from_pretrained(
            args.model_name, revision=args.model_revision, trust_remote_code=True,
            truncation_side="left", padding_side="left",
        )
    trainer = GRPOTrainer(
        model=args.model_name,
        reward_funcs=reward_manager,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=None,
        peft_config=peft_config,
        processing_class=processing_class,
    )
    started_at = time.time()
    trainer.train()
    wall_clock_seconds = time.time() - started_at
    trainer.save_model(str(output_dir))

    # fail-closed post-run checks (E5 v2 pre-registration)
    health = check_teacher_health(
        output_dir, args.reward_strategy, init_weights,
        args.stable_delay_steps, MICRO_TARGET_WEIGHT_SUM,
    )
    from rtw_llm import microcode_sandbox

    backstops = (
        microcode_sandbox._WORKER.backstop_firings if microcode_sandbox._WORKER else 0
    )
    summary = {
        "wall_clock_seconds": wall_clock_seconds,
        "global_step": int(trainer.state.global_step),
        "seed_plan": seed_plan,
        "reward_strategy": args.reward_strategy,
        "init_weights": init_weights,
        "teacher_health": health,
        "sandbox_backstop_firings": int(backstops),
        "bit_stability_qualified": bool(backstops > 0),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
