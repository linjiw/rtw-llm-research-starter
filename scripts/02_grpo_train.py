#!/usr/bin/env python
"""GRPO post-training with static/manual/random/adaptive RTW reward weights."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from rtw_llm.curriculum import CurriculumConfig, CurriculumController, CurriculumSampler
from rtw_llm.data_access import assert_countdown_data_access
from rtw_llm.provenance import build_run_identity, write_intent, write_result
from rtw_llm.rewards import RTWRewardManager
from rtw_llm.seed_protocol import (
    LEGACY_SEED_PROTOCOL,
    SEED_PROTOCOLS,
    TRUE_SEED_PROTOCOL,
    apply_pre_model_seed,
    resolve_grpo_seed_plan,
)
from rtw_llm.teacher import RTWTeacher, TeacherConfig
from rtw_llm.trl_compat import set_first_supported_kwarg, supported_config_kwargs


def plan_model_init(init_adapter_path: str | None, use_lora: bool) -> dict:
    """Decide how the GRPO trainer is initialized (pure, GPU-free — testable).

    Returns a plan dict:
      mode="continue_adapter": load the pre-trained LoRA and continue it
        (is_trainable=True); peft_config MUST be None or TRL stacks a second
        adapter (capacity confound). The v0.13 SFT-warmup path.
      mode="fresh_lora": attach a new zero-init LoRA (the baseline path).
      mode="full_finetune": no LoRA at all.
    """
    if init_adapter_path is not None:
        return {"mode": "continue_adapter", "adapter_path": init_adapter_path, "use_peft_config": False}
    if use_lora:
        return {"mode": "fresh_lora", "adapter_path": None, "use_peft_config": True}
    return {"mode": "full_finetune", "adapter_path": None, "use_peft_config": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--train_path", default="data/countdown/train.jsonl")
    parser.add_argument("--eval_path", default=None)
    parser.add_argument("--output_dir", default="outputs/grpo_rtw_qwen05b")
    parser.add_argument(
        "--reward_strategy",
        choices=[
            "adaptive",
            "adaptive_stable",
            "adaptive_stable_v12",
            "adaptive_phased",
            "static",
            "manual",
            "random",
        ],
        default="adaptive",
    )
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=768)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Teacher/controller seed. Historical runs varied this value while TRL "
            "kept trainer seed 42; use --seed_protocol countdown-true-seeds-v2 "
            "with --trainer_seed N for future true-seed experiments."
        ),
    )
    parser.add_argument(
        "--trainer_seed",
        type=int,
        default=42,
        help="Explicit TRL/GRPO seed; defaults to 42 for archived-protocol compatibility.",
    )
    parser.add_argument(
        "--seed_protocol",
        choices=SEED_PROTOCOLS,
        default=LEGACY_SEED_PROTOCOL,
        help="Fail-closed seed-role contract; never compare banks across protocol ids.",
    )
    parser.add_argument(
        "--task_curriculum",
        choices=["uniform", "manual", "adaptive"],
        default="uniform",
    )
    parser.add_argument("--prompt_field", default="prompt")
    parser.add_argument("--report_to", default="wandb")
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--strict_provenance", action="store_true")
    parser.add_argument("--experiment_protocol", default=None)
    parser.add_argument("--method_arm", default=None)
    parser.add_argument(
        "--init_adapter_path",
        default=None,
        help=(
            "Optional path to a pre-trained LoRA adapter (e.g. an SFT warmup) to "
            "CONTINUE training from. When set, GRPO resumes the given adapter "
            "(is_trainable=True) instead of attaching a fresh zero-init LoRA — the "
            "single-variable warmup change for v0.13. Same base model and adapter "
            "shape as the no-warmup baseline; only the LoRA init differs."
        ),
    )
    args = parser.parse_args()

    if args.experiment_protocol is not None:
        from rtw_llm.v19_protocol import PROTOCOL_ID, validate_v19_grpo_args

        if args.experiment_protocol != PROTOCOL_ID:
            raise ValueError(f"Unsupported experiment protocol: {args.experiment_protocol}")
        validate_v19_grpo_args(vars(args))

    repo_root = Path(__file__).resolve().parents[1]
    assert_countdown_data_access(
        args.train_path, purpose="training", runner="02_grpo_train", repo_root=repo_root
    )
    if args.eval_path:
        assert_countdown_data_access(
            args.eval_path,
            purpose="training_eval",
            runner="02_grpo_train",
            repo_root=repo_root,
        )

    seed_plan = resolve_grpo_seed_plan(
        teacher_seed=args.seed,
        trainer_seed=args.trainer_seed,
        protocol_id=args.seed_protocol,
    )
    if args.seed_protocol == TRUE_SEED_PROTOCOL and not args.strict_provenance:
        raise ValueError("countdown-true-seeds-v2 requires --strict_provenance")
    apply_pre_model_seed(seed_plan)
    print(f"Resolved seed plan: {seed_plan}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = load_dataset("json", data_files=args.train_path, split="train")
    eval_ds = (
        load_dataset("json", data_files=args.eval_path, split="train")
        if args.eval_path
        else None
    )

    # TRL expects a column named 'prompt'. Keep alternative harness fields available.
    if args.prompt_field != "prompt":
        train_ds = train_ds.map(lambda x: {"prompt": x[args.prompt_field]})
        if eval_ds is not None:
            eval_ds = eval_ds.map(lambda x: {"prompt": x[args.prompt_field]})

    use_cuda = torch.cuda.is_available()
    config_kwargs = {
        # Pin the GRPO loss dynamics to TRL 1.7.0's resolved defaults so future
        # TRL bumps cannot silently change training behavior between ladder
        # runs. Note these differ from the archive-era TRL that trained the
        # v0.9B checkpoints (which used a KL penalty and different loss/scaling
        # defaults) — see the Gate 0 report.
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
        GRPOConfig,
        config_kwargs,
        ["max_prompt_length", "max_length"],
        args.max_prompt_length,
    )
    resolved_config_kwargs = supported_config_kwargs(GRPOConfig, config_kwargs)
    train_args = GRPOConfig(**resolved_config_kwargs)
    if int(train_args.seed) != int(seed_plan["trainer_seed"]):
        raise RuntimeError(
            "Resolved GRPOConfig seed does not match the requested trainer seed: "
            f"resolved={train_args.seed}, requested={seed_plan['trainer_seed']}"
        )
    if args.experiment_protocol:
        if str(train_args.eval_strategy) not in {"no", "IntervalStrategy.NO"}:
            raise RuntimeError("V0.19 GRPO requires resolved eval_strategy=no")
        if eval_ds is not None:
            raise RuntimeError("V0.19 GRPO forbids any evaluation dataset")
        expected_resolved = {
            "generation_batch_size": 16,
            "steps_per_generation": 8,
            "num_iterations": 1,
            "loss_type": "dapo",
            "scale_rewards": "group",
            "beta": 0.0,
        }
        mismatches = [
            key for key, value in expected_resolved.items() if getattr(train_args, key) != value
        ]
        if mismatches:
            raise RuntimeError(f"V0.19 resolved GRPO config mismatch: {mismatches}")
    if args.strict_provenance:
        input_files = {"train": args.train_path}
        if args.eval_path:
            input_files["eval"] = args.eval_path
        identity = build_run_identity(
            run_kind="grpo",
            requested_args=vars(args),
            resolved_config=train_args.to_dict(),
            seed_roles=seed_plan,
            input_files=input_files,
            model_name=args.model_name,
            adapter_path=args.init_adapter_path,
            repo_root=repo_root,
            model_revision=args.model_revision,
        )
        write_intent(output_dir, identity)

    teacher = RTWTeacher(
        TeacherConfig(
            strategy=args.reward_strategy,
            seed=int(seed_plan["teacher_seed"]),
            log_path=str(output_dir / "teacher_weights.jsonl"),
        )
    )
    curriculum = None
    if args.task_curriculum != "uniform":
        curriculum = CurriculumController(
            CurriculumConfig(
                mode=args.task_curriculum,
                log_path=str(output_dir / "curriculum_state.jsonl"),
            )
        )

    reward_manager = RTWRewardManager(
        teacher=teacher,
        primary_weight=1.0,
        log_path=str(output_dir / "reward_components.jsonl"),
        curriculum=curriculum,
        group_size=args.num_generations,
    )

    # Model + adapter handling (see plan_model_init / docs/V13_..._PLAN.md A1).
    # Default: base-model string + fresh zero-init LoRA. With --init_adapter_path:
    # load the pre-trained adapter and CONTINUE it (is_trainable=True), with
    # peft_config=None so TRL does not stack a second adapter.
    init_plan = plan_model_init(args.init_adapter_path, args.use_lora)
    model_arg: object = args.model_name
    peft_config = None
    if init_plan["mode"] == "continue_adapter":
        from peft import PeftModel
        from trl.trainer.utils import create_model_from_path

        # Load the base through the SAME helper (and thus same dtype=float32 /
        # device_map="auto" defaults) TRL uses for the fresh-LoRA baseline, so the
        # ONLY difference vs C0 is the LoRA init (SFT-warmed vs zero), not the base
        # load dtype/device_map (v0.13 diff-review F1).
        model_init_kwargs = {"trust_remote_code": True}
        if args.model_revision:
            model_init_kwargs["revision"] = args.model_revision
        base = create_model_from_path(args.model_name, **model_init_kwargs)
        model_arg = PeftModel.from_pretrained(
            base, init_plan["adapter_path"], is_trainable=True
        )
    elif init_plan["use_peft_config"]:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )

    trainer_cls = GRPOTrainer
    if curriculum is not None:
        # The v0.10 cadence identity (1 generation block = 1 controller update
        # = 1 optimizer step) and the curriculum-state log both assume no batch
        # reuse and steps_per_generation == grad_accum. Check before loading
        # the model so misconfiguration fails in seconds, not minutes.
        if getattr(train_args, "num_iterations", 1) != 1:
            raise ValueError("task_curriculum requires num_iterations == 1")
        if train_args.steps_per_generation != train_args.gradient_accumulation_steps:
            raise ValueError(
                "task_curriculum requires steps_per_generation == gradient_accumulation_steps"
            )
        if train_args.remove_unused_columns:
            raise ValueError(
                "task_curriculum requires remove_unused_columns=False: the sampler "
                "override matches the train dataset by identity"
            )
        if train_args.world_size != 1:
            raise ValueError(
                "task_curriculum is single-process only: the sampled index stream "
                "depends on rank-local controller state and would diverge across ranks"
            )
        if train_args.eval_strategy != "no":
            raise ValueError(
                "task_curriculum requires eval_strategy='no': eval reward batches "
                "would advance the controller and pollute its competence EMAs"
            )
        tier_of_index = list(train_ds["difficulty"])

        class CurriculumGRPOTrainer(GRPOTrainer):
            def _get_train_sampler(self, dataset=None):
                if dataset is not None and dataset is not self.train_dataset:
                    # Fail loud rather than silently degrading the experiment
                    # arm to uniform RepeatSampler on an unexpected dataset copy.
                    raise ValueError(
                        "CurriculumGRPOTrainer received a dataset that is not the "
                        "train dataset; curriculum sampling would be silently skipped"
                    )
                return CurriculumSampler(
                    tier_of_index=tier_of_index,
                    controller=curriculum,
                    mini_repeat_count=self.num_generations,
                    batch_size=self.args.generation_batch_size // self.num_generations,
                    repeat_count=self.num_iterations * self.args.steps_per_generation,
                    seed=self.args.seed,
                )

        trainer_cls = CurriculumGRPOTrainer

    processing_class = None
    if args.model_revision:
        from transformers import AutoTokenizer

        processing_class = AutoTokenizer.from_pretrained(
            args.model_name,
            revision=args.model_revision,
            trust_remote_code=True,
            truncation_side="left",
            padding_side="left",
        )
    trainer = trainer_cls(
        model=model_arg,
        reward_funcs=reward_manager,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=processing_class,
    )
    started_at = time.time()
    trainer.train()
    wall_clock_seconds = time.time() - started_at
    trainer.save_model(str(output_dir))
    if args.strict_provenance:
        training_state = {
            "global_step": int(trainer.state.global_step),
            "max_steps": int(trainer.state.max_steps),
            "log_history": trainer.state.log_history,
            "wall_clock_seconds": wall_clock_seconds,
        }
        (output_dir / "training_state.json").write_text(
            json.dumps(training_state, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        artifacts = {
            "adapter_config": output_dir / "adapter_config.json",
            "adapter_weights": output_dir / "adapter_model.safetensors",
            "reward_components": output_dir / "reward_components.jsonl",
            "teacher_weights": output_dir / "teacher_weights.jsonl",
            "tokenizer_config": output_dir / "tokenizer_config.json",
            "training_args": output_dir / "training_args.bin",
            "training_state": output_dir / "training_state.json",
        }
        if curriculum is not None:
            artifacts["curriculum_state"] = output_dir / "curriculum_state.jsonl"
        write_result(
            output_dir,
            artifact_paths=artifacts,
            observations={"wall_clock_seconds": wall_clock_seconds},
        )


if __name__ == "__main__":
    main()
