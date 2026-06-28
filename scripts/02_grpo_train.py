#!/usr/bin/env python
"""GRPO post-training with static/manual/random/adaptive RTW reward weights."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from rtw_llm.rewards import RTWRewardManager
from rtw_llm.teacher import RTWTeacher, TeacherConfig
from rtw_llm.trl_compat import set_first_supported_kwarg, supported_config_kwargs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train_path", default="data/countdown/train.jsonl")
    parser.add_argument("--eval_path", default="data/countdown/validation.jsonl")
    parser.add_argument("--output_dir", default="outputs/grpo_rtw_qwen05b")
    parser.add_argument(
        "--reward_strategy",
        choices=["adaptive", "adaptive_stable", "static", "manual", "random"],
        default="adaptive",
    )
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=768)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt_field", default="prompt")
    parser.add_argument("--report_to", default="wandb")
    parser.add_argument("--use_lora", action="store_true", default=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = load_dataset("json", data_files=args.train_path, split="train")
    eval_ds = load_dataset("json", data_files=args.eval_path, split="train")

    # TRL expects a column named 'prompt'. Keep alternative harness fields available.
    if args.prompt_field != "prompt":
        train_ds = train_ds.map(lambda x: {"prompt": x[args.prompt_field]})
        eval_ds = eval_ds.map(lambda x: {"prompt": x[args.prompt_field]})

    teacher = RTWTeacher(
        TeacherConfig(
            strategy=args.reward_strategy,
            seed=args.seed,
            log_path=str(output_dir / "teacher_weights.jsonl"),
        )
    )
    reward_manager = RTWRewardManager(
        teacher=teacher,
        primary_weight=1.0,
        log_path=str(output_dir / "reward_components.jsonl"),
    )

    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )

    use_cuda = torch.cuda.is_available()
    config_kwargs = {
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
    }
    set_first_supported_kwarg(
        GRPOConfig,
        config_kwargs,
        ["max_prompt_length", "max_length"],
        args.max_prompt_length,
    )
    train_args = GRPOConfig(**supported_config_kwargs(GRPOConfig, config_kwargs))

    trainer = GRPOTrainer(
        model=args.model_name,
        reward_funcs=reward_manager,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))


if __name__ == "__main__":
    main()
