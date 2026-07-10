#!/usr/bin/env python
"""Optional SFT warmup for the Countdown output format."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from rtw_llm.provenance import build_run_identity, write_intent, write_result
from rtw_llm.seed_protocol import (
    LEGACY_SEED_PROTOCOL,
    SEED_PROTOCOLS,
    TRUE_SEED_PROTOCOL,
    apply_pre_model_seed,
    resolve_sft_seed_plan,
)
from rtw_llm.trl_compat import set_first_supported_kwarg, supported_config_kwargs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--train_path", default="data/countdown/train.jsonl")
    parser.add_argument("--eval_path", default=None)
    parser.add_argument("--output_dir", default="outputs/sft_qwen05b")
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seed_protocol",
        choices=SEED_PROTOCOLS,
        default=LEGACY_SEED_PROTOCOL,
        help=(
            "Legacy preserves pre-trainer LoRA RNG behavior; corrected-v2 globally "
            "seeds before model/adapter construction."
        ),
    )
    parser.add_argument("--report_to", default="wandb")
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--strict_provenance", action="store_true")
    parser.add_argument(
        "--completion_only_loss",
        action="store_true",
        default=False,
        help=(
            "Mask the prompt so loss falls only on the completion (v0.13 A3). Uses "
            "TRL's native prompt-completion dataset format instead of a formatting "
            "func, so the small SFT budget teaches expression construction rather "
            "than prompt reproduction."
        ),
    )
    args = parser.parse_args()

    seed_plan = resolve_sft_seed_plan(args.seed, args.seed_protocol)
    if args.seed_protocol == TRUE_SEED_PROTOCOL and not args.strict_provenance:
        raise ValueError("countdown-true-seeds-v2 requires --strict_provenance")
    apply_pre_model_seed(seed_plan)
    print(f"Resolved seed plan: {seed_plan}")

    ds = load_dataset("json", data_files=args.train_path, split="train")
    eval_ds = load_dataset("json", data_files=args.eval_path, split="train") if args.eval_path else None

    # The prompt ends at "Now solve the task." and generation begins on a new
    # line; the original single-string join used "\n". Preserve that seam.
    formatting_func = None
    if args.completion_only_loss:
        # Native prompt-completion format: TRL masks the prompt automatically and
        # a formatting_func is incompatible. Fold the "\n" join seam into prompt.
        keep = {"prompt", "completion"}
        ds = ds.map(lambda x: {"prompt": x["prompt"] + "\n"}).remove_columns(
            [c for c in ds.column_names if c not in keep]
        )
        if eval_ds is not None:
            eval_ds = eval_ds.map(lambda x: {"prompt": x["prompt"] + "\n"}).remove_columns(
                [c for c in eval_ds.column_names if c not in keep]
            )
    else:

        def formatting_func(example: dict) -> str:
            return example["prompt"] + "\n" + example["completion"]

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
        "output_dir": args.output_dir,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "logging_steps": 10,
        "save_steps": 100,
        "bf16": use_cuda,
        "fp16": False,
        "optim": "adamw_torch_fused" if use_cuda else "adamw_torch",
        "report_to": args.report_to,
        "run_name": Path(args.output_dir).name,
        "seed": int(seed_plan["trainer_seed"]),
        "trust_remote_code": True,
    }
    if args.completion_only_loss:
        config_kwargs["completion_only_loss"] = True
    if args.model_revision:
        config_kwargs["model_init_kwargs"] = {
            "revision": args.model_revision,
            "trust_remote_code": True,
        }
    set_first_supported_kwarg(SFTConfig, config_kwargs, ["max_seq_length", "max_length"], 1024)
    train_args = SFTConfig(**supported_config_kwargs(SFTConfig, config_kwargs))

    output_dir = Path(args.output_dir)
    if args.strict_provenance:
        input_files = {"train": args.train_path}
        if args.eval_path:
            input_files["eval"] = args.eval_path
        identity = build_run_identity(
            run_kind="sft",
            requested_args=vars(args),
            resolved_config=train_args.to_dict(),
            seed_roles=seed_plan,
            input_files=input_files,
            model_name=args.model_name,
            repo_root=Path(__file__).resolve().parents[1],
            model_revision=args.model_revision,
        )
        write_intent(output_dir, identity)

    processing_class = None
    if args.model_revision:
        from transformers import AutoTokenizer

        processing_class = AutoTokenizer.from_pretrained(
            args.model_name,
            revision=args.model_revision,
            trust_remote_code=True,
        )
    trainer = SFTTrainer(
        model=args.model_name,
        args=train_args,
        train_dataset=ds,
        eval_dataset=eval_ds,
        formatting_func=formatting_func,
        peft_config=peft_config,
        processing_class=processing_class,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    if args.strict_provenance:
        write_result(
            output_dir,
            artifact_paths={
                "adapter_config": output_dir / "adapter_config.json",
                "adapter_weights": output_dir / "adapter_model.safetensors",
                "tokenizer_config": output_dir / "tokenizer_config.json",
                "training_args": output_dir / "training_args.bin",
            },
        )


if __name__ == "__main__":
    main()
