#!/usr/bin/env python
"""Optional SFT warmup for the Countdown output format."""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train_path", default="data/countdown/train.jsonl")
    parser.add_argument("--output_dir", default="outputs/sft_qwen05b")
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--use_lora", action="store_true", default=True)
    args = parser.parse_args()

    ds = load_dataset("json", data_files=args.train_path, split="train")

    def format_example(example: dict) -> str:
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

    train_args = SFTConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=100,
        max_seq_length=1024,
        bf16=True,
        report_to="wandb",
        run_name=Path(args.output_dir).name,
    )

    trainer = SFTTrainer(
        model=args.model_name,
        args=train_args,
        train_dataset=ds,
        formatting_func=format_example,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
