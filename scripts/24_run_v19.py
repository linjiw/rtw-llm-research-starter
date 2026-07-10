#!/usr/bin/env python
"""Fail-closed orchestrator for v0.19 preflight, training, and validation."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from rtw_llm.provenance import verify_completed_run
from rtw_llm.v19_protocol import (
    ARM_SPECS,
    CONFIRMATION_READY_RECORD,
    CONFIRM_SCORE_MANIFEST,
    CONFIRM_SCORE_REPORT,
    DEV_SCORE_MANIFEST,
    DEV_SCORE_REPORT,
    EVAL_CONFIG,
    MODEL_NAME,
    MODEL_REVISION,
    PROTOCOL_DIR,
    PROTOCOL_ID,
    TRAIN_PATH,
    TRAINING_SEEDS,
    TRUE_SEED_PROTOCOL,
    VALIDATION_PATH,
    VIEW_FILES,
    eval_dir,
    training_dir,
    validate_production_gate,
    validate_confirmation_ready,
    validate_score_artifact,
    verify_v19_training_health,
)


def consumed_single_gpu_hours(runs_root: Path) -> float:
    seconds = 0.0
    if not runs_root.exists():
        return 0.0
    for path in runs_root.rglob("run_result.json"):
        payload = json.loads(path.read_text())
        value = payload.get("observations", {}).get("wall_clock_seconds", 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise RuntimeError(f"Malformed wall-clock observation: {path}")
        seconds += float(value)
    return seconds / 3600.0


def production_train_commands(python: str, runs_root: Path, seed: int) -> list[dict[str, Any]]:
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"Production seed must be one of {TRAINING_SEEDS}")
    common = [
        python,
        "scripts/01_sft_warmup.py",
        "--model_name",
        MODEL_NAME,
        "--model_revision",
        MODEL_REVISION,
        "--train_path",
        TRAIN_PATH.as_posix(),
        "--output_dir",
        str(training_dir(runs_root, "sft_only", seed)),
        "--max_steps",
        "313",
        "--batch_size",
        "2",
        "--grad_accum",
        "8",
        "--learning_rate",
        "5e-5",
        "--seed",
        str(seed),
        "--seed_protocol",
        TRUE_SEED_PROTOCOL,
        "--completion_only_loss",
        "--strict_provenance",
        "--experiment_protocol",
        PROTOCOL_ID,
        "--report_to",
        "none",
    ]
    commands = [
        {
            "name": f"sft_seed{seed}",
            "command": common,
            "output_dir": training_dir(runs_root, "sft_only", seed),
            "health": {"run_kind": "sft", "expected_steps": 313},
        }
    ]
    for arm in ("grpo_static", "grpo_stable", "sft_grpo_static", "sft_grpo_stable"):
        spec = ARM_SPECS[arm]
        command = [
            python,
            "scripts/02_grpo_train.py",
            "--model_name",
            MODEL_NAME,
            "--model_revision",
            MODEL_REVISION,
            "--train_path",
            TRAIN_PATH.as_posix(),
            "--output_dir",
            str(training_dir(runs_root, arm, seed)),
            "--reward_strategy",
            spec["reward_strategy"],
            "--method_arm",
            arm,
            "--seed",
            str(seed),
            "--trainer_seed",
            str(seed),
            "--seed_protocol",
            TRUE_SEED_PROTOCOL,
            "--max_steps",
            "300",
            "--batch_size",
            "2",
            "--grad_accum",
            "8",
            "--num_generations",
            "4",
            "--max_prompt_length",
            "768",
            "--max_completion_length",
            "256",
            "--task_curriculum",
            "uniform",
            "--prompt_field",
            "prompt",
            "--strict_provenance",
            "--experiment_protocol",
            PROTOCOL_ID,
            "--report_to",
            "none",
        ]
        if spec["sft_parent"]:
            command.extend(
                ["--init_adapter_path", str(training_dir(runs_root, "sft_only", seed))]
            )
        commands.append(
            {
                "name": f"{arm}_seed{seed}",
                "command": command,
                "output_dir": training_dir(runs_root, arm, seed),
                "health": {
                    "run_kind": "grpo",
                    "expected_steps": 300,
                    "expected_strategy": spec["reward_strategy"],
                    "require_group_variance": True,
                },
            }
        )
    return commands


def _eval_command(
    python: str,
    runs_root: Path,
    *,
    view: str,
    arm: str,
    seed: int | None,
    device: str,
) -> dict[str, Any]:
    output = eval_dir(runs_root, view, arm, seed)
    command = [
        python,
        "scripts/07_best_of_n_rerank.py",
        "--model_name",
        MODEL_NAME,
        "--model_revision",
        MODEL_REVISION,
        "--data_path",
        VALIDATION_PATH.as_posix(),
        "--task_ids_file",
        (PROTOCOL_DIR / VIEW_FILES[view]).as_posix(),
        "--output_dir",
        str(output),
        "--engine",
        "hf",
        "--hf_gen_mode",
        EVAL_CONFIG["hf_gen_mode"],
        "--device",
        device,
        "--prompt_field",
        "prompt",
        "--batch_size",
        str(EVAL_CONFIG["batch_size"]),
        "--n_values",
        *[str(value) for value in EVAL_CONFIG["n_values"]],
        "--max_n",
        str(EVAL_CONFIG["max_n"]),
        "--max_new_tokens",
        str(EVAL_CONFIG["max_new_tokens"]),
        "--temperature",
        str(EVAL_CONFIG["temperature"]),
        "--top_p",
        str(EVAL_CONFIG["top_p"]),
        "--seed",
        str(EVAL_CONFIG["sampling_seed"]),
        "--method",
        arm,
        "--training_protocol",
        TRUE_SEED_PROTOCOL,
        "--experiment_protocol",
        PROTOCOL_ID,
        "--split",
        view,
        "--strict_provenance",
        "--skip_if_complete",
    ]
    if arm != "base":
        command.extend(
            [
                "--adapter_path",
                str(training_dir(runs_root, arm, int(seed))),
                "--training_seed",
                str(seed),
            ]
        )
    if view == "validation_confirm400":
        command.extend(
            ["--confirmation_ready_record", CONFIRMATION_READY_RECORD.as_posix()]
        )
    return {"name": output.name, "command": command, "output_dir": output, "eval": True}


def production_eval_commands(
    python: str,
    runs_root: Path,
    *,
    view: str,
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    commands = [
        _eval_command(python, runs_root, view=view, arm="base", seed=None, device="cuda")
    ]
    for arm in ARM_SPECS:
        if arm == "base":
            continue
        commands.extend(
            _eval_command(python, runs_root, view=view, arm=arm, seed=seed, device="cuda")
            for seed in seeds
        )
    return commands


def preflight_commands(
    python: str, repo_root: Path, run_label: str, device: str
) -> list[dict[str, Any]]:
    root = Path("outputs/v19/preflight") / run_label
    sft = root / "train/sft_seed0_step1"
    fresh = root / "train/grpo_stable_seed0_step1"
    combined = root / "train/sft_grpo_stable_seed0_step1"
    common_seed = ["--seed", "0", "--seed_protocol", TRUE_SEED_PROTOCOL]
    commands = [
        {
            "name": "preflight_sft",
            "command": [
                python,
                "scripts/01_sft_warmup.py",
                "--model_name",
                MODEL_NAME,
                "--model_revision",
                MODEL_REVISION,
                "--train_path",
                TRAIN_PATH.as_posix(),
                "--output_dir",
                str(sft),
                "--max_steps",
                "1",
                "--batch_size",
                "2",
                "--grad_accum",
                "1",
                "--learning_rate",
                "5e-5",
                *common_seed,
                "--completion_only_loss",
                "--strict_provenance",
                "--report_to",
                "none",
            ],
            "output_dir": sft,
            "health": {"run_kind": "sft", "expected_steps": 1},
        }
    ]
    for name, output, parent in (
        ("preflight_grpo_fresh", fresh, None),
        ("preflight_grpo_from_sft", combined, sft),
    ):
        command = [
            python,
            "scripts/02_grpo_train.py",
            "--model_name",
            MODEL_NAME,
            "--model_revision",
            MODEL_REVISION,
            "--train_path",
            TRAIN_PATH.as_posix(),
            "--output_dir",
            str(output),
            "--reward_strategy",
            "adaptive_stable",
            "--max_steps",
            "1",
            "--batch_size",
            "2",
            "--grad_accum",
            "1",
            "--num_generations",
            "2",
            "--max_prompt_length",
            "768",
            "--max_completion_length",
            "32",
            "--seed",
            "0",
            "--trainer_seed",
            "0",
            "--seed_protocol",
            TRUE_SEED_PROTOCOL,
            "--strict_provenance",
            "--report_to",
            "none",
        ]
        if parent is not None:
            command.extend(["--init_adapter_path", str(parent)])
        commands.append(
            {
                "name": name,
                "command": command,
                "output_dir": output,
                "health": {
                    "run_kind": "grpo",
                    "expected_steps": 1,
                    "expected_strategy": "adaptive_stable",
                    "require_group_variance": False,
                },
            }
        )
    for name, adapter in (("preflight_eval_base", None), ("preflight_eval_combined", combined)):
        output = root / f"eval/{name}"
        command = [
            python,
            "scripts/07_best_of_n_rerank.py",
            "--model_name",
            MODEL_NAME,
            "--model_revision",
            MODEL_REVISION,
            "--data_path",
            VALIDATION_PATH.as_posix(),
            "--task_ids_file",
            (PROTOCOL_DIR / VIEW_FILES["validation_preflight2"]).as_posix(),
            "--output_dir",
            str(output),
            "--engine",
            "hf",
            "--hf_gen_mode",
            "batched",
            "--device",
            device,
            "--batch_size",
            "4",
            "--n_values",
            "1",
            "2",
            "--max_n",
            "2",
            "--max_new_tokens",
            "64",
            "--temperature",
            "0.7",
            "--top_p",
            "0.95",
            "--seed",
            "0",
            "--method",
            name,
            "--training_protocol",
            TRUE_SEED_PROTOCOL,
            "--split",
            "validation_preflight2",
            "--strict_provenance",
        ]
        if adapter is not None:
            command.extend(["--adapter_path", str(adapter), "--training_seed", "0"])
        commands.append({"name": name, "command": command, "output_dir": output, "eval": True})
    return commands


def _require_all_training(repo_root: Path) -> None:
    validate_confirmation_ready(CONFIRMATION_READY_RECORD, repo_root=repo_root)


def execute_commands(
    commands: list[dict[str, Any]],
    *,
    repo_root: Path,
    log_root: Path,
    execute: bool,
    budget: Mapping[str, Any] | None = None,
    runs_root: Path | None = None,
) -> None:
    log_root.mkdir(parents=True, exist_ok=True)
    for item in commands:
        rendered = shlex.join(item["command"])
        print(f"[{item['name']}] {rendered}")
        if not execute:
            continue
        if budget is not None:
            if runs_root is None:
                raise RuntimeError("Production budget enforcement requires runs_root")
            used = consumed_single_gpu_hours(repo_root / runs_root)
            cost = used * float(budget["usd_per_gpu_hour"])
            if used >= float(budget["max_single_gpu_hours"]) or cost >= float(
                budget["max_cost_usd"]
            ):
                raise RuntimeError(
                    f"Production budget exhausted before {item['name']}: "
                    f"gpu_hours={used:.3f} cost_usd={cost:.2f}"
                )
        log_path = log_root / f"{item['name']}.log"
        with log_path.open("x") as log:
            result = subprocess.run(
                item["command"],
                cwd=repo_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if result.returncode:
            tail = "\n".join(log_path.read_text().splitlines()[-30:])
            raise RuntimeError(f"{item['name']} failed; log={log_path}\n{tail}")
        output_dir = repo_root / item["output_dir"] if item.get("output_dir") else None
        if item.get("eval"):
            assert output_dir is not None
            verify_completed_run(
                output_dir,
                required_artifact_roles={"candidates", "metrics", "run_config", "summary"},
            )
        elif item.get("score"):
            validate_score_artifact(
                repo_root=repo_root,
                runs_root=runs_root,
                report_path=item["score"]["report"],
                manifest_path=item["score"]["manifest"],
                expected_view=item["score"]["view"],
                expected_seeds=item["score"]["seeds"],
            )
        elif item.get("health"):
            assert output_dir is not None
            health = verify_v19_training_health(output_dir, **item["health"])
            (log_root / f"{item['name']}.health.json").write_text(
                json.dumps(health, indent=2, sort_keys=True) + "\n"
            )
        if budget is not None:
            used = consumed_single_gpu_hours(repo_root / runs_root)
            cost = used * float(budget["usd_per_gpu_hour"])
            if used > float(budget["max_single_gpu_hours"]) or cost > float(
                budget["max_cost_usd"]
            ):
                raise RuntimeError(
                    f"Production budget reached after {item['name']}; stopping: "
                    f"gpu_hours={used:.3f} cost_usd={cost:.2f}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=[
            "preflight",
            "train_seed",
            "eval_dev",
            "score_dev",
            "prepare_confirm",
            "eval_confirm",
            "score_confirm",
        ],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run_label", default=None)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--runs_root", type=Path, default=Path("outputs/v19/production"))
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    runs_root = args.runs_root
    budget = None
    if args.stage == "preflight":
        if not args.run_label:
            raise ValueError("Preflight requires a unique --run_label")
        commands = preflight_commands(args.python, repo_root, args.run_label, args.device)
        log_root = repo_root / "outputs/v19/logs/preflight" / args.run_label
    else:
        if args.execute:
            budget = validate_production_gate(repo_root)
        if args.stage == "train_seed":
            if args.seed not in TRAINING_SEEDS:
                raise ValueError(f"train_seed requires --seed in {TRAINING_SEEDS}")
            commands = production_train_commands(args.python, runs_root, int(args.seed))
            log_root = repo_root / "outputs/v19/logs/production" / f"train_seed{args.seed}"
        elif args.stage == "eval_dev":
            if args.seed not in (None, 0):
                raise ValueError("Only seed 0 is evaluated on the development view")
            commands = production_eval_commands(
                args.python, runs_root, view="validation_dev100", seeds=(0,)
            )
            log_root = repo_root / "outputs/v19/logs/production/eval_dev_seed0"
        elif args.stage == "score_dev":
            commands = [
                {
                    "name": "score_validation_dev100_seed0",
                    "command": [
                        args.python,
                        "scripts/22_score_v19.py",
                        "--runs_root",
                        str(runs_root),
                        "--view",
                        "validation_dev100",
                        "--development_seed0",
                        "--out_json",
                        DEV_SCORE_REPORT.as_posix(),
                    ],
                    "score": {
                        "report": DEV_SCORE_REPORT,
                        "manifest": DEV_SCORE_MANIFEST,
                        "view": "validation_dev100",
                        "seeds": (0,),
                    },
                }
            ]
            log_root = repo_root / "outputs/v19/logs/production/score_dev_seed0"
        elif args.stage == "prepare_confirm":
            commands = [
                {
                    "name": "prepare_confirmation_ready",
                    "command": [
                        args.python,
                        "scripts/27_prepare_v19_confirmation.py",
                        "--runs_root",
                        str(runs_root),
                    ],
                }
            ]
            log_root = repo_root / "outputs/v19/logs/production/prepare_confirm"
        elif args.stage == "score_confirm":
            if args.execute:
                validate_confirmation_ready(CONFIRMATION_READY_RECORD, repo_root=repo_root)
            commands = [
                {
                    "name": "score_validation_confirm400",
                    "command": [
                        args.python,
                        "scripts/22_score_v19.py",
                        "--runs_root",
                        str(runs_root),
                        "--view",
                        "validation_confirm400",
                        "--out_json",
                        CONFIRM_SCORE_REPORT.as_posix(),
                    ],
                    "score": {
                        "report": CONFIRM_SCORE_REPORT,
                        "manifest": CONFIRM_SCORE_MANIFEST,
                        "view": "validation_confirm400",
                        "seeds": TRAINING_SEEDS,
                    },
                }
            ]
            log_root = repo_root / "outputs/v19/logs/production/score_confirm"
        else:
            if args.execute:
                _require_all_training(repo_root)
            commands = production_eval_commands(
                args.python,
                runs_root,
                view="validation_confirm400",
                seeds=TRAINING_SEEDS,
            )
            log_root = repo_root / "outputs/v19/logs/production/eval_confirm"
    print(json.dumps({"stage": args.stage, "execute": args.execute, "jobs": len(commands)}))
    execute_commands(
        commands,
        repo_root=repo_root,
        log_root=log_root,
        execute=args.execute,
        budget=budget,
        runs_root=runs_root if budget is not None else None,
    )


if __name__ == "__main__":
    main()
