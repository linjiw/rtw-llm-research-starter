"""Frozen protocol helpers for the Countdown-v2 within-v2 baseline reset."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .provenance import (
    RESULT_NAME,
    adapter_record,
    content_sha256,
    file_record,
    runtime_record,
    verify_completed_run,
)

PROTOCOL_ID = "countdown-v19-within-v2-reset-v1"
PROTOCOL_SCHEMA = "countdown-v19-protocol-manifest-v1"
ENVIRONMENT_SCHEMA = "countdown-v19-production-environment-v1"
LAUNCH_SCHEMA = "countdown-v19-production-launch-v1"
TEST_RELEASE_SCHEMA = "countdown-v19-test-release-v1"
SCORE_MANIFEST_SCHEMA = "countdown-v19-score-artifact-v1"
CONFIRMATION_READY_SCHEMA = "countdown-v19-confirmation-ready-v1"

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
TRUE_SEED_PROTOCOL = "countdown-true-seeds-v2"
TRAINING_SEEDS = (0, 1, 2)
VIEW_DOMAIN = "countdown-v19-validation-dev-v1"

PROTOCOL_DIR = Path("protocols/countdown_v2/v19")
PROTOCOL_MANIFEST = PROTOCOL_DIR / "manifest.json"
ENVIRONMENT_LOCK = PROTOCOL_DIR / "production_environment.json"
LAUNCH_RECORD = PROTOCOL_DIR / "production_launch.json"
CONFIRMATION_READY_RECORD = PROTOCOL_DIR / "confirmation_ready.json"
DEV_SCORE_REPORT = Path("outputs/v19/production/scores/validation_dev100_seed0.json")
DEV_SCORE_MANIFEST = Path(str(DEV_SCORE_REPORT) + ".manifest.json")
CONFIRM_SCORE_REPORT = Path("outputs/v19/production/scores/validation_confirm400.json")
CONFIRM_SCORE_MANIFEST = Path(str(CONFIRM_SCORE_REPORT) + ".manifest.json")
VALIDATION_PATH = Path("data/countdown_v2/validation.jsonl")
TRAIN_PATH = Path("data/countdown_v2/train.jsonl")
TEST_PATH = Path("data/countdown_v2/test_in_dist.jsonl")
TEST_IDS_PATH = Path("data/countdown_v2/task_ids/test_in_dist.txt")

VIEW_QUOTAS: dict[str, dict[str, int]] = {
    "validation_dev100": {"easy": 10, "medium": 45, "hard": 45},
    "validation_confirm400": {"easy": 40, "medium": 180, "hard": 180},
}
SOURCE_VALIDATION_QUOTAS = {"easy": 50, "medium": 225, "hard": 225}
VIEW_FILES = {
    "validation_dev100": "validation_dev100.txt",
    "validation_confirm400": "validation_confirm400.txt",
    "validation_preflight2": "validation_preflight2.txt",
}
SOURCE_PATHS = (
    "docs/V19_WITHIN_V2_BASELINE_RESET_PLAN.md",
    "scripts/20_generate_v19_protocol.py",
    "scripts/21_audit_v19_protocol.py",
    "scripts/22_score_v19.py",
    "scripts/23_capture_v19_environment.py",
    "scripts/24_run_v19.py",
    "scripts/25_create_v19_launch_record.py",
    "scripts/27_prepare_v19_confirmation.py",
    "scripts/01_sft_warmup.py",
    "scripts/02_grpo_train.py",
    "scripts/07_best_of_n_rerank.py",
    "src/rtw_llm/data_access.py",
    "src/rtw_llm/engine.py",
    "src/rtw_llm/provenance.py",
    "src/rtw_llm/v19_protocol.py",
)


def protocol_source_paths(repo_root: str | Path) -> tuple[str, ...]:
    root = Path(repo_root).resolve()
    package_sources = {
        path.relative_to(root).as_posix() for path in (root / "src/rtw_llm").glob("*.py")
    }
    return tuple(sorted(set(SOURCE_PATHS) | package_sources))

SFT_CONFIG: dict[str, Any] = {
    "max_steps": 313,
    "batch_size": 2,
    "grad_accum": 8,
    "learning_rate": 5e-5,
    "completion_only_loss": True,
    "max_length": 1024,
    "packing": False,
    "shuffle_dataset": False,
    "lr_scheduler_type": "linear",
    "warmup_steps": 0,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "gradient_checkpointing": True,
    "dataloader_drop_last": False,
    "save_steps": 100,
    "world_size": 1,
}
GRPO_CONFIG: dict[str, Any] = {
    "max_steps": 300,
    "batch_size": 2,
    "grad_accum": 8,
    "learning_rate": 5e-6,
    "num_generations": 4,
    "max_prompt_length": 768,
    "max_completion_length": 256,
    "task_curriculum": "uniform",
    "eval_strategy": "no",
    "generation_batch_size": 16,
    "steps_per_generation": 8,
    "num_iterations": 1,
    "loss_type": "dapo",
    "scale_rewards": "group",
    "beta": 0.0,
    "lr_scheduler_type": "linear",
    "warmup_steps": 0,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "gradient_checkpointing": True,
    "dataloader_drop_last": False,
    "save_steps": 100,
    "world_size": 1,
}
EVAL_CONFIG: dict[str, Any] = {
    "engine": "hf",
    "hf_gen_mode": "batched",
    "batch_size": 16,
    "prompt_field": "prompt",
    "sampling_seed": 0,
    "temperature": 0.7,
    "top_p": 0.95,
    "max_new_tokens": 256,
    "max_n": 8,
    "n_values": [1, 4, 8],
}
INFERENCE_CONFIG: dict[str, Any] = {
    "primary_endpoint": "practical_reranked_exact_at_8",
    "primary_contrast": "sft_grpo_stable_minus_sft_grpo_static",
    "alpha_two_sided": 0.05,
    "confidence": 0.95,
    "bootstrap_draws": 20_000,
    "sign_flip_draws": 20_000,
    "random_seed": 17,
    "secondary_multiplicity": "holm_familywise_0.05",
}

ARM_SPECS: dict[str, dict[str, Any]] = {
    "base": {"family": "untrained", "training_kind": "base", "seeds": [None]},
    "sft_only": {"family": "capability", "training_kind": "sft", "seeds": [0, 1, 2]},
    "grpo_static": {
        "family": "rl_only",
        "training_kind": "grpo",
        "reward_strategy": "static",
        "sft_parent": False,
        "seeds": [0, 1, 2],
    },
    "grpo_stable": {
        "family": "rl_only",
        "training_kind": "grpo",
        "reward_strategy": "adaptive_stable",
        "sft_parent": False,
        "seeds": [0, 1, 2],
    },
    "sft_grpo_static": {
        "family": "sft_initialized",
        "training_kind": "grpo",
        "reward_strategy": "static",
        "sft_parent": True,
        "seeds": [0, 1, 2],
    },
    "sft_grpo_stable": {
        "family": "sft_initialized",
        "training_kind": "grpo",
        "reward_strategy": "adaptive_stable",
        "sft_parent": True,
        "seeds": [0, 1, 2],
    },
}

PRIMARY_CONTRASTS = [
    ("sft_grpo_stable_minus_sft_grpo_static", "sft_grpo_stable", "sft_grpo_static"),
    ("grpo_stable_minus_grpo_static", "grpo_stable", "grpo_static"),
    ("sft_only_minus_base", "sft_only", "base"),
    ("sft_grpo_stable_minus_sft_only", "sft_grpo_stable", "sft_only"),
    ("sft_grpo_static_minus_sft_only", "sft_grpo_static", "sft_only"),
]


class V19ProtocolError(RuntimeError):
    """Raised when a v0.19 artifact or execution gate fails closed."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise V19ProtocolError(f"Cannot read v0.19 JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise V19ProtocolError(f"Expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise V19ProtocolError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise V19ProtocolError(f"Cannot read v0.19 source JSONL {path}: {exc}") from exc
    return rows


def state_labels() -> list[str]:
    labels = ["base"]
    for arm, spec in ARM_SPECS.items():
        if arm == "base":
            continue
        labels.extend(f"{arm}_seed{seed}" for seed in spec["seeds"])
    return labels


def state_label(arm: str, seed: int | None) -> str:
    if arm not in ARM_SPECS:
        raise V19ProtocolError(f"Unknown v0.19 arm: {arm}")
    if arm == "base":
        if seed is not None:
            raise V19ProtocolError("The base state has no training seed")
        return "base"
    if seed not in TRAINING_SEEDS:
        raise V19ProtocolError(f"Arm {arm} requires a true seed in {TRAINING_SEEDS}")
    return f"{arm}_seed{seed}"


def validate_v19_sft_args(args: Mapping[str, Any]) -> None:
    expected = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "train_path": TRAIN_PATH.as_posix(),
        "eval_path": None,
        "max_steps": SFT_CONFIG["max_steps"],
        "batch_size": SFT_CONFIG["batch_size"],
        "grad_accum": SFT_CONFIG["grad_accum"],
        "learning_rate": SFT_CONFIG["learning_rate"],
        "completion_only_loss": True,
        "seed_protocol": TRUE_SEED_PROTOCOL,
        "strict_provenance": True,
    }
    mismatches = [key for key, value in expected.items() if args.get(key) != value]
    if args.get("seed") not in TRAINING_SEEDS:
        mismatches.append("seed")
    if mismatches:
        raise V19ProtocolError(f"V0.19 SFT invocation mismatch: {sorted(set(mismatches))}")


def validate_v19_grpo_args(args: Mapping[str, Any]) -> None:
    arm = args.get("method_arm")
    if arm not in {"grpo_static", "grpo_stable", "sft_grpo_static", "sft_grpo_stable"}:
        raise V19ProtocolError(f"V0.19 GRPO requires an explicit method_arm, found {arm!r}")
    spec = ARM_SPECS[str(arm)]
    expected = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "train_path": TRAIN_PATH.as_posix(),
        "eval_path": None,
        "reward_strategy": spec["reward_strategy"],
        "max_steps": GRPO_CONFIG["max_steps"],
        "learning_rate": GRPO_CONFIG["learning_rate"],
        "batch_size": GRPO_CONFIG["batch_size"],
        "grad_accum": GRPO_CONFIG["grad_accum"],
        "num_generations": GRPO_CONFIG["num_generations"],
        "max_prompt_length": GRPO_CONFIG["max_prompt_length"],
        "max_completion_length": GRPO_CONFIG["max_completion_length"],
        "task_curriculum": "uniform",
        "prompt_field": "prompt",
        "seed_protocol": TRUE_SEED_PROTOCOL,
        "strict_provenance": True,
    }
    mismatches = [key for key, value in expected.items() if args.get(key) != value]
    seed = args.get("seed")
    if seed not in TRAINING_SEEDS or args.get("trainer_seed") != seed:
        mismatches.extend(["seed", "trainer_seed"])
    has_parent = args.get("init_adapter_path") is not None
    if has_parent != bool(spec["sft_parent"]):
        mismatches.append("init_adapter_path")
    if mismatches:
        raise V19ProtocolError(f"V0.19 GRPO invocation mismatch: {sorted(set(mismatches))}")


def validate_v19_eval_args(args: Mapping[str, Any]) -> None:
    expected = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "engine": EVAL_CONFIG["engine"],
        "hf_gen_mode": EVAL_CONFIG["hf_gen_mode"],
        "batch_size": EVAL_CONFIG["batch_size"],
        "prompt_field": EVAL_CONFIG["prompt_field"],
        "seed": EVAL_CONFIG["sampling_seed"],
        "temperature": EVAL_CONFIG["temperature"],
        "top_p": EVAL_CONFIG["top_p"],
        "max_new_tokens": EVAL_CONFIG["max_new_tokens"],
        "max_n": EVAL_CONFIG["max_n"],
        "training_protocol": TRUE_SEED_PROTOCOL,
        "strict_provenance": True,
        "device": "cuda",
    }
    mismatches = [key for key, value in expected.items() if args.get(key) != value]
    if sorted(args.get("n_values", [])) != EVAL_CONFIG["n_values"]:
        mismatches.append("n_values")
    if args.get("method") not in ARM_SPECS:
        mismatches.append("method")
    if args.get("method") == "base":
        if args.get("adapter_path") is not None or args.get("training_seed") is not None:
            mismatches.extend(["adapter_path", "training_seed"])
    elif args.get("training_seed") not in TRAINING_SEEDS or args.get("adapter_path") is None:
        mismatches.extend(["adapter_path", "training_seed"])
    split = args.get("split")
    if split in VIEW_FILES:
        expected_ids = (PROTOCOL_DIR / VIEW_FILES[str(split)]).as_posix()
        if args.get("data_path") != VALIDATION_PATH.as_posix():
            mismatches.append("data_path")
        if args.get("task_ids_file") != expected_ids or args.get("limit") is not None:
            mismatches.extend(["task_ids_file", "limit"])
    elif split == "test_in_dist":
        if args.get("data_path") != TEST_PATH.as_posix():
            mismatches.append("data_path")
        if args.get("task_ids_file") is not None or args.get("limit") is not None:
            mismatches.extend(["task_ids_file", "limit"])
        if args.get("test_release_record") is None:
            mismatches.append("test_release_record")
    else:
        mismatches.append("split")
    if mismatches:
        raise V19ProtocolError(f"V0.19 evaluation invocation mismatch: {sorted(set(mismatches))}")


def _membership_key(task_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{VIEW_DOMAIN}\0{task_id}".encode("utf-8")).hexdigest()
    return digest, task_id


def build_validation_views(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    if len(rows) != 500:
        raise V19ProtocolError(f"Validation source must contain 500 rows, found {len(rows)}")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(task_id, str) or not task_id for task_id in ids):
        raise V19ProtocolError("Validation source contains a missing/non-string ID")
    if len(ids) != len(set(ids)):
        raise V19ProtocolError("Validation source contains duplicate IDs")
    counts = Counter(row.get("difficulty") for row in rows)
    if counts != Counter(SOURCE_VALIDATION_QUOTAS):
        raise V19ProtocolError(
            f"Validation tier quotas mismatch: actual={dict(counts)} "
            f"expected={SOURCE_VALIDATION_QUOTAS}"
        )

    dev_members: set[str] = set()
    for tier, quota in VIEW_QUOTAS["validation_dev100"].items():
        tier_ids = sorted(
            (str(row["id"]) for row in rows if row.get("difficulty") == tier),
            key=_membership_key,
        )
        dev_members.update(tier_ids[:quota])
    source_order = [str(row["id"]) for row in rows]
    dev = [task_id for task_id in source_order if task_id in dev_members]
    confirm = [task_id for task_id in source_order if task_id not in dev_members]
    preflight = dev[:2]
    views = {
        "validation_dev100": dev,
        "validation_confirm400": confirm,
        "validation_preflight2": preflight,
    }
    _validate_view_membership(rows, views)
    return views


def _validate_view_membership(
    rows: Sequence[Mapping[str, Any]], views: Mapping[str, Sequence[str]]
) -> None:
    by_id = {str(row["id"]): row for row in rows}
    dev = list(views.get("validation_dev100", []))
    confirm = list(views.get("validation_confirm400", []))
    preflight = list(views.get("validation_preflight2", []))
    source_order = [str(row["id"]) for row in rows]
    if set(dev) & set(confirm) or set(dev) | set(confirm) != set(source_order):
        raise V19ProtocolError("Development and confirmation views must be disjoint and complete")
    if len(dev) != 100 or len(confirm) != 400:
        raise V19ProtocolError("Validation view sizes must be exactly 100 and 400")
    for name, expected in VIEW_QUOTAS.items():
        actual = Counter(by_id[task_id]["difficulty"] for task_id in views[name])
        if actual != Counter(expected):
            raise V19ProtocolError(
                f"{name} tier quotas mismatch: actual={dict(actual)} expected={expected}"
            )
    if preflight != dev[:2]:
        raise V19ProtocolError("Preflight view must be the first two ordered dev IDs")
    for name in ("validation_dev100", "validation_confirm400"):
        expected_order = [task_id for task_id in source_order if task_id in set(views[name])]
        if list(views[name]) != expected_order:
            raise V19ProtocolError(f"{name} does not preserve source validation row order")


def _ids_bytes(values: Sequence[str]) -> bytes:
    return "".join(f"{value}\n" for value in values).encode("utf-8")


def _artifact_record_from_bytes(value: bytes) -> dict[str, Any]:
    line_count = value.count(b"\n")
    if value and not value.endswith(b"\n"):
        line_count += 1
    return {
        "sha256": hashlib.sha256(value).hexdigest(),
        "size": len(value),
        "line_count": line_count,
    }


def build_protocol_artifacts(repo_root: str | Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    root = Path(repo_root).resolve()
    validation_path = root / VALIDATION_PATH
    dataset_manifest_path = root / "data/countdown_v2/manifest.json"
    rows = _read_jsonl(validation_path)
    views = build_validation_views(rows)
    artifacts = {VIEW_FILES[name]: _ids_bytes(values) for name, values in views.items()}
    core = {
        "schema_version": PROTOCOL_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "dataset": {
            "protocol_id": "countdown-dataset-v2",
            "validation": file_record(validation_path),
            "dataset_manifest": file_record(dataset_manifest_path),
            "dataset_manifest_core_sha256": _read_json(dataset_manifest_path).get(
                "manifest_core_sha256"
            ),
        },
        "protocol_sources": {
            path: file_record(root / path) for path in protocol_source_paths(root)
        },
        "model": {"name": MODEL_NAME, "revision": MODEL_REVISION},
        "seed_protocol": TRUE_SEED_PROTOCOL,
        "training_seeds": list(TRAINING_SEEDS),
        "validation_views": {
            "membership_domain": VIEW_DOMAIN,
            "membership_order": "sha256(domain+NUL+id), then id",
            "published_order": "source_validation_jsonl_row_order",
            "quotas": VIEW_QUOTAS,
            "preflight_source": "first_two_validation_dev100_ids",
        },
        "matrix": ARM_SPECS,
        "state_labels": state_labels(),
        "sft_config": SFT_CONFIG,
        "grpo_config": GRPO_CONFIG,
        "evaluation_config": EVAL_CONFIG,
        "inference_config": INFERENCE_CONFIG,
        "contrasts": [
            {"name": name, "arm": arm, "baseline": baseline}
            for name, arm, baseline in PRIMARY_CONTRASTS
        ],
        "production_environment_lock": ENVIRONMENT_LOCK.as_posix(),
        "production_launch_record": LAUNCH_RECORD.as_posix(),
        "confirmation_ready_record": CONFIRMATION_READY_RECORD.as_posix(),
        "development_score_artifacts": {
            "report": DEV_SCORE_REPORT.as_posix(),
            "manifest": DEV_SCORE_MANIFEST.as_posix(),
        },
        "confirmation_score_artifacts": {
            "report": CONFIRM_SCORE_REPORT.as_posix(),
            "manifest": CONFIRM_SCORE_MANIFEST.as_posix(),
        },
        "test_release_created": False,
        "final_test_release_created": False,
        "artifacts": {
            name: _artifact_record_from_bytes(value) for name, value in sorted(artifacts.items())
        },
        "artifacts_exclude_manifest_and_future_release_records": True,
    }
    manifest = {**core, "manifest_core_sha256": content_sha256(core)}
    return artifacts, manifest


def write_protocol_atomic(
    output_dir: str | Path, *, artifacts: Mapping[str, bytes], manifest: Mapping[str, Any]
) -> None:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite pre-existing v0.19 protocol: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        for relative, value in artifacts.items():
            path = temp / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        (temp / "manifest.json").write_bytes(canonical_json_bytes(dict(manifest)))
        temp.rename(output)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def audit_protocol(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    output = root / PROTOCOL_DIR
    failures: list[str] = []
    try:
        expected_artifacts, expected_manifest = build_protocol_artifacts(root)
        actual_manifest = _read_json(output / "manifest.json")
        if actual_manifest != expected_manifest:
            failures.append("protocol manifest differs from deterministic replay")
        for name, expected in expected_artifacts.items():
            path = output / name
            if not path.is_file() or path.read_bytes() != expected:
                failures.append(f"artifact differs from deterministic replay: {name}")
        allowed = {
            "manifest.json",
            *expected_artifacts,
            ENVIRONMENT_LOCK.name,
            LAUNCH_RECORD.name,
            CONFIRMATION_READY_RECORD.name,
        }
        extras = sorted(
            path.relative_to(output).as_posix()
            for path in output.rglob("*")
            if path.is_file() and path.relative_to(output).as_posix() not in allowed
        )
        if extras:
            failures.append(f"unexpected protocol files: {extras}")
    except Exception as exc:
        failures.append(f"audit exception: {type(exc).__name__}: {exc}")
    return {
        "schema_version": "countdown-v19-protocol-audit-v1",
        "protocol_id": PROTOCOL_ID,
        "status": "ELIGIBLE" if not failures else "INTEGRITY_FAIL",
        "eligible": not failures,
        "failures": failures,
    }


def require_eligible_protocol(repo_root: str | Path) -> None:
    report = audit_protocol(repo_root)
    if not report["eligible"]:
        raise V19ProtocolError(f"V0.19 protocol audit is not eligible: {report['failures']}")


def validate_source_commit(repo_root: str | Path, commit: str) -> None:
    root = Path(repo_root).resolve()
    if not isinstance(commit, str) or len(commit) != 40:
        raise V19ProtocolError(f"Invalid run source commit: {commit!r}")
    manifest = _read_json(root / PROTOCOL_MANIFEST)
    sources = manifest.get("protocol_sources")
    if not isinstance(sources, dict) or not sources:
        raise V19ProtocolError("Protocol manifest has no source fingerprint")
    mismatches = []
    for relative, expected in sources.items():
        try:
            value = subprocess.run(
                ["git", "show", f"{commit}:{relative}"],
                cwd=root,
                check=True,
                capture_output=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            mismatches.append(relative)
            continue
        if _artifact_record_from_bytes(value) != expected:
            mismatches.append(relative)
    if mismatches:
        raise V19ProtocolError(
            f"Run commit does not match the registered source fingerprint: {mismatches[:5]}"
        )


def validate_run_source_identity(identity: Mapping[str, Any], repo_root: str | Path) -> None:
    git = identity.get("git")
    if not isinstance(git, Mapping) or git.get("dirty") is not False:
        raise V19ProtocolError("Run identity is not bound to a clean source commit")
    validate_source_commit(repo_root, str(git.get("commit")))


def protocol_manifest_record(repo_root: str | Path) -> dict[str, Any]:
    return file_record(Path(repo_root) / PROTOCOL_MANIFEST)


def capture_environment_lock(
    repo_root: str | Path, *, container_image_digest: str | None = None
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    require_eligible_protocol(root)
    actual = runtime_record()
    hardware = actual.get("hardware", {})
    if not hardware.get("cuda_available") or len(hardware.get("cuda_devices", [])) != 1:
        raise V19ProtocolError("Production environment capture requires exactly one visible CUDA GPU")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        raise V19ProtocolError(f"Production environment requires WORLD_SIZE=1, found {world_size}")
    core = {
        "schema_version": ENVIRONMENT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_manifest": protocol_manifest_record(root),
        "container_image_digest": container_image_digest,
        "runtime": actual,
        "required_topology": {"world_size": 1, "visible_cuda_devices": 1},
    }
    return {**core, "lock_sha256": content_sha256(core)}


def validate_environment_lock_document(
    lock: Mapping[str, Any], repo_root: str | Path
) -> Mapping[str, Any]:
    root = Path(repo_root).resolve()
    supplied = lock.get("lock_sha256")
    core = {key: value for key, value in lock.items() if key != "lock_sha256"}
    if supplied != content_sha256(core):
        raise V19ProtocolError("Production environment lock digest mismatch")
    expected = {
        "schema_version": ENVIRONMENT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_manifest": protocol_manifest_record(root),
        "required_topology": {"world_size": 1, "visible_cuda_devices": 1},
    }
    mismatches = [key for key, value in expected.items() if lock.get(key) != value]
    if mismatches:
        raise V19ProtocolError(f"Production environment lock mismatch: {mismatches}")
    runtime = lock.get("runtime")
    if not isinstance(runtime, Mapping):
        raise V19ProtocolError("Production environment lock has no runtime object")
    return runtime


def validate_environment_lock(lock: Mapping[str, Any], repo_root: str | Path) -> None:
    locked_runtime = validate_environment_lock_document(lock, repo_root)
    actual = runtime_record()
    if locked_runtime != actual:
        raise V19ProtocolError("Current runtime does not exactly match the production lock")


def _git_output(root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise V19ProtocolError(f"Cannot inspect Git state: {exc}") from exc


def validate_production_gate(
    repo_root: str | Path,
    *,
    environment_path: str | Path = ENVIRONMENT_LOCK,
    launch_path: str | Path = LAUNCH_RECORD,
) -> Mapping[str, Any]:
    root = Path(repo_root).resolve()
    require_eligible_protocol(root)
    env_path = root / environment_path
    run_path = root / launch_path
    if _git_output(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise V19ProtocolError("Production launch requires a clean committed worktree")
    tracked = set(_git_output(root, "ls-files").splitlines())
    for path in (env_path, run_path):
        relative = path.relative_to(root).as_posix()
        if relative not in tracked:
            raise V19ProtocolError(f"Production gate file must be committed: {relative}")
    lock = _read_json(env_path)
    validate_environment_lock(lock, root)
    launch = _read_json(run_path)
    supplied = launch.get("record_sha256")
    core = {key: value for key, value in launch.items() if key != "record_sha256"}
    if supplied != content_sha256(core):
        raise V19ProtocolError("Production launch record digest mismatch")
    expected = {
        "schema_version": LAUNCH_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_manifest": protocol_manifest_record(root),
        "environment_lock": file_record(env_path),
        "max_single_gpu_hours": 60,
        "max_cost_usd": 150,
        "human_approval": True,
    }
    mismatches = [key for key, value in expected.items() if launch.get(key) != value]
    if mismatches:
        raise V19ProtocolError(f"Production launch record mismatch: {mismatches}")
    rate = launch.get("usd_per_gpu_hour")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 0 < rate <= 150:
        raise V19ProtocolError("Production launch record has invalid usd_per_gpu_hour")
    if not isinstance(launch.get("approved_host_label"), str) or not launch[
        "approved_host_label"
    ].strip():
        raise V19ProtocolError("Production launch record has no approved host label")
    return launch


def build_launch_record(
    repo_root: str | Path, *, approved_host_label: str, usd_per_gpu_hour: float
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    require_eligible_protocol(root)
    if not approved_host_label.strip():
        raise V19ProtocolError("Production launch record requires a non-empty host label")
    if isinstance(usd_per_gpu_hour, bool) or not 0 < usd_per_gpu_hour <= 150:
        raise V19ProtocolError("usd_per_gpu_hour must be in (0, 150]")
    environment = root / ENVIRONMENT_LOCK
    lock = _read_json(environment)
    validate_environment_lock(lock, root)
    core = {
        "schema_version": LAUNCH_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_manifest": protocol_manifest_record(root),
        "environment_lock": file_record(environment),
        "approved_host_label": approved_host_label,
        "usd_per_gpu_hour": float(usd_per_gpu_hour),
        "max_single_gpu_hours": 60,
        "max_cost_usd": 150,
        "human_approval": True,
    }
    return {**core, "record_sha256": content_sha256(core)}


def training_dir(root: str | Path, arm: str, seed: int) -> Path:
    base = Path(root) / "train"
    if arm == "sft_only":
        return base / f"sft_seed{seed}"
    return base / f"{arm}_seed{seed}"


def eval_dir(root: str | Path, view: str, arm: str, seed: int | None) -> Path:
    return Path(root) / "eval" / view / state_label(arm, seed)


def verify_adapter_chain(
    *,
    arm: str,
    seed: int,
    training_run: str | Path,
    eval_run: str | Path,
    sft_parent: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> None:
    train = verify_completed_run(training_run)
    evaluation = verify_completed_run(
        eval_run, required_artifact_roles={"candidates", "metrics", "run_config", "summary"}
    )
    train_identity = train["intent"]["identity"]
    eval_identity = evaluation["intent"]["identity"]
    source_root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    validate_run_source_identity(train_identity, source_root)
    validate_run_source_identity(eval_identity, source_root)
    if train_identity["seed_roles"].get("trainer_seed") != seed:
        raise V19ProtocolError(f"{arm}: training manifest seed does not equal {seed}")
    if eval_identity["seed_roles"].get("training_seed_label") != seed:
        raise V19ProtocolError(f"{arm}: evaluation training-seed label does not equal {seed}")
    if eval_identity["model"].get("adapter_identity") != adapter_record(training_run):
        raise V19ProtocolError(f"{arm}: evaluation adapter is not the verified training output")
    expects_parent = bool(ARM_SPECS[arm].get("sft_parent"))
    if expects_parent:
        if sft_parent is None:
            raise V19ProtocolError(f"{arm}: missing SFT parent")
        verify_completed_run(sft_parent)
        if train_identity["model"].get("adapter_identity") != adapter_record(sft_parent):
            raise V19ProtocolError(f"{arm}: GRPO manifest is not bound to its SFT parent")
    elif train_identity["model"].get("adapter_identity") is not None:
        raise V19ProtocolError(f"{arm}: fresh-LoRA arm unexpectedly declares a parent adapter")


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, Sequence):
        return all(_all_finite(item) for item in value)
    return False


def verify_v19_training_health(
    run_dir: str | Path,
    *,
    run_kind: str,
    expected_steps: int,
    expected_strategy: str | None = None,
    require_group_variance: bool = True,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    required = {
        "adapter_config",
        "adapter_weights",
        "tokenizer_config",
        "training_args",
        "training_state",
    }
    if run_kind == "grpo":
        required.update({"reward_components", "teacher_weights"})
    verified = verify_completed_run(run_dir, required_artifact_roles=required)
    source_root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    validate_run_source_identity(verified["intent"]["identity"], source_root)
    root = Path(run_dir)
    state = _read_json(root / "training_state.json")
    if state.get("global_step") != expected_steps or state.get("max_steps") != expected_steps:
        raise V19ProtocolError(
            f"Training did not reach exactly {expected_steps} steps: "
            f"global={state.get('global_step')} max={state.get('max_steps')}"
        )
    if not _all_finite(state):
        raise V19ProtocolError("Training state contains non-finite diagnostics")
    report: dict[str, Any] = {
        "experiment_id": verified["intent"]["experiment_id"],
        "run_kind": run_kind,
        "global_step": expected_steps,
        "wall_clock_seconds": state.get("wall_clock_seconds"),
        "finite_training_state": True,
    }
    if run_kind != "grpo":
        return report
    rewards = _read_jsonl(root / "reward_components.jsonl")
    teachers = _read_jsonl(root / "teacher_weights.jsonl")
    if not rewards or not teachers:
        raise V19ProtocolError("GRPO health requires non-empty reward and teacher logs")
    if expected_strategy and {row.get("strategy") for row in teachers} != {expected_strategy}:
        raise V19ProtocolError("Teacher strategy log does not match the registered arm")
    required_reward_keys = {
        "primary_reward",
        "primary_reward_weighted",
        "aux_reward_weighted",
        "total_reward",
        "reward",
        "components",
    }
    variance_rows = 0
    exact_rows = 0
    for index, row in enumerate(rewards):
        if not required_reward_keys.issubset(row) or not isinstance(row["components"], dict):
            raise V19ProtocolError(f"Reward row {index} conflates or omits reward components")
        exact = float(row["components"].get("exact_correct", -1.0))
        correct = float(row["components"].get("correct", -2.0))
        primary = float(row["primary_reward"])
        if exact not in {0.0, 1.0} or exact != correct or primary != exact:
            raise V19ProtocolError(f"Reward row {index} primary reward disagrees with verifier")
        expected_total = float(row["primary_reward_weighted"]) + float(
            row["aux_reward_weighted"]
        )
        if not math.isclose(float(row["total_reward"]), expected_total, abs_tol=1e-9):
            raise V19ProtocolError(f"Reward row {index} total does not preserve components")
        if not math.isclose(float(row["reward"]), float(row["total_reward"]), abs_tol=1e-9):
            raise V19ProtocolError(f"Reward row {index} trainer reward differs from total")
        if not _all_finite(row):
            raise V19ProtocolError(f"Reward row {index} contains non-finite values")
        variance_rows += int(bool(row.get("group_has_variance")))
        exact_rows += int(exact)
    if require_group_variance and variance_rows == 0:
        raise V19ProtocolError("GRPO run has no within-prompt reward-group variance")
    report.update(
        {
            "reward_rows": len(rewards),
            "teacher_updates": len(teachers),
            "group_variance_row_fraction": variance_rows / len(rewards),
            "verifier_exact_row_fraction": exact_rows / len(rewards),
            "separate_reward_components_verified": True,
        }
    )
    return report


def _normalized_resolved_config(identity: Mapping[str, Any]) -> str:
    config = json.loads(json.dumps(identity.get("resolved_config", {})))
    config.pop("seed", None)
    config.pop("data_seed", None)
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def validate_training_matrix(
    repo_root: str | Path, runs_root: str | Path
) -> dict[str, dict[str, Any]]:
    root = Path(repo_root).resolve()
    runs = Path(runs_root)
    if not runs.is_absolute():
        runs = root / runs
    require_eligible_protocol(root)
    environment = _read_json(root / ENVIRONMENT_LOCK)
    locked_runtime = validate_environment_lock_document(environment, root)
    evidence: dict[str, dict[str, Any]] = {}
    configs: dict[str, set[str]] = {"sft": set(), "grpo": set()}
    for arm, spec in ARM_SPECS.items():
        if arm == "base":
            continue
        for seed in TRAINING_SEEDS:
            label = state_label(arm, seed)
            run_dir = training_dir(runs, arm, seed)
            run_kind = "sft" if arm == "sft_only" else "grpo"
            health = verify_v19_training_health(
                run_dir,
                run_kind=run_kind,
                expected_steps=313 if run_kind == "sft" else 300,
                expected_strategy=spec.get("reward_strategy"),
                require_group_variance=run_kind == "grpo",
                repo_root=root,
            )
            verified = verify_completed_run(run_dir)
            identity = verified["intent"]["identity"]
            requested = identity.get("requested_args", {})
            expected_requested = {
                "experiment_protocol": PROTOCOL_ID,
                "seed": seed,
                "seed_protocol": TRUE_SEED_PROTOCOL,
                "max_steps": 313 if run_kind == "sft" else 300,
                "batch_size": 2,
                "grad_accum": 8,
                "learning_rate": 5e-5 if run_kind == "sft" else 5e-6,
            }
            if run_kind == "sft":
                expected_requested["completion_only_loss"] = True
            if run_kind == "grpo":
                expected_requested.update(
                    {
                        "method_arm": arm,
                        "reward_strategy": spec["reward_strategy"],
                        "trainer_seed": seed,
                        "num_generations": 4,
                        "max_prompt_length": 768,
                        "max_completion_length": 256,
                        "task_curriculum": "uniform",
                        "prompt_field": "prompt",
                    }
                )
            mismatches = [
                key for key, value in expected_requested.items() if requested.get(key) != value
            ]
            if mismatches:
                raise V19ProtocolError(f"Training label mismatch {label}: {mismatches}")
            if identity.get("runtime") != locked_runtime:
                raise V19ProtocolError(f"Training runtime differs from lock: {label}")
            if identity.get("inputs") != {"train": file_record(root / TRAIN_PATH)}:
                raise V19ProtocolError(f"Training input differs from Countdown-v2 train: {label}")
            model = identity.get("model", {})
            if model.get("name") != MODEL_NAME or model.get("revision") != MODEL_REVISION:
                raise V19ProtocolError(f"Training model identity mismatch: {label}")
            configs[run_kind].add(_normalized_resolved_config(identity))
            parent_evidence = {}
            if spec.get("sft_parent"):
                parent = training_dir(runs, "sft_only", seed)
                parent_verified = verify_completed_run(parent)
                if identity["model"].get("adapter_identity") != adapter_record(parent):
                    raise V19ProtocolError(f"Combined arm has wrong SFT parent: {label}")
                parent_evidence = {
                    "sft_parent_experiment_id": parent_verified["intent"]["experiment_id"],
                    "sft_parent_result_manifest": file_record(parent / RESULT_NAME),
                }
            elif identity["model"].get("adapter_identity") is not None:
                raise V19ProtocolError(f"Fresh training arm unexpectedly has a parent: {label}")
            evidence[label] = {
                "training_experiment_id": verified["intent"]["experiment_id"],
                "training_result_manifest": file_record(run_dir / RESULT_NAME),
                "health_sha256": content_sha256(health),
                **parent_evidence,
            }
    for family, values in configs.items():
        if len(values) != 1:
            raise V19ProtocolError(f"Resolved {family} config drift across training matrix")
    if len(evidence) != 15:
        raise V19ProtocolError(f"Training matrix must contain 15 states, found {len(evidence)}")
    return evidence


def validate_score_artifact(
    *,
    repo_root: str | Path,
    runs_root: str | Path,
    report_path: str | Path,
    manifest_path: str | Path,
    expected_view: str,
    expected_seeds: Sequence[int],
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    runs = Path(runs_root)
    if not runs.is_absolute():
        runs = root / runs
    report_file = Path(report_path)
    score_manifest_file = Path(manifest_path)
    if not report_file.is_absolute():
        report_file = root / report_file
    if not score_manifest_file.is_absolute():
        score_manifest_file = root / score_manifest_file
    report = _read_json(report_file)
    score_manifest = _read_json(score_manifest_file)
    supplied = score_manifest.get("manifest_sha256")
    core = {key: value for key, value in score_manifest.items() if key != "manifest_sha256"}
    if supplied != content_sha256(core):
        raise V19ProtocolError("Score artifact manifest digest mismatch")
    expected = {
        "schema_version": SCORE_MANIFEST_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "view": expected_view,
        "observed_training_seeds": list(expected_seeds),
        "report": file_record(report_file),
        "scorer": file_record(root / "scripts/22_score_v19.py"),
        "protocol_manifest": protocol_manifest_record(root),
        "evidence_sha256": content_sha256(report.get("evidence")),
    }
    mismatches = [key for key, value in expected.items() if score_manifest.get(key) != value]
    if mismatches:
        raise V19ProtocolError(f"Score artifact mismatch: {mismatches}")
    if (
        report.get("protocol_id") != PROTOCOL_ID
        or report.get("view") != expected_view
        or report.get("observed_training_seeds") != list(expected_seeds)
    ):
        raise V19ProtocolError("Score report protocol identity mismatch")
    expected_labels = {"base"}
    for arm in ARM_SPECS:
        if arm != "base":
            expected_labels.update(state_label(arm, seed) for seed in expected_seeds)
    state_evidence = report.get("evidence", {}).get("states")
    if report.get("evidence", {}).get("protocol_manifest") != protocol_manifest_record(root):
        raise V19ProtocolError("Score report evidence uses a different protocol manifest")
    if not isinstance(state_evidence, dict) or set(state_evidence) != expected_labels:
        raise V19ProtocolError("Score report does not bind the complete expected state panel")
    for label, evidence in state_evidence.items():
        if label == "base":
            arm, seed = "base", None
        else:
            matches = [
                (arm, seed)
                for arm in ARM_SPECS
                if arm != "base"
                for seed in expected_seeds
                if state_label(arm, seed) == label
            ]
            if len(matches) != 1:
                raise V19ProtocolError(f"Cannot resolve score state label: {label}")
            arm, seed = matches[0]
        current_eval = eval_dir(runs, expected_view, arm, seed)
        verified_eval = verify_completed_run(
            current_eval,
            required_artifact_roles={"candidates", "metrics", "run_config", "summary"},
        )
        validate_run_source_identity(verified_eval["intent"]["identity"], root)
        if (
            evidence.get("eval_experiment_id") != verified_eval["intent"]["experiment_id"]
            or evidence.get("eval_result_manifest")
            != file_record(current_eval / RESULT_NAME)
        ):
            raise V19ProtocolError(f"Score evidence changed for evaluation state: {label}")
        if arm == "base":
            continue
        current_train = training_dir(runs, arm, int(seed))
        verified_train = verify_completed_run(current_train)
        if (
            evidence.get("training_experiment_id")
            != verified_train["intent"]["experiment_id"]
            or evidence.get("training_result_manifest")
            != file_record(current_train / RESULT_NAME)
        ):
            raise V19ProtocolError(f"Score evidence changed for training state: {label}")
        parent = training_dir(runs, "sft_only", int(seed)) if ARM_SPECS[arm].get("sft_parent") else None
        verify_adapter_chain(
            arm=arm,
            seed=int(seed),
            training_run=current_train,
            eval_run=current_eval,
            sft_parent=parent,
            repo_root=root,
        )
    return report


def build_confirmation_ready_record(
    repo_root: str | Path,
    *,
    runs_root: str | Path,
    dev_score_report: str | Path = DEV_SCORE_REPORT,
    dev_score_manifest: str | Path = DEV_SCORE_MANIFEST,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    require_eligible_protocol(root)
    validate_production_gate(root)
    training_evidence = validate_training_matrix(root, runs_root)
    validate_score_artifact(
        repo_root=root,
        runs_root=runs_root,
        report_path=dev_score_report,
        manifest_path=dev_score_manifest,
        expected_view="validation_dev100",
        expected_seeds=(0,),
    )
    runs = Path(runs_root)
    if runs.is_absolute():
        try:
            runs_value = runs.resolve().relative_to(root).as_posix()
        except ValueError as exc:
            raise V19ProtocolError("Production runs root must live inside the repository") from exc
    else:
        runs_value = runs.as_posix()
    report_file = Path(dev_score_report)
    manifest_file = Path(dev_score_manifest)
    if not report_file.is_absolute():
        report_file = root / report_file
    if not manifest_file.is_absolute():
        manifest_file = root / manifest_file
    core = {
        "schema_version": CONFIRMATION_READY_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_manifest": protocol_manifest_record(root),
        "environment_lock": file_record(root / ENVIRONMENT_LOCK),
        "launch_record": file_record(root / LAUNCH_RECORD),
        "runs_root": runs_value,
        "dev_score_report_path": report_file.relative_to(root).as_posix(),
        "dev_score_report": file_record(report_file),
        "dev_score_manifest_path": manifest_file.relative_to(root).as_posix(),
        "dev_score_manifest": file_record(manifest_file),
        "training_states": training_evidence,
        "methods_and_analysis_frozen": True,
    }
    return {**core, "record_sha256": content_sha256(core)}


def validate_confirmation_ready(
    record_path: str | Path, *, repo_root: str | Path
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    require_eligible_protocol(root)
    validate_production_gate(root)
    path = Path(record_path)
    if not path.is_absolute():
        path = root / path
    expected_path = (root / CONFIRMATION_READY_RECORD).resolve()
    if path.resolve() != expected_path:
        raise V19ProtocolError("Confirmation readiness must use the registered record path")
    tracked = set(_git_output(root, "ls-files").splitlines())
    if CONFIRMATION_READY_RECORD.as_posix() not in tracked:
        raise V19ProtocolError("Confirmation readiness record must be committed before access")
    record = _read_json(path)
    supplied = record.get("record_sha256")
    core = {key: value for key, value in record.items() if key != "record_sha256"}
    if supplied != content_sha256(core):
        raise V19ProtocolError("Confirmation readiness record digest mismatch")
    expected = {
        "schema_version": CONFIRMATION_READY_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_manifest": protocol_manifest_record(root),
        "environment_lock": file_record(root / ENVIRONMENT_LOCK),
        "launch_record": file_record(root / LAUNCH_RECORD),
        "methods_and_analysis_frozen": True,
    }
    mismatches = [key for key, value in expected.items() if record.get(key) != value]
    if mismatches:
        raise V19ProtocolError(f"Confirmation readiness mismatch: {mismatches}")
    runs_root = root / str(record.get("runs_root"))
    training_evidence = validate_training_matrix(root, runs_root)
    if record.get("training_states") != training_evidence:
        raise V19ProtocolError("Confirmation readiness training evidence changed")
    report_path = root / str(record.get("dev_score_report_path"))
    manifest_path = root / str(record.get("dev_score_manifest_path"))
    if (
        record.get("dev_score_report") != file_record(report_path)
        or record.get("dev_score_manifest") != file_record(manifest_path)
    ):
        raise V19ProtocolError("Confirmation readiness score artifacts changed")
    validate_score_artifact(
        repo_root=root,
        runs_root=runs_root,
        report_path=report_path,
        manifest_path=manifest_path,
        expected_view="validation_dev100",
        expected_seeds=(0,),
    )
    return record


def validate_test_release(
    release_path: str | Path, *, repo_root: str | Path, runner: str
) -> None:
    root = Path(repo_root).resolve()
    require_eligible_protocol(root)
    release = _read_json(Path(release_path))
    supplied = release.get("record_sha256")
    core = {key: value for key, value in release.items() if key != "record_sha256"}
    if supplied != content_sha256(core):
        raise V19ProtocolError("V0.19 test release digest mismatch")
    expected = {
        "schema_version": TEST_RELEASE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_manifest": protocol_manifest_record(root),
        "test_jsonl": file_record(root / TEST_PATH),
        "test_ordered_ids": file_record(root / TEST_IDS_PATH),
        "scorer": file_record(root / "scripts/22_score_v19.py"),
        "sampling_signature": EVAL_CONFIG,
        "confirmation_score_report": file_record(root / CONFIRM_SCORE_REPORT),
        "confirmation_score_manifest": file_record(root / CONFIRM_SCORE_MANIFEST),
        "authorized_runner": "07_best_of_n_rerank",
        "human_approval": True,
        "frozen_commit": _git_output(root, "rev-parse", "HEAD"),
    }
    mismatches = [key for key, value in expected.items() if release.get(key) != value]
    if mismatches or runner != expected["authorized_runner"]:
        raise V19ProtocolError(f"V0.19 test release mismatch: {mismatches or ['runner']}")
    if _git_output(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise V19ProtocolError("V0.19 test release requires a clean worktree")
    states = release.get("validation_confirm_states")
    if not isinstance(states, dict) or set(states) != set(state_labels()):
        raise V19ProtocolError("V0.19 test release must bind all 16 confirmation states")
    expected_states = {"base": ("base", None)}
    for arm, spec in ARM_SPECS.items():
        if arm != "base":
            expected_states.update(
                {state_label(arm, seed): (arm, seed) for seed in spec["seeds"]}
            )
    expected_eval_inputs = {
        "data": file_record(root / VALIDATION_PATH),
        "ordered_task_ids": file_record(
            root / PROTOCOL_DIR / VIEW_FILES["validation_confirm400"]
        ),
        "confirmation_ready": file_record(root / CONFIRMATION_READY_RECORD),
    }
    environment_path = root / ENVIRONMENT_LOCK
    if not environment_path.is_file():
        raise V19ProtocolError("V0.19 test release requires the production environment lock")
    validate_environment_lock(_read_json(environment_path), root)
    for label, record in states.items():
        if not isinstance(record, dict) or not isinstance(record.get("run_dir"), str):
            raise V19ProtocolError(f"Malformed confirmation-state record: {label}")
        run_dir = root / record["run_dir"]
        verified = verify_completed_run(
            run_dir, required_artifact_roles={"candidates", "metrics", "run_config", "summary"}
        )
        if verified["intent"]["experiment_id"] != record.get("experiment_id"):
            raise V19ProtocolError(f"Confirmation-state experiment ID mismatch: {label}")
        identity = verified["intent"]["identity"]
        config = _read_json(run_dir / "run_config.json")
        arm, seed = expected_states[label]
        expected_config = {
            "experiment_protocol": PROTOCOL_ID,
            "method": arm,
            "training_seed": seed,
            "training_protocol": TRUE_SEED_PROTOCOL,
            "split": "validation_confirm400",
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "hf_gen_mode": EVAL_CONFIG["hf_gen_mode"],
            "batch_size": EVAL_CONFIG["batch_size"],
            "prompt_field": EVAL_CONFIG["prompt_field"],
            "sampling_seed": EVAL_CONFIG["sampling_seed"],
            "temperature": EVAL_CONFIG["temperature"],
            "top_p": EVAL_CONFIG["top_p"],
            "max_new_tokens": EVAL_CONFIG["max_new_tokens"],
            "max_n": EVAL_CONFIG["max_n"],
            "n_values": EVAL_CONFIG["n_values"],
        }
        config_mismatches = [
            key for key, value in expected_config.items() if config.get(key) != value
        ]
        if config_mismatches or identity.get("inputs") != expected_eval_inputs:
            raise V19ProtocolError(
                f"Confirmation-state protocol mismatch {label}: {config_mismatches}"
            )
        if identity.get("runtime") != _read_json(environment_path).get("runtime"):
            raise V19ProtocolError(f"Confirmation-state runtime mismatch: {label}")
        if arm == "base":
            if identity["model"].get("adapter_identity") is not None:
                raise V19ProtocolError("Base confirmation state unexpectedly uses an adapter")
            continue
        training_run_dir = record.get("training_run_dir")
        if not isinstance(training_run_dir, str):
            raise V19ProtocolError(f"Missing training provenance chain: {label}")
        training_run = root / training_run_dir
        training_verified = verify_completed_run(training_run)
        if training_verified["intent"]["experiment_id"] != record.get(
            "training_experiment_id"
        ):
            raise V19ProtocolError(f"Training experiment ID mismatch: {label}")
        parent_path = None
        if ARM_SPECS[arm].get("sft_parent"):
            parent_value = record.get("sft_parent_run_dir")
            if not isinstance(parent_value, str):
                raise V19ProtocolError(f"Missing SFT parent provenance chain: {label}")
            parent_path = root / parent_value
            parent_verified = verify_completed_run(parent_path)
            if parent_verified["intent"]["experiment_id"] != record.get(
                "sft_parent_experiment_id"
            ):
                raise V19ProtocolError(f"SFT parent experiment ID mismatch: {label}")
        verify_adapter_chain(
            arm=arm,
            seed=int(seed),
            training_run=training_run,
            eval_run=run_dir,
            sft_parent=parent_path,
            repo_root=root,
        )
    training_evidence = validate_training_matrix(root, "outputs/v19/production")
    if len(training_evidence) != 15:
        raise V19ProtocolError("V0.19 test release lacks the complete healthy training matrix")
    score_report = validate_score_artifact(
        repo_root=root,
        runs_root="outputs/v19/production",
        report_path=CONFIRM_SCORE_REPORT,
        manifest_path=CONFIRM_SCORE_MANIFEST,
        expected_view="validation_confirm400",
        expected_seeds=TRAINING_SEEDS,
    )
    score_states = score_report.get("evidence", {}).get("states", {})
    for label, record in states.items():
        if score_states.get(label, {}).get("eval_experiment_id") != record.get(
            "experiment_id"
        ):
            raise V19ProtocolError(f"Test release state is absent from confirm score: {label}")
