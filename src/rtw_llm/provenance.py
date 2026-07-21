"""Fail-closed, content-addressed provenance records for research runs."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "rtw-run-manifest-v1"
INTENT_NAME = "run_intent.json"
RESULT_NAME = "run_result.json"
NON_IDENTITY_ARG_KEYS = {"output_dir", "skip_if_complete", "strict_provenance"}
CONTENT_ADDRESSED_ARG_KEYS = {
    "adapter_path",
    "confirmation_ready_record",
    "data_path",
    "eval_path",
    "final_test_release_record",
    "init_adapter_path",
    "model_name",
    "task_ids_file",
    "test_release_record",
    "train_path",
}
NON_IDENTITY_CONFIG_KEYS = {"output_dir", "run_name", "logging_dir"}
CONTENT_ADDRESSED_CONFIG_KEYS = {"adapter_path", "data_path", "model_name", "task_ids_file"}
ADAPTER_IDENTITY_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "chat_template.jinja",
    "config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "run_intent.json",
    "run_result.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
}
WEIGHT_FILE_PATTERNS = (
    re.compile(r"adapter_model(?:-[^.]+)?\.(?:safetensors|bin)$"),
    re.compile(r"model(?:-[^.]+)?\.safetensors$"),
    re.compile(r"pytorch_model(?:-[^.]+)?\.bin$"),
)
HF_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


class ProvenanceError(RuntimeError):
    """Raised when a strict run cannot establish or verify its identity."""


def canonical_json(value: Any) -> str:
    """Canonical JSON used by every content digest."""
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_record(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise ProvenanceError(f"Required provenance input is not a file: {p}")
    digest = hashlib.sha256()
    size = 0
    line_count = 0
    ended_with_newline = False
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
            line_count += chunk.count(b"\n")
            ended_with_newline = chunk.endswith(b"\n")
    if size and not ended_with_newline:
        line_count += 1
    return {"sha256": digest.hexdigest(), "size": size, "line_count": line_count}


def adapter_record(path: str | Path) -> dict[str, Any]:
    """Semantic digest of stable model/adapter identity files, excluding logs."""
    root = Path(path)
    if root.is_file():
        return {"kind": "file", **file_record(root)}
    if not root.is_dir():
        raise ProvenanceError(f"Adapter/model path does not exist: {root}")
    files: dict[str, Any] = {}
    weight_files = []
    for candidate in sorted(root.iterdir()):
        is_weight = any(pattern.fullmatch(candidate.name) for pattern in WEIGHT_FILE_PATTERNS)
        if candidate.is_file() and (candidate.name in ADAPTER_IDENTITY_FILES or is_weight):
            files[candidate.relative_to(root).as_posix()] = file_record(candidate)
            if is_weight:
                weight_files.append(candidate.name)
    if not weight_files:
        raise ProvenanceError(f"No supported model/adapter weight payload found under {root}")
    return {"kind": "adapter_directory", "files": files, "tree_sha256": content_sha256(files)}


def git_record(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)

    def git(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ProvenanceError(f"Cannot resolve Git state at {root}: {exc}") from exc
        return result.stdout.strip()

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    dirty = bool(status)
    return {
        "commit": commit,
        "dirty": dirty,
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest() if dirty else None,
    }


def runtime_record() -> dict[str, Any]:
    packages = {}
    for name in (
        "torch",
        "transformers",
        "trl",
        "datasets",
        "peft",
        "accelerate",
        "numpy",
        "tokenizers",
        "safetensors",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    hardware: dict[str, Any] = {
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    execution: dict[str, Any] = {
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "rank": int(os.environ.get("RANK", "0")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "precision_environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "CUBLAS_WORKSPACE_CONFIG",
                "PYTORCH_CUDA_ALLOC_CONF",
                "NVIDIA_TF32_OVERRIDE",
            )
        },
    }
    try:
        import torch

        hardware.update(
            {
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": torch.version.cuda,
                "cuda_devices": [
                    torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
                ],
                "cuda_device_count": int(torch.cuda.device_count()),
                "cudnn_version": torch.backends.cudnn.version(),
                "mps_available": bool(
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ),
            }
        )
        if torch.cuda.is_available():
            try:
                driver = subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.splitlines()
                hardware["nvidia_driver_versions"] = sorted(
                    {value.strip() for value in driver if value.strip()}
                )
            except (OSError, subprocess.CalledProcessError):
                hardware["nvidia_driver_versions"] = None
        execution.update(
            {
                "torch_num_threads": int(torch.get_num_threads()),
                "torch_num_interop_threads": int(torch.get_num_interop_threads()),
                "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
                "deterministic_warn_only": bool(
                    torch.is_deterministic_algorithms_warn_only_enabled()
                ),
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
            }
        )
    except ImportError:
        hardware["torch_unavailable"] = True
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "hardware": hardware,
        "execution": execution,
    }


def build_run_identity(
    *,
    run_kind: str,
    requested_args: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    seed_roles: Mapping[str, Any],
    input_files: Mapping[str, str | Path],
    model_name: str,
    adapter_path: str | Path | None = None,
    repo_root: str | Path,
    model_revision: str | None = None,
) -> dict[str, Any]:
    git = git_record(repo_root)
    if git["dirty"]:
        raise ProvenanceError(
            "Strict provenance requires a clean Git worktree; commit the code state before compute"
        )
    excluded_args = NON_IDENTITY_ARG_KEYS | CONTENT_ADDRESSED_ARG_KEYS
    args_identity = {key: value for key, value in requested_args.items() if key not in excluded_args}
    excluded_config = NON_IDENTITY_CONFIG_KEYS | CONTENT_ADDRESSED_CONFIG_KEYS
    config_identity = {key: value for key, value in resolved_config.items() if key not in excluded_config}
    inputs = {role: file_record(path) for role, path in sorted(input_files.items()) if path}
    model_path = Path(model_name)
    if model_path.exists():
        model: dict[str, Any] = {"kind": "local", "local_identity": adapter_record(model_path)}
    else:
        if not model_revision or not HF_COMMIT_RE.fullmatch(model_revision):
            raise ProvenanceError(
                "Strict provenance requires --model_revision to be a full 40-hex HF commit"
            )
        model = {"kind": "huggingface", "name": model_name, "revision": model_revision}
    if adapter_path:
        model["adapter_identity"] = adapter_record(adapter_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_kind": run_kind,
        "git": git,
        "requested_args": args_identity,
        "resolved_config": config_identity,
        "seed_roles": dict(seed_roles),
        "inputs": inputs,
        "model": model,
        "runtime": runtime_record(),
    }


def make_intent(identity: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _json_safe(identity)
    experiment_id = content_sha256(normalized)
    core = {
        "manifest_type": "intent",
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "identity": normalized,
    }
    return {**core, "manifest_sha256": content_sha256(core)}


def write_intent(output_dir: str | Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    intent_path = out / INTENT_NAME
    result_path = out / RESULT_NAME
    if intent_path.exists() or result_path.exists():
        raise ProvenanceError(
            f"Strict output directory already owns a manifest and is not reusable: {out}"
        )
    existing = [path for path in out.iterdir() if path.name not in {INTENT_NAME, RESULT_NAME}]
    if existing:
        raise ProvenanceError(
            f"Strict output directory must be empty before compute: {out}; found {existing[:3]}"
        )
    intent = make_intent(identity)
    _write_new_json(intent_path, intent)
    return intent


def write_result(
    output_dir: str | Path,
    *,
    artifact_paths: Mapping[str, str | Path],
    observations: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    intent = verify_intent(out / INTENT_NAME)
    artifacts: dict[str, Any] = {}
    for role, value in sorted(artifact_paths.items()):
        path = Path(value)
        try:
            relative_path = path.resolve().relative_to(out.resolve()).as_posix()
        except ValueError as exc:
            raise ProvenanceError(f"Result artifact must live under {out}: {path}") from exc
        artifacts[role] = {"path": relative_path, **file_record(path)}
    core = {
        "manifest_type": "result",
        "schema_version": SCHEMA_VERSION,
        "experiment_id": intent["experiment_id"],
        "intent_manifest_sha256": intent["manifest_sha256"],
        "artifacts": artifacts,
        "observations": dict(observations or {}),
    }
    result = {**core, "manifest_sha256": content_sha256(core)}
    _write_new_json(out / RESULT_NAME, result)
    return result


def verify_intent(path: str | Path, expected_identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    manifest = _read_json(path)
    if manifest.get("manifest_type") != "intent" or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ProvenanceError(f"Unsupported or malformed intent manifest: {path}")
    supplied = manifest.get("manifest_sha256")
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if supplied != content_sha256(core):
        raise ProvenanceError(f"Intent manifest digest mismatch: {path}")
    if manifest.get("experiment_id") != content_sha256(manifest.get("identity")):
        raise ProvenanceError(f"Intent experiment identity mismatch: {path}")
    if expected_identity is not None and _json_safe(expected_identity) != manifest.get("identity"):
        raise ProvenanceError(f"Requested run identity does not match existing intent: {path}")
    return manifest


def verify_completed_run(
    output_dir: str | Path,
    expected_identity: Mapping[str, Any] | None = None,
    required_artifact_roles: set[str] | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    intent = verify_intent(out / INTENT_NAME, expected_identity)
    result = _read_json(out / RESULT_NAME)
    if result.get("manifest_type") != "result" or result.get("schema_version") != SCHEMA_VERSION:
        raise ProvenanceError(f"Unsupported or malformed result manifest: {out / RESULT_NAME}")
    supplied = result.get("manifest_sha256")
    core = {key: value for key, value in result.items() if key != "manifest_sha256"}
    if supplied != content_sha256(core):
        raise ProvenanceError(f"Result manifest digest mismatch: {out / RESULT_NAME}")
    if result.get("experiment_id") != intent.get("experiment_id"):
        raise ProvenanceError("Result experiment id does not link to the intent")
    if result.get("intent_manifest_sha256") != intent.get("manifest_sha256"):
        raise ProvenanceError("Result intent digest does not link to the current intent")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ProvenanceError("Result manifest has no artifact records")
    if required_artifact_roles and not required_artifact_roles.issubset(artifacts):
        missing = sorted(required_artifact_roles - set(artifacts))
        raise ProvenanceError(f"Result manifest is missing required artifacts: {missing}")
    for role, record in artifacts.items():
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ProvenanceError(f"Malformed artifact record for {role}")
        path = (out / record["path"]).resolve()
        try:
            path.relative_to(out.resolve())
        except ValueError as exc:
            raise ProvenanceError(f"Artifact path escapes output directory for {role}: {path}") from exc
        actual = file_record(path)
        expected = {key: record[key] for key in ("sha256", "size", "line_count")}
        if actual != expected:
            raise ProvenanceError(f"Artifact digest mismatch for {role}: {path}")
    return {"intent": intent, "result": result}


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            _json_safe(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise ProvenanceError(f"Refusing to overwrite provenance manifest: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"Cannot read provenance manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProvenanceError(f"Provenance manifest must be a JSON object: {path}")
    return value


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted((_json_safe(item) for item in value), key=canonical_json)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "value") and isinstance(value.value, (str, int, float, bool)):
        return value.value
    raise TypeError(f"Value is not JSON-safe for provenance: {type(value).__name__}")
