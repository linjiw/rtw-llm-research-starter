"""Deterministic construction and atomic publication of Countdown-v2 data."""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .countdown import difficulty_spec, random_solvable_task, verify_expression
from .prompts import make_prompt, make_sft_completion
from .provenance import content_sha256

PROTOCOL_ID = "countdown-dataset-v2"
GENERATOR_VERSION = "countdown-v2-generator-v1"
MANIFEST_SCHEMA = "countdown-v2-manifest-v1"
BASE_SEED = 20260710
FINAL_TEST_POLICY = (
    "NO_MODEL_EVALUATION_UNTIL_METHODS_CLAIMS_ANALYSIS_AND_STOPPING_RULES_ARE_FROZEN"
)
GENERATION_ORDER = ("easy", "medium", "hard", "ood_long", "ood_division")
IN_DIST_SLICE_ORDER = ("train", "validation", "test_in_dist", "final_test_in_dist")
SPLIT_ORDER = (
    "train",
    "validation",
    "test_in_dist",
    "final_test_in_dist",
    "test_ood_long",
    "test_ood_division",
)
PROPOSAL_SEED_OFFSETS = {
    "easy": 0,
    "medium": 1,
    "hard": 2,
    "ood_long": 3,
    "ood_division": 4,
}
ALLOCATION_SEED_OFFSETS = {"easy": 100, "medium": 101, "hard": 102}
SPLIT_ORDER_SEED_OFFSETS = {
    "train": 200,
    "validation": 201,
    "test_in_dist": 202,
    "final_test_in_dist": 203,
    "test_ood_long": 204,
    "test_ood_division": 205,
}
MAX_PROPOSALS_PER_POOL = 250_000
EASY_LOOSE_KEY_CAPACITY = 1_264
SPLIT_QUOTAS: dict[str, dict[str, int]] = {
    "train": {"easy": 900, "medium": 2_050, "hard": 2_050},
    "validation": {"easy": 50, "medium": 225, "hard": 225},
    "test_in_dist": {"easy": 50, "medium": 225, "hard": 225},
    "final_test_in_dist": {"easy": 50, "medium": 225, "hard": 225},
    "test_ood_long": {"ood_long": 500},
    "test_ood_division": {"ood_division": 500},
}
SOURCE_PATHS = (
    "scripts/18_generate_countdown_v2.py",
    "src/rtw_llm/countdown_v2.py",
    "src/rtw_llm/countdown.py",
    "src/rtw_llm/prompts.py",
)


def canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    kwargs: dict[str, Any] = {
        "sort_keys": True,
        "ensure_ascii": False,
        "allow_nan": False,
    }
    if pretty:
        kwargs["indent"] = 2
    else:
        kwargs["separators"] = (",", ":")
    return (json.dumps(value, **kwargs) + "\n").encode("utf-8")


def canonical_jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(record) for record in records)


def semantic_key(task: Mapping[str, Any]) -> tuple[tuple[int, ...], int, tuple[str, ...]]:
    return (
        tuple(sorted(int(value) for value in task["numbers"])),
        int(task["target"]),
        tuple(sorted(set(str(value) for value in task["allowed_ops"]))),
    )


def loose_key(task: Mapping[str, Any]) -> tuple[tuple[int, ...], int]:
    exact = semantic_key(task)
    return exact[0], exact[1]


def _stable_order(
    tasks: list[dict[str, Any]], *, seed: int, domain: str
) -> list[dict[str, Any]]:
    def key(task: dict[str, Any]) -> tuple[str, str]:
        canonical = json.dumps(semantic_key(task), separators=(",", ":"))
        digest = hashlib.sha256(f"{seed}\0{domain}\0{canonical}".encode()).hexdigest()
        return digest, canonical

    return sorted(tasks, key=key)


def _pool_targets(quotas: Mapping[str, Mapping[str, int]]) -> dict[str, int]:
    return {
        difficulty: sum(int(by_difficulty.get(difficulty, 0)) for by_difficulty in quotas.values())
        for difficulty in GENERATION_ORDER
    }


def _validate_recipe(quotas: Mapping[str, Mapping[str, int]]) -> None:
    if tuple(quotas) != SPLIT_ORDER:
        raise ValueError(f"Split order must be {SPLIT_ORDER}, found {tuple(quotas)}")
    expected_difficulties = {
        "train": ("easy", "medium", "hard"),
        "validation": ("easy", "medium", "hard"),
        "test_in_dist": ("easy", "medium", "hard"),
        "final_test_in_dist": ("easy", "medium", "hard"),
        "test_ood_long": ("ood_long",),
        "test_ood_division": ("ood_division",),
    }
    for split, expected in expected_difficulties.items():
        if tuple(quotas[split]) != expected:
            raise ValueError(
                f"Difficulty order for {split} must be {expected}, found {tuple(quotas[split])}"
            )
        if any(isinstance(value, bool) or int(value) < 0 for value in quotas[split].values()):
            raise ValueError(f"Invalid quota for {split}")
    easy_target = _pool_targets(quotas)["easy"]
    if easy_target > EASY_LOOSE_KEY_CAPACITY:
        raise ValueError(
            f"Easy quota {easy_target} exceeds finite loose-key capacity "
            f"{EASY_LOOSE_KEY_CAPACITY}"
        )


def _generate_pool(
    difficulty: str,
    target_count: int,
    *,
    proposal_seed: int,
    global_seen: set[tuple[tuple[int, ...], int]],
    max_proposals: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rng = random.Random(proposal_seed)
    spec = difficulty_spec(difficulty)
    accepted: list[dict[str, Any]] = []
    duplicate_rejections = 0
    for proposal_index in range(max_proposals):
        task = random_solvable_task(rng, **spec)
        if len(task["numbers"]) != int(spec["n_numbers"]):
            raise RuntimeError(f"Corrected generator returned wrong operand count for {difficulty}")
        verification = verify_expression(
            task["solution"], task["numbers"], task["target"], task["allowed_ops"]
        )
        if not verification.correct:
            raise RuntimeError(f"Corrected generator returned unverifiable task for {difficulty}")
        key = loose_key(task)
        if key in global_seen:
            duplicate_rejections += 1
            continue
        global_seen.add(key)
        accepted.append(
            {
                **task,
                "_proposal_index": proposal_index,
                "_proposal_seed": proposal_seed,
            }
        )
        if len(accepted) == target_count:
            return accepted, {
                "target": target_count,
                "proposals": proposal_index + 1,
                "duplicate_rejections": duplicate_rejections,
            }
    raise RuntimeError(
        f"Pool {difficulty} reached proposal budget {max_proposals} at "
        f"{len(accepted)}/{target_count} accepted tasks"
    )


def _record_for_task(
    task: dict[str, Any],
    *,
    split: str,
    difficulty: str,
    index: int,
    allocation_seed: int | None,
) -> dict[str, Any]:
    numbers = list(task["numbers"])
    target = int(task["target"])
    allowed_ops = list(task["allowed_ops"])
    solution = str(task["solution"])
    return {
        "id": f"v2_{split}_{difficulty}_{index:06d}",
        "split": split,
        "difficulty": difficulty,
        "numbers": numbers,
        "target": target,
        "allowed_ops": allowed_ops,
        "solution": solution,
        "prompt_low": make_prompt(numbers, target, allowed_ops, level="low"),
        "prompt_mid": make_prompt(numbers, target, allowed_ops, level="mid"),
        "prompt_high": make_prompt(numbers, target, allowed_ops, level="high"),
        "prompt": make_prompt(numbers, target, allowed_ops, level="high"),
        "completion": make_sft_completion(solution, target),
        "metadata": {
            "dataset_protocol": PROTOCOL_ID,
            "generator_version": GENERATOR_VERSION,
            "n_numbers": len(numbers),
            "source": "synthetic_by_construction",
            "proposal_seed": int(task["_proposal_seed"]),
            "proposal_index": int(task["_proposal_index"]),
            "allocation_seed": allocation_seed,
        },
    }


def build_records(
    *,
    base_seed: int = BASE_SEED,
    split_quotas: Mapping[str, Mapping[str, int]] = SPLIT_QUOTAS,
    max_proposals: int = MAX_PROPOSALS_PER_POOL,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Build all splits in memory with global loose-key disjointness."""
    _validate_recipe(split_quotas)
    targets = _pool_targets(split_quotas)
    global_seen: set[tuple[tuple[int, ...], int]] = set()
    pools: dict[str, list[dict[str, Any]]] = {}
    pool_stats: dict[str, Any] = {}
    for difficulty in GENERATION_ORDER:
        proposal_seed = base_seed + PROPOSAL_SEED_OFFSETS[difficulty]
        pool, stats = _generate_pool(
            difficulty,
            targets[difficulty],
            proposal_seed=proposal_seed,
            global_seen=global_seen,
            max_proposals=max_proposals,
        )
        pools[difficulty] = pool
        pool_stats[difficulty] = {"proposal_seed": proposal_seed, **stats}

    allocated: dict[str, dict[str, list[dict[str, Any]]]] = {
        split: {} for split in SPLIT_ORDER
    }
    for difficulty in ("easy", "medium", "hard"):
        allocation_seed = base_seed + ALLOCATION_SEED_OFFSETS[difficulty]
        ordered = _stable_order(
            pools[difficulty], seed=allocation_seed, domain=f"allocation/{difficulty}"
        )
        cursor = 0
        for split in IN_DIST_SLICE_ORDER:
            count = int(split_quotas[split][difficulty])
            allocated[split][difficulty] = ordered[cursor : cursor + count]
            cursor += count
        if cursor != len(ordered):
            raise RuntimeError(f"Allocation did not consume complete {difficulty} pool")
    allocated["test_ood_long"]["ood_long"] = pools["ood_long"]
    allocated["test_ood_division"]["ood_division"] = pools["ood_division"]

    split_records: dict[str, list[dict[str, Any]]] = {}
    for split in SPLIT_ORDER:
        records: list[dict[str, Any]] = []
        for difficulty, tasks in allocated[split].items():
            allocation_seed = (
                base_seed + ALLOCATION_SEED_OFFSETS[difficulty]
                if difficulty in ALLOCATION_SEED_OFFSETS
                else None
            )
            records.extend(
                _record_for_task(
                    task,
                    split=split,
                    difficulty=difficulty,
                    index=index,
                    allocation_seed=allocation_seed,
                )
                for index, task in enumerate(tasks)
            )
        order_seed = base_seed + SPLIT_ORDER_SEED_OFFSETS[split]
        split_records[split] = _stable_order(
            records, seed=order_seed, domain=f"split/{split}"
        )

    stats = {
        "pool_stats": pool_stats,
        "global_loose_keys": len(global_seen),
        "max_proposals_per_pool": max_proposals,
    }
    return split_records, stats


def build_artifact_bytes(
    *,
    base_seed: int = BASE_SEED,
    split_quotas: Mapping[str, Mapping[str, int]] = SPLIT_QUOTAS,
    max_proposals: int = MAX_PROPOSALS_PER_POOL,
) -> tuple[dict[str, bytes], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    records, stats = build_records(
        base_seed=base_seed,
        split_quotas=split_quotas,
        max_proposals=max_proposals,
    )
    artifacts: dict[str, bytes] = {}
    for split in SPLIT_ORDER:
        artifacts[f"{split}.jsonl"] = canonical_jsonl_bytes(records[split])
        ids = "".join(f"{record['id']}\n" for record in records[split])
        artifacts[f"task_ids/{split}.txt"] = ids.encode("utf-8")
    return artifacts, stats, records


def bytes_record(payload: bytes) -> dict[str, Any]:
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
        "line_count": payload.count(b"\n") + int(bool(payload) and not payload.endswith(b"\n")),
    }


def _set_digest(values: list[str]) -> str:
    return hashlib.sha256("".join(f"{value}\n" for value in sorted(values)).encode()).hexdigest()


def build_manifest(
    *,
    source_commit: str,
    source_records: Mapping[str, Mapping[str, Any]],
    artifacts: Mapping[str, bytes],
    stats: Mapping[str, Any],
    records: Mapping[str, list[dict[str, Any]]],
    base_seed: int = BASE_SEED,
    split_quotas: Mapping[str, Mapping[str, int]] = SPLIT_QUOTAS,
) -> dict[str, Any]:
    final_records = records["final_test_in_dist"]
    final_ids = [str(record["id"]) for record in final_records]
    final_semantics = [json.dumps(semantic_key(record), separators=(",", ":")) for record in final_records]
    final_loose_keys = [json.dumps(loose_key(record), separators=(",", ":")) for record in final_records]
    final_row_digests = [hashlib.sha256(canonical_json_bytes(record)).hexdigest() for record in final_records]
    core = {
        "schema_version": MANIFEST_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "generator_version": GENERATOR_VERSION,
        "source_commit": source_commit,
        "source_records": {key: dict(value) for key, value in sorted(source_records.items())},
        "runtime_contract": {
            "python_implementation": "cpython",
            "python_major_minor": "3.11",
            "rng": "random.Random",
            "serialization": "utf8_sorted_keys_compact_json_one_lf_per_record",
            "ordering": "sha256_seed_domain_semantic_key_then_semantic_key",
        },
        "base_seed": base_seed,
        "generation_order": list(GENERATION_ORDER),
        "in_dist_slice_order": list(IN_DIST_SLICE_ORDER),
        "split_order": list(SPLIT_ORDER),
        "proposal_seed_offsets": PROPOSAL_SEED_OFFSETS,
        "allocation_seed_offsets": ALLOCATION_SEED_OFFSETS,
        "split_order_seed_offsets": SPLIT_ORDER_SEED_OFFSETS,
        "max_proposals_per_pool": int(stats["max_proposals_per_pool"]),
        "easy_loose_key_capacity": EASY_LOOSE_KEY_CAPACITY,
        "split_quotas": {split: dict(values) for split, values in split_quotas.items()},
        "generation_stats": dict(stats),
        "final_test_policy": FINAL_TEST_POLICY,
        "final_test_protection": {
            "row_count": len(final_records),
            "jsonl_sha256": bytes_record(artifacts["final_test_in_dist.jsonl"])["sha256"],
            "ordered_ids_sha256": bytes_record(
                artifacts["task_ids/final_test_in_dist.txt"]
            )["sha256"],
            "id_set_sha256": _set_digest(final_ids),
            "semantic_key_set_sha256": _set_digest(final_semantics),
            "loose_key_set_sha256": _set_digest(final_loose_keys),
            "canonical_row_digest_set_sha256": _set_digest(final_row_digests),
        },
        "artifacts_exclude_manifest": True,
        "artifacts": {path: bytes_record(payload) for path, payload in sorted(artifacts.items())},
    }
    return {**core, "manifest_core_sha256": content_sha256(core)}


def write_dataset_atomic(
    output_dir: str | Path,
    *,
    artifacts: Mapping[str, bytes],
    manifest: Mapping[str, Any],
    legacy_dir: str | Path,
) -> None:
    output = Path(output_dir)
    legacy = Path(legacy_dir).resolve()
    resolved_output = output.resolve()
    if resolved_output == legacy or legacy in resolved_output.parents:
        raise ValueError(f"Refusing to overwrite legacy dataset: {legacy}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing pre-existing Countdown-v2 target: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        for relative_path, payload in sorted(artifacts.items()):
            destination = temporary / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("xb") as handle:
            handle.write(canonical_json_bytes(dict(manifest), pretty=True))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
