"""Explicit seed-role contracts for reproducible training protocols."""
from __future__ import annotations

from collections.abc import Callable

LEGACY_SEED_PROTOCOL = "countdown-legacy-v1"
TRUE_SEED_PROTOCOL = "countdown-true-seeds-v2"
SEED_PROTOCOLS = (LEGACY_SEED_PROTOCOL, TRUE_SEED_PROTOCOL)


def resolve_grpo_seed_plan(
    teacher_seed: int,
    trainer_seed: int = 42,
    protocol_id: str = LEGACY_SEED_PROTOCOL,
) -> dict[str, int | str]:
    """Resolve GRPO seed roles while preserving the archived protocol."""
    _validate_protocol_and_seeds(protocol_id, teacher_seed, trainer_seed)
    if protocol_id == LEGACY_SEED_PROTOCOL and trainer_seed != 42:
        raise ValueError(
            f"{LEGACY_SEED_PROTOCOL} requires trainer_seed=42; got {trainer_seed}. "
            f"Use {TRUE_SEED_PROTOCOL} for corrected true-seed runs."
        )
    if protocol_id == TRUE_SEED_PROTOCOL and trainer_seed != teacher_seed:
        raise ValueError(
            f"{TRUE_SEED_PROTOCOL} requires trainer_seed == teacher_seed; "
            f"got trainer_seed={trainer_seed}, teacher_seed={teacher_seed}."
        )
    return {
        "protocol_id": protocol_id,
        "teacher_seed": int(teacher_seed),
        "trainer_seed": int(trainer_seed),
        # CurriculumSampler inherits train_args.seed.
        "curriculum_seed": int(trainer_seed),
    }


def resolve_sft_seed_plan(
    trainer_seed: int,
    protocol_id: str = LEGACY_SEED_PROTOCOL,
) -> dict[str, int | str]:
    """Resolve SFT seed roles under the same legacy/corrected contract."""
    _validate_protocol_and_seeds(protocol_id, trainer_seed)
    return {
        "protocol_id": protocol_id,
        "trainer_seed": int(trainer_seed),
    }


def apply_pre_model_seed(
    seed_plan: dict[str, int | str],
    seed_setter: Callable[[int], None] | None = None,
) -> bool:
    """Seed all process RNGs before model/LoRA creation for corrected-v2 only.

    TRL 1.7 constructs fresh PEFT adapters before its Trainer-level set_seed.
    Corrected runs must seed here; legacy runs intentionally preserve their
    archived, uncontrolled pre-trainer initialization behavior.
    """
    if seed_plan["protocol_id"] != TRUE_SEED_PROTOCOL:
        return False
    if seed_setter is None:
        from transformers import set_seed

        seed_setter = set_seed
    seed_setter(int(seed_plan["trainer_seed"]))
    return True


def _validate_protocol_and_seeds(protocol_id: str, *seeds: int) -> None:
    if protocol_id not in SEED_PROTOCOLS:
        raise ValueError(f"Unknown seed protocol: {protocol_id!r}; choose from {SEED_PROTOCOLS}")
    if any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be non-negative integers")
