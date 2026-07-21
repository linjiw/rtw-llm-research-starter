"""Task-clustered inference for correlated candidate/run panels.

The functions in this module deliberately make the semantic task the
resampling unit. They do not turn repeated candidates or repeated runs into
independent observations.
"""
from __future__ import annotations

import json
import hashlib
import math
import numbers as numeric_types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


class ClusterInferenceError(ValueError):
    """Raised when a panel cannot support the declared clustered estimand."""


def _canonical_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numeric_types.Real):
        raise ClusterInferenceError(f"{label} must be an integer, found {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ClusterInferenceError(f"{label} must be an integer, found {value!r}")
    return int(value)


def semantic_task_key(row: Mapping[str, Any]) -> str:
    """Return the canonical Countdown semantic key for one candidate/task row."""
    try:
        raw_numbers = row["numbers"]
        raw_ops_value = row["allowed_ops"]
        if isinstance(raw_numbers, (str, bytes)) or not isinstance(raw_numbers, Sequence):
            raise TypeError("numbers must be a sequence")
        if isinstance(raw_ops_value, (str, bytes)) or not isinstance(raw_ops_value, Sequence):
            raise TypeError("allowed_ops must be a sequence")
        numbers = sorted(
            _canonical_integer(value, label="number") for value in raw_numbers
        )
        target = _canonical_integer(row["target"], label="target")
        raw_ops = list(raw_ops_value)
    except (KeyError, TypeError, ValueError) as exc:
        raise ClusterInferenceError(f"malformed task identity: {exc}") from exc
    if not numbers or not raw_ops or any(not isinstance(value, str) for value in raw_ops):
        raise ClusterInferenceError("task identity requires numbers and string allowed_ops")
    allowed_ops = sorted(set(raw_ops))
    return json.dumps(
        {"allowed_ops": allowed_ops, "numbers": numbers, "target": target},
        sort_keys=True,
        separators=(",", ":"),
    )


def _input_artifact_identity(value: Any) -> dict[str, Any] | None:
    if value in (None, ""):
        return None
    path = Path(value)
    if not path.is_file():
        return {"declared_path": path.as_posix(), "content_verified": False}
    digest = hashlib.sha256()
    size = 0
    line_count = 0
    ended_with_newline = True
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
            line_count += chunk.count(b"\n")
            ended_with_newline = chunk.endswith(b"\n")
    if size and not ended_with_newline:
        line_count += 1
    return {
        "sha256": digest.hexdigest(),
        "size": size,
        "line_count": line_count,
        "content_verified": True,
    }


def evaluation_protocol_signature(
    config: Mapping[str, Any],
    *,
    split: str | None = None,
    strict_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Identity of frozen evaluation choices that every compared bank must share."""
    resolved = dict((strict_identity or {}).get("resolved_config", {}))
    requested = dict((strict_identity or {}).get("requested_args", {}))

    def get_value(key: str, *aliases: str, default: Any = None) -> Any:
        for source in (config, resolved, requested):
            for candidate in (key, *aliases):
                if candidate in source and source[candidate] is not None:
                    return source[candidate]
        return default

    mode = get_value("hf_gen_mode", default="loop")
    strict_inputs = dict((strict_identity or {}).get("inputs", {}))
    data_identity = strict_inputs.get("data") or _input_artifact_identity(
        get_value("data_path")
    )
    task_identity = strict_inputs.get("ordered_task_ids") or _input_artifact_identity(
        get_value("task_ids_file")
    )
    return {
        "sampling_seed": get_value("sampling_seed", "seed"),
        "temperature": get_value("temperature"),
        "top_p": get_value("top_p"),
        "max_new_tokens": get_value("max_new_tokens"),
        "prompt_field": get_value("prompt_field", default="prompt"),
        "engine": get_value("engine", default="hf"),
        "device": get_value("device"),
        "hf_gen_mode": mode,
        "batch_size": get_value("batch_size") if mode == "batched" else None,
        "effective_generation_config": get_value("effective_generation_config"),
        "split": split or get_value("split"),
        "limit": get_value("limit"),
        "n_examples": get_value("n_examples"),
        "max_n": get_value("max_n"),
        "data_input": data_identity,
        "ordered_task_ids_input": task_identity,
    }


def require_matching_evaluation_signatures(
    signatures: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if not signatures:
        raise ClusterInferenceError("no evaluation signatures to compare")
    iterator = iter(signatures.items())
    reference_name, reference = next(iterator)
    reference_dict = dict(reference)
    for name, signature in iterator:
        if dict(signature) != reference_dict:
            differing = sorted(
                key
                for key in set(reference_dict) | set(signature)
                if reference_dict.get(key) != signature.get(key)
            )
            raise ClusterInferenceError(
                f"evaluation protocol mismatch: {reference_name!r} vs {name!r}; "
                f"differing_fields={differing}"
            )
    return reference_dict


def require_complete_evaluation_signature(
    signature: Mapping[str, Any], *, label: str
) -> None:
    required_scalars = (
        "sampling_seed",
        "temperature",
        "top_p",
        "max_new_tokens",
        "prompt_field",
        "engine",
        "hf_gen_mode",
        "split",
    )
    missing = [key for key in required_scalars if signature.get(key) is None]
    for key in ("data_input", "ordered_task_ids_input"):
        identity = signature.get(key)
        if not isinstance(identity, Mapping) or not identity.get("sha256"):
            missing.append(key)
    if missing:
        raise ClusterInferenceError(
            f"{label}: incomplete evaluation protocol signature; missing={sorted(missing)}"
        )


def stack_task_runs(
    runs: Sequence[Mapping[str, float]], *, label: str
) -> dict[str, tuple[float, ...]]:
    """Stack balanced semantic-task maps into a task-by-run panel."""
    if not runs:
        raise ClusterInferenceError(f"{label}: no runs")
    expected = set(runs[0])
    if not expected:
        raise ClusterInferenceError(f"{label}: run has no tasks")
    for index, run in enumerate(runs[1:], start=1):
        current = set(run)
        if current != expected:
            missing = sorted(expected - current)
            extra = sorted(current - expected)
            raise ClusterInferenceError(
                f"{label}: incomplete task grid in run {index}; "
                f"missing={missing[:3]} extra={extra[:3]}"
            )
    return {key: tuple(float(run[key]) for run in runs) for key in sorted(expected)}


def _validated_panel(
    panel: Mapping[str, Sequence[float]], *, label: str, bounded: bool = True
) -> tuple[list[str], np.ndarray]:
    if not panel:
        raise ClusterInferenceError(f"{label}: panel has no tasks")
    keys = sorted(panel)
    rows: list[np.ndarray] = []
    run_count: int | None = None
    for key in keys:
        values = np.asarray(panel[key], dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ClusterInferenceError(f"{label}: task {key!r} has no one-dimensional run vector")
        if run_count is None:
            run_count = int(values.size)
        elif values.size != run_count:
            raise ClusterInferenceError(
                f"{label}: unbalanced run count for task {key!r}; "
                f"expected {run_count}, found {values.size}"
            )
        if not np.all(np.isfinite(values)):
            raise ClusterInferenceError(f"{label}: task {key!r} contains non-finite values")
        if bounded and (np.any(values < 0.0) or np.any(values > 1.0)):
            raise ClusterInferenceError(f"{label}: task {key!r} contains outcomes outside [0, 1]")
        rows.append(values)
    return keys, np.stack(rows)


def _paired_panels(
    arm: Mapping[str, Sequence[float]],
    baseline: Mapping[str, Sequence[float]],
    *,
    bounded: bool = True,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    arm_keys, arm_matrix = _validated_panel(arm, label="arm", bounded=bounded)
    baseline_keys, baseline_matrix = _validated_panel(
        baseline, label="baseline", bounded=bounded
    )
    if arm_keys != baseline_keys:
        missing = sorted(set(baseline_keys) - set(arm_keys))
        extra = sorted(set(arm_keys) - set(baseline_keys))
        raise ClusterInferenceError(
            f"semantic task mismatch; missing_from_arm={missing[:3]} extra_in_arm={extra[:3]}"
        )
    return arm_keys, arm_matrix, baseline_matrix


def _percentile_interval(samples: np.ndarray, confidence: float) -> tuple[float, float]:
    if not 0.0 < confidence < 1.0:
        raise ClusterInferenceError("confidence must be strictly between 0 and 1")
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def _task_bootstrap_means(values: np.ndarray, draws: int, rng: np.random.Generator) -> np.ndarray:
    if draws <= 0:
        raise ClusterInferenceError("bootstrap draws must be positive")
    task_count = int(values.size)
    output = np.empty(draws, dtype=float)
    chunk_size = max(1, min(1024, 2_000_000 // max(task_count, 1)))
    for start in range(0, draws, chunk_size):
        stop = min(start + chunk_size, draws)
        indices = rng.integers(0, task_count, size=(stop - start, task_count))
        output[start:stop] = values[indices].mean(axis=1)
    return output


def _sign_flip_test(
    effects: np.ndarray,
    *,
    draws: int,
    rng: np.random.Generator,
    exact_max_nonzero: int,
) -> dict[str, Any]:
    nonzero = effects[~np.isclose(effects, 0.0, rtol=0.0, atol=1e-15)]
    nonzero_count = int(nonzero.size)
    if nonzero_count == 0:
        return {
            "method": "exact_task_sign_flip",
            "p_value_two_sided": 1.0,
            "nonzero_task_clusters": 0,
            "assignments": 1,
        }

    observed = abs(float(nonzero.sum()))
    tolerance = 1e-12
    if nonzero_count <= exact_max_nonzero:
        assignments = 1 << nonzero_count
        extreme = 0
        chunk_size = 65_536
        bit_positions = np.arange(nonzero_count, dtype=np.uint64)
        for start in range(0, assignments, chunk_size):
            stop = min(start + chunk_size, assignments)
            codes = np.arange(start, stop, dtype=np.uint64)[:, None]
            signs = 1.0 - 2.0 * ((codes >> bit_positions) & 1)
            signed_sums = signs @ nonzero
            extreme += int(np.count_nonzero(np.abs(signed_sums) >= observed - tolerance))
        return {
            "method": "exact_task_sign_flip",
            "p_value_two_sided": float(extreme / assignments),
            "nonzero_task_clusters": nonzero_count,
            "assignments": assignments,
        }

    if draws <= 0:
        raise ClusterInferenceError("sign-flip draws must be positive")
    extreme = 0
    chunk_size = max(1, min(4096, 2_000_000 // nonzero_count))
    for start in range(0, draws, chunk_size):
        count = min(chunk_size, draws - start)
        signs = rng.integers(0, 2, size=(count, nonzero_count), dtype=np.int8)
        signed_sums = (1.0 - 2.0 * signs) @ nonzero
        extreme += int(np.count_nonzero(np.abs(signed_sums) >= observed - tolerance))
    return {
        "method": "monte_carlo_task_sign_flip",
        "p_value_two_sided": float((extreme + 1) / (draws + 1)),
        "nonzero_task_clusters": nonzero_count,
        "draws": draws,
        "finite_sample_correction": "(extreme+1)/(draws+1)",
    }


def task_clustered_difference(
    arm: Mapping[str, Sequence[float]],
    baseline: Mapping[str, Sequence[float]],
    *,
    bootstrap_draws: int = 20_000,
    sign_flip_draws: int = 20_000,
    seed: int = 17,
    confidence: float = 0.95,
    exact_sign_flip_max_nonzero: int = 20,
) -> dict[str, Any]:
    """Estimate an equal-task contrast conditional on observed run panels."""
    keys, arm_matrix, baseline_matrix = _paired_panels(arm, baseline)
    task_effects = arm_matrix.mean(axis=1) - baseline_matrix.mean(axis=1)
    seed_sequence = np.random.SeedSequence(seed)
    bootstrap_seed, sign_flip_seed = seed_sequence.spawn(2)
    bootstrap = _task_bootstrap_means(
        task_effects, bootstrap_draws, np.random.default_rng(bootstrap_seed)
    )
    lower, upper = _percentile_interval(bootstrap, confidence)
    sign_flip = _sign_flip_test(
        task_effects,
        draws=sign_flip_draws,
        rng=np.random.default_rng(sign_flip_seed),
        exact_max_nonzero=exact_sign_flip_max_nonzero,
    )
    return {
        "schema_version": "rtw-task-cluster-inference-v1",
        "available": True,
        "estimand": "equal_task_mean_difference_conditional_on_observed_run_panels",
        "estimate": float(task_effects.mean()),
        "task_clusters": len(keys),
        "arm_runs_per_task": int(arm_matrix.shape[1]),
        "baseline_runs_per_task": int(baseline_matrix.shape[1]),
        "confidence_interval": {
            "method": "task_cluster_percentile_bootstrap",
            "confidence": confidence,
            "lower": lower,
            "upper": upper,
        },
        "bootstrap": {
            "draws": bootstrap_draws,
            "seed": seed,
            "seed_derivation": "SeedSequence(seed).spawn(2)[0]",
            "bit_generator": "PCG64",
            "resampling_unit": "semantic_task",
        },
        "sign_flip": sign_flip,
        "sign_flip_seed_derivation": "SeedSequence(seed).spawn(2)[1]",
        "claim_scope": "observed_run_panels_only_not_training_seed_population",
    }


def _validated_ratio_components(
    numerators: Mapping[str, Sequence[float]],
    denominators: Mapping[str, Sequence[float]],
    *,
    label: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    numerator_keys, numerator_matrix = _validated_panel(
        numerators, label=f"{label}_numerators", bounded=False
    )
    denominator_keys, denominator_matrix = _validated_panel(
        denominators, label=f"{label}_denominators", bounded=False
    )
    if numerator_keys != denominator_keys or numerator_matrix.shape != denominator_matrix.shape:
        raise ClusterInferenceError(f"{label}: numerator/denominator task-run grid mismatch")
    if np.any(numerator_matrix < 0.0) or np.any(denominator_matrix < 0.0):
        raise ClusterInferenceError(f"{label}: ratio counts must be nonnegative")
    if np.any(numerator_matrix > denominator_matrix):
        raise ClusterInferenceError(f"{label}: numerator exceeds denominator")
    return numerator_keys, numerator_matrix, denominator_matrix


def task_clustered_ratio_difference(
    arm_numerators: Mapping[str, Sequence[float]],
    arm_denominators: Mapping[str, Sequence[float]],
    baseline_numerators: Mapping[str, Sequence[float]],
    baseline_denominators: Mapping[str, Sequence[float]],
    *,
    bootstrap_draws: int = 20_000,
    seed: int = 17,
    confidence: float = 0.95,
    max_dropped_fraction: float = 0.01,
) -> dict[str, Any]:
    """Compare pooled conditional ratios with whole-task bootstrap resampling."""
    arm_keys, arm_num, arm_den = _validated_ratio_components(
        arm_numerators, arm_denominators, label="arm"
    )
    baseline_keys, baseline_num, baseline_den = _validated_ratio_components(
        baseline_numerators, baseline_denominators, label="baseline"
    )
    if arm_keys != baseline_keys:
        raise ClusterInferenceError("semantic task mismatch between ratio panels")
    arm_den_total = float(arm_den.sum())
    baseline_den_total = float(baseline_den.sum())
    common = {
        "schema_version": "rtw-task-cluster-ratio-inference-v1",
        "estimand": "pooled_exact_given_legal_difference_conditional_on_observed_panels",
        "task_clusters": len(arm_keys),
        "arm_runs_per_task": int(arm_num.shape[1]),
        "baseline_runs_per_task": int(baseline_num.shape[1]),
        "post_treatment_noncausal": True,
        "bootstrap": {
            "draws": bootstrap_draws,
            "seed": seed,
            "bit_generator": "PCG64",
            "resampling_unit": "semantic_task",
        },
    }
    if arm_den_total == 0.0 or baseline_den_total == 0.0:
        return {
            **common,
            "available": False,
            "reason": "zero_observed_legal_denominator",
            "arm_legal_denominator": arm_den_total,
            "baseline_legal_denominator": baseline_den_total,
        }
    if bootstrap_draws <= 0:
        raise ClusterInferenceError("bootstrap draws must be positive")
    if not 0.0 <= max_dropped_fraction < 1.0:
        raise ClusterInferenceError("max_dropped_fraction must be in [0, 1)")

    arm_num_task = arm_num.sum(axis=1)
    arm_den_task = arm_den.sum(axis=1)
    baseline_num_task = baseline_num.sum(axis=1)
    baseline_den_task = baseline_den.sum(axis=1)
    task_count = len(arm_keys)
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    samples = np.empty(bootstrap_draws, dtype=float)
    valid_draws = 0
    dropped_draws = 0
    chunk_size = max(1, min(1024, 2_000_000 // max(task_count, 1)))
    for start in range(0, bootstrap_draws, chunk_size):
        count = min(chunk_size, bootstrap_draws - start)
        indices = rng.integers(0, task_count, size=(count, task_count))
        arm_den_draw = arm_den_task[indices].sum(axis=1)
        base_den_draw = baseline_den_task[indices].sum(axis=1)
        valid = (arm_den_draw > 0.0) & (base_den_draw > 0.0)
        valid_count = int(np.count_nonzero(valid))
        if valid_count:
            arm_ratio = arm_num_task[indices][valid].sum(axis=1) / arm_den_draw[valid]
            base_ratio = baseline_num_task[indices][valid].sum(axis=1) / base_den_draw[valid]
            samples[valid_draws : valid_draws + valid_count] = arm_ratio - base_ratio
            valid_draws += valid_count
        dropped_draws += count - valid_count
    if valid_draws == 0:
        return {
            **common,
            "available": False,
            "reason": "all_bootstrap_draws_have_zero_legal_denominator",
            "dropped_bootstrap_draws": dropped_draws,
        }
    samples = samples[:valid_draws]
    arm_ratio = float(arm_num.sum() / arm_den_total)
    baseline_ratio = float(baseline_num.sum() / baseline_den_total)
    estimate = arm_ratio - baseline_ratio
    dropped_fraction = dropped_draws / bootstrap_draws
    support = {
        "valid_bootstrap_draws": valid_draws,
        "dropped_bootstrap_draws": dropped_draws,
        "dropped_bootstrap_fraction": dropped_fraction,
        "max_allowed_dropped_fraction": max_dropped_fraction,
    }
    if dropped_fraction > max_dropped_fraction:
        return {
            **common,
            **support,
            "available": False,
            "reason": "sparse_legal_support_exceeds_dropped_draw_threshold",
            "descriptive_estimate": estimate,
            "arm_ratio": arm_ratio,
            "baseline_ratio": baseline_ratio,
        }
    lower, upper = _percentile_interval(samples, confidence)
    return {
        **common,
        **support,
        "available": True,
        "estimate": estimate,
        "arm_ratio": arm_ratio,
        "baseline_ratio": baseline_ratio,
        "confidence_interval": {
            "method": "task_cluster_ratio_percentile_bootstrap",
            "confidence": confidence,
            "lower": lower,
            "upper": upper,
        },
        "claim_scope": "post_treatment_association_not_causal_conversion",
    }


def task_seed_product_bootstrap_difference(
    arm: Mapping[str, Sequence[float]],
    baseline: Mapping[str, Sequence[float]],
    *,
    true_seed_protocol: bool,
    bootstrap_draws: int = 20_000,
    seed: int = 17,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Exploratory product bootstrap over paired tasks and true training seeds."""
    if not true_seed_protocol:
        raise ClusterInferenceError("task-by-seed bootstrap requires countdown-true-seeds-v2")
    keys, arm_matrix, baseline_matrix = _paired_panels(arm, baseline)
    if arm_matrix.shape != baseline_matrix.shape:
        raise ClusterInferenceError("task-by-seed bootstrap requires paired seed grids")
    seed_count = int(arm_matrix.shape[1])
    if seed_count < 3:
        raise ClusterInferenceError("task-by-seed bootstrap requires at least three true seeds")
    if bootstrap_draws <= 0:
        raise ClusterInferenceError("bootstrap draws must be positive")
    effects = arm_matrix - baseline_matrix
    task_count = len(keys)
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    samples = np.empty(bootstrap_draws, dtype=float)
    chunk_size = max(1, min(512, 2_000_000 // max(task_count * seed_count, 1)))
    for start in range(0, bootstrap_draws, chunk_size):
        count = min(chunk_size, bootstrap_draws - start)
        task_indices = rng.integers(0, task_count, size=(count, task_count))
        seed_indices = rng.integers(0, seed_count, size=(count, seed_count))
        sampled = effects[task_indices[:, :, None], seed_indices[:, None, :]]
        samples[start : start + count] = sampled.mean(axis=(1, 2))
    lower, upper = _percentile_interval(samples, confidence)
    status = (
        "exploratory_underpowered_seed_generalization"
        if seed_count == 3
        else "exploratory_seed_generalization"
    )
    return {
        "schema_version": "rtw-task-seed-product-bootstrap-v1",
        "available": True,
        "estimand": "paired_task_and_true_training_seed_population_difference",
        "estimate": float(effects.mean()),
        "task_clusters": task_count,
        "true_training_seeds": seed_count,
        "status": status,
        "confirmatory_p_value": None,
        "confidence_interval": {
            "method": "task_by_seed_product_percentile_bootstrap",
            "confidence": confidence,
            "lower": lower,
            "upper": upper,
        },
        "bootstrap": {"draws": bootstrap_draws, "seed": seed, "bit_generator": "PCG64"},
    }
