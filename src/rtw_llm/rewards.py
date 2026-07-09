"""Reward manager used by GRPO/RLOO-style post-training."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .countdown import reward_breakdown, score_completion, verify_completion
from .teacher import RTWTeacher


def normalize_completion(completion: Any) -> str:
    """TRL may pass strings or chat-message structures depending on dataset format."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        # Common chat format: [{'role': 'assistant', 'content': '...'}]
        if completion and isinstance(completion[-1], dict) and "content" in completion[-1]:
            return str(completion[-1]["content"])
        return "\n".join(str(x) for x in completion)
    if isinstance(completion, dict) and "content" in completion:
        return str(completion["content"])
    return str(completion)


class RTWRewardManager:
    """Scores model completions and updates the RTW teacher.

    This object can be passed as a custom reward function to TRL's GRPOTrainer.
    For multi-GPU training, prefer a single-process smoke run first; then move the
    teacher update to rank 0 and broadcast weights.
    """

    def __init__(
        self,
        teacher: RTWTeacher,
        primary_weight: float = 1.0,
        log_path: str | None = None,
        curriculum: Any = None,
        group_size: int | None = None,
    ) -> None:
        self.__name__ = "rtw_reward"
        self.teacher = teacher
        self.primary_weight = primary_weight
        # Observe-only hook: the curriculum controller reads component scores to
        # steer task sampling; it must never alter rewards or logged components.
        self.curriculum = curriculum
        # GRPO forms advantage groups positionally (each prompt's generations
        # are consecutive), so set group_size=num_generations when used as a
        # TRL reward function. id-based grouping would merge distinct groups
        # that happen to share a prompt id within one batch.
        self.group_size = group_size
        self.log_path = Path(log_path) if log_path else None
        self.batch_index = 0
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("")

    def score_batch(self, completions: list[Any], examples: list[dict[str, Any]]) -> list[float]:
        weights = self.teacher.get_weights()
        rewards: list[float] = []
        component_records: list[dict[str, float]] = []
        log_records: list[dict[str, Any]] = []

        for completion_obj, example in zip(completions, examples):
            completion = normalize_completion(completion_obj)
            total, components, result = score_completion(
                completion,
                example,
                aux_weights=weights,
                primary_weight=self.primary_weight,
            )
            breakdown = reward_breakdown(components, weights, self.primary_weight)
            rewards.append(total)
            component_records.append(components)
            log_records.append(
                {
                    "id": example.get("id"),
                    "difficulty": example.get("difficulty"),
                    "reward_batch_index": self.batch_index,
                    "reward": total,
                    "primary_reward": breakdown["primary_reward"],
                    "primary_reward_weighted": breakdown["primary_reward_weighted"],
                    "aux_reward_weighted": breakdown["aux_reward_weighted"],
                    "total_reward": breakdown["total_reward"],
                    "weights": weights,
                    "components": components,
                    "expression": result.expression,
                    "value": result.value,
                    "correct": result.correct,
                    "error": result.error,
                }
            )

        reward_std = _population_std(rewards)
        for rec in log_records:
            rec["reward_batch_size"] = len(rewards)
            rec["reward_batch_reward_std"] = reward_std
            rec["reward_batch_has_variance"] = reward_std > 1e-9
        # GRPO advantages are computed within a prompt's generation group, so the
        # per-group std (not the cross-prompt batch std) is the signal that
        # determines whether a step carries gradient information. Groups are
        # positional (consecutive slices of group_size), matching TRL's layout.
        if self.group_size and len(rewards) % self.group_size == 0:
            group_stds = [
                _population_std(rewards[i : i + self.group_size])
                for i in range(0, len(rewards), self.group_size)
            ]
            groups_with_variance = sum(1 for s in group_stds if s > 1e-9)
            for i, rec in enumerate(log_records):
                std = group_stds[i // self.group_size]
                rec["group_reward_std"] = std
                rec["group_has_variance"] = std > 1e-9
                rec["batch_group_variance_fraction"] = groups_with_variance / len(group_stds)

        self.teacher.update(component_records)
        if self.curriculum is not None:
            self.curriculum.observe(
                [(ex.get("difficulty"), comp) for ex, comp in zip(examples, component_records)]
            )
        self._log(log_records)
        self.batch_index += 1
        return rewards

    def __call__(self, completions: list[Any], **kwargs: Any) -> list[float]:
        # The dataset columns are passed as keyword lists by TRL.
        n = len(completions)
        examples: list[dict[str, Any]] = []
        for i in range(n):
            examples.append(
                {
                    "id": _get_i(kwargs.get("id"), i),
                    "difficulty": _get_i(kwargs.get("difficulty"), i),
                    "numbers": _get_i(kwargs.get("numbers"), i),
                    "target": _get_i(kwargs.get("target"), i),
                    "allowed_ops": _get_i(kwargs.get("allowed_ops"), i),
                }
            )
        return self.score_batch(completions, examples)

    def _log(self, records: list[dict[str, Any]]) -> None:
        if not self.log_path:
            return
        with self.log_path.open("a") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")


def _get_i(value: Any, i: int) -> Any:
    if isinstance(value, (list, tuple)):
        return value[i]
    return value


def _population_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return float(variance**0.5)


def metrics_for_completion(completion: str, example: dict[str, Any]) -> dict[str, Any]:
    result = verify_completion(completion, example)
    comps = result.to_components()
    return {
        **comps,
        "expression": result.expression,
        "value": result.value,
        "error": result.error,
        "reward_hacking_candidate": float(
            result.found_answer_tag and not result.correct and (not result.uses_all_numbers or not result.numeric_ok)
        ),
    }
