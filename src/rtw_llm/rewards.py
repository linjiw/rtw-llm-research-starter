"""Reward manager used by GRPO/RLOO-style post-training."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .countdown import score_completion, verify_completion
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
    ) -> None:
        self.teacher = teacher
        self.primary_weight = primary_weight
        self.log_path = Path(log_path) if log_path else None
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
            rewards.append(total)
            component_records.append(components)
            log_records.append(
                {
                    "id": example.get("id"),
                    "difficulty": example.get("difficulty"),
                    "reward": total,
                    "weights": weights,
                    "components": components,
                    "expression": result.expression,
                    "value": result.value,
                    "correct": result.correct,
                    "error": result.error,
                }
            )

        self.teacher.update(component_records)
        self._log(log_records)
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
