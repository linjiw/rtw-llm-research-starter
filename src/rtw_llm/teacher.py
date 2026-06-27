"""Reward Training Wheels teacher/controller for adaptive auxiliary rewards."""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

AUX_KEYS = ["format", "valid_expression", "uses_numbers", "allowed_ops", "brevity"]


@dataclass
class TeacherConfig:
    strategy: str = "adaptive"  # adaptive | static | manual | random
    min_weight: float = 0.02
    max_weight: float = 0.35
    init_weight: float = 0.20
    ema_beta: float = 0.90
    lr: float = 0.30
    primary_success_decay: float = 0.75
    manual_warmup_steps: int = 100
    seed: int = 0
    log_path: str | None = None
    aux_keys: list[str] = field(default_factory=lambda: AUX_KEYS.copy())


class RTWTeacher:
    """Adaptive controller for auxiliary reward weights.

    The teacher state mirrors the RTW idea: it stores recent/EMA primary reward,
    auxiliary component scores, and previous weights. The default strategy is a
    simple, robust controller instead of an outer-loop PPO teacher so the first
    LLM experiment stays inexpensive and stable.
    """

    def __init__(self, config: TeacherConfig | None = None):
        self.config = config or TeacherConfig()
        if self.config.strategy not in {"adaptive", "static", "manual", "random"}:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")
        self.step = 0
        self.rng = random.Random(self.config.seed)
        self.ema_primary = 0.0
        self.ema_aux = {k: 0.0 for k in self.config.aux_keys}
        self.weights = {k: float(self.config.init_weight) for k in self.config.aux_keys}
        self.history: list[dict] = []
        self.log_path = Path(self.config.log_path) if self.config.log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("")

    def get_weights(self) -> dict[str, float]:
        if self.config.strategy == "random":
            return {
                k: self.rng.uniform(self.config.min_weight, self.config.max_weight)
                for k in self.config.aux_keys
            }
        if self.config.strategy == "manual":
            progress = min(1.0, self.step / max(1, self.config.manual_warmup_steps))
            # Manual training wheels: start high and decay linearly.
            return {
                k: self.config.max_weight * (1 - progress) + self.config.min_weight * progress
                for k in self.config.aux_keys
            }
        return dict(self.weights)

    def update(self, batch_components: Iterable[dict[str, float]]) -> dict[str, float]:
        comps = list(batch_components)
        if not comps:
            return self.get_weights()

        beta = self.config.ema_beta
        batch_primary = sum(c.get("correct", 0.0) for c in comps) / len(comps)
        self.ema_primary = beta * self.ema_primary + (1 - beta) * batch_primary

        batch_aux: dict[str, float] = {}
        for key in self.config.aux_keys:
            batch_aux[key] = sum(c.get(key, 0.0) for c in comps) / len(comps)
            self.ema_aux[key] = beta * self.ema_aux[key] + (1 - beta) * batch_aux[key]

        if self.config.strategy == "adaptive":
            self._adaptive_update()
        elif self.config.strategy == "static":
            self.weights = {k: float(self.config.init_weight) for k in self.config.aux_keys}
        elif self.config.strategy == "manual":
            self.weights = self.get_weights()
        elif self.config.strategy == "random":
            self.weights = self.get_weights()

        self.step += 1
        record = {
            "step": self.step,
            "strategy": self.config.strategy,
            "ema_primary": self.ema_primary,
            "ema_aux": dict(self.ema_aux),
            "batch_primary": batch_primary,
            "batch_aux": batch_aux,
            "weights": dict(self.weights),
        }
        self.history.append(record)
        self._log(record)
        return dict(self.weights)

    def _adaptive_update(self) -> None:
        # Core RTW intuition:
        #   1. If the student is failing an auxiliary behavior, increase that wheel.
        #   2. As primary success rises, gradually remove dependence on wheels.
        competence = max(0.0, min(1.0, self.ema_primary))
        global_decay = max(0.0, 1.0 - self.config.primary_success_decay * competence)
        new_weights: dict[str, float] = {}
        for key, old in self.weights.items():
            need = 1.0 - max(0.0, min(1.0, self.ema_aux.get(key, 0.0)))
            target = self.config.min_weight + (self.config.max_weight - self.config.min_weight) * need * global_decay
            target = max(self.config.min_weight, min(self.config.max_weight, target))
            updated = (1 - self.config.lr) * old + self.config.lr * target
            new_weights[key] = float(updated)
        self.weights = new_weights

    def _log(self, record: dict) -> None:
        if not self.log_path:
            return
        with self.log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def state_dict(self) -> dict:
        return {
            "config": asdict(self.config),
            "step": self.step,
            "ema_primary": self.ema_primary,
            "ema_aux": dict(self.ema_aux),
            "weights": dict(self.weights),
            "history": self.history[-1000:],
        }
