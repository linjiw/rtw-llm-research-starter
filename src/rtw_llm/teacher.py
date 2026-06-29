"""Reward Training Wheels teacher/controller for adaptive auxiliary rewards."""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

AUX_KEYS = [
    "format",
    "valid_expression",
    "number_multiset_f1",
    "allowed_ops",
    "numeric_distance_reward",
    "brevity",
]

CONSTRAINT_AUX_KEYS = ("valid_expression", "number_multiset_f1", "allowed_ops")

STABLE_FLOORS = {
    "format": 0.03,
    "valid_expression": 0.16,
    "number_multiset_f1": 0.18,
    "allowed_ops": 0.12,
    "numeric_distance_reward": 0.00,
    "brevity": 0.02,
}

STABLE_CAPS = {
    "numeric_distance_reward": 0.20,
}

VALID_STRATEGIES = {"adaptive", "adaptive_stable", "adaptive_phased", "static", "manual", "random"}


@dataclass
class TeacherConfig:
    strategy: str = "adaptive"
    min_weight: float = 0.02
    max_weight: float = 0.35
    init_weight: float = 0.20
    ema_beta: float = 0.90
    lr: float = 0.30
    primary_success_decay: float = 0.75
    manual_warmup_steps: int = 100
    stable_delay_steps: int = 50
    stable_lr: float = 0.10
    stable_alpha: float = 0.10
    stable_target_weight_sum: float = 1.20
    stable_floors: dict[str, float] = field(default_factory=lambda: STABLE_FLOORS.copy())
    stable_caps: dict[str, float] = field(default_factory=lambda: STABLE_CAPS.copy())
    phased_number_f1_enter: float = 0.80
    phased_valid_enter: float = 0.35
    phased_number_f1_exit: float = 0.75
    phased_valid_exit: float = 0.30
    phased_min_dwell_updates: int = 5
    phased_phase_a_numeric_cap: float = 0.16
    phased_phase_b_numeric_cap: float = 0.25
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
        if self.config.strategy not in VALID_STRATEGIES:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")
        self.step = 0
        self.rng = random.Random(self.config.seed)
        self.ema_primary = 0.0
        self.ema_aux = {k: 0.0 for k in self.config.aux_keys}
        self.weights = {k: float(self.config.init_weight) for k in self.config.aux_keys}
        self.history: list[dict] = []
        self.last_diagnostics: dict = {}
        self.teacher_phase = "A" if self.config.strategy == "adaptive_phased" else None
        self.phase_switch_step: int | None = None
        self.phase_flip_count = 0
        self._phase_enter_streak = 0
        self._phase_exit_streak = 0
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

        previous_weights = dict(self.weights)
        raw_weights: dict[str, float] | None = None
        floor_hits: list[str] = []
        cap_hits: list[str] = []
        delay_active = False

        if self.config.strategy == "adaptive":
            self._adaptive_update()
        elif self.config.strategy == "adaptive_stable":
            raw_weights, floor_hits, cap_hits, delay_active = self._adaptive_stable_update(previous_weights)
        elif self.config.strategy == "adaptive_phased":
            raw_weights, floor_hits, cap_hits, delay_active = self._adaptive_phased_update(previous_weights)
        elif self.config.strategy == "static":
            self.weights = {k: float(self.config.init_weight) for k in self.config.aux_keys}
        elif self.config.strategy == "manual":
            self.weights = self.get_weights()
        elif self.config.strategy == "random":
            self.weights = self.get_weights()

        diagnostics = self._weight_diagnostics(
            previous_weights=previous_weights,
            raw_weights=raw_weights,
            floor_hits=floor_hits,
            cap_hits=cap_hits,
            delay_active=delay_active,
        )
        self.last_diagnostics = diagnostics
        self.step += 1
        record = {
            "step": self.step,
            "strategy": self.config.strategy,
            "ema_primary": self.ema_primary,
            "ema_aux": dict(self.ema_aux),
            "batch_primary": batch_primary,
            "batch_aux": batch_aux,
            "weights": dict(self.weights),
            "diagnostics": diagnostics,
        }
        self.history.append(record)
        self._log(record)
        return dict(self.weights)

    def _adaptive_update(self) -> None:
        self.weights = self._adaptive_candidate(self.weights)

    def _adaptive_candidate(
        self,
        base_weights: dict[str, float],
        lr: float | None = None,
    ) -> dict[str, float]:
        # Core RTW intuition:
        #   1. If the student is failing an auxiliary behavior, increase that wheel.
        #   2. As primary success rises, gradually remove dependence on wheels.
        competence = max(0.0, min(1.0, self.ema_primary))
        global_decay = max(0.0, 1.0 - self.config.primary_success_decay * competence)
        new_weights: dict[str, float] = {}
        for key, old in base_weights.items():
            need = 1.0 - max(0.0, min(1.0, self.ema_aux.get(key, 0.0)))
            target = self.config.min_weight + (self.config.max_weight - self.config.min_weight) * need * global_decay
            target = max(self.config.min_weight, min(self.config.max_weight, target))
            update_lr = self.config.lr if lr is None else lr
            updated = (1 - update_lr) * old + update_lr * target
            new_weights[key] = float(updated)
        return new_weights

    def _adaptive_stable_update(
        self,
        previous_weights: dict[str, float],
    ) -> tuple[dict[str, float], list[str], list[str], bool]:
        if self.step < self.config.stable_delay_steps:
            self.weights = {k: float(self.config.init_weight) for k in self.config.aux_keys}
            return dict(self.weights), [], [], True

        raw_weights = self._adaptive_candidate(previous_weights, lr=self.config.stable_lr)
        alpha = max(0.0, min(1.0, self.config.stable_alpha))
        smoothed = {
            key: (1 - alpha) * previous_weights[key] + alpha * raw_weights[key]
            for key in self.config.aux_keys
        }
        self.weights, floor_hits, cap_hits = self._project_stable_weights(smoothed)
        return raw_weights, floor_hits, cap_hits, False

    def _adaptive_phased_update(
        self,
        previous_weights: dict[str, float],
    ) -> tuple[dict[str, float], list[str], list[str], bool]:
        if self.step < self.config.stable_delay_steps:
            self.weights = {k: float(self.config.init_weight) for k in self.config.aux_keys}
            self.teacher_phase = "A"
            return dict(self.weights), [], [], True

        self._update_phase_state()
        raw_weights = self._adaptive_candidate(previous_weights, lr=self.config.stable_lr)
        alpha = max(0.0, min(1.0, self.config.stable_alpha))
        smoothed = {
            key: (1 - alpha) * previous_weights[key] + alpha * raw_weights[key]
            for key in self.config.aux_keys
        }
        floors, caps = self._phased_constraints()
        self.weights, floor_hits, cap_hits = self._project_stable_weights(smoothed, floors=floors, caps=caps)
        return raw_weights, floor_hits, cap_hits, False

    def _update_phase_state(self) -> None:
        if self.teacher_phase is None:
            self.teacher_phase = "A"
        number_f1 = self.ema_aux.get("number_multiset_f1", 0.0)
        valid_expression = self.ema_aux.get("valid_expression", 0.0)
        dwell = max(1, int(self.config.phased_min_dwell_updates))

        enter_b = (
            number_f1 >= self.config.phased_number_f1_enter
            and valid_expression >= self.config.phased_valid_enter
        )
        exit_b = (
            number_f1 < self.config.phased_number_f1_exit
            or valid_expression < self.config.phased_valid_exit
        )

        if self.teacher_phase == "A":
            self._phase_enter_streak = self._phase_enter_streak + 1 if enter_b else 0
            self._phase_exit_streak = 0
            if self._phase_enter_streak >= dwell:
                self.teacher_phase = "B"
                self.phase_flip_count += 1
                self.phase_switch_step = self.step
                self._phase_enter_streak = 0
        else:
            self._phase_exit_streak = self._phase_exit_streak + 1 if exit_b else 0
            self._phase_enter_streak = 0
            if self._phase_exit_streak >= dwell:
                self.teacher_phase = "A"
                self.phase_flip_count += 1
                self.phase_switch_step = self.step
                self._phase_exit_streak = 0

    def _phased_constraints(self) -> tuple[dict[str, float], dict[str, float]]:
        floors = self.config.stable_floors.copy()
        caps = self.config.stable_caps.copy()
        if self.teacher_phase == "B":
            caps["numeric_distance_reward"] = self.config.phased_phase_b_numeric_cap
            floors["valid_expression"] = max(floors.get("valid_expression", 0.0), 0.18)
            floors["number_multiset_f1"] = max(floors.get("number_multiset_f1", 0.0), 0.20)
        else:
            caps["numeric_distance_reward"] = self.config.phased_phase_a_numeric_cap
            floors["valid_expression"] = max(floors.get("valid_expression", 0.0), 0.22)
            floors["number_multiset_f1"] = max(floors.get("number_multiset_f1", 0.0), 0.28)
            floors["allowed_ops"] = max(floors.get("allowed_ops", 0.0), 0.12)
        return floors, caps

    def _project_stable_weights(
        self,
        candidate: dict[str, float],
        floors: dict[str, float] | None = None,
        caps: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], list[str], list[str]]:
        floor_config = self.config.stable_floors if floors is None else floors
        cap_config = self.config.stable_caps if caps is None else caps
        projected_floors = {
            key: max(self.config.min_weight, float(floor_config.get(key, self.config.min_weight)))
            for key in self.config.aux_keys
        }
        projected_caps = {
            key: min(self.config.max_weight, float(cap_config.get(key, self.config.max_weight)))
            for key in self.config.aux_keys
        }
        projected_caps = {key: max(projected_caps[key], projected_floors[key]) for key in self.config.aux_keys}

        weights = {
            key: min(projected_caps[key], max(projected_floors[key], float(candidate.get(key, self.config.init_weight))))
            for key in self.config.aux_keys
        }

        min_budget = sum(projected_floors.values())
        max_budget = sum(projected_caps.values())
        target = max(min_budget, min(float(self.config.stable_target_weight_sum), max_budget))
        eps = 1e-12
        for _ in range(32):
            total = sum(weights.values())
            diff = target - total
            if abs(diff) <= 1e-9:
                break
            if diff > 0:
                movable = [key for key in self.config.aux_keys if weights[key] < projected_caps[key] - eps]
                capacity = sum(projected_caps[key] - weights[key] for key in movable)
                if capacity <= eps:
                    break
                for key in movable:
                    weights[key] += diff * ((projected_caps[key] - weights[key]) / capacity)
                    weights[key] = min(projected_caps[key], weights[key])
            else:
                movable = [key for key in self.config.aux_keys if weights[key] > projected_floors[key] + eps]
                capacity = sum(weights[key] - projected_floors[key] for key in movable)
                if capacity <= eps:
                    break
                for key in movable:
                    weights[key] += diff * ((weights[key] - projected_floors[key]) / capacity)
                    weights[key] = max(projected_floors[key], weights[key])

        floor_hits = [key for key in self.config.aux_keys if weights[key] <= projected_floors[key] + 1e-8]
        cap_hits = [key for key in self.config.aux_keys if weights[key] >= projected_caps[key] - 1e-8]
        return {key: float(value) for key, value in weights.items()}, floor_hits, cap_hits

    def _weight_diagnostics(
        self,
        previous_weights: dict[str, float],
        raw_weights: dict[str, float] | None,
        floor_hits: list[str],
        cap_hits: list[str],
        delay_active: bool,
    ) -> dict:
        raw_weights = raw_weights or dict(self.weights)
        weight_sum = sum(self.weights.values())
        raw_weight_sum = sum(raw_weights.values())
        constraint_weight_mass = sum(self.weights.get(key, 0.0) for key in CONSTRAINT_AUX_KEYS)
        numeric_distance_weight = self.weights.get("numeric_distance_reward", 0.0)
        update_deltas = [
            abs(self.weights.get(key, 0.0) - previous_weights.get(key, 0.0))
            for key in self.config.aux_keys
        ]
        return {
            "delay_active": bool(delay_active),
            "teacher_phase": self.teacher_phase,
            "phase_switch_step": self.phase_switch_step,
            "phase_flip_count": int(self.phase_flip_count),
            "phase_enter_streak": int(self._phase_enter_streak),
            "phase_exit_streak": int(self._phase_exit_streak),
            "weight_sum": float(weight_sum),
            "raw_weight_sum": float(raw_weight_sum),
            "constraint_weight_mass": float(constraint_weight_mass),
            "numeric_distance_weight": float(numeric_distance_weight),
            "numeric_distance_to_constraint_ratio": float(
                numeric_distance_weight / max(constraint_weight_mass, 1e-12)
            ),
            "update_l1": float(sum(update_deltas)),
            "update_linf": float(max(update_deltas) if update_deltas else 0.0),
            "floor_hits": {key: key in set(floor_hits) for key in self.config.aux_keys},
            "cap_hits": {key: key in set(cap_hits) for key in self.config.aux_keys},
        }

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
            "teacher_phase": self.teacher_phase,
            "phase_switch_step": self.phase_switch_step,
            "phase_flip_count": self.phase_flip_count,
            "history": self.history[-1000:],
        }
