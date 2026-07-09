"""GACL-style task-difficulty curriculum: controller and GRPO-compatible sampler.

The controller observes per-completion reward components (observe-only; it must
never alter rewards) and maintains per-tier competence EMAs. The sampler mirrors
TRL's RepeatSampler yield contract but draws prompts from per-tier shuffled
queues with tier probabilities read live from the controller.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import Sampler

TIERS = ["easy", "medium", "hard"]

MANUAL_SCHEDULE = [
    # (first update this row applies from, probs for easy/medium/hard)
    (0, (0.60, 0.30, 0.10)),
    (100, (0.34, 0.33, 0.33)),
    (200, (0.10, 0.30, 0.60)),
]

VALID_MODES = {"uniform", "manual", "adaptive"}


@dataclass
class CurriculumConfig:
    mode: str = "uniform"
    beta: float = 0.90
    epsilon: float = 0.20
    p_min: float = 0.10
    delay_updates: int = 25
    valid_gate: float = 0.50
    tau_valid: float = 0.50
    sigma_valid: float = 0.25
    tau_exact: float = 0.175
    sigma_exact: float = 0.15
    log_path: str | None = None
    tiers: list[str] = field(default_factory=lambda: TIERS.copy())


class CurriculumController:
    """Tracks per-tier competence and exposes tier sampling probabilities.

    Competence is gated: while a tier's valid-expression EMA is below
    `valid_gate`, competence is the valid EMA (legality phase, tau_valid);
    afterwards it is the exact-correct EMA (exact phase, tau_exact). A flat
    valid/exact blend cannot discriminate because a tier at valid=1/exact=0
    would sit exactly at the blended target.
    """

    def __init__(self, config: CurriculumConfig | None = None):
        self.config = config or CurriculumConfig()
        if self.config.mode not in VALID_MODES:
            raise ValueError(f"Unknown curriculum mode: {self.config.mode}")
        self.update_count = 0
        self.ema_valid: dict[str, float | None] = {t: None for t in self.config.tiers}
        self.ema_exact: dict[str, float | None] = {t: None for t in self.config.tiers}
        self.cumulative_draws = {t: 0 for t in self.config.tiers}
        self.seen_indices: set[int] = set()
        self.log_path = Path(self.config.log_path) if self.config.log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("")

    def observe(self, records: Iterable[tuple[str | None, dict[str, float]]]) -> None:
        """Update per-tier EMAs from one reward batch. Observe-only."""
        by_tier: dict[str, list[dict[str, float]]] = {}
        for difficulty, components in records:
            if difficulty in self.ema_valid:
                by_tier.setdefault(difficulty, []).append(components)
        for tier, comps in by_tier.items():
            k = len(comps)
            valid_mean = sum(c.get("valid_expression", 0.0) for c in comps) / k
            exact_mean = sum(c.get("correct", 0.0) for c in comps) / k
            # Sample-weighted EMA: beta^k so sparsely observed tiers are not
            # dominated by single-sample noise; first observation initializes.
            beta_eff = self.config.beta**k
            if self.ema_valid[tier] is None:
                self.ema_valid[tier] = valid_mean
                self.ema_exact[tier] = exact_mean
            else:
                self.ema_valid[tier] = beta_eff * self.ema_valid[tier] + (1 - beta_eff) * valid_mean
                self.ema_exact[tier] = beta_eff * self.ema_exact[tier] + (1 - beta_eff) * exact_mean
        self.update_count += 1
        self._log()

    def record_chunk(self, tier_counts: dict[str, int], indices: Sequence[int]) -> None:
        for tier, count in tier_counts.items():
            self.cumulative_draws[tier] = self.cumulative_draws.get(tier, 0) + count
        self.seen_indices.update(int(i) for i in indices)

    def competence(self, tier: str) -> tuple[float | None, str]:
        valid = self.ema_valid[tier]
        exact = self.ema_exact[tier]
        if valid is None or exact is None:
            return None, "unobserved"
        if valid < self.config.valid_gate:
            return valid, "legality"
        return exact, "exact"

    def tier_probs(self) -> dict[str, float]:
        tiers = self.config.tiers
        uniform = {t: 1.0 / len(tiers) for t in tiers}
        if self.config.mode == "uniform":
            return uniform
        if self.config.mode == "manual":
            probs = MANUAL_SCHEDULE[0][1]
            for start, row in MANUAL_SCHEDULE:
                if self.update_count >= start:
                    probs = row
            return dict(zip(tiers, probs))
        if self.update_count < self.config.delay_updates:
            return uniform
        scores: dict[str, float] = {}
        for tier in tiers:
            c, phase = self.competence(tier)
            if c is None:
                scores[tier] = 1.0  # force exploration of unobserved tiers
                continue
            tau = self.config.tau_valid if phase == "legality" else self.config.tau_exact
            sigma = self.config.sigma_valid if phase == "legality" else self.config.sigma_exact
            scores[tier] = math.exp(-((c - tau) ** 2) / (2 * sigma**2))
        total = sum(scores.values())
        eps = self.config.epsilon
        mixed = {t: (1 - eps) * scores[t] / total + eps / len(tiers) for t in tiers}
        # Floor-mixing keeps every tier >= p_min while summing to 1 exactly.
        floor = self.config.p_min
        return {t: floor + (1 - len(tiers) * floor) * mixed[t] for t in tiers}

    def _log(self) -> None:
        if not self.log_path:
            return
        record = {
            "update": self.update_count,
            "mode": self.config.mode,
            "ema_valid": dict(self.ema_valid),
            "ema_exact": dict(self.ema_exact),
            "phase": {t: self.competence(t)[1] for t in self.config.tiers},
            "tier_probs": self.tier_probs(),
            "cumulative_draws": dict(self.cumulative_draws),
            "unique_tasks_seen": len(self.seen_indices),
        }
        with self.log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def state_dict(self) -> dict:
        return {
            "config": asdict(self.config),
            "update_count": self.update_count,
            "ema_valid": dict(self.ema_valid),
            "ema_exact": dict(self.ema_exact),
            "cumulative_draws": dict(self.cumulative_draws),
            "unique_tasks_seen": len(self.seen_indices),
        }


class CurriculumSampler(Sampler):
    """RepeatSampler-compatible sampler with tier-weighted prompt selection.

    Yield contract (identical to trl.trainer.grpo_trainer.RepeatSampler): for
    each chunk of `batch_size` unique prompt indices, the chunk is yielded
    `repeat_count` times, each index repeated `mini_repeat_count` times, and
    only full chunks are yielded. In "uniform" mode the index sequence is
    bit-for-bit identical to RepeatSampler for the same seed.

    Non-uniform modes build each chunk by drawing a tier per slot from the
    controller's live `tier_probs()` and popping from a per-tier shuffled queue
    (reshuffled on exhaustion), so exposure within a tier is without
    replacement across chunks. Each chunk is materialized once and re-yielded
    for all repeats; controller state is read lazily per chunk. Note that
    accelerate's dataloader prefetches one batch ahead, so chunk N+1 is drawn
    before chunk N's rewards are observed — a one-generation-block lag between
    controller updates and the probabilities that produced each chunk.

    Single-process only: the chunk stream depends on controller state built
    from rank-local rewards, so multi-process runs would draw divergent index
    streams per rank (guarded at trainer level). Resume-from-checkpoint is
    unsupported: controller state is not restored, so a resumed curriculum run
    would replay a different stream than the original.
    """

    def __init__(
        self,
        tier_of_index: Sequence[str],
        controller: CurriculumController,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: int | None = None,
    ):
        self.tier_of_index = list(tier_of_index)
        unknown = sorted(set(self.tier_of_index) - set(controller.config.tiers))
        if unknown:
            raise ValueError(
                f"Dataset difficulty labels {unknown} not in curriculum tiers "
                f"{controller.config.tiers}; those rows would silently never be sampled"
            )
        self.controller = controller
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(self.tier_of_index)
        self.seed = seed
        # Persistent generators (like RepeatSampler): epoch N+1 continues the stream.
        self.torch_generator = torch.Generator()
        if seed is not None:
            self.torch_generator.manual_seed(seed)
        self.rng = random.Random(seed)
        self.tier_indices: dict[str, list[int]] = {}
        for i, tier in enumerate(self.tier_of_index):
            self.tier_indices.setdefault(tier, []).append(i)
        self._queues: dict[str, list[int]] = {t: [] for t in self.tier_indices}

    def _refill(self, tier: str) -> None:
        fresh = list(self.tier_indices[tier])
        self.rng.shuffle(fresh)
        self._queues[tier] = fresh

    def _pop_index(self, tier: str, chunk_set: set[int]) -> int:
        queue = self._queues[tier]
        if not queue:
            self._refill(tier)
            queue = self._queues[tier]
        # Take the last queued index not already in this chunk. Only when the
        # entire tier is a subset of the chunk (tier smaller than the slots
        # drawn for it) do we knowingly return an in-chunk duplicate.
        for pos in range(len(queue) - 1, -1, -1):
            if queue[pos] not in chunk_set:
                return queue.pop(pos)
        return queue.pop()

    def _draw_tier(self, probs: dict[str, float]) -> str:
        tiers = [t for t in probs if self.tier_indices.get(t)]
        weights = [probs[t] for t in tiers]
        total = sum(weights)
        r = self.rng.random() * total
        acc = 0.0
        for tier, w in zip(tiers, weights):
            acc += w
            if r <= acc:
                return tier
        return tiers[-1]

    def __iter__(self):
        num_chunks = self.num_samples // self.batch_size
        if self.controller.config.mode == "uniform":
            # Bit-for-bit RepeatSampler equivalence for the fairness test. Only
            # the first num_chunks full chunks are consumed below, so no
            # partial-chunk filter is needed.
            indexes = torch.randperm(self.num_samples, generator=self.torch_generator).tolist()
            chunks = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        else:
            chunks = None
        for chunk_i in range(num_chunks):
            if chunks is not None:
                chunk = chunks[chunk_i]
            else:
                probs = self.controller.tier_probs()
                chunk = []
                chunk_set: set[int] = set()
                tier_counts: dict[str, int] = {}
                for _ in range(self.batch_size):
                    tier = self._draw_tier(probs)
                    index = self._pop_index(tier, chunk_set)
                    chunk.append(index)
                    chunk_set.add(index)
                    tier_counts[tier] = tier_counts.get(tier, 0) + 1
                self.controller.record_chunk(tier_counts, chunk)
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return (self.num_samples // self.batch_size) * self.batch_size * self.mini_repeat_count * self.repeat_count
