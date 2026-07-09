import json

import pytest

from rtw_llm.curriculum import (
    MANUAL_SCHEDULE,
    CurriculumConfig,
    CurriculumController,
    CurriculumSampler,
)


def make_controller(mode="adaptive", **kwargs) -> CurriculumController:
    return CurriculumController(CurriculumConfig(mode=mode, **kwargs))


def batch(tier: str, valid: float, exact: float, k: int = 4):
    return [(tier, {"valid_expression": valid, "correct": exact}) for _ in range(k)]


def feed(controller, tier_stats, updates):
    for _ in range(updates):
        records = []
        for tier, (valid, exact) in tier_stats.items():
            records.extend(batch(tier, valid, exact))
        controller.observe(records)


def test_uniform_before_delay():
    controller = make_controller(delay_updates=25)
    feed(controller, {"easy": (1.0, 0.5), "medium": (0.0, 0.0), "hard": (0.0, 0.0)}, 10)
    probs = controller.tier_probs()
    assert probs == pytest.approx({"easy": 1 / 3, "medium": 1 / 3, "hard": 1 / 3})


def test_adaptive_prefers_band_tier():
    controller = make_controller(delay_updates=5)
    # easy saturated (exact far above tau_exact), medium in the legality band,
    # hard hopeless: medium should get the most mass.
    feed(controller, {"easy": (1.0, 0.9), "medium": (0.45, 0.0), "hard": (0.02, 0.0)}, 30)
    probs = controller.tier_probs()
    assert probs["medium"] > probs["easy"]
    assert probs["medium"] > probs["hard"]
    assert sum(probs.values()) == pytest.approx(1.0)


def test_gating_switches_signal():
    controller = make_controller(delay_updates=0)
    feed(controller, {"easy": (1.0, 0.0), "medium": (0.3, 0.0), "hard": (0.3, 0.0)}, 30)
    # easy is past the valid gate -> exact phase; others still legality phase.
    assert controller.competence("easy")[1] == "exact"
    assert controller.competence("medium")[1] == "legality"
    # valid=1.0/exact=0.0 must NOT look ideal: exact-phase competence is 0,
    # which is off the tau_exact band but scores nonzero; the tier sitting at
    # the legality band target (0.3 vs tau 0.5 is closer than 0 vs 0.175 in
    # sigma units) is not automatically dominated by the saturated tier.
    probs = controller.tier_probs()
    assert probs["easy"] < max(probs["medium"], probs["hard"]) + 1e-9


def test_probability_floor_holds():
    controller = make_controller(delay_updates=0, p_min=0.10)
    feed(controller, {"easy": (0.5, 0.175), "medium": (0.0, 0.0), "hard": (0.0, 0.0)}, 50)
    probs = controller.tier_probs()
    for tier in ("easy", "medium", "hard"):
        assert probs[tier] >= 0.10 - 1e-9
    assert sum(probs.values()) == pytest.approx(1.0)


def test_unobserved_tier_gets_exploration_mass():
    controller = make_controller(delay_updates=0)
    feed(controller, {"easy": (0.9, 0.9)}, 10)  # medium/hard never observed
    probs = controller.tier_probs()
    assert probs["medium"] > probs["easy"]
    assert probs["hard"] > probs["easy"]


def test_ema_absent_tier_unchanged_and_sample_weighted():
    controller = make_controller(delay_updates=0)
    controller.observe(batch("easy", 1.0, 1.0, k=4))
    before = controller.ema_valid["easy"]
    controller.observe(batch("medium", 0.5, 0.0, k=4))
    assert controller.ema_valid["easy"] == before  # absent tier keeps EMA
    # k=8 moves the EMA further toward the batch mean than k=1 does.
    a = make_controller(delay_updates=0)
    a.observe(batch("easy", 1.0, 0.0, k=4))
    a.observe(batch("easy", 0.0, 0.0, k=8))
    b = make_controller(delay_updates=0)
    b.observe(batch("easy", 1.0, 0.0, k=4))
    b.observe(batch("easy", 0.0, 0.0, k=1))
    assert a.ema_valid["easy"] < b.ema_valid["easy"]


def test_manual_schedule_breakpoints():
    controller = make_controller(mode="manual")
    assert controller.tier_probs()["easy"] == pytest.approx(0.60)
    controller.update_count = 100
    assert controller.tier_probs()["easy"] == pytest.approx(0.34)
    controller.update_count = 250
    assert controller.tier_probs()["hard"] == pytest.approx(0.60)
    assert [row[0] for row in MANUAL_SCHEDULE] == [0, 100, 200]


def test_controller_log_written(tmp_path):
    log = tmp_path / "curriculum_state.jsonl"
    controller = make_controller(delay_updates=0, log_path=str(log))
    feed(controller, {"easy": (0.5, 0.1)}, 3)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 3
    assert rows[-1]["update"] == 3
    assert set(rows[-1]["tier_probs"]) == {"easy", "medium", "hard"}


TIERS_600 = ["easy", "medium", "hard"] * 200  # index i has tier TIERS_600[i]


def test_uniform_mode_matches_repeat_sampler_bit_for_bit():
    trl = pytest.importorskip("trl.trainer.grpo_trainer")
    controller = make_controller(mode="uniform")
    ours = CurriculumSampler(
        tier_of_index=TIERS_600,
        controller=controller,
        mini_repeat_count=4,
        batch_size=4,
        repeat_count=8,
        seed=0,
    )
    theirs = trl.RepeatSampler(
        data_source=list(range(len(TIERS_600))),
        mini_repeat_count=4,
        batch_size=4,
        repeat_count=8,
        shuffle=True,
        seed=0,
    )
    assert list(ours) == list(theirs)
    assert len(ours) == len(theirs)


def test_sampler_yield_structure_and_len():
    controller = make_controller(delay_updates=0)
    sampler = CurriculumSampler(
        tier_of_index=TIERS_600,
        controller=controller,
        mini_repeat_count=4,
        batch_size=4,
        repeat_count=8,
        seed=0,
    )
    out = list(sampler)
    assert len(out) == len(sampler) == (600 // 4) * 4 * 4 * 8
    block = 4 * 4  # one chunk yield: 4 unique indices x 4 mini-repeats
    first = out[:block]
    # Each index appears in a run of mini_repeat_count.
    assert first == [i for i in first[::4] for _ in range(4)]
    # The chunk is re-yielded repeat_count times unchanged (materialized once).
    for r in range(8):
        assert out[r * block : (r + 1) * block] == first
    # Unique indices within a chunk.
    assert len(set(first)) == 4


def test_sampler_respects_tier_probs_and_no_replacement():
    controller = make_controller(delay_updates=0, p_min=0.05, epsilon=0.1)
    # Push the controller hard toward "hard" tier: easy/medium saturated.
    feed(
        controller,
        {"easy": (1.0, 0.9), "medium": (1.0, 0.9), "hard": (0.45, 0.0)},
        40,
    )
    sampler = CurriculumSampler(
        tier_of_index=TIERS_600,
        controller=controller,
        mini_repeat_count=1,
        batch_size=4,
        repeat_count=1,
        seed=0,
    )
    out = list(sampler)
    unique = out  # mini_repeat_count=1, repeat_count=1 -> raw draws
    hard_draws = [i for i in unique if TIERS_600[i] == "hard"]
    assert len(hard_draws) > 0.5 * len(unique)
    # Without replacement across chunks: no hard index repeats until the whole
    # hard tier (200 tasks) has been consumed once.
    seen = set()
    for idx in hard_draws:
        if idx in seen:
            assert len(seen) >= 200
            break
        seen.add(idx)


def test_sampler_determinism():
    def draws():
        controller = make_controller(delay_updates=0)
        feed(controller, {"easy": (0.5, 0.1), "medium": (0.3, 0.0), "hard": (0.1, 0.0)}, 10)
        sampler = CurriculumSampler(
            tier_of_index=TIERS_600,
            controller=controller,
            mini_repeat_count=2,
            batch_size=4,
            repeat_count=2,
            seed=7,
        )
        return list(sampler)

    assert draws() == draws()


def test_pop_index_refill_keeps_queue_duplicate_free():
    # Regression: a refill occurring while some indices were skipped for being
    # in the current chunk must not leave duplicate entries in the tier queue.
    controller = make_controller(delay_updates=0)
    sampler = CurriculumSampler(
        tier_of_index=["hard", "hard"],  # tiny tier: refill happens constantly
        controller=controller,
        mini_repeat_count=1,
        batch_size=1,
        repeat_count=1,
        seed=0,
    )
    chunk_set = {0, 1}
    for _ in range(20):
        sampler._pop_index("hard", chunk_set)
        queue = sampler._queues["hard"]
        assert len(queue) == len(set(queue)), f"duplicates in queue: {queue}"
        assert set(queue).issubset({0, 1})


def test_pop_index_without_replacement_across_refill():
    controller = make_controller(delay_updates=0)
    tiers = ["hard"] * 10
    sampler = CurriculumSampler(
        tier_of_index=tiers,
        controller=controller,
        mini_repeat_count=1,
        batch_size=1,
        repeat_count=1,
        seed=3,
    )
    draws = [sampler._pop_index("hard", set()) for _ in range(30)]
    # Every full pass of 10 draws covers each index exactly once.
    for start in range(0, 30, 10):
        assert sorted(draws[start : start + 10]) == list(range(10))


def test_sampler_rejects_unknown_difficulty_labels():
    controller = make_controller(delay_updates=0)
    with pytest.raises(ValueError, match="ood_long"):
        CurriculumSampler(
            tier_of_index=["easy", "ood_long"],
            controller=controller,
            mini_repeat_count=1,
            batch_size=1,
            repeat_count=1,
            seed=0,
        )


def test_reward_manager_observe_hook_is_observe_only():
    from rtw_llm.rewards import RTWRewardManager
    from rtw_llm.teacher import RTWTeacher, TeacherConfig

    example = {"id": "t1", "difficulty": "easy", "numbers": [4, 6, 4], "target": 14, "allowed_ops": ["+", "-"]}
    completions = ["<answer>(4+6)+4</answer>", "<answer>4+6</answer>"]

    def run(curriculum):
        teacher = RTWTeacher(TeacherConfig(strategy="static", seed=0))
        manager = RTWRewardManager(teacher=teacher, curriculum=curriculum)
        return manager.score_batch(completions, [example, example])

    controller = make_controller(delay_updates=0)
    rewards_with = run(controller)
    rewards_without = run(None)
    assert rewards_with == rewards_without  # hook must never alter rewards
    assert controller.update_count == 1
    assert controller.ema_valid["easy"] is not None


def test_group_variance_is_positional(tmp_path):
    from rtw_llm.rewards import RTWRewardManager
    from rtw_llm.teacher import RTWTeacher, TeacherConfig

    # Two positional groups of 2 sharing the same prompt id: group A has
    # variance (correct vs junk), group B does not (junk vs junk). id-based
    # grouping would merge them and report variance for all four rows.
    example = {"id": "t1", "difficulty": "easy", "numbers": [4, 6, 4], "target": 14, "allowed_ops": ["+", "-"]}
    completions = ["<answer>(4+6)+4</answer>", "no tags", "no tags", "no tags"]
    log = tmp_path / "reward_components.jsonl"
    teacher = RTWTeacher(TeacherConfig(strategy="static", seed=0))
    manager = RTWRewardManager(teacher=teacher, group_size=2, log_path=str(log))
    manager.score_batch(completions, [example] * 4)
    records = [json.loads(line) for line in log.read_text().splitlines()]

    assert [r["group_has_variance"] for r in records] == [True, True, False, False]
    assert records[0]["batch_group_variance_fraction"] == pytest.approx(0.5)


def test_controller_uses_configured_graded_channel():
    # framework-bug (a): with a dense graded_key, per-tier competence must
    # reflect fractional progress, not a binary-derived EMA.
    controller = CurriculumController(
        CurriculumConfig(
            mode="adaptive",
            delay_updates=0,
            gate_key="syntax_parses",
            graded_key="held_out_pass_rate",
        )
    )
    for _ in range(20):
        controller.observe(
            [("easy", {"syntax_parses": 1.0, "held_out_pass_rate": 0.6}) for _ in range(4)]
        )
    c, phase = controller.competence("easy")
    assert phase == "exact"
    assert c == pytest.approx(0.6, abs=0.05)  # graded value, not 0/1


def test_reward_manager_custom_scorer_and_fields():
    # framework-bug (b): the manager must dispatch to a task-agnostic scorer
    # and forward configured dataset columns instead of Countdown's.
    from types import SimpleNamespace

    from rtw_llm.rewards import RTWRewardManager
    from rtw_llm.teacher import RTWTeacher, TeacherConfig

    seen = []

    def fake_scorer(completion, example, aux_weights, primary_weight):
        seen.append(example)
        components = {"correct": 1.0, "valid_expression": 1.0}
        result = SimpleNamespace(expression="f", value=1, correct=True, error=None)
        return 2.0, components, result

    teacher = RTWTeacher(TeacherConfig(strategy="static", seed=0))
    manager = RTWRewardManager(
        teacher=teacher,
        scorer=fake_scorer,
        example_fields=("signature", "visible_tests"),
    )
    rewards = manager(
        ["<answer>def f(): pass</answer>"],
        id=["t1"],
        difficulty=["easy"],
        signature=["def f()"],
        visible_tests=[["assert f() is None"]],
    )
    assert rewards == [2.0]
    assert seen[0]["signature"] == "def f()"
    assert seen[0]["visible_tests"] == ["assert f() is None"]
    assert "numbers" not in seen[0]  # Countdown fields not forced on other tasks
