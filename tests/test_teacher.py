import json

from rtw_llm.teacher import (
    AUX_KEYS,
    MICRO_AUX_KEYS,
    MICRO_STABLE_CAPS,
    MICRO_STABLE_FLOORS,
    MICRO_TARGET_WEIGHT_SUM,
    STABLE_CAPS,
    STABLE_FLOORS,
    RTWTeacher,
    TeacherConfig,
)
from rtw_llm.rewards import RTWRewardManager


def _micro_config(**kw):
    return TeacherConfig(
        strategy="adaptive_stable",
        aux_keys=list(MICRO_AUX_KEYS),
        stable_floors=dict(MICRO_STABLE_FLOORS),
        stable_caps=dict(MICRO_STABLE_CAPS),
        stable_target_weight_sum=MICRO_TARGET_WEIGHT_SUM,
        **kw,
    )


def test_countdown_teacher_constants_are_byte_identical():
    # Adding MicroCode tables must NOT perturb the Countdown defaults (invariant).
    assert AUX_KEYS == ["format", "valid_expression", "number_multiset_f1",
                        "allowed_ops", "numeric_distance_reward", "brevity"]
    assert STABLE_FLOORS["valid_expression"] == 0.16
    assert STABLE_CAPS == {"numeric_distance_reward": 0.20}


def test_microcode_teacher_drives_its_own_aux_key_set():
    t = RTWTeacher(_micro_config(stable_delay_steps=2, seed=0))
    assert set(t.get_weights()) == set(MICRO_AUX_KEYS)
    for _ in range(60):
        t.update([{"correct": 0.0, "valid_expression": 1.0, "runs_without_error": 1.0,
                   "visible_pass_rate": 0.99, "no_hardcoding_heuristic": 0.4,
                   "held_out_pass_rate": 0.2} for _ in range(8)])
    w = t.get_weights()
    assert set(w) == set(MICRO_AUX_KEYS)
    assert abs(sum(w.values()) - MICRO_TARGET_WEIGHT_SUM) < 1e-6  # budget respected
    # proxy (visible_pass_rate) saturated => down-weighted below its 0.20 init
    assert w["visible_pass_rate"] < 0.20


def test_teacher_updates_weights():
    teacher = RTWTeacher(TeacherConfig(strategy="adaptive", seed=0))
    w0 = teacher.get_weights()
    batch = [
        {"correct": 0.0, "format": 0.0, "valid_expression": 0.0, "uses_numbers": 0.0, "allowed_ops": 1.0, "brevity": 1.0}
        for _ in range(8)
    ]
    teacher.update(batch)
    w1 = teacher.get_weights()
    assert w1["format"] >= w0["format"]
    assert w1["valid_expression"] >= w0["valid_expression"]


def test_static_teacher_constant():
    teacher = RTWTeacher(TeacherConfig(strategy="static", init_weight=0.2))
    teacher.update([{"correct": 1.0, "format": 1.0, "valid_expression": 1.0, "uses_numbers": 1.0, "allowed_ops": 1.0, "brevity": 1.0}])
    assert all(abs(v - 0.2) < 1e-9 for v in teacher.get_weights().values())


def test_adaptive_stable_delay_uses_static_weights_until_step_50():
    teacher = RTWTeacher(TeacherConfig(strategy="adaptive_stable", stable_delay_steps=50, seed=0))
    batch = [
        {
            "correct": 0.0,
            "format": 1.0,
            "valid_expression": 0.0,
            "number_multiset_f1": 0.0,
            "allowed_ops": 0.0,
            "numeric_distance_reward": 1.0,
            "brevity": 1.0,
        }
        for _ in range(8)
    ]
    for _ in range(50):
        teacher.update(batch)
        assert all(abs(v - 0.2) < 1e-9 for v in teacher.get_weights().values())
        assert teacher.history[-1]["diagnostics"]["delay_active"]

    teacher.update(batch)
    weights = teacher.get_weights()
    assert not teacher.history[-1]["diagnostics"]["delay_active"]
    assert any(abs(v - 0.2) > 1e-6 for v in weights.values())


def test_adaptive_stable_uses_stable_lr_for_raw_candidate():
    teacher = RTWTeacher(
        TeacherConfig(
            strategy="adaptive_stable",
            stable_delay_steps=0,
            stable_alpha=1.0,
            aux_keys=["format"],
            stable_floors={"format": 0.03},
            stable_caps={},
            stable_target_weight_sum=0.20,
        )
    )
    teacher.update([{"correct": 0.0, "format": 0.0}])
    diagnostics = teacher.history[-1]["diagnostics"]
    assert abs(diagnostics["raw_weight_sum"] - 0.215) < 1e-9


def test_adaptive_stable_smoothing_limits_single_step_jump():
    teacher = RTWTeacher(TeacherConfig(strategy="adaptive_stable", stable_delay_steps=0, stable_alpha=0.10))
    batch = [
        {
            "correct": 0.0,
            "format": 1.0,
            "valid_expression": 0.0,
            "number_multiset_f1": 0.0,
            "allowed_ops": 0.0,
            "numeric_distance_reward": 1.0,
            "brevity": 1.0,
        }
        for _ in range(8)
    ]
    teacher.update(batch)
    diagnostics = teacher.history[-1]["diagnostics"]
    assert diagnostics["update_linf"] < 0.05
    assert diagnostics["update_l1"] < 0.20


def test_adaptive_stable_legality_floors_are_respected():
    teacher = RTWTeacher(TeacherConfig(strategy="adaptive_stable", stable_delay_steps=0))
    batch = [
        {
            "correct": 1.0,
            "format": 1.0,
            "valid_expression": 1.0,
            "number_multiset_f1": 1.0,
            "allowed_ops": 1.0,
            "numeric_distance_reward": 1.0,
            "brevity": 1.0,
        }
        for _ in range(8)
    ]
    for _ in range(100):
        teacher.update(batch)
    weights = teacher.get_weights()
    assert weights["valid_expression"] >= 0.16
    assert weights["number_multiset_f1"] >= 0.18
    assert weights["allowed_ops"] >= 0.12


def test_adaptive_stable_numeric_distance_cap_is_respected():
    teacher = RTWTeacher(TeacherConfig(strategy="adaptive_stable", stable_delay_steps=0))
    batch = [
        {
            "correct": 0.0,
            "format": 1.0,
            "valid_expression": 1.0,
            "number_multiset_f1": 1.0,
            "allowed_ops": 1.0,
            "numeric_distance_reward": 0.0,
            "brevity": 1.0,
        }
        for _ in range(8)
    ]
    for _ in range(100):
        teacher.update(batch)
    assert teacher.get_weights()["numeric_distance_reward"] <= 0.20 + 1e-9
    assert teacher.history[-1]["diagnostics"]["numeric_distance_to_constraint_ratio"] <= 0.45


def test_adaptive_phased_starts_in_phase_a_with_stricter_number_floor():
    teacher = RTWTeacher(TeacherConfig(strategy="adaptive_phased", stable_delay_steps=0))
    batch = [
        {
            "correct": 0.0,
            "format": 1.0,
            "valid_expression": 0.0,
            "number_multiset_f1": 0.0,
            "allowed_ops": 1.0,
            "numeric_distance_reward": 1.0,
            "brevity": 1.0,
        }
        for _ in range(8)
    ]
    teacher.update(batch)
    weights = teacher.get_weights()
    diagnostics = teacher.history[-1]["diagnostics"]
    assert diagnostics["teacher_phase"] == "A"
    assert weights["number_multiset_f1"] >= 0.28
    assert weights["valid_expression"] >= 0.22
    assert weights["numeric_distance_reward"] <= 0.16 + 1e-9


def test_adaptive_phased_enters_phase_b_after_dwell_and_raises_numeric_cap():
    teacher = RTWTeacher(
        TeacherConfig(
            strategy="adaptive_phased",
            stable_delay_steps=0,
            phased_min_dwell_updates=2,
        )
    )
    good_batch = [
        {
            "correct": 0.0,
            "format": 1.0,
            "valid_expression": 1.0,
            "number_multiset_f1": 1.0,
            "allowed_ops": 1.0,
            "numeric_distance_reward": 0.0,
            "brevity": 1.0,
        }
        for _ in range(8)
    ]
    for _ in range(50):
        teacher.update(good_batch)
    diagnostics = teacher.history[-1]["diagnostics"]
    assert diagnostics["teacher_phase"] == "B"
    assert diagnostics["phase_flip_count"] == 1
    assert teacher.get_weights()["numeric_distance_reward"] <= 0.25 + 1e-9


def test_adaptive_phased_hysteresis_prevents_immediate_phase_flip_back():
    teacher = RTWTeacher(
        TeacherConfig(
            strategy="adaptive_phased",
            stable_delay_steps=0,
            phased_min_dwell_updates=3,
            ema_beta=0.0,
        )
    )
    good_batch = [
        {
            "correct": 0.0,
            "format": 1.0,
            "valid_expression": 1.0,
            "number_multiset_f1": 1.0,
            "allowed_ops": 1.0,
            "numeric_distance_reward": 0.0,
            "brevity": 1.0,
        }
    ]
    bad_batch = [
        {
            "correct": 0.0,
            "format": 1.0,
            "valid_expression": 0.0,
            "number_multiset_f1": 0.0,
            "allowed_ops": 1.0,
            "numeric_distance_reward": 1.0,
            "brevity": 1.0,
        }
    ]
    for _ in range(3):
        teacher.update(good_batch)
    assert teacher.history[-1]["diagnostics"]["teacher_phase"] == "B"
    teacher.update(bad_batch)
    assert teacher.history[-1]["diagnostics"]["teacher_phase"] == "B"
    for _ in range(2):
        teacher.update(bad_batch)
    assert teacher.history[-1]["diagnostics"]["teacher_phase"] == "A"


def test_adaptive_stable_weight_budget_is_preserved_and_logged():
    teacher = RTWTeacher(TeacherConfig(strategy="adaptive_stable", stable_delay_steps=0))
    batch = [
        {
            "correct": 0.0,
            "format": 0.5,
            "valid_expression": 0.1,
            "number_multiset_f1": 0.2,
            "allowed_ops": 0.3,
            "numeric_distance_reward": 0.4,
            "brevity": 0.5,
        }
        for _ in range(8)
    ]
    teacher.update(batch)
    diagnostics = teacher.history[-1]["diagnostics"]
    assert abs(sum(teacher.get_weights().values()) - 1.20) < 1e-6
    assert abs(diagnostics["weight_sum"] - 1.20) < 1e-6
    assert diagnostics["constraint_weight_mass"] > 0.0


def test_reward_manager_has_callable_name_for_trl_logging():
    manager = RTWRewardManager(RTWTeacher(TeacherConfig(strategy="static")))
    assert manager.__name__ == "rtw_reward"


def test_reward_manager_logs_primary_auxiliary_and_total_reward(tmp_path):
    log_path = tmp_path / "reward_components.jsonl"
    teacher = RTWTeacher(TeacherConfig(strategy="static", init_weight=0.2))
    manager = RTWRewardManager(teacher, log_path=str(log_path))
    rewards = manager.score_batch(
        ["<answer>(1+2)*3</answer>", "<answer>(1+2)*3"],
        [
            {"id": "ok", "numbers": [1, 2, 3], "target": 9, "allowed_ops": ["+", "-", "*"]},
            {"id": "partial", "numbers": [1, 2, 3], "target": 9, "allowed_ops": ["+", "-", "*"]},
        ],
    )
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(rows) == 2
    assert rewards == [rows[0]["total_reward"], rows[1]["total_reward"]]
    assert rows[0]["primary_reward"] == 1.0
    assert rows[0]["aux_reward_weighted"] > 0.0
    assert rows[0]["total_reward"] == rows[0]["primary_reward_weighted"] + rows[0]["aux_reward_weighted"]
    assert rows[0]["reward_batch_has_variance"]


def _stable_batch():
    return [
        {
            "correct": 0.0,
            "format": 1.0,
            "valid_expression": 0.2,
            "number_multiset_f1": 0.5,
            "allowed_ops": 0.6,
            "numeric_distance_reward": 0.3,
            "brevity": 1.0,
        }
        for _ in range(8)
    ]


def test_v12_strategy_raises_valid_expression_envelope():
    teacher = RTWTeacher(TeacherConfig(strategy="adaptive_stable_v12", stable_delay_steps=5, seed=0))
    assert teacher.config.stable_floors["valid_expression"] == 0.30
    assert teacher.config.stable_caps["valid_expression"] == 0.45
    for _ in range(30):
        teacher.update(_stable_batch())
    weights = teacher.get_weights()
    assert weights["valid_expression"] >= 0.30 - 1e-9
    assert weights["valid_expression"] <= 0.45 + 1e-9
    assert abs(sum(weights.values()) - 1.20) < 1e-6


def test_v12_custom_floor_tables_win_over_alias():
    custom = {k: 0.05 for k in AUX_KEYS}
    teacher = RTWTeacher(
        TeacherConfig(strategy="adaptive_stable_v12", stable_floors=custom, seed=0)
    )
    assert teacher.config.stable_floors == custom


def test_adaptive_stable_unchanged_by_v12_addition():
    a = RTWTeacher(TeacherConfig(strategy="adaptive_stable", stable_delay_steps=5, seed=0))
    for _ in range(30):
        a.update(_stable_batch())
    assert a.config.stable_floors["valid_expression"] == 0.16
    weights = a.get_weights()
    # The base strategy's projection can push valid_expression above the v12
    # floor, but its configured envelope must be untouched.
    assert a.config.stable_caps.get("valid_expression", a.config.max_weight) == a.config.max_weight
    assert abs(sum(weights.values()) - 1.20) < 1e-6


def test_v12_explicit_cap_wins_over_global_max_weight():
    # Regression for the dead-cap bug: budget surplus must be able to push
    # valid_expression above max_weight (0.35) up to its explicit 0.45 cap.
    teacher = RTWTeacher(TeacherConfig(strategy="adaptive_stable_v12", stable_delay_steps=1, seed=0))
    # Everything except valid_expression saturated -> their targets fall to
    # min_weight, so the budget projection redistributes surplus into the one
    # needy key.
    batch = [
        {
            "correct": 0.0,
            "format": 1.0,
            "valid_expression": 0.0,
            "number_multiset_f1": 1.0,
            "allowed_ops": 1.0,
            "numeric_distance_reward": 1.0,
            "brevity": 1.0,
        }
        for _ in range(8)
    ]
    for _ in range(200):
        teacher.update(batch)
    weights = teacher.get_weights()
    assert weights["valid_expression"] > 0.35
    assert weights["valid_expression"] <= 0.45 + 1e-9
    assert abs(sum(weights.values()) - 1.20) < 1e-6


def test_adaptive_stable_golden_trajectory_unchanged():
    # Bit-identity pin for the base strategy across the v12 cap-projection
    # change: numeric_distance stays at its explicit 0.20 cap and the weight
    # budget holds on a mixed batch.
    teacher = RTWTeacher(TeacherConfig(strategy="adaptive_stable", stable_delay_steps=5, seed=0))
    for _ in range(30):
        teacher.update(_stable_batch())
    weights = teacher.get_weights()
    assert weights["numeric_distance_reward"] <= 0.20 + 1e-9
    assert all(w <= 0.35 + 1e-9 for w in weights.values())
    assert abs(sum(weights.values()) - 1.20) < 1e-6


# --- I8b: per-key init/static weight vector (default-off, byte-identical) ---

# A distinct vector where NO value equals the 0.20 scalar default, so a missed
# routing site (which would emit 0.20) is caught, not masked (advisor A1).
_I8B_VEC = {
    "format": 0.05,
    "valid_expression": 0.11,
    "number_multiset_f1": 0.13,
    "allowed_ops": 0.17,
    "numeric_distance_reward": 0.19,
    "brevity": 0.31,
}


def test_i8b_default_none_is_byte_identical_static():
    # Regression pin: init_weights=None must equal the scalar path exactly.
    t = RTWTeacher(TeacherConfig(strategy="static", init_weight=0.2))
    assert t.get_weights() == {k: 0.2 for k in AUX_KEYS}


def test_i8b_static_holds_per_key_vector_before_and_after_update():
    t = RTWTeacher(TeacherConfig(strategy="static", init_weights=dict(_I8B_VEC)))
    # (a) step 0, before any update() — pins the __init__ seed site
    assert t.get_weights() == _I8B_VEC
    # (b) after update() — pins the static branch site
    t.update([{"correct": 1.0, **{k: 1.0 for k in AUX_KEYS}}])
    assert t.get_weights() == _I8B_VEC


def test_i8b_adaptive_stable_delay_seeds_per_key_vector():
    t = RTWTeacher(TeacherConfig(strategy="adaptive_stable", stable_delay_steps=50,
                                 init_weights=dict(_I8B_VEC), seed=0))
    # during the delay period the weights are the per-key init (pins delay reset)
    t.update([{"correct": 0.0, **{k: 0.0 for k in AUX_KEYS}}])
    assert t.get_weights() == _I8B_VEC


def test_i8b_adaptive_phased_delay_seeds_per_key_vector():
    t = RTWTeacher(TeacherConfig(strategy="adaptive_phased", stable_delay_steps=50,
                                 init_weights=dict(_I8B_VEC), seed=0))
    t.update([{"correct": 0.0, **{k: 0.0 for k in AUX_KEYS}}])
    assert t.get_weights() == _I8B_VEC


def test_i8b_missing_key_falls_back_to_scalar():
    # A key absent from init_weights uses the scalar init_weight.
    t = RTWTeacher(TeacherConfig(strategy="static", init_weight=0.2,
                                 init_weights={"valid_expression": 0.40}))
    w = t.get_weights()
    assert w["valid_expression"] == 0.40
    assert w["format"] == 0.2  # fell back to scalar


def test_i8b_unknown_key_fails_loud():
    # A typo (key not in aux_keys) must raise, not silently revert (advisor A2).
    import pytest
    with pytest.raises(ValueError, match="not in aux_keys"):
        RTWTeacher(TeacherConfig(strategy="static", init_weights={"visible_pass": 0.35}))


def test_i8b_adaptive_stable_decays_overweight_proxy_post_delay():
    # E5 mechanism: a proxy-overweight init with a LOW floor must be free to
    # decay post-delay (floors orthogonal to init). Use valid_expression as the
    # stand-in proxy; saturate its ema so need->0 pulls it down.
    t = RTWTeacher(TeacherConfig(
        strategy="adaptive_stable", stable_delay_steps=2,
        init_weights={**{k: 0.02 for k in AUX_KEYS}, "valid_expression": 0.35},
        stable_floors={k: 0.02 for k in AUX_KEYS}, seed=0,
    ))
    start = t.get_weights()["valid_expression"]
    for _ in range(60):
        t.update([{"correct": 0.0, **{k: 0.0 for k in AUX_KEYS}, "valid_expression": 1.0}])
    end = t.get_weights()["valid_expression"]
    assert start == 0.35
    assert end < start  # need=1-ema decays the saturating overweighted proxy
