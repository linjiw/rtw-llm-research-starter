import json

from rtw_llm.teacher import RTWTeacher, TeacherConfig
from rtw_llm.rewards import RTWRewardManager


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
