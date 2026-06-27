from rtw_llm.teacher import RTWTeacher, TeacherConfig


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
