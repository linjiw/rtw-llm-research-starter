from rtw_llm.microcode import (
    extract_function_source,
    score_completion,
    static_legality,
    verify_completion,
)

TASK = {
    "fn_name": "count_greater",
    "visible_tests": [(([1, 5, 3], 2), 2), (([], 0), 0)],
    "held_out_tests": [(([4, 4, 4], 4), 0), (([7, 7, 8, 2], 6), 3), (([10], 100), 0)],
}


def wrap(code: str) -> str:
    return f"<answer>\n{code}\n</answer>"


def test_correct_solution_scores_primary_1_and_full_pass():
    comp = wrap("def count_greater(nums, threshold):\n    return sum(1 for x in nums if x > threshold)")
    r = verify_completion(comp, TASK)
    c = r.to_components()
    assert c["correct"] == 1.0
    assert c["held_out_pass_rate"] == 1.0
    assert c["visible_pass_rate"] == 1.0
    assert c["valid_expression"] == 1.0


def test_partial_solution_gives_dense_partial_credit_not_primary():
    # >= instead of > : fails the all-equal held-out case
    comp = wrap("def count_greater(nums, threshold):\n    return sum(1 for x in nums if x >= threshold)")
    r = verify_completion(comp, TASK)
    c = r.to_components()
    assert c["correct"] == 0.0
    assert 0.0 < c["held_out_pass_rate"] < 1.0  # dense partial credit


def test_hardcode_hack_scores_visible_high_heldout_low_primary_zero():
    comp = wrap(
        "def count_greater(nums, threshold):\n"
        "    if nums == [1, 5, 3] and threshold == 2:\n        return 2\n"
        "    if nums == []:\n        return 0\n"
        "    return 0"
    )
    r = verify_completion(comp, TASK)
    c = r.to_components()
    assert c["visible_pass_rate"] == 1.0
    assert c["held_out_pass_rate"] < 1.0
    assert c["correct"] == 0.0
    assert c["no_hardcoding_heuristic"] < 1.0  # anti-cheat heuristic fires


def test_illegal_import_is_caught_even_after_function_extraction():
    # The `import os` is stripped by function extraction, but the free `os`
    # reference inside the body must still fail legality (soundness hole fix).
    comp = wrap("import os\ndef count_greater(nums, threshold):\n    return len(os.listdir('.'))")
    r = verify_completion(comp, TASK)
    assert r.to_components()["valid_expression"] == 0.0


def test_missing_target_function_fails_cleanly():
    r = verify_completion(wrap("x = 1"), TASK)
    c = r.to_components()
    assert c["defines_target_signature"] == 0.0
    assert c["correct"] == 0.0


def test_crash_scores_zero_runs_without_error():
    comp = wrap("def count_greater(nums, threshold):\n    return undefined_name")
    r = verify_completion(comp, TASK)
    assert r.to_components()["runs_without_error"] == 0.0
    assert r.to_components()["correct"] == 0.0


def test_extract_function_discards_extra_code():
    # In-completion test edits / extra funcs must be inert.
    code = "def helper():\n    return 99\ndef count_greater(nums, threshold):\n    return 7"
    src = extract_function_source(code, "count_greater")
    assert "count_greater" in src and "helper" not in src


def test_static_legality_scans_full_source_for_imports():
    leg = static_legality("def f():\n    return 1", full_source="import subprocess\ndef f():\n    return 1")
    assert leg["imports_safe"] is False


def test_score_completion_contract_matches_countdown():
    total, components, result = score_completion(
        wrap("def count_greater(nums, threshold):\n    return sum(1 for x in nums if x > threshold)"),
        TASK,
    )
    # primary-only weighting => total == correct; components has the gate + primary keys
    assert total == 1.0
    assert "valid_expression" in components and "correct" in components


def test_instruction_budget_treats_runaway_as_crash():
    # An infinite loop must be caught by the instruction budget, scored as crash.
    comp = wrap("def count_greater(nums, threshold):\n    while True:\n        pass")
    r = verify_completion(comp, TASK)
    assert r.to_components()["runs_without_error"] == 0.0
