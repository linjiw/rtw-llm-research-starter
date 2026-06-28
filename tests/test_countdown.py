import random

from rtw_llm.countdown import (
    answer_tag_signals,
    extract_answer,
    random_solvable_task,
    reward_breakdown,
    verify_completion,
    verify_expression,
)


def test_extract_answer():
    expr, tag = extract_answer("hello <answer>(1+2)*3</answer> bye")
    assert tag
    assert expr == "(1+2)*3"


def test_answer_tag_signals_are_dense():
    signals = answer_tag_signals("prefix <answer>(1+2)")
    assert signals["contains_open_answer_tag"]
    assert not signals["contains_close_answer_tag"]
    assert not signals["has_extractable_answer_span"]


def test_format_component_rewards_partial_tags():
    ex = {"numbers": [1, 2, 3], "target": 9, "allowed_ops": ["+", "-", "*"]}
    result = verify_completion("<answer>(1+2)*3", ex)
    comps = result.to_components()
    assert comps["contains_open_answer_tag"] == 1.0
    assert comps["contains_close_answer_tag"] == 0.0
    assert comps["has_extractable_answer_span"] == 0.0
    assert comps["format"] == 1.0 / 3.0


def test_verify_correct_expression():
    result = verify_expression("(1+2)*3", [1, 2, 3], 9, ["+", "-", "*"])
    assert result.correct
    assert result.uses_all_numbers
    assert result.uses_allowed_ops
    comps = result.to_components()
    assert comps["exact_correct"] == 1.0
    assert comps["uses_allowed_numbers"] == 1.0
    assert comps["number_precision"] == 1.0
    assert comps["number_recall"] == 1.0
    assert comps["number_multiset_f1"] == 1.0
    assert comps["uses_no_extra_numbers"] == 1.0
    assert comps["uses_all_required_numbers"] == 1.0
    assert comps["numeric_distance_reward"] == 1.0


def test_reward_breakdown_splits_primary_and_auxiliary_reward():
    components = {"correct": 1.0, "format": 0.5, "brevity": 1.0}
    breakdown = reward_breakdown(components, {"format": 0.2, "brevity": 0.1})
    assert breakdown["primary_reward"] == 1.0
    assert breakdown["primary_reward_weighted"] == 1.0
    assert breakdown["aux_reward_weighted"] == 0.2
    assert breakdown["total_reward"] == 1.2


def test_reject_invented_constant():
    result = verify_expression("9", [1, 2, 3], 9, ["+", "-", "*"])
    assert not result.correct
    assert not result.uses_all_numbers
    comps = result.to_components()
    assert comps["uses_allowed_numbers"] == 0.0
    assert comps["number_precision"] == 0.0
    assert comps["number_recall"] == 0.0
    assert comps["number_multiset_f1"] == 0.0
    assert comps["uses_no_extra_numbers"] == 0.0
    assert comps["uses_all_required_numbers"] == 0.0
    assert comps["allowed_ops"] == 0.0
    assert comps["numeric_distance_reward"] == 0.0


def test_near_target_invented_constant_gets_no_numeric_distance_reward():
    result = verify_completion(
        "<answer>149</answer>",
        {"numbers": [12, 13, 15, 11], "target": 150, "allowed_ops": ["+", "-", "*"]},
    )
    comps = result.to_components()
    assert comps["number_multiset_f1"] == 0.0
    assert comps["uses_no_extra_numbers"] == 0.0
    assert comps["uses_all_required_numbers"] == 0.0
    assert comps["uses_allowed_numbers"] == 0.0
    assert comps["allowed_ops"] == 0.0
    assert comps["valid_expression"] == 0.0
    assert comps["exact_correct"] == 0.0
    assert comps["numeric_distance_reward"] == 0.0


def test_partial_legal_number_use_gets_fractional_credit():
    result = verify_completion(
        "<answer>12+13</answer>",
        {"numbers": [12, 13, 15, 11], "target": 150, "allowed_ops": ["+", "-", "*"]},
    )
    comps = result.to_components()
    assert comps["number_precision"] == 1.0
    assert comps["number_recall"] == 0.5
    assert comps["number_multiset_f1"] == 2.0 / 3.0
    assert comps["uses_no_extra_numbers"] == 1.0
    assert comps["uses_all_required_numbers"] == 0.0
    assert comps["uses_allowed_numbers"] == 0.0
    assert comps["valid_expression"] == 0.0
    assert comps["exact_correct"] == 0.0


def test_legal_number_expression_wrong_target_is_valid_but_not_correct():
    result = verify_completion(
        "<answer>((12+13)*(15-11))</answer>",
        {"numbers": [12, 13, 15, 11], "target": 150, "allowed_ops": ["+", "-", "*"]},
    )
    comps = result.to_components()
    assert comps["number_multiset_f1"] == 1.0
    assert comps["uses_allowed_numbers"] == 1.0
    assert comps["uses_allowed_ops"] == 1.0
    assert comps["valid_expression"] == 1.0
    assert comps["exact_correct"] == 0.0


def test_fractional_number_reward_for_missing_number():
    result = verify_expression("(1+2)", [1, 2, 3], 3, ["+", "-", "*"])
    assert not result.correct
    comps = result.to_components()
    assert comps["parse_ok"] == 1.0
    assert comps["uses_allowed_numbers"] == 0.0
    assert comps["number_precision"] == 1.0
    assert comps["number_recall"] == 2.0 / 3.0
    assert comps["number_multiset_f1"] == 0.8
    assert comps["uses_no_extra_numbers"] == 1.0
    assert comps["uses_all_required_numbers"] == 0.0


def test_fractional_number_reward_for_extra_number():
    result = verify_expression("(1+2+4)", [1, 2, 3], 7, ["+", "-", "*"])
    assert not result.correct
    comps = result.to_components()
    assert comps["parse_ok"] == 1.0
    assert comps["uses_allowed_numbers"] == 0.0
    assert comps["number_precision"] == 2.0 / 3.0
    assert comps["number_recall"] == 2.0 / 3.0
    assert comps["number_multiset_f1"] == 2.0 / 3.0
    assert comps["uses_no_extra_numbers"] == 0.0
    assert comps["uses_all_required_numbers"] == 0.0


def test_fractional_number_reward_for_repeated_number():
    result = verify_expression("(1+2+2)", [1, 2, 3], 5, ["+", "-", "*"])
    assert not result.correct
    comps = result.to_components()
    assert comps["parse_ok"] == 1.0
    assert comps["number_precision"] == 2.0 / 3.0
    assert comps["number_recall"] == 2.0 / 3.0
    assert comps["number_multiset_f1"] == 2.0 / 3.0
    assert comps["uses_no_extra_numbers"] == 0.0
    assert comps["uses_all_required_numbers"] == 0.0


def test_numeric_distance_reward_for_wrong_target():
    result = verify_expression("(1+2)*3", [1, 2, 3], 10, ["+", "-", "*"])
    assert not result.correct
    comps = result.to_components()
    assert comps["valid_expression"] == 1.0
    assert comps["numeric_distance_reward"] == 0.5


def test_reject_disallowed_op():
    result = verify_expression("(8/2)+3", [8, 2, 3], 7, ["+", "-", "*"])
    assert not result.correct
    assert not result.uses_allowed_ops


def test_random_generator_is_valid():
    rng = random.Random(0)
    task = random_solvable_task(rng, n_numbers=4, allowed_ops=["+", "-", "*"], max_target=300)
    result = verify_expression(task["solution"], task["numbers"], task["target"], task["allowed_ops"])
    assert result.correct


def test_verify_completion():
    ex = {"numbers": [1, 2, 3], "target": 9, "allowed_ops": ["+", "-", "*"]}
    result = verify_completion("<answer>(1+2)*3</answer>", ex)
    assert result.correct
