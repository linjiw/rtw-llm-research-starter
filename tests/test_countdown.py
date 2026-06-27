from rtw_llm.countdown import extract_answer, verify_expression, verify_completion, random_solvable_task
import random


def test_extract_answer():
    expr, tag = extract_answer("hello <answer>(1+2)*3</answer> bye")
    assert tag
    assert expr == "(1+2)*3"


def test_verify_correct_expression():
    result = verify_expression("(1+2)*3", [1, 2, 3], 9, ["+", "-", "*"])
    assert result.correct
    assert result.uses_all_numbers
    assert result.uses_allowed_ops


def test_reject_invented_constant():
    result = verify_expression("9", [1, 2, 3], 9, ["+", "-", "*"])
    assert not result.correct
    assert not result.uses_all_numbers


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
