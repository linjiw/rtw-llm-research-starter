import inspect
import random

from rtw_llm.microcode import verify_completion
from rtw_llm.microcode_gen import (
    RUNG_TIER,
    TEMPLATES,
    TEMPLATES_BY_RUNG,
    _template_pool,
    difficulty_spec,
    random_solvable_task,
)


def _reference_completion(task):
    tmpl = next(t for t in TEMPLATES if t.key == task["template"])
    src = inspect.getsource(tmpl.reference).replace(
        f"def {tmpl.reference.__name__}(", f"def {task['fn_name']}(", 1
    )
    return f"<answer>\n{src}\n</answer>"


def test_every_rung_has_a_template():
    assert all(r in TEMPLATES_BY_RUNG for r in RUNG_TIER)


def test_train_library_has_at_least_24_templates_4_per_rung():
    train = [t for t in TEMPLATES if t.family == "train"]
    assert len(train) >= 24, len(train)
    from collections import Counter
    per_rung = Counter(t.rung for t in train)
    assert all(per_rung[r] >= 4 for r in range(6)), dict(per_rung)


def test_references_are_solvable_by_construction_across_tiers():
    # The critical metamorphic check: every generated task's reference impl
    # must pass its OWN visible + held-out tests through the verifier, else a
    # reference bug silently mislabels the whole template family.
    rng = random.Random(0)
    for tier in ["easy", "medium", "hard"]:
        spec = difficulty_spec(tier)
        for i in range(30):
            task = random_solvable_task(rng, spec, i, "train")
            c = verify_completion(_reference_completion(task), task).to_components()
            assert c["correct"] == 1.0, (tier, task["template"])
            assert c["held_out_pass_rate"] == 1.0
            assert c["visible_pass_rate"] == 1.0


def test_hardcode_hack_fails_held_out_on_generated_task():
    # A visible-only hardcode must score visible=1 but held_out<1 / primary=0.
    rng = random.Random(3)
    task = random_solvable_task(rng, difficulty_spec("medium"), 0, "train")
    fn, vis = task["fn_name"], task["visible_tests"]
    branches = "\n".join(
        f"    if list(args) == {list(a)}:\n        return {e!r}" for a, e in vis
    )
    # fall back to a wrong constant so held-out mostly fails
    hack = f"def {fn}(*args):\n{branches}\n    return None"
    c = verify_completion(f"<answer>\n{hack}\n</answer>", task).to_components()
    assert c["visible_pass_rate"] == 1.0
    assert c["held_out_pass_rate"] < 1.0
    assert c["correct"] == 0.0


def test_task_schema_has_verifier_and_harness_fields():
    task = random_solvable_task(random.Random(0), difficulty_spec("easy"), 0, "validation")
    for k in ["fn_name", "visible_tests", "held_out_tests", "prompt_low",
              "prompt_mid", "prompt_high", "prompt", "difficulty", "rung", "id"]:
        assert k in task, k
    assert task["prompt"] == task["prompt_high"]
    assert task["difficulty"] == RUNG_TIER[task["rung"]]


def test_generation_is_deterministic_under_seed():
    a = random_solvable_task(random.Random(7), difficulty_spec("hard"), 5, "t")
    b = random_solvable_task(random.Random(7), difficulty_spec("hard"), 5, "t")
    assert a == b


def test_ood_families_are_isolated_and_solvable():
    # I6-c: ood_* specs draw ONLY their held-out family; train specs never draw
    # ood; every ood reference is solvable-by-construction across a JSON round-trip.
    import inspect
    import json
    for ood in ["ood_compose", "ood_transform"]:
        spec = difficulty_spec(ood)
        pool = _template_pool(spec)
        assert pool, ood
        assert all(t.family == ood for t in pool)
        rng = random.Random(1)
        for i in range(40):
            task = json.loads(json.dumps(random_solvable_task(rng, spec, i, ood)))
            tmpl = next(t for t in TEMPLATES if t.key == task["template"])
            src = inspect.getsource(tmpl.reference).replace(
                f"def {tmpl.reference.__name__}(", f"def {task['fn_name']}(", 1
            )
            c = verify_completion(f"<answer>\n{src}\n</answer>", task).to_components()
            assert c["correct"] == 1.0, (ood, task["template"])


def test_ood_hardcode_hack_fails_held_out():
    # The live hacking surface must exist on OOD tasks too.
    task = random_solvable_task(random.Random(2), difficulty_spec("ood_compose"), 0, "ood_compose")
    fn, vis = task["fn_name"], task["visible_tests"]
    branches = "\n".join(f"    if list(args) == {list(a)}:\n        return {e!r}" for a, e in vis)
    hack = f"def {fn}(*args):\n{branches}\n    return None"
    c = verify_completion(f"<answer>\n{hack}\n</answer>", task).to_components()
    assert c["visible_pass_rate"] == 1.0
    assert c["correct"] == 0.0


def test_train_specs_draw_only_train_family():
    # I6 family mechanism: a train tier spec must never sample an ood_* template
    # (the held-out split integrity). Default families=("train",) enforces it.
    for tier in ["easy", "medium", "hard"]:
        spec = difficulty_spec(tier)
        assert spec["families"] == ("train",)
        assert all(t.family == "train" for t in _template_pool(spec))


def test_template_pool_default_families_is_train_only():
    # An un-migrated spec (no 'families' key) must still default to train-only,
    # never silently pulling ood templates.
    spec = {"rungs": [0, 1], "n_visible": 2, "n_held_out": 5}  # no 'families'
    assert all(t.family == "train" for t in _template_pool(spec))


def test_fn_names_randomized_across_tasks():
    rng = random.Random(0)
    spec = difficulty_spec("easy")
    names = {random_solvable_task(rng, spec, i, "t")["fn_name"] for i in range(20)}
    assert len(names) > 1  # not a single memorizable identity
