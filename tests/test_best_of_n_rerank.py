import importlib.util
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "07_best_of_n_rerank.py"
    spec = importlib.util.spec_from_file_location("best_of_n", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def candidate(idx, exact=0.0, valid=0.0, number_f1=0.0, distance=0.0, hack=0.0):
    return {
        "candidate_index": idx,
        "practical_score": 0.0,
        "metrics": {
            "exact_correct": exact,
            "valid_expression": valid,
            "number_multiset_f1": number_f1,
            "uses_allowed_numbers": valid,
            "uses_allowed_ops": valid,
            "uses_all_required_numbers": valid,
            "uses_no_extra_numbers": valid,
            "numeric_distance_reward": distance,
            "reward_hacking_candidate": hack,
            "brevity": 1.0,
        },
    }


def test_practical_score_does_not_depend_on_exact_correct():
    mod = load_module()
    base = candidate(0, exact=0.0, valid=1.0, number_f1=1.0)["metrics"]
    exact = dict(base)
    exact["exact_correct"] = 1.0
    assert mod.practical_score(base) == mod.practical_score(exact)


def test_oracle_selection_prefers_exact_candidate():
    mod = load_module()
    weak_exact = candidate(0, exact=1.0, valid=0.1, number_f1=0.1)
    strong_wrong = candidate(1, exact=0.0, valid=1.0, number_f1=1.0)
    weak_exact["practical_score"] = mod.practical_score(weak_exact["metrics"])
    strong_wrong["practical_score"] = mod.practical_score(strong_wrong["metrics"])
    assert mod.choose_oracle([strong_wrong, weak_exact]) is weak_exact


def test_practical_selection_prefers_legality_score_without_oracle_exact():
    mod = load_module()
    weak_exact = candidate(0, exact=1.0, valid=0.1, number_f1=0.1)
    strong_wrong = candidate(1, exact=0.0, valid=1.0, number_f1=1.0)
    weak_exact["practical_score"] = mod.practical_score(weak_exact["metrics"])
    strong_wrong["practical_score"] = mod.practical_score(strong_wrong["metrics"])
    assert mod.choose_practical([weak_exact, strong_wrong]) is strong_wrong


def test_evaluate_candidates_reports_monotonic_oracle_exact_at_n():
    mod = load_module()
    rows_by_id = {
        "a": [candidate(0, exact=0.0), candidate(1, exact=1.0)],
        "b": [candidate(0, exact=0.0), candidate(1, exact=0.0)],
    }
    for rows in rows_by_id.values():
        for row in rows:
            row["practical_score"] = mod.practical_score(row["metrics"])
    report = mod.evaluate_candidates(rows_by_id, [1, 2])
    assert report["by_n"]["1"]["oracle_exact_at_n"] == 0.0
    assert report["by_n"]["2"]["oracle_exact_at_n"] == 0.5
