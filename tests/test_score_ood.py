import importlib.util
from pathlib import Path


def load_ood_scorer():
    path = Path(__file__).resolve().parents[1] / "scripts" / "15_score_ood.py"
    spec = importlib.util.spec_from_file_location("score_ood", path)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _cand(idx, expr, exact=0.0, valid=0.0, f1=0.0, tokens=40):
    return {
        "id": "t0",
        "candidate_index": idx,
        "extracted_expression": expr,
        "completion_token_count": tokens,
        "metrics": {
            "exact_correct": exact,
            "valid_expression": valid,
            "number_multiset_f1": f1,
        },
    }


def test_uses_division_detects_slash():
    m = load_ood_scorer()
    assert m.uses_division("(a/b)+c")
    assert not m.uses_division("a+b*c")
    assert not m.uses_division(None)
    assert not m.uses_division("")


def test_arm_stats_empty_bank():
    m = load_ood_scorer()
    assert m.arm_stats([]) == {"present": False}


def test_arm_stats_division_and_legality_rates():
    m = load_ood_scorer()
    rows = [
        _cand(0, "a/b", exact=0.0, valid=1.0, f1=1.0),   # legal, uses /
        _cand(1, "a+b", exact=1.0, valid=1.0, f1=1.0),   # legal exact, no /
        _cand(2, "a-b-c", exact=0.0, valid=0.0, f1=0.5),  # illegal
        _cand(3, "a*b", exact=0.0, valid=0.0, f1=0.0, tokens=256),  # illegal, truncated
    ]
    s = m.arm_stats(rows)
    assert s["n_cand"] == 4
    assert s["legal_rate"] == 0.5
    assert s["div_adoption_rate"] == 0.25
    assert s["truncation_rate"] == 0.25
    assert s["p_exact_given_legal"] == 0.5  # 1 exact of 2 legal


def test_oracle_at_8_counts_task_with_any_exact():
    m = load_ood_scorer()
    rows = [_cand(i, "a+b", exact=(1.0 if i == 3 else 0.0), valid=1.0) for i in range(8)]
    assert m.arm_stats(rows)["oracle_at_8"] == 1
