import importlib.util
from pathlib import Path


def load_scorer():
    path = Path(__file__).resolve().parents[1] / "scripts" / "16_score_harness_shift.py"
    spec = importlib.util.spec_from_file_location("score_harness", path)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _cand(valid=0.0, f1=0.0, evals=0.0, expr="a+b"):
    return {
        "extracted_expression": expr,
        "metrics": {
            "valid_expression": valid,
            "number_multiset_f1": f1,
            "evaluates_without_exception": evals,
        },
    }


def test_field_stats_empty():
    m = load_scorer()
    assert m.field_stats([]) == {"present": False}


def test_field_stats_legality_and_parseable_f1():
    m = load_scorer()
    rows = [
        _cand(valid=1.0, f1=1.0, evals=1.0),   # legal, parseable, f1=1
        _cand(valid=0.0, f1=0.5, evals=1.0),   # parseable-but-illegal, f1=0.5
        _cand(valid=0.0, f1=0.0, evals=0.0, expr=None),  # unparseable
    ]
    s = m.field_stats(rows)
    assert s["n_cand"] == 3
    assert abs(s["legal_rate"] - 1 / 3) < 1e-9
    # parseable set = the two with evals/valid; f1 mean over them = (1.0+0.5)/2
    assert s["n_parseable"] == 2
    assert abs(s["number_f1_parseable_mean"] - 0.75) < 1e-9


def test_parseable_restriction_excludes_fulltext_fallback():
    # A candidate with no expression and f1=0 (extract_answer full-text fallback)
    # must NOT be counted as parseable, so it can't dilute the f1 read.
    m = load_scorer()
    rows = [_cand(valid=1.0, f1=1.0, evals=1.0), _cand(valid=0.0, f1=0.0, evals=0.0, expr=None)]
    s = m.field_stats(rows)
    assert s["n_parseable"] == 1
    assert s["number_f1_parseable_mean"] == 1.0
