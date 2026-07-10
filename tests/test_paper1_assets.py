import importlib.util
from pathlib import Path


def load_mod():
    p = Path(__file__).resolve().parents[1] / "scripts" / "17_paper1_assets.py"
    spec = importlib.util.spec_from_file_location("assets", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _cand(idx, exact=0.0, practical=0.0):
    return {"id": "t", "candidate_index": idx, "practical_score": practical,
            "metrics": {"exact_correct": exact}}


def test_oracle_and_reranked_at():
    m = load_mod()
    # task with an exact candidate at idx1 that also has the top practical score
    rows = [_cand(0, exact=0.0, practical=1.0), _cand(1, exact=1.0, practical=9.0)]
    assert m._oracle_at(rows, 8) == 1
    assert m._reranked_at(rows, 8) == 1  # selector picks the exact one


def test_reranked_misses_when_selector_prefers_nonexact():
    m = load_mod()
    rows = [_cand(0, exact=1.0, practical=1.0), _cand(1, exact=0.0, practical=9.0)]
    assert m._oracle_at(rows, 8) == 1     # exact exists
    assert m._reranked_at(rows, 8) == 0   # selector picks the non-exact high-score one


def test_md_table_shape():
    m = load_mod()
    t = m.md_table([{"a": 1, "b": 2}], ["a", "b"])
    assert t.count("\n") == 2  # header + separator + one row
