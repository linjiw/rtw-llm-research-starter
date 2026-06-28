import importlib.util
import json
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "06_failure_taxonomy.py"
    spec = importlib.util.spec_from_file_location("failure_taxonomy", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_classify_primary_failure_modes():
    mod = load_module()
    assert mod.classify({"exact_correct": 1.0}, "<answer>1+2</answer>") == "exact_correct"
    assert mod.classify({}, "no tags") == "no_answer_span"
    assert mod.classify({"has_extractable_answer_span": 1.0}, "<answer>x</answer>") == "parse_failure"
    assert (
        mod.classify(
            {
                "has_extractable_answer_span": 1.0,
                "parse_ok": 1.0,
                "uses_no_extra_numbers": 1.0,
                "uses_all_required_numbers": 1.0,
                "uses_allowed_ops": 1.0,
                "evaluates_without_exception": 1.0,
                "valid_expression": 1.0,
                "exact_correct": 0.0,
            },
            "<answer>1+2</answer>",
        )
        == "legal_but_wrong_value"
    )


def test_summarize_counts_failure_modes(tmp_path):
    mod = load_module()
    path = tmp_path / "generations.jsonl"
    records = [
        {
            "id": "a",
            "completion": "<answer>1+2</answer>",
            "metrics": {"has_extractable_answer_span": 1.0, "parse_ok": 0.0},
        },
        {
            "id": "b",
            "completion": "<answer>1+2</answer>",
            "metrics": {"exact_correct": 1.0, "valid_expression": 1.0},
        },
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    summary = mod.summarize(path)
    assert summary["n"] == 2
    assert summary["failure_counts"] == {"parse_failure": 1, "exact_correct": 1}
    assert summary["metric_means"]["exact_correct"] == 0.5
