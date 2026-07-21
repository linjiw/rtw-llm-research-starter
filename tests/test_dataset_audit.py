import json

from rtw_llm.dataset_audit import (
    FROZEN_FILES,
    SPLIT_FILES,
    assert_safe_report_path,
    audit_frozen_ids,
    audit_records,
    audit_repository,
    loose_key,
    overlap_stats,
    semantic_key,
    write_report,
)
from rtw_llm.prompts import make_prompt, make_sft_completion
from rtw_llm.provenance import file_record


def record(task_id="task-1", *, numbers=None, target=6, allowed_ops=None, split="validation"):
    numbers = numbers or [1, 2, 3]
    allowed_ops = allowed_ops or ["+", "-"]
    solution = "((1+2)+3)"
    return {
        "id": task_id,
        "split": split,
        "difficulty": "easy",
        "numbers": numbers,
        "target": target,
        "allowed_ops": allowed_ops,
        "solution": solution,
        "prompt_low": make_prompt(numbers, target, allowed_ops, "low"),
        "prompt_mid": make_prompt(numbers, target, allowed_ops, "mid"),
        "prompt_high": make_prompt(numbers, target, allowed_ops, "high"),
        "prompt": make_prompt(numbers, target, allowed_ops, "high"),
        "completion": make_sft_completion(solution, target),
        "metadata": {"n_numbers": 3},
    }


def test_clean_record_passes_integrity_checks():
    report = audit_records("validation", [record()])
    assert report["schema_errors"] == []
    assert report["solution_failure_ids"] == []
    assert report["completion_failure_ids"] == []
    assert report["within_split_semantic_duplicate_groups"] == 0
    assert all(item["count"] == 0 for item in report["prompt_drift"].values())


def test_semantic_duplicate_ignores_number_and_operator_order():
    first = record("a")
    second = record("b", numbers=[3, 1, 2], allowed_ops=["-", "+"])
    assert semantic_key(first) == semantic_key(second)
    report = audit_records("validation", [first, second])
    assert report["within_split_semantic_duplicate_groups"] == 1
    assert report["operator_order_warning_ids"] == ["b"]


def test_loose_only_overlap_warns_without_exact_semantic_overlap():
    left = [record("a")]
    right = [record("b", allowed_ops=["+", "-", "*"])]
    assert overlap_stats(left, right)["shared_key_groups"] == 0
    assert overlap_stats(left, right, key_fn=loose_key)["shared_key_groups"] == 1


def test_invalid_solution_and_completion_fail_verifier():
    bad = record()
    bad["solution"] = "1+2"
    bad["completion"] = make_sft_completion("1+2", bad["target"])
    report = audit_records("validation", [bad])
    assert report["solution_failure_ids"] == ["task-1"]
    assert report["completion_failure_ids"] == ["task-1"]


def test_frozen_ids_detect_duplicate_missing_and_wrong_split():
    splits = {
        "train": [record("train-1", split="train")],
        "validation": [record("val-1")],
        "test_in_dist": [record("test-1", split="test_in_dist")],
    }
    report = audit_frozen_ids(
        "validation", ["val-1", "val-1", "test-1", "missing"], splits
    )
    assert report["duplicate_ids"] == ["val-1"]
    assert report["wrong_split_ids"] == ["test-1"]
    assert report["missing_ids"] == ["missing"]


def test_prompt_drift_is_reported_not_a_verifier_failure():
    changed = record()
    changed["prompt_low"] = "legacy prompt"
    report = audit_records("validation", [changed])
    assert report["prompt_drift"]["prompt_low"]["ids"] == ["task-1"]
    assert report["solution_failure_ids"] == []


def test_report_bytes_are_deterministic(tmp_path):
    payload = {"b": [2, 1], "a": {"x": 1}}
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_report(first, payload)
    write_report(second, payload)
    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text()) == payload


def test_line_reorder_changes_raw_hash_not_semantic_overlap(tmp_path):
    first = record("a")
    second = record("b", numbers=[4, 1, 1], target=6)
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    first_hash = file_record(path)["sha256"]
    path.write_text(json.dumps(second) + "\n" + json.dumps(first) + "\n")
    second_hash = file_record(path)["sha256"]
    assert first_hash != second_hash
    assert overlap_stats([first, second], [first, second])["shared_key_groups"] == 2


def test_report_path_cannot_overwrite_research_inputs(tmp_path):
    assert assert_safe_report_path(tmp_path, "docs/artifacts/audit.json") == (
        tmp_path / "docs/artifacts/audit.json"
    )
    for unsafe in ("data/countdown/train.jsonl", "src/rtw_llm/countdown.py"):
        try:
            assert_safe_report_path(tmp_path, unsafe)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected protected output rejection: {unsafe}")


def test_repository_malformed_row_returns_integrity_fail_not_exception(monkeypatch, tmp_path):
    import rtw_llm.dataset_audit as dataset_audit

    for relative in SPLIT_FILES.values():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(('{"id":"broken"}\n' if relative.endswith("train.jsonl") else ""))
    for relative in FROZEN_FILES.values():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    for relative in (
        "scripts/00_generate_countdown_dataset.py",
        "src/rtw_llm/prompts.py",
        "src/rtw_llm/countdown.py",
        "src/rtw_llm/dataset_audit.py",
        "README.md",
        "Makefile",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    monkeypatch.setattr(dataset_audit, "_git_head", lambda _: "a" * 40)
    report = audit_repository(tmp_path, replay_generator=True)
    assert report["verdict"]["status"] == "INTEGRITY_FAIL"
    assert report["generator_replay"]["skipped_due_to_structural_errors"] is True


def test_overlap_counts_rows_even_when_duplicate_ids_exist():
    left = [record("duplicate"), record("duplicate")]
    right = [record("right")]
    stats = overlap_stats(left, right)
    assert stats["affected_left_records"] == 2
    assert stats["left_ids"] == ["duplicate"]
    assert stats["record_pair_combinations"] == 2
