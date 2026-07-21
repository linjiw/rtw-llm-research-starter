import importlib.util
import json
from pathlib import Path

import pytest

from rtw_llm.provenance import ProvenanceError, write_intent, write_result


def load_bestofn_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "07_best_of_n_rerank.py"
    spec = importlib.util.spec_from_file_location("best_of_n", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_aggregator_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "08_summarize_v09_seed_expansion.py"
    spec = importlib.util.spec_from_file_location("v09_agg", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def candidate(idx, exact=0.0, valid=0.0, number_f1=0.0, distance=0.0, hack=0.0, tokens=1):
    return {
        "id": "task",
        "candidate_index": idx,
        "practical_score": 0.0,
        "completion_token_count": tokens,
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


def score_rows(mod, rows):
    for row in rows:
        row["practical_score"] = mod.practical_score(row["metrics"])
    return rows


def test_practical_score_does_not_use_exact_correct():
    mod = load_bestofn_module()
    base = candidate(0, exact=0.0, valid=1.0, number_f1=1.0)["metrics"]
    exact = dict(base)
    exact["exact_correct"] = 1.0
    assert mod.practical_score(base) == mod.practical_score(exact)


def test_oracle_selection_prefers_exact_candidate():
    mod = load_bestofn_module()
    weak_exact = candidate(0, exact=1.0, valid=0.1, number_f1=0.1)
    strong_wrong = candidate(1, exact=0.0, valid=1.0, number_f1=1.0)
    score_rows(mod, [weak_exact, strong_wrong])
    assert mod.choose_oracle([strong_wrong, weak_exact]) is weak_exact


def test_practical_selection_prefers_legality_score_without_oracle_exact():
    mod = load_bestofn_module()
    weak_exact = candidate(0, exact=1.0, valid=0.1, number_f1=0.1)
    strong_wrong = candidate(1, exact=0.0, valid=1.0, number_f1=1.0)
    score_rows(mod, [weak_exact, strong_wrong])
    assert mod.choose_practical([weak_exact, strong_wrong]) is strong_wrong


def test_oracle_exact_at_n_is_monotonic():
    mod = load_bestofn_module()
    rows_by_id = {
        "a": score_rows(mod, [candidate(0, exact=0.0), candidate(1, exact=1.0)]),
        "b": score_rows(mod, [candidate(0, exact=0.0), candidate(1, exact=0.0)]),
    }
    report = mod.evaluate_candidates(rows_by_id, [1, 2])
    assert report["by_n"]["1"]["oracle_exact_at_n"] == 0.0
    assert report["by_n"]["2"]["oracle_exact_at_n"] == 0.5
    assert report["by_n"]["2"]["oracle_exact_at_n"] >= report["by_n"]["1"]["oracle_exact_at_n"]


def test_prefix_n_values_reuse_candidate_bank():
    mod = load_bestofn_module()
    rows_by_id = {"a": score_rows(mod, [candidate(0, tokens=3), candidate(1, exact=1.0, valid=1.0, number_f1=1.0, tokens=5)])}
    report = mod.evaluate_candidates(rows_by_id, [1, 2], wall_clock_seconds=20.0, max_n=2)
    assert report["by_n"]["1"]["tokens_generated"] == 3
    assert report["by_n"]["2"]["tokens_generated"] == 8
    assert report["by_n"]["1"]["wall_clock_seconds_estimated"] == 10.0
    assert rows_by_id["a"][0]["selected_by_practical_n"] == [1]
    assert rows_by_id["a"][1]["selected_by_oracle_n"] == [2]


def test_same_task_ids_required_for_pairing(tmp_path):
    agg = load_aggregator_module()
    stable = tmp_path / "v09b_bestofn_stable_v06c_seed1_validation_limit50_n8"
    static = tmp_path / "v09b_bestofn_static_v06b_seed1_validation_limit50_n8"
    stable.mkdir()
    static.mkdir()
    minimal_metrics = {"by_n": {"1": {}}, "total_candidates": 1, "total_tokens_generated": 1}
    for path, method in [(stable, "stable"), (static, "static")]:
        (path / "metrics.json").write_text(json.dumps(minimal_metrics))
        (path / "run_config.json").write_text(json.dumps({"method": method, "training_seed": 1, "split": "validation"}))
    stable_row = candidate(0, exact=1.0, valid=1.0, number_f1=1.0)
    static_row = candidate(0, exact=1.0, valid=1.0, number_f1=1.0)
    stable_row["id"] = "a"
    static_row["id"] = "b"
    for row in [stable_row, static_row]:
        row["practical_score"] = 1.0
    (stable / "candidates.jsonl").write_text(json.dumps(stable_row) + "\n")
    (static / "candidates.jsonl").write_text(json.dumps(static_row) + "\n")
    runs = [agg.load_run(stable), agg.load_run(static)]
    try:
        agg.paired_overlap(runs)
    except ValueError as exc:
        assert "Task ID/order mismatch" in str(exc)
    else:
        raise AssertionError("expected task ID mismatch")


def test_paired_overlap_counts(tmp_path):
    agg = load_aggregator_module()
    dirs = {}
    for method in ["stable", "static"]:
        path = tmp_path / f"v09b_bestofn_{method}_seed1_validation_limit50_n8"
        path.mkdir()
        (path / "metrics.json").write_text(json.dumps({"by_n": {"1": {}}, "total_candidates": 2, "total_tokens_generated": 2}))
        (path / "run_config.json").write_text(json.dumps({"method": method, "training_seed": 1, "split": "validation"}))
        dirs[method] = path
    stable_rows = [candidate(0, exact=1.0), candidate(0, exact=0.0)]
    static_rows = [candidate(0, exact=0.0), candidate(0, exact=0.0)]
    for idx, row in enumerate(stable_rows):
        row["id"] = f"t{idx}"
        row["practical_score"] = 1.0
    for idx, row in enumerate(static_rows):
        row["id"] = f"t{idx}"
        row["practical_score"] = 1.0
    dirs["stable"].joinpath("candidates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in stable_rows))
    dirs["static"].joinpath("candidates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in static_rows))
    pairs = agg.paired_overlap([agg.load_run(dirs["stable"]), agg.load_run(dirs["static"])])
    assert pairs[0]["stable_only"] == 1
    assert pairs[0]["static_only"] == 0
    assert pairs[0]["neither"] == 1


def test_mcnemar_exact_small_sample():
    agg = load_aggregator_module()
    assert agg.exact_mcnemar_p(4, 0) == 0.125
    assert agg.exact_mcnemar_p(2, 1) == 1.0
    assert agg.exact_mcnemar_p(0, 0) == 1.0


def test_cost_per_exact_handles_zero_exact():
    mod = load_bestofn_module()
    assert mod.cost_per_exact(8, 0.0) == 8_000_000_000_000.0
    assert mod.cost_per_exact(8, 0.25) == 32.0


def _strict_identity():
    return {
        "schema_version": "rtw-run-manifest-v1",
        "run_kind": "best_of_n",
        "git": {"commit": "a" * 40, "dirty": False, "status_sha256": None},
        "requested_args": {},
        "resolved_config": {},
        "seed_roles": {"sampling_seed": 0},
        "inputs": {},
        "model": {"name": "model", "revision": "b" * 40},
        "runtime": {"python": "test"},
    }


def test_strict_skip_requires_verified_completed_manifest(tmp_path):
    mod = load_bestofn_module()
    out = tmp_path / "run"
    out.mkdir()
    assert mod.is_strict_complete(out, _strict_identity()) is False

    write_intent(out, _strict_identity())
    with pytest.raises(ProvenanceError):
        mod.is_strict_complete(out, _strict_identity())

    artifact = out / "metrics.json"
    artifact.write_text("{}\n")
    write_result(out, artifact_paths={"metrics": artifact})
    with pytest.raises(ProvenanceError, match="missing required artifacts"):
        mod.is_strict_complete(out, _strict_identity())

    complete = tmp_path / "complete"
    write_intent(complete, _strict_identity())
    artifacts = {}
    for role, name in {
        "candidates": "candidates.jsonl",
        "metrics": "metrics.json",
        "run_config": "run_config.json",
        "summary": "summary.csv",
    }.items():
        path = complete / name
        path.write_text("{}\n")
        artifacts[role] = path
    write_result(complete, artifact_paths=artifacts)
    assert mod.is_strict_complete(complete, _strict_identity()) is True
