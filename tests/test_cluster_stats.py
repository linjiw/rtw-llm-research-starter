import importlib.util
import json
from pathlib import Path

import pytest

from rtw_llm.cluster_stats import (
    ClusterInferenceError,
    evaluation_protocol_signature,
    require_complete_evaluation_signature,
    require_matching_evaluation_signatures,
    semantic_task_key,
    stack_task_runs,
    task_clustered_difference,
    task_clustered_ratio_difference,
    task_seed_product_bootstrap_difference,
)


def load_script(name: str, filename: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate(
    task_id: str,
    index: int,
    *,
    numbers: list[int],
    target: int,
    valid: float,
    exact: float,
) -> dict:
    return {
        "id": task_id,
        "candidate_index": index,
        "difficulty": "easy",
        "numbers": numbers,
        "target": target,
        "allowed_ops": ["+", "-", "*"],
        "practical_score": valid + exact,
        "completion_token_count": 4,
        "metrics": {
            "valid_expression": valid,
            "exact_correct": exact,
            "expression": f"expr-{task_id}-{index}",
        },
    }


def test_identical_panels_have_zero_effect_and_unit_sign_flip_p():
    panel = {"a": [0.0, 1.0], "b": [1.0, 0.0], "c": [1.0, 1.0]}
    result = task_clustered_difference(panel, panel, bootstrap_draws=200, seed=4)
    assert result["estimate"] == 0.0
    assert result["confidence_interval"]["lower"] == 0.0
    assert result["confidence_interval"]["upper"] == 0.0
    assert result["sign_flip"]["p_value_two_sided"] == 1.0
    assert result["task_clusters"] == 3


def test_all_positive_task_effects_use_exact_sign_flip():
    arm = {f"t{i}": [1.0] for i in range(4)}
    baseline = {f"t{i}": [0.0] for i in range(4)}
    result = task_clustered_difference(arm, baseline, bootstrap_draws=100, seed=1)
    assert result["estimate"] == 1.0
    assert result["sign_flip"] == {
        "method": "exact_task_sign_flip",
        "p_value_two_sided": 0.125,
        "nonzero_task_clusters": 4,
        "assignments": 16,
    }


def test_candidate_correlation_is_reduced_to_task_clusters():
    # Each value is already a within-task candidate rate. Eight correlated
    # candidates per row must not produce 24 independent observations.
    arm = {"a": [8 / 8], "b": [0 / 8], "c": [8 / 8]}
    baseline = {"a": [0 / 8], "b": [0 / 8], "c": [0 / 8]}
    result = task_clustered_difference(arm, baseline, bootstrap_draws=200, seed=2)
    assert result["task_clusters"] == 3
    assert result["arm_runs_per_task"] == 1
    assert result["estimate"] == pytest.approx(2 / 3)


def test_bootstrap_and_monte_carlo_are_deterministic():
    arm = {f"t{i}": [float(i % 2), float((i + 1) % 2)] for i in range(21)}
    baseline = {f"t{i}": [0.0, 0.0] for i in range(21)}
    first = task_clustered_difference(
        arm,
        baseline,
        bootstrap_draws=257,
        sign_flip_draws=263,
        seed=91,
        exact_sign_flip_max_nonzero=2,
    )
    second = task_clustered_difference(
        arm,
        baseline,
        bootstrap_draws=257,
        sign_flip_draws=263,
        seed=91,
        exact_sign_flip_max_nonzero=2,
    )
    assert first == second
    assert first["confidence_interval"] == {
        "method": "task_cluster_percentile_bootstrap",
        "confidence": 0.95,
        "lower": 0.5,
        "upper": 0.5,
    }
    assert first["sign_flip"]["method"] == "monte_carlo_task_sign_flip"


def test_task_rows_are_resampled_jointly_across_runs():
    # Within each task the two run values are anti-correlated, so every task
    # mean and every task-resampled contrast is exactly zero.
    arm = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    baseline = {"a": [0.5, 0.5], "b": [0.5, 0.5]}
    result = task_clustered_difference(arm, baseline, bootstrap_draws=200, seed=8)
    assert result["confidence_interval"]["lower"] == 0.0
    assert result["confidence_interval"]["upper"] == 0.0


def test_incomplete_or_unbalanced_task_grid_fails_closed():
    with pytest.raises(ClusterInferenceError, match="incomplete task grid"):
        stack_task_runs([{"a": 1.0, "b": 0.0}, {"a": 1.0}], label="arm")
    with pytest.raises(ClusterInferenceError, match="unbalanced run count"):
        task_clustered_difference({"a": [1.0], "b": [0.0, 1.0]}, {"a": [0.0], "b": [0.0]})
    with pytest.raises(ClusterInferenceError, match="semantic task mismatch"):
        task_clustered_difference({"a": [1.0]}, {"b": [0.0]})


def test_semantic_key_ignores_presentation_order_but_keeps_operators():
    left = semantic_task_key(
        {"numbers": [3, 1, 2], "target": 6, "allowed_ops": ["*", "+", "+"]}
    )
    reordered = semantic_task_key(
        {"numbers": [2, 3, 1], "target": 6, "allowed_ops": ["+", "*"]}
    )
    different_ops = semantic_task_key(
        {"numbers": [2, 3, 1], "target": 6, "allowed_ops": ["+"]}
    )
    assert left == reordered
    assert left != different_ops


@pytest.mark.parametrize(
    "row",
    [
        {"numbers": [1.2, 2], "target": 3, "allowed_ops": ["+"]},
        {"numbers": [1, 2], "target": 3.8, "allowed_ops": ["+"]},
        {"numbers": [True, 2], "target": 3, "allowed_ops": ["+"]},
    ],
)
def test_semantic_key_rejects_nonintegral_or_boolean_values(row):
    with pytest.raises(ClusterInferenceError, match="must be an integer"):
        semantic_task_key(row)


def test_evaluation_signature_rejects_sampling_protocol_mismatch(tmp_path):
    task_ids = tmp_path / "ids.txt"
    task_ids.write_text("a\nb\n")
    base = {
        "sampling_seed": 0,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_new_tokens": 256,
        "prompt_field": "prompt",
        "engine": "hf",
        "hf_gen_mode": "loop",
        "split": "validation",
        "task_ids_file": str(task_ids),
    }
    signature = evaluation_protocol_signature(base)
    mismatch = evaluation_protocol_signature({**base, "temperature": 0.8})
    with pytest.raises(ClusterInferenceError, match="temperature"):
        require_matching_evaluation_signatures({"arm": signature, "base": mismatch})


def test_incomplete_evaluation_signature_fails_closed():
    signature = evaluation_protocol_signature({"prompt_field": "prompt"})
    with pytest.raises(ClusterInferenceError, match="incomplete evaluation protocol"):
        require_complete_evaluation_signature(signature, label="arm")


def test_ratio_bootstrap_reports_zero_legal_denominator():
    result = task_clustered_ratio_difference(
        {"a": [0.0]},
        {"a": [0.0]},
        {"a": [1.0]},
        {"a": [1.0]},
        bootstrap_draws=100,
    )
    assert result["available"] is False
    assert result["reason"] == "zero_observed_legal_denominator"


def test_ratio_bootstrap_resamples_tasks_and_is_deterministic():
    args = (
        {"a": [1.0], "b": [0.0]},
        {"a": [1.0], "b": [1.0]},
        {"a": [0.0], "b": [0.0]},
        {"a": [1.0], "b": [1.0]},
    )
    first = task_clustered_ratio_difference(*args, bootstrap_draws=200, seed=12)
    second = task_clustered_ratio_difference(*args, bootstrap_draws=200, seed=12)
    assert first == second
    assert first["estimate"] == 0.5
    assert first["task_clusters"] == 2
    assert first["dropped_bootstrap_draws"] == 0
    assert first["post_treatment_noncausal"] is True


def test_sparse_ratio_bootstrap_withholds_unstable_interval():
    keys = [f"t{i}" for i in range(50)]
    arm_den = {key: [float(index == 0)] for index, key in enumerate(keys)}
    arm_num = dict(arm_den)
    base_den = dict(arm_den)
    base_num = {key: [0.0] for key in keys}
    result = task_clustered_ratio_difference(
        arm_num,
        arm_den,
        base_num,
        base_den,
        bootstrap_draws=2_000,
        seed=3,
    )
    assert result["available"] is False
    assert result["reason"] == "sparse_legal_support_exceeds_dropped_draw_threshold"
    assert result["dropped_bootstrap_fraction"] > 0.30
    assert "confidence_interval" not in result


def test_task_seed_bootstrap_requires_true_seed_protocol_and_three_seeds():
    three_arm = {"a": [1.0, 1.0, 1.0], "b": [0.0, 1.0, 0.0]}
    three_base = {"a": [0.0, 0.0, 0.0], "b": [0.0, 0.0, 0.0]}
    with pytest.raises(ClusterInferenceError, match="countdown-true-seeds-v2"):
        task_seed_product_bootstrap_difference(
            three_arm, three_base, true_seed_protocol=False, bootstrap_draws=50
        )
    with pytest.raises(ClusterInferenceError, match="at least three"):
        task_seed_product_bootstrap_difference(
            {"a": [1.0, 1.0]},
            {"a": [0.0, 0.0]},
            true_seed_protocol=True,
            bootstrap_draws=50,
        )
    result = task_seed_product_bootstrap_difference(
        three_arm,
        three_base,
        true_seed_protocol=True,
        bootstrap_draws=100,
        seed=5,
    )
    assert result["status"] == "exploratory_underpowered_seed_generalization"
    assert result["confirmatory_p_value"] is None
    assert result["true_training_seeds"] == 3


def test_v09_aggregate_marks_pooled_mcnemar_invalid():
    mod = load_script("v09_cluster_test", "08_summarize_v09_seed_expansion.py")
    summary = mod.aggregate_paired(
        [
            {
                "split": "validation",
                "training_seed": 0,
                "N": 8,
                "selector": "practical",
                "both": 1,
                "stable_only": 2,
                "static_only": 0,
                "neither": 1,
                "delta_reranked_exact": 0.5,
            },
            {
                "split": "validation",
                "training_seed": 1,
                "N": 8,
                "selector": "practical",
                "both": 1,
                "stable_only": 2,
                "static_only": 0,
                "neither": 1,
                "delta_reranked_exact": 0.5,
            },
        ]
    )[0]
    assert "mcnemar_p" not in summary
    legacy = summary["legacy_pseudoreplicated_descriptive_only"]
    assert legacy["inference_valid"] is False
    assert legacy["pooled_mcnemar_p"] == 0.125


def test_v09_clustered_panel_uses_semantic_tasks_and_withdraws_legacy_seed_claim(tmp_path):
    mod = load_script("v09_cluster_panel_test", "08_summarize_v09_seed_expansion.py")
    runs = []
    for method, exact_values in (("stable", [1.0, 1.0]), ("static", [0.0, 0.0])):
        path = tmp_path / method / "candidates.jsonl"
        path.parent.mkdir()
        rows = [
            candidate("id-a", 0, numbers=[1, 2, 3], target=6, valid=1.0, exact=exact_values[0]),
            candidate("id-b", 0, numbers=[2, 2, 3], target=7, valid=1.0, exact=exact_values[1]),
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        runs.append(
            {
                "run_dir": str(path.parent),
                "metrics": {"by_n": {"1": {}}},
                "config": {"training_protocol": "countdown-legacy-v1"},
                "method": method,
                "training_seed": 0,
                "split": "validation",
                "candidates_path": path,
            }
        )
    result = mod.clustered_paired(runs, selector="practical", bootstrap_draws=100)[0]
    clustered = result["task_clustered_observed_panel"]
    assert clustered["task_clusters"] == 2
    assert clustered["estimate"] == 1.0
    assert result["training_seed_generalization"]["available"] is False


def test_v09_clustered_panel_rejects_duplicate_method_run_cell():
    mod = load_script("v09_duplicate_cell_test", "08_summarize_v09_seed_expansion.py")
    base = {
        "run_dir": "unused",
        "metrics": {"by_n": {"1": {}}},
        "config": {},
        "training_seed": 0,
        "split": "validation",
        "candidates_path": Path("unused"),
    }
    runs = [
        {**base, "method": "stable"},
        {**base, "method": "stable"},
        {**base, "method": "static"},
    ]
    with pytest.raises(ClusterInferenceError, match="duplicate method-by-run-label"):
        mod.clustered_paired(runs, selector="practical", bootstrap_draws=10)


def test_v09_pairing_rejects_temperature_mismatch():
    mod = load_script("v09_protocol_mismatch_test", "08_summarize_v09_seed_expansion.py")
    base = {
        "run_dir": "unused",
        "metrics": {"by_n": {"1": {}}},
        "training_seed": 0,
        "split": "validation",
        "candidates_path": Path("unused"),
    }
    runs = [
        {**base, "method": "stable", "config": {"temperature": 0.7}},
        {**base, "method": "static", "config": {"temperature": 0.8}},
    ]
    with pytest.raises(ValueError, match="Evaluation-identity mismatch.*temperature"):
        mod.paired_overlap(runs)


def test_tracked_legacy_v09_summary_has_no_generic_aggregate_p_values():
    artifact = Path(__file__).resolve().parents[1] / "outputs" / "v09_seed_expansion_paired.json"
    payload = json.loads(artifact.read_text())
    assert payload["protocol_correction"]["cross_run_pooled_mcnemar_inference_valid"] is False
    for key in ("paired_summary", "paired_oracle_summary"):
        for row in payload[key]:
            assert "mcnemar_p" not in row
            assert row["legacy_pseudoreplicated_descriptive_only"]["inference_valid"] is False


def test_v13_scorer_replaces_candidate_z_test_with_task_clustered_inference():
    mod = load_script("v13_cluster_test", "12_score_v13.py")
    arm = {
        "a": [
            candidate("a", 0, numbers=[1, 2, 3], target=6, valid=1.0, exact=1.0),
            candidate("a", 1, numbers=[1, 2, 3], target=6, valid=1.0, exact=0.0),
        ],
        "b": [
            candidate("b", 0, numbers=[2, 2, 3], target=7, valid=1.0, exact=0.0),
            candidate("b", 1, numbers=[2, 2, 3], target=7, valid=1.0, exact=0.0),
        ],
    }
    baseline = {
        task_id: [dict(row, metrics={**row["metrics"], "valid_expression": 0.0, "exact_correct": 0.0}) for row in rows]
        for task_id, rows in arm.items()
    }
    result = mod.score_arm(
        "arm",
        arm,
        {"baseline-0": baseline},
        overlap=set(),
        gold_by_task={},
        n=2,
        bootstrap_draws=100,
    )
    easy = result["easy_legality_all"]
    assert "two_proportion_p_vs_pooled" not in easy
    assert easy["legacy_pseudoreplicated_descriptive_only"]["inference_valid"] is False
    assert easy["task_clustered_vs_baseline_panel"]["task_clusters"] == 2
    ratio = result["p_exact_given_legal_task_clustered_vs_baseline_panel"]
    assert ratio["available"] is False
    assert ratio["reason"] == "zero_observed_legal_denominator"


def test_v13_overlap_identity_includes_allowed_operators(tmp_path):
    mod = load_script("v13_overlap_ops_test", "12_score_v13.py")
    train = tmp_path / "train.jsonl"
    train.write_text(
        json.dumps(
            {
                "numbers": [1, 2, 3],
                "target": 6,
                "allowed_ops": ["+"],
                "solution": "1 + 2 + 3",
            }
        )
        + "\n"
    )
    row = candidate("a", 0, numbers=[1, 2, 3], target=6, valid=1.0, exact=1.0)
    row["allowed_ops"] = ["*"]
    overlap, gold = mod.overlap_task_ids({"a": [row]}, train)
    assert overlap == set()
    assert gold == {}


def test_v13_combined_arm_panel_averages_observed_runs():
    mod = load_script("v13_observed_panel_test", "12_score_v13.py")
    arm_high = {
        "a": [candidate("a", 0, numbers=[1, 2, 3], target=6, valid=1.0, exact=0.0)]
    }
    arm_low = {
        "a": [candidate("a", 0, numbers=[1, 2, 3], target=6, valid=0.0, exact=0.0)]
    }
    baseline = {
        "a": [candidate("a", 0, numbers=[1, 2, 3], target=6, valid=0.0, exact=0.0)]
    }
    result = mod.clustered_candidate_panel_comparison(
        [arm_high, arm_low],
        {"baseline": baseline},
        tier="easy",
        bootstrap_draws=50,
        seed=2,
    )
    assert result["arm_observed_runs"] == 2
    assert result["legality_rate_difference"]["estimate"] == 0.5


def test_v13_panel_rejects_unequal_candidate_n():
    mod = load_script("v13_unequal_n_test", "12_score_v13.py")
    one = {
        "a": [candidate("a", 0, numbers=[1, 2, 3], target=6, valid=1.0, exact=0.0)]
    }
    two = {
        "a": [
            candidate("a", 0, numbers=[1, 2, 3], target=6, valid=1.0, exact=0.0),
            candidate("a", 1, numbers=[1, 2, 3], target=6, valid=1.0, exact=0.0),
        ]
    }
    with pytest.raises(ClusterInferenceError, match="unequal candidate counts"):
        mod.clustered_candidate_panel_comparison(
            [one], {"baseline": two}, bootstrap_draws=20, seed=2
        )


def test_v13_comparison_rejects_prompt_signature_mismatch():
    mod = load_script("v13_signature_test", "12_score_v13.py")
    arm = {
        "name": "arm",
        "evaluation_signature": evaluation_protocol_signature({"prompt_field": "prompt_high"}),
    }
    baseline = {
        "base": {
            "evaluation_signature": evaluation_protocol_signature(
                {"prompt_field": "prompt_low"}
            )
        }
    }
    with pytest.raises(ClusterInferenceError, match="prompt_field"):
        mod.comparison_evaluation_signature([arm], baseline)


def test_v13_scorer_rejects_duplicate_semantic_tasks():
    mod = load_script("v13_duplicate_test", "12_score_v13.py")
    bank = {
        "a": [candidate("a", 0, numbers=[1, 2, 3], target=6, valid=1.0, exact=0.0)],
        "b": [candidate("b", 0, numbers=[3, 2, 1], target=6, valid=1.0, exact=0.0)],
    }
    with pytest.raises(ClusterInferenceError, match="duplicate semantic tasks"):
        mod.task_candidate_components(bank)


def test_v13_scorer_rejects_duplicate_candidate_index():
    mod = load_script("v13_candidate_cell_test", "12_score_v13.py")
    rows = [
        candidate("a", 0, numbers=[1, 2, 3], target=6, valid=1.0, exact=0.0),
        candidate("a", 0, numbers=[1, 2, 3], target=6, valid=1.0, exact=0.0),
    ]
    with pytest.raises(ClusterInferenceError, match="candidate indices"):
        mod.task_candidate_components({"a": rows})
