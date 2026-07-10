import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest
import torch

import rtw_llm.v19_protocol as v19
from rtw_llm.data_access import DataAccessError, assert_countdown_data_access
from rtw_llm.engine import HFEngine


def load_script(name: str, filename: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validation_rows():
    rows = []
    index = 0
    for tier, count in v19.SOURCE_VALIDATION_QUOTAS.items():
        for _ in range(count):
            rows.append({"id": f"task-{index:03d}", "difficulty": tier})
            index += 1
    # The view files must preserve this non-tier-grouped source order.
    return rows[::2] + rows[1::2]


def test_validation_views_are_deterministic_disjoint_complete_and_stratified():
    rows = validation_rows()
    first = v19.build_validation_views(rows)
    second = v19.build_validation_views(rows)
    assert first == second
    dev = first["validation_dev100"]
    confirm = first["validation_confirm400"]
    assert len(dev) == 100
    assert len(confirm) == 400
    assert set(dev).isdisjoint(confirm)
    assert set(dev) | set(confirm) == {row["id"] for row in rows}
    by_id = {row["id"]: row for row in rows}
    assert Counter(by_id[value]["difficulty"] for value in dev) == Counter(
        {"easy": 10, "medium": 45, "hard": 45}
    )
    assert Counter(by_id[value]["difficulty"] for value in confirm) == Counter(
        {"easy": 40, "medium": 180, "hard": 180}
    )
    assert first["validation_preflight2"] == dev[:2]
    assert dev == [row["id"] for row in rows if row["id"] in set(dev)]


def test_validation_view_builder_rejects_wrong_source_grid():
    rows = validation_rows()
    with pytest.raises(v19.V19ProtocolError, match="500 rows"):
        v19.build_validation_views(rows[:-1])
    rows[-1] = {**rows[-1], "id": rows[0]["id"]}
    with pytest.raises(v19.V19ProtocolError, match="duplicate IDs"):
        v19.build_validation_views(rows)


def test_repository_protocol_replays_and_has_sixteen_states():
    root = Path(__file__).resolve().parents[1]
    report = v19.audit_protocol(root)
    assert report["status"] == "ELIGIBLE"
    assert len(v19.state_labels()) == 16
    assert len(set(v19.state_labels())) == 16
    manifest = json.loads((root / v19.PROTOCOL_MANIFEST).read_text())
    assert manifest["state_labels"] == v19.state_labels()
    assert manifest["test_release_created"] is False
    assert manifest["final_test_release_created"] is False
    for required_source in (
        "src/rtw_llm/countdown.py",
        "src/rtw_llm/rewards.py",
        "src/rtw_llm/teacher.py",
        "src/rtw_llm/seed_protocol.py",
        "src/rtw_llm/cluster_stats.py",
    ):
        assert required_source in manifest["protocol_sources"]


def test_v19_invocation_contracts_fail_closed():
    sft = {
        "model_name": v19.MODEL_NAME,
        "model_revision": v19.MODEL_REVISION,
        "train_path": v19.TRAIN_PATH.as_posix(),
        "eval_path": None,
        "max_steps": 313,
        "batch_size": 2,
        "grad_accum": 8,
        "learning_rate": 5e-5,
        "completion_only_loss": True,
        "seed_protocol": v19.TRUE_SEED_PROTOCOL,
        "strict_provenance": True,
        "seed": 0,
    }
    v19.validate_v19_sft_args(sft)
    with pytest.raises(v19.V19ProtocolError, match="eval_path"):
        v19.validate_v19_sft_args({**sft, "eval_path": "validation.jsonl"})

    grpo = {
        "model_name": v19.MODEL_NAME,
        "model_revision": v19.MODEL_REVISION,
        "train_path": v19.TRAIN_PATH.as_posix(),
        "eval_path": None,
        "method_arm": "sft_grpo_stable",
        "reward_strategy": "adaptive_stable",
        "max_steps": 300,
        "learning_rate": 5e-6,
        "batch_size": 2,
        "grad_accum": 8,
        "num_generations": 4,
        "max_prompt_length": 768,
        "max_completion_length": 256,
        "task_curriculum": "uniform",
        "prompt_field": "prompt",
        "seed_protocol": v19.TRUE_SEED_PROTOCOL,
        "strict_provenance": True,
        "seed": 1,
        "trainer_seed": 1,
        "init_adapter_path": "sft-parent",
    }
    v19.validate_v19_grpo_args(grpo)
    with pytest.raises(v19.V19ProtocolError, match="reward_strategy"):
        v19.validate_v19_grpo_args({**grpo, "reward_strategy": "static"})


def test_production_command_matrix_has_one_sft_and_four_grpo_jobs_per_seed():
    runner = load_script("v19_runner_test", "24_run_v19.py")
    commands = runner.production_train_commands("python", Path("outputs/v19/production"), 2)
    assert [item["name"] for item in commands] == [
        "sft_seed2",
        "grpo_static_seed2",
        "grpo_stable_seed2",
        "sft_grpo_static_seed2",
        "sft_grpo_stable_seed2",
    ]
    rendered = [item["command"] for item in commands]
    assert all("--strict_provenance" in command for command in rendered)
    assert all(v19.PROTOCOL_ID in command for command in rendered)
    assert all("--eval_path" not in command for command in rendered)
    combined = [command for command in rendered if "--init_adapter_path" in command]
    assert len(combined) == 2


def test_production_eval_matrix_and_signature_are_frozen():
    runner = load_script("v19_eval_runner_test", "24_run_v19.py")
    commands = runner.production_eval_commands(
        "python",
        Path("outputs/v19/production"),
        view="validation_confirm400",
        seeds=v19.TRAINING_SEEDS,
    )
    assert len(commands) == 16
    for item in commands:
        command = item["command"]
        assert command[command.index("--hf_gen_mode") + 1] == "batched"
        assert command[command.index("--batch_size") + 1] == "16"
        assert command[command.index("--max_n") + 1] == "8"
        assert command[command.index("--seed") + 1] == "0"
        assert "test_in_dist" not in " ".join(command)


def test_registered_eval_rejects_non_cuda_device():
    args = {
        "model_name": v19.MODEL_NAME,
        "model_revision": v19.MODEL_REVISION,
        "engine": "hf",
        "hf_gen_mode": "batched",
        "batch_size": 16,
        "prompt_field": "prompt",
        "seed": 0,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_new_tokens": 256,
        "max_n": 8,
        "n_values": [1, 4, 8],
        "training_protocol": v19.TRUE_SEED_PROTOCOL,
        "strict_provenance": True,
        "device": "mps",
        "method": "base",
        "adapter_path": None,
        "training_seed": None,
        "split": "validation_dev100",
        "data_path": v19.VALIDATION_PATH.as_posix(),
        "task_ids_file": (v19.PROTOCOL_DIR / v19.VIEW_FILES["validation_dev100"]).as_posix(),
        "limit": None,
    }
    with pytest.raises(v19.V19ProtocolError, match="device"):
        v19.validate_v19_eval_args(args)


def test_actual_confirm_view_is_blocked_without_committed_readiness():
    root = Path(__file__).resolve().parents[1]
    assert_countdown_data_access(
        v19.VALIDATION_PATH,
        purpose="model_eval",
        runner="07_best_of_n_rerank",
        experiment_protocol=v19.PROTOCOL_ID,
        ordered_task_ids_file=v19.PROTOCOL_DIR / v19.VIEW_FILES["validation_dev100"],
        repo_root=root,
    )
    with pytest.raises(DataAccessError, match="requires a readiness record"):
        assert_countdown_data_access(
            v19.VALIDATION_PATH,
            purpose="model_eval",
            runner="07_best_of_n_rerank",
            experiment_protocol=v19.PROTOCOL_ID,
            ordered_task_ids_file=(
                v19.PROTOCOL_DIR / v19.VIEW_FILES["validation_confirm400"]
            ),
            repo_root=root,
        )


def test_exact_generation_metadata_distinguishes_eos_and_length():
    engine = object.__new__(HFEngine)
    engine.eos_token_ids = {7}
    eos = engine._metadata_for_tokens(torch.tensor([1, 7]), 2)
    capped = engine._metadata_for_tokens(torch.tensor([1, 2]), 2)
    other = engine._metadata_for_tokens(torch.tensor([1]), 2)
    assert eos == {
        "generated_token_count": 2,
        "finish_reason": "eos",
        "completion_hit_cap": False,
    }
    assert capped["finish_reason"] == "length"
    assert capped["completion_hit_cap"] is True
    assert other["finish_reason"] == "other"


def test_holm_adjustment_is_monotone_and_familywise():
    scorer = load_script("v19_score_test", "22_score_v19.py")
    result = scorer.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.20, "d": 0.50})
    assert result["a"]["holm_adjusted_p_value"] == pytest.approx(0.04)
    assert result["b"]["holm_adjusted_p_value"] == pytest.approx(0.12)
    assert result["a"]["reject_familywise_0.05"] is True
    assert result["b"]["reject_familywise_0.05"] is False


def test_scorer_rejects_task_payload_drift_and_impossible_token_metadata(
    monkeypatch, tmp_path
):
    scorer = load_script("v19_score_payload_test", "22_score_v19.py")
    monkeypatch.setattr(
        scorer,
        "verify_completed_run",
        lambda *args, **kwargs: {"intent": {"identity": {}}},
    )
    task = {
        "id": "task-1",
        "difficulty": "easy",
        "numbers": [1, 2, 3],
        "target": 6,
        "allowed_ops": ["+", "-", "*"],
    }
    completion = "<answer>((1+2)+3)</answer>"
    metrics = scorer.metrics_for_completion(completion, task)
    base_row = {
        **task,
        "candidate_index": 0,
        "raw_generation": completion,
        "metrics": metrics,
        "token_count_source": "generated_token_ids",
        "completion_token_count": 4,
        "finish_reason": "eos",
        "completion_hit_cap": False,
    }
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_config.json").write_text("{}")
    (run_dir / "candidates.jsonl").write_text(json.dumps({**base_row, "target": 7}) + "\n")
    with pytest.raises(scorer.V19ScoreError, match="differs from frozen source"):
        scorer.load_bank(run_dir, {"task-1": task})

    with pytest.raises(scorer.V19ScoreError, match="exceeds the registered"):
        scorer._recomputed_candidate(
            {
                **base_row,
                "completion_token_count": 257,
                "finish_reason": "length",
                "completion_hit_cap": True,
            },
            expected_task=task,
        )


def test_training_health_rejects_primary_reward_disagreement(monkeypatch, tmp_path):
    monkeypatch.setattr(
        v19,
        "verify_completed_run",
        lambda *args, **kwargs: {
            "intent": {"experiment_id": "x", "identity": {"git": {"dirty": False}}}
        },
    )
    monkeypatch.setattr(v19, "validate_run_source_identity", lambda *args, **kwargs: None)
    (tmp_path / "training_state.json").write_text(
        json.dumps(
            {"global_step": 1, "max_steps": 1, "log_history": [], "wall_clock_seconds": 1.0}
        )
    )
    reward = {
        "primary_reward": 1.0,
        "primary_reward_weighted": 1.0,
        "aux_reward_weighted": 0.0,
        "total_reward": 1.0,
        "reward": 1.0,
        "components": {"exact_correct": 0.0, "correct": 0.0},
        "group_has_variance": True,
    }
    (tmp_path / "reward_components.jsonl").write_text(json.dumps(reward) + "\n")
    (tmp_path / "teacher_weights.jsonl").write_text(
        json.dumps({"strategy": "static"}) + "\n"
    )
    with pytest.raises(v19.V19ProtocolError, match="disagrees with verifier"):
        v19.verify_v19_training_health(
            tmp_path,
            run_kind="grpo",
            expected_steps=1,
            expected_strategy="static",
        )


def test_adapter_chain_rejects_wrong_sft_parent(monkeypatch, tmp_path):
    train = tmp_path / "train"
    evaluation = tmp_path / "eval"
    parent = tmp_path / "parent"
    identities = {
        str(train): {
            "seed_roles": {"trainer_seed": 0},
            "model": {"adapter_identity": {"tree": "wrong-parent"}},
        },
        str(evaluation): {
            "seed_roles": {"training_seed_label": 0},
            "model": {"adapter_identity": {"tree": "train"}},
        },
        str(parent): {"seed_roles": {"trainer_seed": 0}, "model": {}},
    }

    def completed(path, *args, **kwargs):
        return {"intent": {"identity": identities[str(path)], "experiment_id": str(path)}}

    monkeypatch.setattr(v19, "verify_completed_run", completed)
    monkeypatch.setattr(v19, "validate_run_source_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        v19,
        "adapter_record",
        lambda path: {"tree": Path(path).name},
    )
    with pytest.raises(v19.V19ProtocolError, match="SFT parent"):
        v19.verify_adapter_chain(
            arm="sft_grpo_stable",
            seed=0,
            training_run=train,
            eval_run=evaluation,
            sft_parent=parent,
            repo_root=tmp_path,
        )


def test_score_matrix_exercises_complete_state_grid(monkeypatch, tmp_path):
    scorer = load_script("v19_score_matrix_test", "22_score_v19.py")
    runtime = {"runtime": "locked"}
    lock_path = tmp_path / v19.ENVIRONMENT_LOCK
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("{}")
    monkeypatch.setattr(scorer, "require_eligible_protocol", lambda _root: None)
    monkeypatch.setattr(scorer, "_load_ids", lambda _path: ["task-1"])
    task = {
        "id": "task-1",
        "difficulty": "easy",
        "numbers": [1, 2, 3],
        "target": 6,
        "allowed_ops": ["+", "-", "*"],
    }
    monkeypatch.setattr(scorer, "load_frozen_tasks", lambda *_args: {"task-1": task})
    monkeypatch.setattr(scorer, "validate_run_source_identity", lambda *args: None)
    monkeypatch.setattr(scorer, "verify_adapter_chain", lambda **kwargs: None)
    monkeypatch.setattr(
        scorer, "validate_environment_lock_document", lambda *_args: runtime
    )
    monkeypatch.setattr(
        scorer,
        "file_record",
        lambda path: {"sha256": str(path), "size": 1, "line_count": 1},
    )

    def fake_bank(run_dir, expected_tasks):
        label = Path(run_dir).name
        if label == "base":
            arm, seed = "base", None
        else:
            arm, raw_seed = label.rsplit("_seed", 1)
            seed = int(raw_seed)
        return {
            "config": {
                "experiment_protocol": v19.PROTOCOL_ID,
                "method": arm,
                "training_seed": seed,
            },
            "signature": {"frozen": True},
            "verified": {
                "intent": {
                    "experiment_id": f"eval-{label}",
                    "identity": {"runtime": runtime},
                }
            },
            "state": label,
        }

    def fake_outcomes(bank):
        label = bank["state"]
        exact = float("stable" in label or label.startswith("sft_only"))
        return {
            "practical": {"task-1": exact},
            "oracle": {"task-1": exact},
            "tiers": {"task-1": "easy"},
            "generated_tokens": 8,
            "candidate_count": 8,
            "completion_cap_hit_fraction": 0.0,
            "candidate_legality": exact,
            "tokens_per_practical_exact_task": 8.0 if exact else None,
            "mean_generated_tokens_per_task": 8.0,
        }

    def fake_completed(path, *args, **kwargs):
        label = Path(path).name
        raw_seed = label.rsplit("seed", 1)[-1]
        seed = int(raw_seed)
        is_sft = label.startswith("sft_seed")
        arm = "sft_only" if is_sft else label.rsplit("_seed", 1)[0]
        requested = {
            "experiment_protocol": v19.PROTOCOL_ID,
            "seed": seed,
            "seed_protocol": v19.TRUE_SEED_PROTOCOL,
        }
        if not is_sft:
            requested.update(
                {
                    "method_arm": arm,
                    "reward_strategy": v19.ARM_SPECS[arm]["reward_strategy"],
                    "trainer_seed": seed,
                }
            )
        return {
            "intent": {
                "experiment_id": f"train-{label}",
                "identity": {
                    "runtime": runtime,
                    "requested_args": requested,
                    "resolved_config": {
                        "seed": seed,
                        "family": "sft" if is_sft else "grpo",
                    },
                },
            }
        }

    monkeypatch.setattr(scorer, "load_bank", fake_bank)
    monkeypatch.setattr(scorer, "task_outcomes", fake_outcomes)
    monkeypatch.setattr(
        scorer, "_summary_for_runs", lambda runs: {"observed_states": len(runs)}
    )
    monkeypatch.setattr(scorer, "verify_completed_run", fake_completed)
    monkeypatch.setattr(
        scorer,
        "verify_v19_training_health",
        lambda *args, **kwargs: {"wall_clock_seconds": 1.0},
    )
    report = scorer.score_matrix(
        tmp_path,
        tmp_path / "outputs/v19/production",
        view="validation_confirm400",
    )
    assert set(report["arm_summaries"]) == set(v19.ARM_SPECS)
    assert len(report["evidence"]["states"]) == 16
    primary = report["contrasts"][v19.INFERENCE_CONFIG["primary_contrast"]]
    assert primary["estimate"] == 1.0
    assert primary["positive_claim_criteria_met"] is False


def test_environment_lock_rejects_runtime_drift(monkeypatch, tmp_path):
    manifest = tmp_path / v19.PROTOCOL_MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n")
    runtime = {
        "hardware": {"cuda_available": True, "cuda_devices": ["A10G"]},
        "packages": {"torch": "x"},
    }
    monkeypatch.setattr(v19, "runtime_record", lambda: runtime)
    monkeypatch.setattr(v19, "require_eligible_protocol", lambda _root: None)
    lock = v19.capture_environment_lock(tmp_path)
    v19.validate_environment_lock(lock, tmp_path)
    monkeypatch.setattr(v19, "runtime_record", lambda: {**runtime, "packages": {"torch": "y"}})
    with pytest.raises(v19.V19ProtocolError, match="does not exactly match"):
        v19.validate_environment_lock(lock, tmp_path)
