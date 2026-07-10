import hashlib
import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

import rtw_llm.data_access as data_access
import rtw_llm.countdown_v2_audit as v2_audit
from rtw_llm.countdown import (
    difficulty_spec,
    random_solvable_task,
    random_solvable_task_legacy_v1,
    verify_expression,
)
from rtw_llm.countdown_v2 import (
    FINAL_TEST_POLICY,
    SPLIT_ORDER,
    build_artifact_bytes,
    build_manifest,
    build_records,
    canonical_json_bytes,
    loose_key,
    write_dataset_atomic,
)
from rtw_llm.countdown_v2_audit import PINNED_LEGACY_HASHES, audit_countdown_v2
from rtw_llm.data_access import DataAccessError, assert_countdown_data_access
from rtw_llm.provenance import file_record


SMALL_QUOTAS = {
    "train": {"easy": 2, "medium": 2, "hard": 2},
    "validation": {"easy": 1, "medium": 1, "hard": 1},
    "test_in_dist": {"easy": 1, "medium": 1, "hard": 1},
    "final_test_in_dist": {"easy": 1, "medium": 1, "hard": 1},
    "test_ood_long": {"ood_long": 2},
    "test_ood_division": {"ood_division": 2},
}


def load_script00():
    path = Path(__file__).resolve().parents[1] / "scripts/00_generate_countdown_dataset.py"
    spec = importlib.util.spec_from_file_location("legacy_generator_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def small_build(seed=101):
    artifacts, stats, records = build_artifact_bytes(
        base_seed=seed,
        split_quotas=SMALL_QUOTAS,
        max_proposals=10_000,
    )
    manifest = build_manifest(
        source_commit="a" * 40,
        source_records={},
        artifacts=artifacts,
        stats=stats,
        records=records,
        base_seed=seed,
        split_quotas=SMALL_QUOTAS,
    )
    return artifacts, stats, records, manifest


def publish_small_dataset(tmp_path):
    artifacts, _, records, manifest = small_build()
    output = tmp_path / "data/countdown_v2"
    write_dataset_atomic(
        output,
        artifacts=artifacts,
        manifest=manifest,
        legacy_dir=tmp_path / "data/countdown",
    )
    return output, records, manifest


def test_legacy_leftover_defect_is_preserved_but_corrected_generator_rejects_it():
    legacy = random_solvable_task_legacy_v1(
        random.Random(136), max_attempts=1, **difficulty_spec("hard")
    )
    assert legacy["numbers"] == [14]
    with pytest.raises(RuntimeError):
        random_solvable_task(random.Random(136), max_attempts=1, **difficulty_spec("hard"))


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard", "ood_long", "ood_division"])
def test_corrected_generator_always_uses_requested_operand_count(difficulty):
    spec = difficulty_spec(difficulty)
    rng = random.Random(700 + spec["n_numbers"])
    for _ in range(100):
        task = random_solvable_task(rng, **spec)
        assert len(task["numbers"]) == spec["n_numbers"]
        assert verify_expression(
            task["solution"], task["numbers"], task["target"], task["allowed_ops"]
        ).correct


def test_legacy_script_replays_all_committed_non_prompt_fields_and_hashes():
    root = Path(__file__).resolve().parents[1]
    module = load_script00()
    recipe = {
        "train": (["easy", "medium", "hard"], 42),
        "validation": (["easy", "medium", "hard"], 43),
        "test_in_dist": (["easy", "medium", "hard"], 44),
        "test_ood_long": (["ood_long"], 45),
        "test_ood_division": (["ood_division"], 46),
    }
    for split, (difficulties, seed) in recipe.items():
        path = root / f"data/countdown/{split}.jsonl"
        stored = [json.loads(line) for line in path.read_text().splitlines()]
        replayed = module.build_records(len(stored), split, difficulties, seed)
        assert replayed == stored
        replayed_bytes = "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in replayed
        ).encode()
        assert replayed_bytes == path.read_bytes()
    for relative, expected_hash in PINNED_LEGACY_HASHES.items():
        assert file_record(root / relative)["sha256"] == expected_hash


def test_legacy_replay_script_refuses_frozen_evidence_paths(tmp_path):
    module = load_script00()
    frozen = tmp_path / "data/countdown"
    for unsafe in (Path("data/countdown"), Path("data/countdown/subdir"), frozen):
        with pytest.raises(ValueError, match="frozen legacy evidence"):
            module.assert_safe_legacy_replay_output(unsafe, repo_root=tmp_path)
    safe = module.assert_safe_legacy_replay_output(
        Path("outputs/legacy_replay/countdown"), repo_root=tmp_path
    )
    assert safe == (tmp_path / "outputs/legacy_replay/countdown").resolve()


def test_small_dataset_is_byte_deterministic_and_globally_loose_disjoint():
    first_artifacts, first_stats, first_records, first_manifest = small_build()
    second_artifacts, second_stats, second_records, second_manifest = small_build()
    assert first_artifacts == second_artifacts
    assert first_stats == second_stats
    assert first_records == second_records
    assert first_manifest == second_manifest
    keys = []
    for split in SPLIT_ORDER:
        keys.extend(loose_key(row) for row in first_records[split])
    assert len(keys) == len(set(keys))
    assert first_stats["global_loose_keys"] == len(keys)
    assert "manifest.json" not in first_manifest["artifacts"]
    assert first_manifest["artifacts_exclude_manifest"] is True


def test_easy_capacity_and_proposal_budget_fail_closed():
    impossible = {split: dict(values) for split, values in SMALL_QUOTAS.items()}
    impossible["train"]["easy"] = 1_265
    for split in ("validation", "test_in_dist", "final_test_in_dist"):
        impossible[split]["easy"] = 0
    with pytest.raises(ValueError, match="exceeds finite loose-key capacity"):
        build_records(split_quotas=impossible)
    with pytest.raises(RuntimeError, match="proposal budget"):
        build_records(split_quotas=SMALL_QUOTAS, max_proposals=0)


def test_atomic_publish_refuses_legacy_and_any_existing_target(tmp_path):
    artifacts, _, _, manifest = small_build()
    legacy = tmp_path / "data/countdown"
    with pytest.raises(ValueError, match="legacy"):
        write_dataset_atomic(
            legacy,
            artifacts=artifacts,
            manifest=manifest,
            legacy_dir=legacy,
        )
    existing = tmp_path / "data/countdown_v2"
    existing.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="pre-existing"):
        write_dataset_atomic(
            existing,
            artifacts=artifacts,
            manifest=manifest,
            legacy_dir=legacy,
        )


def test_atomic_publish_writes_only_declared_artifacts_and_manifest(tmp_path):
    output, _, manifest = publish_small_dataset(tmp_path)
    assert json.loads((output / "manifest.json").read_text()) == manifest
    assert (output / "train.jsonl").exists()
    assert (output / "task_ids/final_test_in_dist.txt").exists()
    assert not any(path.name.startswith(f".{output.name}.tmp-") for path in output.parent.iterdir())


def test_data_guard_rejects_final_subset_reserialization_and_changed_id(tmp_path):
    _, records, _ = publish_small_dataset(tmp_path)
    final_row = records["final_test_in_dist"][0]
    subset = tmp_path / "subset.jsonl"
    subset.write_text(json.dumps(final_row, sort_keys=False) + "\n")
    with pytest.raises(DataAccessError, match="forbidden for training"):
        assert_countdown_data_access(
            subset,
            purpose="training",
            runner="test",
            repo_root=tmp_path,
        )
    changed_id = {**final_row, "id": "changed-id"}
    subset.write_bytes(canonical_json_bytes(changed_id))
    with pytest.raises(DataAccessError, match="semantic_keys"):
        assert_countdown_data_access(
            subset,
            purpose="training_eval",
            runner="test",
            repo_root=tmp_path,
        )
    changed_ops = {**final_row, "id": "changed-again", "allowed_ops": ["+"]}
    subset.write_bytes(canonical_json_bytes(changed_ops))
    with pytest.raises(DataAccessError, match="loose_keys"):
        assert_countdown_data_access(
            subset,
            purpose="training",
            runner="test",
            repo_root=tmp_path,
        )


def test_data_guard_rejects_final_row_appended_to_training_file(tmp_path):
    output, records, _ = publish_small_dataset(tmp_path)
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_bytes(
        (output / "train.jsonl").read_bytes()
        + canonical_json_bytes(records["final_test_in_dist"][0])
    )
    with pytest.raises(DataAccessError, match="forbidden for training"):
        assert_countdown_data_access(
            mixed,
            purpose="training",
            runner="test",
            repo_root=tmp_path,
        )


def test_data_guard_allows_nonfinal_rows_and_blocks_unreleased_final(tmp_path):
    output, _, _ = publish_small_dataset(tmp_path)
    assert_countdown_data_access(
        output / "train.jsonl",
        purpose="training",
        runner="test",
        repo_root=tmp_path,
    ) is None
    with pytest.raises(DataAccessError, match="release record"):
        assert_countdown_data_access(
            output / "final_test_in_dist.jsonl",
            purpose="model_eval",
            runner="03_eval",
            repo_root=tmp_path,
        )


def test_missing_manifest_blocks_renamed_final_subset_and_all_runner_access(tmp_path):
    output, records, _ = publish_small_dataset(tmp_path)
    copied = tmp_path / "renamed.jsonl"
    copied.write_bytes(canonical_json_bytes(records["final_test_in_dist"][0]))
    (output / "manifest.json").unlink()
    for path in (copied, output / "train.jsonl"):
        with pytest.raises(DataAccessError, match="exists without its required manifest"):
            assert_countdown_data_access(
                path,
                purpose="training",
                runner="test",
                repo_root=tmp_path,
            )


def test_full_final_release_is_bound_to_manifest_runner_and_frozen_head(monkeypatch, tmp_path):
    output, _, manifest = publish_small_dataset(tmp_path)
    frozen_commit = "b" * 40
    release = {
        "schema_version": data_access.FINAL_RELEASE_SCHEMA,
        "dataset_manifest_sha256": file_record(output / "manifest.json")["sha256"],
        "final_jsonl_sha256": manifest["final_test_protection"]["jsonl_sha256"],
        "final_ordered_ids_sha256": manifest["final_test_protection"][
            "ordered_ids_sha256"
        ],
        "final_test_policy": FINAL_TEST_POLICY,
        "frozen_commit": frozen_commit,
        "authorized_runners": ["07_best_of_n_rerank"],
        "human_approval": True,
    }
    release_path = tmp_path / "release.json"
    release_path.write_text(json.dumps(release))

    def fake_git(_root, *args):
        return frozen_commit if args[:2] == ("rev-parse", "HEAD") else ""

    monkeypatch.setattr(data_access, "_git_output", fake_git)
    assert_countdown_data_access(
        output / "final_test_in_dist.jsonl",
        purpose="model_eval",
        runner="07_best_of_n_rerank",
        release_record=release_path,
        repo_root=tmp_path,
    )
    with pytest.raises(DataAccessError, match="never final-release authorized"):
        assert_countdown_data_access(
            output / "final_test_in_dist.jsonl",
            purpose="model_eval",
            runner="03_eval",
            release_record=release_path,
            repo_root=tmp_path,
        )
    release["dataset_manifest_sha256"] = hashlib.sha256(b"wrong").hexdigest()
    release_path.write_text(json.dumps(release))
    with pytest.raises(DataAccessError, match="release record mismatch"):
        assert_countdown_data_access(
            output / "final_test_in_dist.jsonl",
            purpose="model_eval",
            runner="07_best_of_n_rerank",
            release_record=release_path,
            repo_root=tmp_path,
        )


def test_manifest_tampering_blocks_data_access(tmp_path):
    output, _, _ = publish_small_dataset(tmp_path)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["final_test_policy"] = "tampered"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(DataAccessError, match="manifest core digest mismatch"):
        assert_countdown_data_access(
            output / "train.jsonl",
            purpose="training",
            runner="test",
            repo_root=tmp_path,
        )


def test_independent_auditor_rejects_operator_order_and_malformed_rows(monkeypatch):
    _, _, records, _ = small_build()
    monkeypatch.setattr(v2_audit, "EXPECTED_QUOTAS", SMALL_QUOTAS)
    validation = [dict(row) for row in records["validation"]]
    assert v2_audit._record_errors("validation", validation) == []
    validation[0] = {**validation[0], "allowed_ops": list(reversed(validation[0]["allowed_ops"]))}
    errors = v2_audit._record_errors("validation", validation)
    assert any("wrong ordered operator list" in error for error in errors)
    malformed = [dict(validation[0])]
    malformed[0].pop("numbers")
    assert any("missing fields" in error for error in v2_audit._record_errors("validation", malformed))


@pytest.mark.parametrize("corruption", ["manifest", "jsonl", "missing_file"])
def test_repository_audit_returns_deterministic_integrity_fail_for_malformed_inputs(
    tmp_path, corruption
):
    output, _, _ = publish_small_dataset(tmp_path)
    if corruption == "manifest":
        (output / "manifest.json").write_text("{not-json\n")
    elif corruption == "jsonl":
        (output / "train.jsonl").write_text("{not-json\n")
    else:
        (output / "validation.jsonl").unlink()
    first = audit_countdown_v2(tmp_path, replay=False)
    second = audit_countdown_v2(tmp_path, replay=False)
    assert first == second
    assert first["verdict"]["status"] == "INTEGRITY_FAIL"
    assert first["verdict"]["eligible_for_corrected_v2"] is False


def test_audit_cli_exits_two_and_writes_failure_report(tmp_path):
    output, _, _ = publish_small_dataset(tmp_path)
    (output / "manifest.json").write_text("{not-json\n")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/19_audit_countdown_v2.py"),
            "--repo_root",
            str(tmp_path),
            "--out_json",
            "docs/artifacts/failure.json",
            "--skip_replay",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    report = json.loads((tmp_path / "docs/artifacts/failure.json").read_text())
    assert report["verdict"]["status"] == "INTEGRITY_FAIL"
