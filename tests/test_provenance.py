import json

import pytest

from rtw_llm.provenance import (
    ProvenanceError,
    build_run_identity,
    canonical_json,
    content_sha256,
    file_record,
    make_intent,
    verify_completed_run,
    verify_intent,
    write_intent,
    write_result,
)


def identity():
    return {
        "schema_version": "rtw-run-manifest-v1",
        "run_kind": "test",
        "git": {"commit": "a" * 40, "dirty": False, "status_sha256": None},
        "requested_args": {"seed": 2},
        "resolved_config": {"beta": 0.0},
        "seed_roles": {"trainer_seed": 2},
        "inputs": {"train": {"sha256": "b" * 64, "size": 3, "line_count": 1}},
        "model": {"name": "model", "revision": "c" * 40},
        "runtime": {"python": "3.11"},
    }


def test_canonical_hash_is_key_order_invariant_and_value_sensitive():
    left = {"b": [2, 1], "a": 1}
    right = {"a": 1, "b": [2, 1]}
    reordered_values = {"a": 1, "b": [1, 2]}
    assert canonical_json(left) == canonical_json(right)
    assert content_sha256(left) == content_sha256(right)
    assert content_sha256(left) != content_sha256(reordered_values)


def test_file_record_is_byte_and_order_sensitive(tmp_path):
    path = tmp_path / "rows.txt"
    path.write_text("a\nb\n")
    first = file_record(path)
    path.write_text("b\na\n")
    second = file_record(path)
    assert first["size"] == second["size"]
    assert first["line_count"] == second["line_count"] == 2
    assert first["sha256"] != second["sha256"]


def test_intent_is_atomic_write_once_and_incomplete_dir_is_not_reusable(tmp_path):
    out = tmp_path / "run"
    manifest = write_intent(out, identity())
    assert verify_intent(out / "run_intent.json") == manifest
    with pytest.raises(ProvenanceError, match="not reusable"):
        write_intent(out, identity())


def test_strict_intent_refuses_preexisting_owned_artifacts(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "metrics.json").write_text("{}")
    with pytest.raises(ProvenanceError, match="must be empty"):
        write_intent(out, identity())


def test_expected_identity_mismatch_fails(tmp_path):
    out = tmp_path / "run"
    write_intent(out, identity())
    changed = identity()
    changed["seed_roles"] = {"trainer_seed": 3}
    with pytest.raises(ProvenanceError, match="does not match"):
        verify_intent(out / "run_intent.json", changed)


def test_result_links_intent_and_detects_artifact_tamper(tmp_path):
    out = tmp_path / "run"
    intent = write_intent(out, identity())
    artifact = out / "metrics.json"
    artifact.write_text('{"exact": 1}\n')
    result = write_result(out, artifact_paths={"metrics": artifact})
    assert result["intent_manifest_sha256"] == intent["manifest_sha256"]
    verify_completed_run(out, identity())
    artifact.write_text('{"exact": 0}\n')
    with pytest.raises(ProvenanceError, match="Artifact digest mismatch"):
        verify_completed_run(out, identity())


def test_manifest_digest_tamper_is_detected(tmp_path):
    out = tmp_path / "run"
    write_intent(out, identity())
    path = out / "run_intent.json"
    payload = json.loads(path.read_text())
    payload["identity"]["run_kind"] = "tampered"
    path.write_text(json.dumps(payload))
    with pytest.raises(ProvenanceError, match="digest mismatch"):
        verify_intent(path)


def test_make_intent_excludes_no_hidden_volatility():
    first = make_intent(identity())
    second = make_intent(identity())
    assert first == second


def test_build_identity_rejects_dirty_tree(monkeypatch, tmp_path):
    import rtw_llm.provenance as provenance

    data = tmp_path / "data.jsonl"
    data.write_text("{}\n")
    monkeypatch.setattr(
        provenance,
        "git_record",
        lambda _: {"commit": "a" * 40, "dirty": True, "status_sha256": "b" * 64},
    )
    with pytest.raises(ProvenanceError, match="clean Git worktree"):
        build_run_identity(
            run_kind="sft",
            requested_args={"output_dir": "one", "seed": 1},
            resolved_config={"output_dir": "one", "run_name": "one", "seed": 1},
            seed_roles={"trainer_seed": 1},
            input_files={"train": data},
            model_name="remote/model",
            model_revision="c" * 40,
            repo_root=tmp_path,
        )


def test_build_identity_excludes_storage_location_and_requires_remote_revision(
    monkeypatch, tmp_path
):
    import rtw_llm.provenance as provenance

    data = tmp_path / "data.jsonl"
    data.write_text("{}\n")
    monkeypatch.setattr(
        provenance,
        "git_record",
        lambda _: {"commit": "a" * 40, "dirty": False, "status_sha256": None},
    )
    monkeypatch.setattr(provenance, "runtime_record", lambda: {"python": "test"})
    common = {
        "run_kind": "sft",
        "seed_roles": {"trainer_seed": 1},
        "input_files": {"train": data},
        "model_name": "remote/model",
        "model_revision": "c" * 40,
        "repo_root": tmp_path,
    }
    first = build_run_identity(
        requested_args={"output_dir": "one", "seed": 1},
        resolved_config={"output_dir": "one", "run_name": "one", "seed": 1},
        **common,
    )
    second = build_run_identity(
        requested_args={"output_dir": "two", "seed": 1},
        resolved_config={"output_dir": "two", "run_name": "two", "seed": 1},
        **common,
    )
    assert first == second
    with pytest.raises(ProvenanceError, match="model_revision"):
        build_run_identity(
            requested_args={"seed": 1},
            resolved_config={"seed": 1},
            **{**common, "model_revision": None},
        )
    with pytest.raises(ProvenanceError, match="40-hex"):
        build_run_identity(
            requested_args={"seed": 1},
            resolved_config={"seed": 1},
            **{**common, "model_revision": "main"},
        )


def test_adapter_identity_requires_and_hashes_weight_payload(tmp_path):
    from rtw_llm.provenance import adapter_record

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    with pytest.raises(ProvenanceError, match="weight payload"):
        adapter_record(adapter)
    shard = adapter / "model-00001-of-00002.safetensors"
    shard.write_bytes(b"first")
    first = adapter_record(adapter)["tree_sha256"]
    shard.write_bytes(b"second")
    second = adapter_record(adapter)["tree_sha256"]
    assert first != second


def test_completed_run_enforces_required_roles(tmp_path):
    out = tmp_path / "run"
    write_intent(out, identity())
    metrics = out / "metrics.json"
    metrics.write_text("{}\n")
    write_result(out, artifact_paths={"metrics": metrics})
    with pytest.raises(ProvenanceError, match="missing required artifacts"):
        verify_completed_run(out, required_artifact_roles={"metrics", "candidates"})


def test_content_addressed_identity_is_invariant_to_local_storage_paths(monkeypatch, tmp_path):
    import rtw_llm.provenance as provenance

    monkeypatch.setattr(
        provenance,
        "git_record",
        lambda _: {"commit": "a" * 40, "dirty": False, "status_sha256": None},
    )
    monkeypatch.setattr(provenance, "runtime_record", lambda: {"python": "test"})

    copies = []
    for suffix in ("one", "two"):
        root = tmp_path / suffix
        root.mkdir()
        data = root / "train.jsonl"
        data.write_text("{}\n")
        model = root / "model"
        model.mkdir()
        (model / "config.json").write_text("{}")
        (model / "model.safetensors").write_bytes(b"same-model")
        adapter = root / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}")
        (adapter / "adapter_model.safetensors").write_bytes(b"same-adapter")
        copies.append((data, model, adapter))

    identities = []
    for data, model, adapter in copies:
        identities.append(
            build_run_identity(
                run_kind="grpo",
                requested_args={
                    "train_path": data,
                    "model_name": model,
                    "init_adapter_path": adapter,
                    "seed": 1,
                },
                resolved_config={
                    "seed": 1,
                    "model_name": str(model),
                    "adapter_path": str(adapter),
                    "data_path": str(data),
                },
                seed_roles={"trainer_seed": 1},
                input_files={"train": data},
                model_name=str(model),
                adapter_path=adapter,
                repo_root=tmp_path,
            )
        )
    assert identities[0] == identities[1]
