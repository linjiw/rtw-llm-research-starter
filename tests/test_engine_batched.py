import importlib.util
import json
from pathlib import Path

import pytest

from rtw_llm.engine import GEN_MODES, resolve_batch_pad_token_id, slice_new_tokens


def load_script_module(name: str, filename: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    def __init__(self, pad_token_id, vocab):
        self.pad_token_id = pad_token_id
        self._vocab = vocab

    def get_vocab(self):
        return dict(self._vocab)


QWEN_LIKE_VOCAB = {
    "<|endoftext|>": 151643,
    "<|im_end|>": 151645,
    "<|fim_pad|>": 151662,
    "<|video_pad|>": 151656,
    "<|image_pad|>": 151655,
}
QWEN_EOS_IDS = {151645, 151643}


def test_pad_token_never_in_eos_set():
    # Qwen's default pad (<|endoftext|>) IS an EOS id; the resolver must skip it
    # or the repetition penalty biases termination on padded rows (ADV-1).
    tok = FakeTokenizer(pad_token_id=151643, vocab=QWEN_LIKE_VOCAB)
    pad_id = resolve_batch_pad_token_id(tok, QWEN_EOS_IDS)
    assert pad_id not in QWEN_EOS_IDS
    assert pad_id == 151662  # <|fim_pad|>, first safe candidate


def test_pad_token_keeps_existing_safe_pad():
    tok = FakeTokenizer(pad_token_id=42, vocab=QWEN_LIKE_VOCAB)
    assert resolve_batch_pad_token_id(tok, QWEN_EOS_IDS) == 42


def test_pad_token_errors_when_no_safe_candidate():
    tok = FakeTokenizer(pad_token_id=1, vocab={"<|fim_pad|>": 2})
    with pytest.raises(ValueError, match="repetition penalty"):
        resolve_batch_pad_token_id(tok, {1, 2})


def test_slice_new_tokens_uses_uniform_padded_length():
    import torch

    # Batch of 2, padded input length 3, then 2 generated tokens each.
    sequences = torch.tensor([[9, 1, 2, 100, 101], [1, 2, 3, 200, 201]])
    new = slice_new_tokens(sequences, padded_input_len=3, pad_token_id=9)
    assert [t.tolist() for t in new] == [[100, 101], [200, 201]]


def test_slice_new_tokens_strips_right_fill_from_early_eos_rows():
    import torch

    # Row 0 hit EOS (7) after one token; generate() right-fills it with the
    # pad id (9) until row 1 finishes. Qwen's <|fim_pad|> is special=False so
    # decode would NOT strip it — the slice must (ADV-10: early-EOS rows must
    # not leak pad text into completions or inflate token counts).
    pad = 9
    sequences = torch.tensor(
        [
            [pad, 1, 2, 100, 7, pad, pad, pad],
            [1, 2, 3, 200, 201, 202, 203, 7],
        ]
    )
    new = slice_new_tokens(sequences, padded_input_len=3, pad_token_id=pad)
    assert new[0].tolist() == [100, 7]
    assert new[1].tolist() == [200, 201, 202, 203, 7]


def test_gen_modes_are_loop_and_batched():
    assert GEN_MODES == ("loop", "batched")


# --- sampling identity / skip_if_complete hardening in 07 ---


def bestofn():
    return load_script_module("best_of_n_batched", "07_best_of_n_rerank.py")


def make_complete_artifacts(tmp_path: Path, config: dict):
    metrics = {"by_n": {"8": {}}, "n_examples": 1}
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    (tmp_path / "summary.csv").write_text("n\n")
    (tmp_path / "candidates.jsonl").write_text(
        "".join(json.dumps({"id": "t", "candidate_index": i}) + "\n" for i in range(8))
    )
    (tmp_path / "run_config.json").write_text(json.dumps(config))


BASE_CONFIG = {
    "hf_gen_mode": "loop",
    "model_name": "m",
    "adapter_path": None,
    "sampling_seed": 0,
    "temperature": 0.7,
    "top_p": 0.95,
    "max_new_tokens": 256,
    "batch_size": 8,
}


def test_is_complete_skips_when_identity_matches(tmp_path):
    mod = bestofn()
    make_complete_artifacts(tmp_path, BASE_CONFIG)
    identity = mod.sampling_identity(BASE_CONFIG)
    assert mod.is_complete(tmp_path, max_n=8, n_examples=1, requested_identity=identity)


def test_is_complete_refuses_mode_mismatch(tmp_path):
    mod = bestofn()
    make_complete_artifacts(tmp_path, BASE_CONFIG)
    requested = mod.sampling_identity({**BASE_CONFIG, "hf_gen_mode": "batched"})
    with pytest.raises(ValueError, match="sampling identity"):
        mod.is_complete(tmp_path, max_n=8, n_examples=1, requested_identity=requested)


def test_is_complete_refuses_batched_batch_size_mismatch(tmp_path):
    mod = bestofn()
    make_complete_artifacts(tmp_path, {**BASE_CONFIG, "hf_gen_mode": "batched"})
    requested = mod.sampling_identity(
        {**BASE_CONFIG, "hf_gen_mode": "batched", "batch_size": 32}
    )
    with pytest.raises(ValueError, match="sampling identity"):
        mod.is_complete(tmp_path, max_n=8, n_examples=1, requested_identity=requested)


def test_loop_identity_ignores_batch_size(tmp_path):
    mod = bestofn()
    make_complete_artifacts(tmp_path, BASE_CONFIG)
    requested = mod.sampling_identity({**BASE_CONFIG, "batch_size": 32})
    assert mod.is_complete(tmp_path, max_n=8, n_examples=1, requested_identity=requested)


def test_is_complete_refuses_prompt_field_mismatch(tmp_path):
    # Harness-shift: reusing a prompt_high bank for a prompt_mid request must NOT skip.
    mod = bestofn()
    make_complete_artifacts(tmp_path, {**BASE_CONFIG, "prompt_field": "prompt_high"})
    requested = mod.sampling_identity({**BASE_CONFIG, "prompt_field": "prompt_mid"})
    with pytest.raises(ValueError, match="sampling identity"):
        mod.is_complete(tmp_path, max_n=8, n_examples=1, requested_identity=requested)


def test_legacy_config_without_prompt_field_counts_as_prompt(tmp_path):
    mod = bestofn()
    legacy = {k: v for k, v in BASE_CONFIG.items() if k != "prompt_field"}
    make_complete_artifacts(tmp_path, legacy)
    requested = mod.sampling_identity({**BASE_CONFIG, "prompt_field": "prompt"})
    assert mod.is_complete(tmp_path, max_n=8, n_examples=1, requested_identity=requested)


def test_legacy_config_without_gen_mode_counts_as_loop(tmp_path):
    mod = bestofn()
    legacy = {k: v for k, v in BASE_CONFIG.items() if k != "hf_gen_mode"}
    make_complete_artifacts(tmp_path, legacy)
    requested = mod.sampling_identity(BASE_CONFIG)
    assert mod.is_complete(tmp_path, max_n=8, n_examples=1, requested_identity=requested)


# --- paired-overlap identity guard in 08 ---


def test_paired_overlap_refuses_mixed_generation_modes(tmp_path):
    agg = load_script_module("v09_agg_batched", "08_summarize_v09_seed_expansion.py")
    candidate = {
        "id": "t0",
        "candidate_index": 0,
        "practical_score": 1.0,
        "metrics": {"exact_correct": 1.0},
    }
    runs = []
    for method, mode in [("stable", "batched"), ("static", "loop")]:
        run_dir = tmp_path / f"{method}_run"
        run_dir.mkdir()
        (run_dir / "candidates.jsonl").write_text(json.dumps(candidate) + "\n")
        runs.append(
            {
                "run_dir": str(run_dir),
                "metrics": {"by_n": {"1": {}}},
                "config": {"method": method, "hf_gen_mode": mode, "batch_size": 8},
                "method": method,
                "training_seed": 0,
                "split": "validation",
                "candidates_path": run_dir / "candidates.jsonl",
            }
        )
    with pytest.raises(ValueError, match="Generation-identity mismatch"):
        agg.paired_overlap(runs)
