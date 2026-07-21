from rtw_llm.seed_protocol import (
    LEGACY_SEED_PROTOCOL,
    TRUE_SEED_PROTOCOL,
    apply_pre_model_seed,
    resolve_sft_seed_plan,
)


def test_sft_legacy_preserves_pretrainer_rng_behavior():
    calls = []
    plan = resolve_sft_seed_plan(2, LEGACY_SEED_PROTOCOL)
    assert apply_pre_model_seed(plan, calls.append) is False
    assert calls == []


def test_sft_corrected_protocol_preseeds_fresh_lora():
    calls = []
    plan = resolve_sft_seed_plan(2, TRUE_SEED_PROTOCOL)
    assert apply_pre_model_seed(plan, calls.append) is True
    assert calls == [2]


def test_sft_seed_plan_rejects_negative_seed():
    try:
        resolve_sft_seed_plan(-1, TRUE_SEED_PROTOCOL)
    except ValueError:
        pass
    else:
        raise AssertionError("Negative SFT seed must fail before model loading")
