import importlib.util
from pathlib import Path

from rtw_llm.seed_protocol import TRUE_SEED_PROTOCOL


def load_grpo_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "02_grpo_train.py"
    spec = importlib.util.spec_from_file_location("grpo_train", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_init_adapter_continues_lora_without_fresh_peft_config():
    # v0.13 A1: with --init_adapter_path the SFT LoRA is CONTINUED, and no fresh
    # peft_config is attached (else TRL stacks a second adapter — capacity confound).
    mod = load_grpo_module()
    plan = mod.plan_model_init("outputs/checkpoints/sft_warmup", use_lora=True)
    assert plan["mode"] == "continue_adapter"
    assert plan["adapter_path"] == "outputs/checkpoints/sft_warmup"
    assert plan["use_peft_config"] is False


def test_no_adapter_uses_fresh_lora_baseline_path():
    mod = load_grpo_module()
    plan = mod.plan_model_init(None, use_lora=True)
    assert plan["mode"] == "fresh_lora"
    assert plan["adapter_path"] is None
    assert plan["use_peft_config"] is True


def test_adapter_path_overrides_use_lora_flag():
    # Even with use_lora True (the default), an adapter path must not add a
    # second fresh LoRA on top of the continued one.
    mod = load_grpo_module()
    plan = mod.plan_model_init("some/adapter", use_lora=True)
    assert plan["use_peft_config"] is False


def test_no_lora_no_adapter_is_full_finetune():
    mod = load_grpo_module()
    plan = mod.plan_model_init(None, use_lora=False)
    assert plan["mode"] == "full_finetune"
    assert plan["use_peft_config"] is False


def test_legacy_seed_protocol_preserves_archived_grpo_seed():
    mod = load_grpo_module()
    plan = mod.resolve_grpo_seed_plan(
        teacher_seed=2,
        trainer_seed=42,
        protocol_id=mod.LEGACY_SEED_PROTOCOL,
    )
    assert plan == {
        "protocol_id": "countdown-legacy-v1",
        "teacher_seed": 2,
        "trainer_seed": 42,
        "curriculum_seed": 42,
    }


def test_true_seed_protocol_routes_one_seed_to_teacher_trainer_and_curriculum():
    mod = load_grpo_module()
    plan = mod.resolve_grpo_seed_plan(
        teacher_seed=2,
        trainer_seed=2,
        protocol_id=TRUE_SEED_PROTOCOL,
    )
    assert plan == {
        "protocol_id": "countdown-true-seeds-v2",
        "teacher_seed": 2,
        "trainer_seed": 2,
        "curriculum_seed": 2,
    }


def test_seed_protocol_rejects_mislabeled_combinations():
    mod = load_grpo_module()
    for teacher_seed, trainer_seed, protocol_id in [
        (1, 1, mod.LEGACY_SEED_PROTOCOL),
        (1, 42, TRUE_SEED_PROTOCOL),
    ]:
        try:
            mod.resolve_grpo_seed_plan(teacher_seed, trainer_seed, protocol_id)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Expected protocol validation failure for {protocol_id}: "
                f"teacher={teacher_seed}, trainer={trainer_seed}"
            )


def test_seed_protocol_rejects_negative_seed():
    mod = load_grpo_module()
    try:
        mod.resolve_grpo_seed_plan(-1, 42, mod.LEGACY_SEED_PROTOCOL)
    except ValueError:
        pass
    else:
        raise AssertionError("Negative seeds must fail before model loading")


def test_true_seed_protocol_preseeds_before_model_initialization():
    mod = load_grpo_module()
    calls = []
    plan = mod.resolve_grpo_seed_plan(7, 7, TRUE_SEED_PROTOCOL)
    applied = mod.apply_pre_model_seed(plan, seed_setter=calls.append)
    assert applied is True
    assert calls == [7]


def test_legacy_seed_protocol_preserves_uncontrolled_pretrainer_rng_behavior():
    mod = load_grpo_module()
    calls = []
    plan = mod.resolve_grpo_seed_plan(2, 42, mod.LEGACY_SEED_PROTOCOL)
    applied = mod.apply_pre_model_seed(plan, seed_setter=calls.append)
    assert applied is False
    assert calls == []


def test_requested_trainer_seed_survives_actual_grpo_config(tmp_path):
    from trl import GRPOConfig

    mod = load_grpo_module()
    plan = mod.resolve_grpo_seed_plan(9, 9, TRUE_SEED_PROTOCOL)
    config = GRPOConfig(output_dir=str(tmp_path), seed=plan["trainer_seed"], report_to="none")
    assert config.seed == 9
