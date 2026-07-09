import importlib.util
from pathlib import Path


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
