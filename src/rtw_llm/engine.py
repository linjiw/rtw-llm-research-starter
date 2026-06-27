"""Small inference engine abstraction for evaluation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenerationConfigLite:
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False


class HFEngine:
    def __init__(self, model_name: str, adapter_path: str | None = None, device_map: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map,
            trust_remote_code=True,
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def generate(self, prompts: list[str], config: GenerationConfigLite) -> list[str]:
        import torch

        outputs: list[str] = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=config.do_sample,
                    temperature=config.temperature if config.do_sample else None,
                    top_p=config.top_p if config.do_sample else None,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = gen[0][inputs["input_ids"].shape[1] :]
            outputs.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        return outputs


class VLLMEngine:
    def __init__(self, model_name: str):
        from vllm import LLM

        self.llm = LLM(model=model_name)

    def generate(self, prompts: list[str], config: GenerationConfigLite) -> list[str]:
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_new_tokens,
        )
        outputs = self.llm.generate(prompts, params)
        return [out.outputs[0].text for out in outputs]
