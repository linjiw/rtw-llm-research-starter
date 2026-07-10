"""Small inference engine abstraction for evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Registered special tokens Qwen2.5 never generates, in preference order.
# Left-padding with a token from the EOS set is unsafe: the repetition penalty
# inherited from the model's generation_config counts pad tokens as generated
# text and suppresses termination on padded rows (ADV-1 in
# docs/THROUGHPUT_BATCHED_BESTOFN_PLAN.md).
_SAFE_PAD_CANDIDATES = ("<|fim_pad|>", "<|video_pad|>", "<|image_pad|>")

GEN_MODES = ("loop", "batched")


def resolve_eos_token_ids(model: Any, tokenizer: Any) -> set[int]:
    """Collect every token id that can terminate generation for this model."""
    ids: set[int] = set()
    for source in (getattr(model, "generation_config", None), getattr(model, "config", None)):
        eos = getattr(source, "eos_token_id", None)
        if eos is None:
            continue
        if isinstance(eos, int):
            ids.add(eos)
        else:
            ids.update(int(e) for e in eos)
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    return ids


def resolve_batch_pad_token_id(tokenizer: Any, eos_token_ids: set[int]) -> int:
    """Pick a pad token for batched generation that is outside the EOS set."""
    if tokenizer.pad_token_id is not None and int(tokenizer.pad_token_id) not in eos_token_ids:
        return int(tokenizer.pad_token_id)
    vocab = tokenizer.get_vocab()
    for token in _SAFE_PAD_CANDIDATES:
        token_id = vocab.get(token)
        if token_id is not None and int(token_id) not in eos_token_ids:
            return int(token_id)
    raise ValueError(
        "No pad token outside the EOS set is available; batched generation would "
        "bias termination via the repetition penalty. Use gen_mode='loop'."
    )


def slice_new_tokens(sequences: Any, padded_input_len: int, pad_token_id: int) -> list[Any]:
    """Per-row new tokens from a batched generate output (uniform padded input).

    Rows that hit EOS before the batch finishes are right-filled with
    pad_token_id by generate(); that fill must be stripped here — Qwen's
    <|fim_pad|> is special=False, so skip_special_tokens does NOT remove it
    at decode and it would leak into completions, corrupting verifier metrics
    and token counts for every early-terminating row.
    """
    rows = []
    for i in range(sequences.shape[0]):
        row = sequences[i][padded_input_len:]
        rows.append(row[row != pad_token_id])
    return rows


@dataclass
class GenerationConfigLite:
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False


class HFEngine:
    def __init__(
        self,
        model_name: str,
        adapter_path: str | None = None,
        model_revision: str | None = None,
        device: str = "auto",
        gen_mode: str = "loop",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if gen_mode not in GEN_MODES:
            raise ValueError(f"gen_mode must be one of {GEN_MODES}, got {gen_mode!r}")
        self.gen_mode = gen_mode

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        use_cuda = device == "cuda" and torch.cuda.is_available()
        use_mps = (
            device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        dtype = torch.bfloat16 if use_cuda else torch.float32
        model_kwargs: dict[str, object] = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
        }
        if use_cuda:
            model_kwargs["device_map"] = "auto"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            revision=model_revision,
        )
        if model_revision:
            model_kwargs["revision"] = model_revision
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        if use_mps or device == "cpu":
            self.model = self.model.to(device)
        self.device = next(self.model.parameters()).device
        self.model.eval()

        self.eos_token_ids = resolve_eos_token_ids(self.model, self.tokenizer)
        if self.gen_mode == "batched":
            self.batch_pad_token_id = resolve_batch_pad_token_id(
                self.tokenizer, self.eos_token_ids
            )
            # The tokenizer does the left-padding, so the safe pad id must be
            # installed on it — otherwise inputs are padded with the default
            # pad token (EOS for Qwen) and ADV-1 fires anyway.
            self.tokenizer.pad_token_id = self.batch_pad_token_id
            self.tokenizer.pad_token = self.tokenizer.convert_ids_to_tokens(
                self.batch_pad_token_id
            )
            self.tokenizer.padding_side = "left"

    def effective_generation_config(self) -> dict[str, Any]:
        """Sampling knobs the model's generation_config silently injects.

        Recorded in run artifacts so the frozen protocol is auditable (ADV-4).
        Only knobs our generate() calls do NOT override are reported —
        temperature/top_p come from the caller and live elsewhere in
        run_config; repeating the model-default values here would record
        numbers that were never in effect.
        """
        gen_config = getattr(self.model, "generation_config", None)
        inherited_keys = ("top_k", "repetition_penalty")
        out: dict[str, Any] = {
            f"inherited_{key}": getattr(gen_config, key, None) for key in inherited_keys
        }
        out["eos_token_id"] = sorted(self.eos_token_ids)
        out["pad_token_id_batched"] = getattr(self, "batch_pad_token_id", None)
        out["gen_mode"] = self.gen_mode
        return out

    def generate(self, prompts: list[str], config: GenerationConfigLite) -> list[str]:
        if self.gen_mode == "batched":
            return self._generate_batched(prompts, config)
        return self._generate_loop(prompts, config)

    def _generate_loop(self, prompts: list[str], config: GenerationConfigLite) -> list[str]:
        import torch

        outputs: list[str] = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
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

    def _generate_batched(self, prompts: list[str], config: GenerationConfigLite) -> list[str]:
        import torch

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        padded_input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            gen = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=config.max_new_tokens,
                do_sample=config.do_sample,
                temperature=config.temperature if config.do_sample else None,
                top_p=config.top_p if config.do_sample else None,
                pad_token_id=self.batch_pad_token_id,
            )
        return [
            self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            for new_tokens in slice_new_tokens(gen, padded_input_len, self.batch_pad_token_id)
        ]


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
