from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ReflectionCall:
    call_index: int
    prompt_token_count: int
    completion_token_count: int
    raw_output: str


class LocalQwenReflectionLM:
    """GEPA reflection callable using the already-loaded frozen Qwen model."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        device: str,
        seed: int,
        max_new_tokens: int = 192,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.calls: list[ReflectionCall] = []

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = self.tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)
        call_index = len(self.calls)
        torch.manual_seed(self.seed + call_index)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed + call_index)
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=int(self.tokenizer.pad_token_id),
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = generated[0, encoded.input_ids.shape[1] :]
        raw_output = self.tokenizer.decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        self.calls.append(
            ReflectionCall(
                call_index=call_index,
                prompt_token_count=int(encoded.input_ids.shape[1]),
                completion_token_count=int(new_ids.shape[0]),
                raw_output=raw_output,
            )
        )
        return raw_output

    @property
    def total_cost(self) -> float:
        return 0.0

    def manifest(self) -> dict[str, Any]:
        return {
            "provider": "local",
            "model_class": type(self.model).__name__,
            "seed": self.seed,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "calls": [call.__dict__ for call in self.calls],
        }
