from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name_or_path: str
    revision: str | None = None
    local_files_only: bool = True
    local_snapshot_path: str | None = None
    dtype: str = "bfloat16"
    device: str = "cuda:0"

    @property
    def resolved_name_or_path(self) -> str:
        if self.local_files_only and self.local_snapshot_path:
            snapshot = Path(self.local_snapshot_path)
            if not snapshot.is_dir():
                raise FileNotFoundError(f"Configured model snapshot does not exist: {snapshot}")
            return str(snapshot)
        return self.name_or_path


@dataclass(frozen=True)
class PromptConfig:
    system_prompt: str
    seed_instruction: str
    label_names: dict[str, str]
    enable_thinking: bool = False


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str = "fancyzhx/ag_news"
    seed: int = 20261002
    train_size: int = 256
    validation_size: int = 512


@dataclass(frozen=True)
class PilotConfig:
    model: ModelConfig
    prompt: PromptConfig
    data: DataConfig
    manual_instructions: tuple[str, ...]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PilotConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle)

        model = ModelConfig(**raw["model"])
        prompt_raw = dict(raw["prompt"])
        prompt_raw["label_names"] = {
            str(key): str(value) for key, value in prompt_raw["label_names"].items()
        }
        prompt = PromptConfig(**prompt_raw)
        data = DataConfig(**raw.get("data", {}))
        manual = tuple(str(item) for item in raw.get("manual_instructions", ()))

        expected_labels = ("A", "B", "C", "D")
        if tuple(prompt.label_names) != expected_labels:
            raise ValueError(
                "prompt.label_names must preserve the order A, B, C, D; "
                f"received {tuple(prompt.label_names)}"
            )
        if len(manual) != 3:
            raise ValueError("Exactly three manual alternative instructions are required.")
        return cls(model=model, prompt=prompt, data=data, manual_instructions=manual)
