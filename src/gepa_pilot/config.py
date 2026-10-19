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
    # "letter": answer with the label letter (one token; the original setup). "name": answer with the label
    # name itself (usually several tokens; scored by teacher-forced sequence likelihood).
    answer_style: str = "letter"


@dataclass(frozen=True)
class TaskConfig:
    """Task wording and dataset fields. Defaults reproduce the AG News pilot exactly."""

    name: str = "AG News"
    dataset_config: str | None = None
    # One dataset field, or several (e.g. ["premise", "hypothesis"]); several fields are written into the
    # input block one per line as "Heading: value", with headings from field_headings (default: field names).
    text_field: str | tuple[str, ...] = "text"
    field_headings: tuple[str, ...] = ()
    label_field: str = "label"
    # Multi-label tasks, reduced to binary relevance: each (text, candidate class) pair is one example and
    # the binary labels are prompt.label_names (A = applies, B = does not apply). label_field then holds a
    # list of class ids; class_ids picks and orders the classes used, class_names names them. Optimization
    # splits pair every text with its true classes plus as many sampled false ones (balanced); report sets
    # pair every text with every class. multi_label_share = fraction of texts with two or more classes.
    multi_label: bool = False
    class_ids: tuple[int, ...] = ()
    class_names: tuple[str, ...] = ()
    candidate_heading: str = "Candidate label"
    multi_label_share: float = 0.5
    # Heading of the input section in the classification prompt.
    input_heading: str = "Article"
    # Phrases used by the reflection template and the per-example feedback.
    input_noun: str = "news article"
    input_short: str = "article"
    focus: str = "the article's central topic"


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
    task: TaskConfig = TaskConfig()

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
        if prompt.answer_style not in ("letter", "name"):
            raise ValueError(f"prompt.answer_style must be 'letter' or 'name', got {prompt.answer_style!r}")
        data = DataConfig(**raw.get("data", {}))
        task_raw = dict(raw.get("task", {}))
        for key in ("text_field", "field_headings", "class_ids", "class_names"):
            if isinstance(task_raw.get(key), list):
                task_raw[key] = tuple(str(item) for item in task_raw[key])
        if "class_ids" in task_raw:
            task_raw["class_ids"] = tuple(int(item) for item in task_raw["class_ids"])
        task = TaskConfig(**task_raw)
        if task.multi_label and (len(task.class_ids) != len(task.class_names) or len(prompt_raw["label_names"]) != 2):
            raise ValueError("Multi-label tasks need class_ids and class_names of equal length and two binary labels.")
        if isinstance(task.text_field, tuple) and task.field_headings and len(task.field_headings) != len(task.text_field):
            raise ValueError("task.field_headings must have one heading per text field.")
        manual = tuple(str(item) for item in raw.get("manual_instructions", ()))

        expected_labels = tuple(chr(ord("A") + index) for index in range(len(prompt.label_names)))
        if not 2 <= len(expected_labels) <= 26 or tuple(prompt.label_names) != expected_labels:
            raise ValueError(
                "prompt.label_names must be consecutive capital letters starting at A; "
                f"received {tuple(prompt.label_names)}"
            )
        if len(manual) != 3:
            raise ValueError("Exactly three manual alternative instructions are required.")
        return cls(model=model, prompt=prompt, data=data, manual_instructions=manual, task=task)
