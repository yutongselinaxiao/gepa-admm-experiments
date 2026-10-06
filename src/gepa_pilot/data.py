from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


LABEL_TO_CLASS_ID = {chr(ord("A") + index): index for index in range(26)}
CLASS_ID_TO_LABEL = {value: key for key, value in LABEL_TO_CLASS_ID.items()}


@dataclass(frozen=True)
class NewsExample:
    example_id: str
    article: str
    gold_label: str

    def __post_init__(self) -> None:
        if self.gold_label not in LABEL_TO_CLASS_ID:
            raise ValueError(f"Unsupported gold label: {self.gold_label!r}")
        if not self.article.strip():
            raise ValueError("Article text must not be empty.")


def load_examples_json(path: str | Path) -> list[NewsExample]:
    with Path(path).open("r", encoding="utf-8") as handle:
        rows: list[dict[str, Any]] = json.load(handle)
    return [NewsExample(**row) for row in rows]


def stratified_split_indices(
    labels: Sequence[int],
    train_size: int,
    validation_size: int,
    seed: int,
    num_classes: int = 4,
) -> tuple[list[int], list[int]]:
    if train_size % num_classes != 0 or validation_size % num_classes != 0:
        raise ValueError(f"Stratified sample sizes must be divisible by {num_classes}.")
    train_per_class = train_size // num_classes
    validation_per_class = validation_size // num_classes
    rng = random.Random(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for class_id in range(num_classes):
        candidates = [index for index, label in enumerate(labels) if int(label) == class_id]
        rng.shuffle(candidates)
        required = train_per_class + validation_per_class
        if len(candidates) < required:
            raise ValueError(f"Not enough examples for class {class_id}.")
        train_indices.extend(candidates[:train_per_class])
        validation_indices.extend(candidates[train_per_class:required])
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices


def load_ag_news_development_splits(
    dataset_name: str,
    train_size: int,
    validation_size: int,
    seed: int,
    *,
    num_classes: int = 4,
    dataset_config: str | None = None,
    text_field: str = "text",
    label_field: str = "label",
) -> tuple[list[NewsExample], list[NewsExample]]:
    """Load only the official training split and create disjoint stratified splits.

    Despite the name (kept for existing callers), this works for any single-label
    classification dataset whose integer labels 0..num_classes-1 map to letters A, B, ...
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, dataset_config, split="train")
    labels = list(dataset[label_field])
    train_indices, validation_indices = stratified_split_indices(
        labels, train_size, validation_size, seed, num_classes=num_classes
    )

    def convert(indices: Iterable[int], split_name: str) -> list[NewsExample]:
        return [
            NewsExample(
                example_id=f"official-train:{index}:{split_name}",
                article=str(dataset[index][text_field]),
                gold_label=CLASS_ID_TO_LABEL[int(dataset[index][label_field])],
            )
            for index in indices
        ]

    return convert(train_indices, "train"), convert(validation_indices, "validation")


def load_ag_news_test_split(dataset_name: str, *, allow_test: bool = False) -> list[NewsExample]:
    """Load the official test split only after an explicit final-evaluation opt-in."""
    if not allow_test:
        raise PermissionError(
            "The official test split is locked. Pass allow_test=True only for final evaluation."
        )
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split="test")
    return [
        NewsExample(
            example_id=f"official-test:{index}",
            article=str(row["text"]),
            gold_label=CLASS_ID_TO_LABEL[int(row["label"])],
        )
        for index, row in enumerate(dataset)
    ]


def split_selection_report(
    examples: Sequence[NewsExample], seed: int
) -> tuple[list[NewsExample], list[NewsExample]]:
    """Class-stratified half split: GEPA selects on the first half, results are reported on the second.

    Reporting on examples that no selection step saw removes the best-of-k optimism that
    arises when the same set both chooses the winning candidate and scores it.
    """
    by_label: dict[str, list[NewsExample]] = {}
    for example in sorted(examples, key=lambda item: item.example_id):
        by_label.setdefault(example.gold_label, []).append(example)
    rng = random.Random(seed)
    selection_ids: set[str] = set()
    for label in sorted(by_label):
        group = by_label[label]
        if len(group) % 2:
            raise ValueError(f"Class {label} has an odd number of examples; cannot halve it.")
        rng.shuffle(group)
        selection_ids.update(example.example_id for example in group[: len(group) // 2])
    selection = [example for example in examples if example.example_id in selection_ids]
    report = [example for example in examples if example.example_id not in selection_ids]
    return selection, report
