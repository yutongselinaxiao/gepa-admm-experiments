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


def format_input(row: Any, text_field: str | Sequence[str], field_headings: Sequence[str] = ()) -> str:
    """The model input for one dataset row: one field as is, or several fields as "Heading: value" lines."""
    if isinstance(text_field, str):
        return str(row[text_field])
    headings = list(field_headings) or list(text_field)
    return "\n".join(f"{heading}: {row[field]}" for heading, field in zip(headings, text_field))


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
    text_field: str | Sequence[str] = "text",
    label_field: str = "label",
    field_headings: Sequence[str] = (),
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
                article=format_input(dataset[index], text_field, field_headings),
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


def pair_id(index: int, split: str, class_position: int) -> str:
    """Example id of one (text, candidate class) pair in a multi-label task."""
    return f"official-train:{index}:{split}:{class_position}"


def text_index(example_id: str) -> int:
    return int(example_id.split(":")[1])


def multilabel_pairs(row: Any, index: int, split: str, task: Any, positions: Sequence[int]) -> list[NewsExample]:
    """Binary-relevance examples for one text: one per candidate class position (A = applies, B = not)."""
    gold = {task.class_ids.index(c) for c in row[task.label_field] if c in task.class_ids}
    text = format_input(row, task.text_field, task.field_headings)
    return [
        NewsExample(
            example_id=pair_id(index, split, k),
            article=f"{text}\n{task.candidate_heading}: {task.class_names[k]}",
            gold_label="A" if k in gold else "B",
        )
        for k in positions
    ]


def load_multilabel_splits(
    dataset_name: str, dataset_config: str | None, task: Any, train_texts: int, validation_texts: int, seed: int,
) -> tuple[list[NewsExample], list[NewsExample]]:
    """Multi-label development splits as balanced binary-relevance pairs (see TaskConfig.multi_label).

    Texts with at least one of task.class_ids are sampled (a multi_label_share of them with two or more);
    each text gets its true classes plus an equal number of randomly drawn false ones."""
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, dataset_config, split="train")
    wanted = set(task.class_ids)
    counts = [len(set(labels) & wanted) for labels in dataset[task.label_field]]
    rng = random.Random(seed)
    multi = [i for i, n in enumerate(counts) if n >= 2]
    single = [i for i, n in enumerate(counts) if n == 1]
    rng.shuffle(multi)
    rng.shuffle(single)
    total = train_texts + validation_texts
    n_multi = round(total * task.multi_label_share)
    chosen = multi[:n_multi] + single[: total - n_multi]
    rng.shuffle(chosen)

    def expand(indices: Sequence[int], split: str) -> list[NewsExample]:
        pairs: list[NewsExample] = []
        for index in indices:
            row = dataset[index]
            positives = [task.class_ids.index(c) for c in row[task.label_field] if c in wanted]
            negatives = [k for k in range(len(task.class_ids)) if k not in positives]
            rng.shuffle(negatives)
            pairs += multilabel_pairs(row, index, split, task, sorted(positives + negatives[: len(positives)]))
        return pairs

    return expand(chosen[:train_texts], "train"), expand(chosen[train_texts:], "validation")


def split_selection_report(
    examples: Sequence[NewsExample], seed: int
) -> tuple[list[NewsExample], list[NewsExample]]:
    """Class-stratified half split: GEPA selects on the first half, results are reported on the second.

    Reporting on examples that no selection step saw removes the best-of-k optimism that
    arises when the same set both chooses the winning candidate and scores it.
    """
    if all(example.example_id.count(":") == 3 for example in examples):  # multi-label pairs: split by text
        texts = sorted({text_index(example.example_id) for example in examples})
        random.Random(seed).shuffle(texts)
        chosen = set(texts[: len(texts) // 2])
        selection = [example for example in examples if text_index(example.example_id) in chosen]
        return selection, [example for example in examples if text_index(example.example_id) not in chosen]
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
