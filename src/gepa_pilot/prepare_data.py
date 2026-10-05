from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import datasets

from .config import PilotConfig
from .data import NewsExample, load_ag_news_development_splits


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_examples(path: Path, examples: Sequence[NewsExample]) -> str:
    rows = [asdict(example) for example in examples]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return _sha256(rows)


def _class_counts(examples: Sequence[NewsExample]) -> dict[str, int]:
    counts = Counter(example.gold_label for example in examples)
    return {label: counts[label] for label in ("A", "B", "C", "D")}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create locked, stratified AG News development manifests."
    )
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--output-dir", default="artifacts/data")
    args = parser.parse_args()

    config = PilotConfig.from_yaml(args.config)
    output_dir = Path(args.output_dir)
    train_path = output_dir / "train.json"
    validation_path = output_dir / "validation.json"
    manifest_path = output_dir / "data_manifest.json"

    train, validation = load_ag_news_development_splits(
        dataset_name=config.data.dataset_name,
        train_size=config.data.train_size,
        validation_size=config.data.validation_size,
        seed=config.data.seed,
    )
    if set(item.example_id for item in train) & set(item.example_id for item in validation):
        raise AssertionError("Training and validation IDs overlap.")

    train_sha256 = _write_examples(train_path, train)
    validation_sha256 = _write_examples(validation_path, validation)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": config.data.dataset_name,
        "source_split_requested": "train",
        "official_test_examples_read_or_selected": False,
        "sampling_seed": config.data.seed,
        "datasets_version": datasets.__version__,
        "python_version": platform.python_version(),
        "train": {
            "path": str(train_path),
            "size": len(train),
            "class_counts": _class_counts(train),
            "canonical_content_sha256": train_sha256,
            "file_sha256": _file_sha256(train_path),
            "example_ids": [item.example_id for item in train],
        },
        "validation": {
            "path": str(validation_path),
            "size": len(validation),
            "class_counts": _class_counts(validation),
            "canonical_content_sha256": validation_sha256,
            "file_sha256": _file_sha256(validation_path),
            "example_ids": [item.example_id for item in validation],
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"PASS train: {len(train)} examples, classes={manifest['train']['class_counts']}")
    print(
        "PASS validation: "
        f"{len(validation)} examples, classes={manifest['validation']['class_counts']}"
    )
    print("PASS train/validation IDs are disjoint")
    print("PASS no official test examples were read or selected")
    print(f"WROTE {manifest_path}")


if __name__ == "__main__":
    main()
