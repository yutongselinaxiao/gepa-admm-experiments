from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
import transformers

from .config import PilotConfig
from .data import NewsExample, load_examples_json
from .evaluator import CommonEvaluator


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _diagnostic_subset(examples: Sequence[NewsExample]) -> list[NewsExample]:
    return [
        item
        for label in sorted({entry.gold_label for entry in examples})
        for item in [entry for entry in examples if entry.gold_label == label][:2]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run seed and manual prompt baselines.")
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--validation", default="artifacts/data/validation.json")
    parser.add_argument("--output", default="artifacts/baseline_results.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--diagnostic-max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    config = PilotConfig.from_yaml(args.config)
    validation_path = Path(args.validation)
    examples = load_examples_json(validation_path)
    expected_ids = {item.example_id for item in examples}
    if len(expected_ids) != len(examples):
        raise AssertionError("Validation manifest contains duplicate example IDs.")
    if any(item.example_id.startswith("official-test:") for item in examples):
        raise AssertionError("Refusing to run development baselines on official test examples.")

    evaluator = CommonEvaluator.from_pretrained(
        name_or_path=config.model.resolved_name_or_path,
        revision=None if config.model.local_snapshot_path else config.model.revision,
        local_files_only=config.model.local_files_only,
        dtype=config.model.dtype,
        device=config.model.device,
        system_prompt=config.prompt.system_prompt,
        label_names=config.prompt.label_names,
        enable_thinking=config.prompt.enable_thinking,
        input_heading=config.task.input_heading,
        answer_style=config.prompt.answer_style,
    )

    deterministic_probe = examples[:8]
    first_probe = evaluator.evaluate(deterministic_probe, config.prompt.seed_instruction)
    second_probe = evaluator.evaluate(deterministic_probe, config.prompt.seed_instruction)
    if first_probe.to_dict() != second_probe.to_dict():
        raise AssertionError("Repeated validation probe was not deterministic.")

    instructions = {
        "seed": config.prompt.seed_instruction,
        **{
            f"manual_{index}": instruction
            for index, instruction in enumerate(config.manual_instructions, start=1)
        },
    }
    started = time.perf_counter()
    results: dict[str, Any] = {}
    for name, instruction in instructions.items():
        prompt_started = time.perf_counter()
        summary = evaluator.evaluate(
            examples,
            instruction,
            batch_size=args.batch_size,
        )
        result = summary.to_dict()
        result["wall_time_seconds"] = time.perf_counter() - prompt_started
        result["forward_batches"] = math.ceil(len(examples) / args.batch_size)
        results[name] = result
        print(
            f"RESULT {name}: accuracy={summary.accuracy:.4f}, "
            f"mean_loss={summary.mean_loss:.4f}, "
            f"seconds={result['wall_time_seconds']:.2f}"
        )

    diagnostic_examples = _diagnostic_subset(examples)
    diagnostic_generations = {
        name: evaluator.diagnostic_generations(
            diagnostic_examples,
            instruction,
            max_new_tokens=args.diagnostic_max_new_tokens,
        )
        for name, instruction in instructions.items()
    }

    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": args.config,
        "config": {
            "model": asdict(config.model),
            "prompt": asdict(config.prompt),
            "data": asdict(config.data),
            "manual_instructions": list(config.manual_instructions),
        },
        "environment": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "device": str(evaluator.device),
            "cuda_device_name": torch.cuda.get_device_name(evaluator.device)
            if evaluator.device.type == "cuda"
            else None,
            "dtype": config.model.dtype,
        },
        "validation": {
            "path": str(validation_path),
            "sha256": _file_sha256(validation_path),
            "size": len(examples),
            "contains_official_test_examples": False,
        },
        "batch_size": args.batch_size,
        "batching_policy": {
            "padding_side": "right",
            "example_order": "validation_manifest_order",
            "logits": "only unique answer positions are projected through the LM head",
            "note": "Batch size and ordering are fixed because BF16 logits can vary with batch shape.",
        },
        "deterministic_probe_passed": True,
        "total_wall_time_seconds": time.perf_counter() - started,
        "diagnostic_generations": diagnostic_generations,
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("PASS deterministic validation probe")
    print("PASS no official test examples were evaluated")
    print(f"WROTE {output_path}")


if __name__ == "__main__":
    main()
