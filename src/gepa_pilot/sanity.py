from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .config import PilotConfig
from .data import load_ag_news_test_split, load_examples_json
from .evaluator import CommonEvaluator
from .prompting import PromptBuilder


def _load_tokenizer(config: PilotConfig) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        config.model.resolved_name_or_path,
        revision=None if config.model.local_snapshot_path else config.model.revision,
        local_files_only=config.model.local_files_only,
        use_fast=True,
    )


def run_tokenizer_gates(config: PilotConfig, examples_path: str | Path) -> dict[str, Any]:
    examples = load_examples_json(examples_path)
    tokenizer = _load_tokenizer(config)
    builder = PromptBuilder(
        tokenizer,
        config.prompt.system_prompt,
        config.prompt.label_names,
        enable_thinking=config.prompt.enable_thinking,
    )
    label_token_ids = builder.label_token_ids()
    encodings = builder.assert_seed_gates(
        config.prompt.seed_instruction, [example.article for example in examples]
    )

    test_lock_works = False
    try:
        load_ag_news_test_split(config.data.dataset_name)
    except PermissionError:
        test_lock_works = True
    if not test_lock_works:
        raise AssertionError("The test-split lock did not reject an unauthorized load.")

    return {
        "label_token_ids": {label: list(ids) for label, ids in label_token_ids.items()},
        "all_labels_single_token": all(len(ids) == 1 for ids in label_token_ids.values()),
        "seed_slot_token_count": len(encodings[0].slot_ids),
        "prompt_boundary_compositional": all(item.compositional for item in encodings),
        "full_roundtrip_exact": all(item.full_roundtrip_exact for item in encodings),
        "slot_roundtrip_exact": all(item.slot_roundtrip_exact for item in encodings),
        "slot_ids_independent_of_article": len({item.slot_ids for item in encodings}) == 1,
        "test_split_locked": test_lock_works,
        "examples_checked": len(examples),
    }


def run_model_gates(config: PilotConfig, examples_path: str | Path) -> dict[str, Any]:
    examples = load_examples_json(examples_path)
    evaluator = CommonEvaluator.from_pretrained(
        name_or_path=config.model.resolved_name_or_path,
        revision=None if config.model.local_snapshot_path else config.model.revision,
        local_files_only=config.model.local_files_only,
        dtype=config.model.dtype,
        device=config.model.device,
        system_prompt=config.prompt.system_prompt,
        label_names=config.prompt.label_names,
        enable_thinking=config.prompt.enable_thinking,
    )

    first = evaluator.evaluate(examples, config.prompt.seed_instruction)
    second = evaluator.evaluate(examples, config.prompt.seed_instruction)
    first_scores = [record.label_scores for record in first.records]
    second_scores = [record.label_scores for record in second.records]
    deterministic = first_scores == second_scores
    if not deterministic:
        raise AssertionError("Repeated seed evaluation produced different label scores.")

    single_records = [
        evaluator.evaluate([example], config.prompt.seed_instruction).records[0]
        for example in examples
    ]
    batch_predictions = [record.predicted_label for record in first.records]
    single_predictions = [record.predicted_label for record in single_records]
    batch_consistency_agreement = sum(
        left == right for left, right in zip(batch_predictions, single_predictions)
    ) / len(examples)
    batch_score_max_difference = max(
        abs(batch_record.label_scores[label] - single_record.label_scores[label])
        for batch_record, single_record in zip(first.records, single_records)
        for label in evaluator.labels
    )
    batch_tolerance = 1e-5 if config.model.dtype == "float32" else 5e-2

    parity = evaluator.parity_check(examples, config.prompt.seed_instruction)
    tolerance = 1e-5 if config.model.dtype == "float32" else 1e-2
    if parity.prediction_agreement != 1.0:
        raise AssertionError("input_ids and inputs_embeds predictions disagree.")
    if parity.maximum_absolute_label_score_difference > tolerance:
        raise AssertionError(
            "input_ids and inputs_embeds label logits differ by "
            f"{parity.maximum_absolute_label_score_difference:.6g}, tolerance={tolerance}"
        )

    gradient_examples = examples[:2]
    seed_encoding = evaluator.prompt_builder.encode(
        gradient_examples[0].article, config.prompt.seed_instruction
    )
    seed_slot_ids = torch.tensor(
        seed_encoding.slot_ids, dtype=torch.long, device=evaluator.device
    )
    with torch.enable_grad():
        soft_slot = (
            evaluator.model.get_input_embeddings()(seed_slot_ids)
            .detach()
            .clone()
            .requires_grad_(True)
        )
        soft_scores = evaluator.score_slot_embeddings(
            gradient_examples, config.prompt.seed_instruction, soft_slot
        )
        targets = torch.tensor(
            [evaluator.labels.index(example.gold_label) for example in gradient_examples],
            dtype=torch.long,
            device=evaluator.device,
        )
        soft_loss = torch.nn.functional.cross_entropy(soft_scores, targets)
        soft_loss.backward()
        if soft_slot.grad is None:
            raise AssertionError("No gradient reached the editable soft slot.")
        gradient_norm = float(soft_slot.grad.float().norm())
        gradient_finite = bool(torch.isfinite(soft_slot.grad).all())
    if not gradient_finite or gradient_norm <= 0.0:
        raise AssertionError(
            f"Invalid editable-slot gradient: finite={gradient_finite}, norm={gradient_norm}"
        )

    instruction_results = {}
    instructions = (config.prompt.seed_instruction,) + config.manual_instructions
    for index, instruction in enumerate(instructions):
        key = "seed" if index == 0 else f"manual_{index}"
        instruction_results[key] = evaluator.evaluate(examples, instruction).to_dict()

    diagnostic_generations = evaluator.diagnostic_generations(
        examples[:4], config.prompt.seed_instruction, max_new_tokens=4
    )

    return {
        "device": str(evaluator.device),
        "dtype": config.model.dtype,
        "deterministic_repeated_evaluation": deterministic,
        "batch_consistency": {
            "prediction_agreement": batch_consistency_agreement,
            "maximum_absolute_label_score_difference": batch_score_max_difference,
            "tolerance": batch_tolerance,
        },
        "parity_tolerance": tolerance,
        "parity": parity.to_dict(),
        "soft_slot_gradient": {
            "examples_checked": len(gradient_examples),
            "loss": float(soft_loss.detach()),
            "gradient_norm": gradient_norm,
            "all_gradient_values_finite": gradient_finite,
            "slot_token_count": int(soft_slot.shape[0]),
            "model_parameters_trainable": sum(
                parameter.requires_grad for parameter in evaluator.model.parameters()
            ),
        },
        "diagnostic_generations": diagnostic_generations,
        "instruction_results": instruction_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GEPA/ADMM evaluator sanity gates.")
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--examples", default="configs/sanity_examples.json")
    parser.add_argument("--output", default="artifacts/sanity_report.json")
    parser.add_argument(
        "--tokenizer-only",
        action="store_true",
        help="Run gates that do not require loading the task model.",
    )
    args = parser.parse_args()

    config = PilotConfig.from_yaml(args.config)
    report: dict[str, Any] = {
        "config": {
            "model": asdict(config.model),
            "prompt": asdict(config.prompt),
            "data": asdict(config.data),
            "manual_instructions": list(config.manual_instructions),
        },
        "tokenizer_gates": run_tokenizer_gates(config, args.examples),
    }
    if not args.tokenizer_only:
        report["model_gates"] = run_model_gates(config, args.examples)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    gates = report["tokenizer_gates"]
    print(f"PASS label tokens: {gates['label_token_ids']}")
    print(f"PASS prompt compositionality: {gates['examples_checked']} examples")
    print("PASS tokenizer round-trip: full prompt and editable slot")
    print("PASS test split is locked")
    if "model_gates" in report:
        model = report["model_gates"]
        parity = model["parity"]
        print("PASS repeated evaluation is deterministic")
        batch_consistency = model["batch_consistency"]
        batch_status = (
            "PASS"
            if batch_consistency["prediction_agreement"] == 1.0
            and batch_consistency["maximum_absolute_label_score_difference"]
            <= batch_consistency["tolerance"]
            else "DIAGNOSTIC"
        )
        print(
            f"{batch_status} batched/single-example consistency: "
            f"agreement={batch_consistency['prediction_agreement']:.3f}, "
            "max_abs_diff="
            f"{batch_consistency['maximum_absolute_label_score_difference']:.6g}"
        )
        print(
            "PASS input_ids/inputs_embeds parity: "
            f"agreement={parity['prediction_agreement']:.3f}, "
            f"max_abs_diff={parity['maximum_absolute_label_score_difference']:.6g}"
        )
        gradient = model["soft_slot_gradient"]
        print(
            "PASS soft-slot gradient: "
            f"tokens={gradient['slot_token_count']}, "
            f"norm={gradient['gradient_norm']:.6g}, "
            f"trainable_model_parameters={gradient['model_parameters_trainable']}"
        )
        for name, result in model["instruction_results"].items():
            print(
                f"RESULT {name}: accuracy={result['accuracy']:.3f}, "
                f"mean_loss={result['mean_loss']:.4f}"
            )
    print(f"WROTE {output_path}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
