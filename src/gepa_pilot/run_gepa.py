from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers
import gepa
from gepa import optimize

from .config import PilotConfig
from .data import NewsExample, load_examples_json
from .evaluator import CommonEvaluator, EvaluationSummary
from .gepa_adapter import AGNewsGEPAAdapter
from .gepa_report import write_gepa_report
from .gepa_tracking import IterationSafetyStopper, ObservationTracker, ProposalCountStopper
from .local_reflection import LocalQwenReflectionLM


REFLECTION_PROMPT_TEMPLATE = """You are improving one editable instruction for a four-class AG News classifier.

The task interface is immutable:
- A = World
- B = Sports
- C = Business
- D = Sci/Tech
- The classifier receives one news article and must output exactly one capital letter: A, B, C, or D.
- The label mapping and response format are supplied elsewhere in the fixed prompt. Do not alter, reorder, or invent labels.

Current editable instruction:
```
<curr_param>
```

Observed examples, predictions, and feedback:
```
<side_info>
```

Infer a general improvement from the errors without copying article-specific facts. Write one complete replacement instruction of at most 90 words. It should help identify the article's central topic, preserve the immutable mapping, and require the one-letter response. Output only the replacement instruction enclosed in triple backticks, with no heading or explanation."""


class ExampleDataLoader:
    def __init__(self, examples: list[NewsExample]) -> None:
        self._order = [example.example_id for example in examples]
        self._examples = {example.example_id: example for example in examples}
        if len(self._examples) != len(examples):
            raise ValueError("Example IDs must be unique.")

    def all_ids(self) -> list[str]:
        return list(self._order)

    def fetch(self, ids: list[str]) -> list[NewsExample]:
        return [self._examples[example_id] for example_id in ids]

    def __len__(self) -> int:
        return len(self._order)


def _summary_cache_entry(summary: EvaluationSummary) -> dict[str, Any]:
    return {
        "accuracy": summary.accuracy,
        "mean_loss": summary.mean_loss,
        "per_class_accuracy": summary.per_class_accuracy,
        "confusion_matrix": summary.confusion_matrix,
        "correct_by_id": {record.example_id: record.correct for record in summary.records},
        "loss_by_id": {record.example_id: record.loss for record in summary.records},
        "predicted_by_id": {
            record.example_id: record.predicted_label for record in summary.records
        },
    }


def _gepa_source() -> dict[str, Any]:
    """Record which GEPA code ran: the installed wheel or the editable fork commit."""
    source = Path(gepa.__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain", "--", "."],
            capture_output=True, text=True, check=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"gepa_source": str(source), "gepa_git_commit": commit, "gepa_git_dirty": dirty}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the bounded GEPA observation pilot.")
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--train", default="artifacts/data/train.json")
    parser.add_argument("--validation", default="artifacts/data/validation.json")
    parser.add_argument("--run-dir", default="artifacts/gepa_observation")
    parser.add_argument("--proposals", type=int, default=12)
    parser.add_argument("--reflection-minibatch-size", type=int, default=4)
    parser.add_argument("--evaluation-batch-size", type=int, default=32)
    parser.add_argument("--reflection-max-new-tokens", type=int, default=192)
    parser.add_argument("--skip-posthoc-audit", action="store_true")
    args = parser.parse_args()

    if args.proposals < 1 or args.reflection_minibatch_size < 1:
        raise ValueError("Proposal and reflection minibatch counts must be positive.")
    config = PilotConfig.from_yaml(args.config)
    train = load_examples_json(args.train)
    validation = load_examples_json(args.validation)
    if any(item.example_id.startswith("official-test:") for item in train + validation):
        raise AssertionError("GEPA optimization must not receive official test examples.")

    run_dir = Path(args.run_dir)
    state_path = run_dir / "gepa_state.bin"
    if state_path.exists():
        raise FileExistsError(
            f"{state_path} already exists. Choose a new --run-dir to avoid accidental resume."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

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
    adapter = AGNewsGEPAAdapter(
        evaluator, evaluation_batch_size=args.evaluation_batch_size
    )
    reflector = LocalQwenReflectionLM(
        evaluator.model,
        evaluator.tokenizer,
        device=config.model.device,
        seed=config.data.seed + 1000,
        max_new_tokens=args.reflection_max_new_tokens,
    )
    tracker = ObservationTracker(run_dir / "callback_events.jsonl")
    stoppers = [
        ProposalCountStopper(tracker, args.proposals),
        IterationSafetyStopper(maximum=args.proposals * 3),
    ]

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model": asdict(config.model),
            "prompt": asdict(config.prompt),
            "data": asdict(config.data),
        },
        "environment": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "gepa_version": "0.1.4",
            **_gepa_source(),
            "device": config.model.device,
            "cuda_device_name": torch.cuda.get_device_name(torch.device(config.model.device)),
        },
        "train_size": len(train),
        "validation_size": len(validation),
        "official_test_examples_present": False,
        "proposal_cap": args.proposals,
        "reflection_minibatch_size": args.reflection_minibatch_size,
        "evaluation_batch_size": args.evaluation_batch_size,
        "use_merge": False,
        "frontier_type": "instance",
        "reflection_model": "same frozen local Qwen checkpoint as task model",
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    started = time.perf_counter()
    result = optimize(
        seed_candidate={"instruction": config.prompt.seed_instruction},
        trainset=ExampleDataLoader(train),
        valset=ExampleDataLoader(validation),
        adapter=adapter,
        reflection_lm=reflector,
        reflection_prompt_template=REFLECTION_PROMPT_TEMPLATE,
        candidate_selection_strategy="pareto",
        frontier_type="instance",
        skip_perfect_score=True,
        reflection_minibatch_size=args.reflection_minibatch_size,
        perfect_score=1.0,
        use_merge=False,
        stop_callbacks=stoppers,
        run_dir=str(run_dir),
        callbacks=[tracker],
        track_best_outputs=True,
        display_progress_bar=False,
        cache_evaluation=True,
        seed=config.data.seed,
        raise_on_exception=True,
        val_evaluation_policy="full_eval",
        acceptance_criterion="strict_improvement",
    )
    optimization_seconds = time.perf_counter() - started

    best_idx = max(
        range(len(result.val_aggregate_scores)),
        key=lambda index: result.val_aggregate_scores[index],
    )
    best_instruction = result.candidates[best_idx]["instruction"]
    tracker_snapshot = tracker.snapshot()

    audit_cache: dict[str, dict[str, Any]] = {}
    if not args.skip_posthoc_audit:
        texts = {config.prompt.seed_instruction, best_instruction}
        for proposal in tracker.proposals:
            parent = proposal.get("parent_candidate", {}).get("instruction")
            child = proposal.get("new_instructions", {}).get("instruction")
            if parent:
                texts.add(parent)
            if child:
                texts.add(child)
        for index, text in enumerate(sorted(texts)):
            print(f"AUDIT {index + 1}/{len(texts)}")
            audit_cache[text] = _summary_cache_entry(
                evaluator.evaluate(
                    validation,
                    text,
                    batch_size=args.evaluation_batch_size,
                )
            )

    candidate_table: list[dict[str, Any]] = []
    for proposal_number, proposal in enumerate(tracker.proposals, start=1):
        parent_text = proposal.get("parent_candidate", {}).get("instruction")
        child_text = proposal.get("new_instructions", {}).get("instruction")
        child_audit = audit_cache.get(child_text, {})
        parent_audit = audit_cache.get(parent_text, {})
        child_correct = child_audit.get("correct_by_id", {})
        parent_correct = parent_audit.get("correct_by_id", {})
        fixed = [
            example_id
            for example_id in child_correct
            if child_correct[example_id] and not parent_correct.get(example_id, False)
        ]
        introduced = [
            example_id
            for example_id in child_correct
            if not child_correct[example_id] and parent_correct.get(example_id, False)
        ]
        slot_token_count = None
        compositional = None
        if child_text:
            encoding = evaluator.prompt_builder.encode(validation[0].article, child_text)
            slot_token_count = len(encoding.slot_ids)
            compositional = encoding.compositional
        candidate_table.append(
            {
                "proposal_number": proposal_number,
                "iteration": proposal.get("iteration"),
                "status": proposal.get("status"),
                "candidate_idx": proposal.get("candidate_idx"),
                "parent_candidate_idx": proposal.get("parent_candidate_idx"),
                "parent_validation_score": proposal.get("parent_validation_score"),
                "parent_instruction": parent_text,
                "instruction": child_text,
                "slot_token_count": slot_token_count,
                "tokenizer_compositional": compositional,
                "selected_best": child_text == best_instruction,
                "minibatch_scores_after": proposal.get("minibatch_scores_after"),
                "validation_accuracy": child_audit.get("accuracy"),
                "validation_mean_loss": child_audit.get("mean_loss"),
                "per_class_accuracy": child_audit.get("per_class_accuracy"),
                "errors_fixed_count": len(fixed),
                "new_errors_count": len(introduced),
                "errors_fixed_ids": fixed,
                "new_error_ids": introduced,
            }
        )

    output = {
        "optimization_wall_time_seconds": optimization_seconds,
        "proposal_cap": args.proposals,
        "proposal_count": tracker.proposal_count,
        "candidate_count_accepted_including_seed": len(result.candidates),
        "best_candidate_idx": best_idx,
        "best_instruction": best_instruction,
        "best_validation_accuracy": result.val_aggregate_scores[best_idx],
        "all_accepted_candidates": result.candidates,
        "accepted_candidate_parents": result.parents,
        "accepted_candidate_validation_accuracy": result.val_aggregate_scores,
        "total_metric_calls": result.total_metric_calls,
        "num_full_validation_evaluations": result.num_full_val_evals,
        "adapter_counters": adapter.counters(),
        "reflection": reflector.manifest(),
        "tracker": tracker_snapshot,
        "candidate_table": candidate_table,
        "audit_by_instruction": audit_cache,
    }
    output_path = run_dir / "gepa_result.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    if audit_cache:
        write_gepa_report(output_path)
    print(
        f"DONE proposals={tracker.proposal_count}, accepted={len(result.candidates) - 1}, "
        f"best_val_accuracy={result.val_aggregate_scores[best_idx]:.4f}"
    )
    print(f"WROTE {output_path}")


if __name__ == "__main__":
    main()
