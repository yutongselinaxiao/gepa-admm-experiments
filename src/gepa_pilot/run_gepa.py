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
from gepa.proposer.reflective_mutation.reflection_lm import StatelessReflectionLM

from .admm import (
    ADMMHyperparameters,
    ADMMRefiner,
    build_allowed_token_ids,
    hyperparameters_dict,
)
from .admm_proposer import (
    ADMMHeldOutAcceptance,
    ADMMProposer,
    HeldOutJudgedReflectionLM,
    PolishingReflectionLM,
)
from .config import PilotConfig
from .data import NewsExample, load_examples_json, split_selection_report
from .evaluator import CommonEvaluator, EvaluationSummary
from .gepa_adapter import AGNewsGEPAAdapter
from .gepa_report import write_gepa_report
from .gated_admm import GatedConfig
from .gepa_tracking import IterationSafetyStopper, ObservationTracker, ProposalCountStopper
from .local_reflection import LocalQwenReflectionLM
from .prompting import label_list


_NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}


def reflection_prompt_template(config: PilotConfig) -> str:
    """GEPA reflection template for the configured task (identical to the pilot's for AG News)."""
    task = config.task
    labels = tuple(config.prompt.label_names)
    mapping = "\n".join(f"- {label} = {name}" for label, name in config.prompt.label_names.items())
    classes = _NUMBER_WORDS.get(len(labels), str(len(labels)))
    return f"""You are improving one editable instruction for a {classes}-class {task.name} classifier.

The task interface is immutable:
{mapping}
- The classifier receives one {task.input_noun} and must output exactly one capital letter: {label_list(labels)}.
- The label mapping and response format are supplied elsewhere in the fixed prompt. Do not alter, reorder, or invent labels.

Current editable instruction:
```
<curr_param>
```

Observed examples, predictions, and feedback:
```
<side_info>
```

Infer a general improvement from the errors without copying {task.input_short}-specific facts. Write one complete replacement instruction of at most 90 words. It should help identify {task.focus}, preserve the immutable mapping, and require the one-letter response. Output only the replacement instruction enclosed in triple backticks, with no heading or explanation."""


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
    parser.add_argument(
        "--run-seed",
        type=int,
        default=0,
        help="Offsets the GEPA, reflection and ADMM seeds; the data split is unchanged.",
    )
    parser.add_argument(
        "--selection-split",
        action="store_true",
        help="GEPA selects on half of validation; the other half is only used for reporting.",
    )
    parser.add_argument(
        "--proposer", choices=("reflective", "admm", "reflect_polish"), default="reflective"
    )
    parser.add_argument(
        "--admm-projection",
        choices=("euclidean", "context_top_p", "context_beam"),
        default="euclidean",
    )
    parser.add_argument("--admm-beam-width", type=int, default=4)
    parser.add_argument(
        "--admm-dual-step",
        type=float,
        default=0.0,
        help="> 0 enables per-class reward constraints with this dual step size.",
    )
    parser.add_argument("--admm-dual-batch-size", type=int, default=64)
    parser.add_argument(
        "--admm-length-offsets",
        type=int,
        nargs="+",
        default=[0],
        help="Slot-length offsets per ADMM proposal, e.g. 0 -2 2 (multi-length). Must include 0.",
    )
    parser.add_argument(
        "--admm-insertion",
        choices=("lm", "gradient"),
        default="lm",
        help="How a longer length variant picks inserted tokens: the LM's top token, or by task-loss gradient.",
    )
    parser.add_argument(
        "--admm-gated",
        action="store_true",
        help="Gated variable-length ADMM (keep-gates + blank-capable beam projection; gated_admm.py).",
    )
    parser.add_argument(
        "--admm-starts",
        type=int,
        default=1,
        help="Independent ADMM runs per proposal from the same parent, pooled and chosen on S (multi-start).",
    )
    parser.add_argument("--admm-spare-slots", type=int, default=4)
    parser.add_argument("--admm-gate-weight", type=float, default=10.0)
    parser.add_argument(
        "--admm-acceptance-batch-size",
        type=int,
        default=0,
        help="> 0: held-out verdict on its own batch (not the one that picked the iterate).",
    )
    parser.add_argument("--admm-fluency-weight", type=float, default=1.0)
    parser.add_argument("--admm-top-p", type=float, default=0.9)
    parser.add_argument("--admm-rho", type=float, default=1.0)
    parser.add_argument("--admm-learning-rate", type=float, default=0.005)
    parser.add_argument("--admm-outer-iterations", type=int, default=10)
    parser.add_argument("--admm-e-steps", type=int, default=3)
    parser.add_argument("--admm-train-batch-size", type=int, default=16)
    parser.add_argument("--admm-selection-batch-size", type=int, default=32)
    parser.add_argument(
        "--reflection-acceptance",
        choices=("minibatch", "heldout"),
        default="minibatch",
        help="heldout: reflective children are judged by the same held-out test as ADMM children.",
    )
    parser.add_argument(
        "--admm-acceptance",
        choices=("minibatch", "heldout"),
        default="minibatch",
        help="heldout: accept ADMM children by the proposer's held-out batch, not GEPA's minibatch.",
    )
    args = parser.parse_args()

    if args.proposals < 1 or args.reflection_minibatch_size < 1:
        raise ValueError("Proposal and reflection minibatch counts must be positive.")
    config = PilotConfig.from_yaml(args.config)
    optimization_seed = config.data.seed + 100_000 * args.run_seed
    train = load_examples_json(args.train)
    validation = load_examples_json(args.validation)
    gepa_valset, report_set = validation, []
    if args.selection_split:
        if args.skip_posthoc_audit:
            raise ValueError("--selection-split reports through the post-hoc audit; do not skip it.")
        gepa_valset, report_set = split_selection_report(validation, seed=config.data.seed)
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
        input_heading=config.task.input_heading,
    )
    adapter = AGNewsGEPAAdapter(
        evaluator, evaluation_batch_size=args.evaluation_batch_size, focus=config.task.focus
    )
    reflection_template = reflection_prompt_template(config)
    reflector: LocalQwenReflectionLM | None = None
    heldout_judge: HeldOutJudgedReflectionLM | None = None
    admm_proposer: ADMMProposer | None = None
    if args.proposer in ("reflective", "reflect_polish"):
        reflector = LocalQwenReflectionLM(
            evaluator.model,
            evaluator.tokenizer,
            device=config.model.device,
            seed=optimization_seed + 1000,
            max_new_tokens=args.reflection_max_new_tokens,
        )
        proposal_kwargs: dict[str, Any] = {
            "reflection_lm": reflector,
            "reflection_prompt_template": reflection_template,
        }
        if args.proposer == "reflective" and args.reflection_acceptance == "heldout":
            heldout_judge = HeldOutJudgedReflectionLM(
                StatelessReflectionLM(reflector, reflection_template),
                evaluator,
                train,
                selection_batch_size=args.admm_selection_batch_size,
                seed=optimization_seed + 2000,
                log_path=run_dir / "heldout_judgements.jsonl",
            )
            proposal_kwargs = {"reflection_strategy": heldout_judge}
    if args.proposer in ("admm", "reflect_polish"):
        admm_hp = ADMMHyperparameters(
            rho=args.admm_rho,
            learning_rate=args.admm_learning_rate,
            outer_iterations=args.admm_outer_iterations,
            e_steps=args.admm_e_steps,
            train_batch_size=args.admm_train_batch_size,
            evaluation_batch_size=args.evaluation_batch_size,
            projection=args.admm_projection,
            top_p=args.admm_top_p,
            beam_width=args.admm_beam_width,
            fluency_weight=args.admm_fluency_weight,
        )
        embedding_rows = evaluator.model.get_input_embeddings().weight.shape[0]
        admm_proposer = ADMMProposer(
            ADMMRefiner(
                evaluator,
                admm_hp,
                build_allowed_token_ids(evaluator.tokenizer, embedding_rows),
            ),
            train,
            selection_batch_size=args.admm_selection_batch_size,
            seed=optimization_seed + 2000,
            log_path=run_dir / "admm_proposals.jsonl",
            dual_step=args.admm_dual_step,
            dual_batch_size=args.admm_dual_batch_size,
            acceptance_batch_size=args.admm_acceptance_batch_size,
            length_offsets=args.admm_length_offsets,
            insertion=args.admm_insertion,
            gated=GatedConfig(spare_slots=args.admm_spare_slots, gate_weight=args.admm_gate_weight)
            if args.admm_gated else None,
            starts=args.admm_starts,
        )
        if args.proposer == "admm":
            proposal_kwargs = {"custom_candidate_proposer": admm_proposer}
        else:
            polishing = PolishingReflectionLM(
                StatelessReflectionLM(reflector, reflection_template), admm_proposer
            )
            if args.reflection_acceptance == "heldout":
                # Fair reflect-then-polish: the final child (polished or not) is judged against the
                # GEPA parent, with the reflective arms' rule, on the polish's independent slice S'.
                if args.admm_acceptance_batch_size <= 0:
                    raise ValueError("reflect_polish with held-out acceptance needs --admm-acceptance-batch-size > 0.")
                heldout_judge = HeldOutJudgedReflectionLM(
                    polishing,
                    evaluator,
                    train,
                    selection_batch_size=args.admm_selection_batch_size,
                    seed=optimization_seed + 2000,
                    log_path=run_dir / "heldout_judgements.jsonl",
                    acceptance_slice=args.admm_acceptance_batch_size,
                )
                proposal_kwargs = {"reflection_strategy": heldout_judge}
            else:
                proposal_kwargs = {"reflection_strategy": polishing}
    acceptance_criterion: Any = "strict_improvement"
    if admm_proposer is not None and args.admm_acceptance == "heldout":
        acceptance_criterion = ADMMHeldOutAcceptance(admm_proposer)
    if heldout_judge is not None:
        acceptance_criterion = ADMMHeldOutAcceptance(heldout_judge)
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
        "selection_split": args.selection_split,
        "gepa_valset_size": len(gepa_valset),
        "report_set_size": len(report_set),
        "official_test_examples_present": False,
        "proposal_cap": args.proposals,
        "reflection_minibatch_size": args.reflection_minibatch_size,
        "evaluation_batch_size": args.evaluation_batch_size,
        "use_merge": False,
        "frontier_type": "instance",
        "proposer": args.proposer,
        "run_seed": args.run_seed,
        "reflection_acceptance": args.reflection_acceptance,
        "optimization_seed": optimization_seed,
        "reflection_model": (
            "same frozen local Qwen checkpoint as task model"
            if reflector is not None
            else None
        ),
    }
    if admm_proposer is not None:
        manifest["admm"] = {
            **hyperparameters_dict(admm_proposer.refiner.hp),
            "selection_batch_size": args.admm_selection_batch_size,
            "acceptance": args.admm_acceptance,
            "dual_step": args.admm_dual_step,
            "dual_batch_size": args.admm_dual_batch_size,
            "acceptance_batch_size": args.admm_acceptance_batch_size,
            "length_offsets": args.admm_length_offsets,
            "insertion": args.admm_insertion,
            "starts": args.admm_starts,
            "gated": vars(admm_proposer.gated) if admm_proposer.gated is not None else None,
            "seed": admm_proposer.seed,
            "allowed_vocabulary_size": len(admm_proposer.refiner.allowed_token_ids),
        }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    started = time.perf_counter()
    result = optimize(
        seed_candidate={"instruction": config.prompt.seed_instruction},
        trainset=ExampleDataLoader(train),
        valset=ExampleDataLoader(gepa_valset),
        adapter=adapter,
        **proposal_kwargs,
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
        seed=optimization_seed,
        raise_on_exception=True,
        val_evaluation_policy="full_eval",
        acceptance_criterion=acceptance_criterion,
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
        "best_validation_accuracy_scope": "selection half" if args.selection_split else "full validation",
        "report_half": (
            {
                "ids": [example.example_id for example in report_set],
                "best_accuracy": sum(
                    audit_cache[best_instruction]["correct_by_id"][example.example_id]
                    for example in report_set
                ) / len(report_set),
                "seed_accuracy": sum(
                    audit_cache[config.prompt.seed_instruction]["correct_by_id"][example.example_id]
                    for example in report_set
                ) / len(report_set),
            }
            if args.selection_split
            else None
        ),
        "all_accepted_candidates": result.candidates,
        "accepted_candidate_parents": result.parents,
        "accepted_candidate_validation_accuracy": result.val_aggregate_scores,
        "total_metric_calls": result.total_metric_calls,
        "num_full_validation_evaluations": result.num_full_val_evals,
        "adapter_counters": adapter.counters(),
        "reflection": reflector.manifest() if reflector is not None else None,
        "admm_proposer": admm_proposer.manifest() if admm_proposer is not None else None,
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
        + (
            f", report_half_accuracy={output['report_half']['best_accuracy']:.4f}"
            if output["report_half"]
            else ""
        )
    )
    print(f"WROTE {output_path}")


if __name__ == "__main__":
    main()
