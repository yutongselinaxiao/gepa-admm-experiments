from __future__ import annotations

import argparse
import json
import math
import platform
import random
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers

from .admm import (
    ADMMHyperparameters,
    ADMMRefiner,
    build_allowed_token_ids,
    calibration_record,
    decode_and_check_slot,
    evaluate_discrete_slot,
    evaluate_soft_slot,
    hyperparameters_dict,
)
from .config import PilotConfig
from .data import load_examples_json
from .evaluator import CommonEvaluator


def _json_without_records(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "records_by_id"}


def _choose_best(checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        checkpoints,
        key=lambda row: (
            row["discrete_validation"]["accuracy"],
            -row["discrete_validation"]["mean_loss"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the minimal fixed-length ADMM pilot.")
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--train", default="artifacts/data/train.json")
    parser.add_argument("--validation", default="artifacts/data/validation.json")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--rho", type=float, required=True)
    parser.add_argument("--rho-grid", type=float, nargs="*", default=[1.0, 10.0, 100.0])
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--outer-iterations", type=int, default=30)
    parser.add_argument("--e-steps", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--evaluation-batch-size", type=int, default=32)
    parser.add_argument("--validation-frequency", type=int, default=5)
    parser.add_argument("--projection-chunk-size", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--calibrate-only", action="store_true")
    args = parser.parse_args()

    hp = ADMMHyperparameters(
        rho=args.rho,
        learning_rate=args.learning_rate,
        outer_iterations=args.outer_iterations,
        e_steps=args.e_steps,
        train_batch_size=args.train_batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        validation_frequency=args.validation_frequency,
        projection_chunk_size=args.projection_chunk_size,
    )
    hp.validate()
    config = PilotConfig.from_yaml(args.config)
    seed = config.data.seed if args.seed is None else args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train = load_examples_json(args.train)
    validation = load_examples_json(args.validation)
    if any(item.example_id.startswith("official-test:") for item in train + validation):
        raise AssertionError("ADMM must not receive official test examples.")
    run_dir = Path(args.run_dir)
    result_path = run_dir / "admm_result.json"
    if result_path.exists():
        raise FileExistsError(f"{result_path} already exists; choose a new --run-dir.")
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
    instruction = config.prompt.seed_instruction
    seed_encodings = evaluator.prompt_builder.assert_seed_gates(
        instruction, [train[0].article, validation[0].article]
    )
    seed_ids = torch.tensor(
        seed_encodings[0].slot_ids, dtype=torch.long, device=evaluator.device
    )
    if not 24 <= len(seed_ids) <= 32:
        raise AssertionError(f"Seed slot length {len(seed_ids)} is outside 24--32 tokens.")
    embedding_weight = evaluator.model.get_input_embeddings().weight
    seed_embeddings = embedding_weight.index_select(0, seed_ids).detach().float()
    allowed_ids = build_allowed_token_ids(
        evaluator.tokenizer, embedding_weight.shape[0]
    )
    if any(int(token_id) not in set(allowed_ids.tolist()) for token_id in seed_ids.cpu()):
        raise AssertionError("The seed contains a token excluded from the allowed vocabulary.")

    calibration_batch = train[: hp.train_batch_size]
    calibration = calibration_record(
        evaluator,
        calibration_batch,
        instruction,
        seed_embeddings,
        learning_rate=hp.learning_rate,
        rho_grid=args.rho_grid,
    )
    parity = evaluator.parity_check(validation[:8], instruction).to_dict()
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model": asdict(config.model),
            "prompt": asdict(config.prompt),
            "data": asdict(config.data),
        },
        "hyperparameters": hyperparameters_dict(hp),
        "seed": seed,
        "slot_token_count": len(seed_ids),
        "allowed_vocabulary_size": len(allowed_ids),
        "excluded_vocabulary_size": embedding_weight.shape[0] - len(allowed_ids),
        "train_size": len(train),
        "validation_size": len(validation),
        "official_test_examples_present": False,
        "environment": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "device": config.model.device,
            "cuda_device_name": torch.cuda.get_device_name(torch.device(config.model.device)),
        },
        "calibration": calibration,
        "initial_parity": parity,
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    if args.calibrate_only:
        print(json.dumps(calibration, indent=2, sort_keys=True))
        return

    refiner = ADMMRefiner(evaluator, hp, allowed_ids)
    state = refiner.init_state(instruction, seed_ids)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(len(train), generator=generator).tolist()
    batch_cursor = 0
    checkpoints: list[dict[str, Any]] = []
    started = time.perf_counter()

    def next_batch() -> list[Any]:
        nonlocal permutation, batch_cursor
        if batch_cursor + hp.train_batch_size > len(permutation):
            permutation = torch.randperm(len(train), generator=generator).tolist()
            batch_cursor = 0
        indices = permutation[batch_cursor : batch_cursor + hp.train_batch_size]
        batch_cursor += hp.train_batch_size
        return [train[index] for index in indices]

    def checkpoint(iteration: int, train_loss: float | None, previous_z: torch.Tensor) -> None:
        e, z, u = state.e, state.z, state.u
        continuous = evaluate_soft_slot(
            evaluator,
            validation,
            instruction,
            e.detach(),
            batch_size=hp.evaluation_batch_size,
        )
        discrete = evaluate_discrete_slot(
            evaluator,
            validation,
            instruction,
            z,
            batch_size=hp.evaluation_batch_size,
        )
        state.forward_calls += 2 * math.ceil(len(validation) / hp.evaluation_batch_size)
        ez = embedding_weight.index_select(0, z).detach().float()
        previous_ez = embedding_weight.index_select(0, previous_z).detach().float()
        primal = e.detach() - ez
        dual = hp.rho * (ez - previous_ez)
        roundtrip = decode_and_check_slot(evaluator, validation[0].article, z)
        reconstructed = None
        if not roundtrip["roundtrip_exact"] and not roundtrip["reconstruction_error"]:
            reconstructed_summary = evaluator.evaluate(
                validation,
                roundtrip["reconstructed_instruction"],
                batch_size=hp.evaluation_batch_size,
            )
            state.forward_calls += math.ceil(len(validation) / hp.evaluation_batch_size)
            reconstructed = {
                "accuracy": reconstructed_summary.accuracy,
                "mean_loss": reconstructed_summary.mean_loss,
                "per_class_accuracy": reconstructed_summary.per_class_accuracy,
            }
        record = {
            "iteration": iteration,
            "train_soft_task_loss": train_loss,
            "continuous_validation": _json_without_records(continuous),
            "discrete_validation": _json_without_records(discrete),
            "primal_residual_norm": float(primal.norm()),
            "dual_residual_norm": float(dual.norm()),
            "changed_token_count_from_previous": int(z.ne(previous_z).sum()),
            "changed_token_count_from_seed": int(z.ne(seed_ids).sum()),
            "slot_token_ids": z.detach().cpu().tolist(),
            "roundtrip": roundtrip,
            "reconstructed_text_validation": reconstructed,
            "forward_calls_so_far": state.forward_calls,
            "backward_calls_so_far": state.backward_calls,
        }
        checkpoints.append(record)
        torch.save(
            {
                "iteration": iteration,
                "e": e.detach().cpu(),
                "z": z.detach().cpu(),
                "u": u.detach().cpu(),
                "optimizer": state.optimizer.state_dict(),
                "diagnostics": record,
            },
            run_dir / f"checkpoint_{iteration:03d}.pt",
        )
        (run_dir / "history.json").write_text(
            json.dumps(checkpoints, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(
            f"CHECKPOINT iteration={iteration} discrete_acc={discrete['accuracy']:.4f} "
            f"discrete_loss={discrete['mean_loss']:.4f} soft_loss={continuous['mean_loss']:.4f} "
            f"r={record['primal_residual_norm']:.4f} s={record['dual_residual_norm']:.4f} "
            f"changed={record['changed_token_count_from_seed']} "
            f"roundtrip={roundtrip['roundtrip_exact']}"
        )

    checkpoint(0, None, state.z.clone())
    for iteration in range(1, hp.outer_iterations + 1):
        last_task_loss, previous_z = refiner.step(state, next_batch())
        if iteration % hp.validation_frequency == 0 or iteration == hp.outer_iterations:
            checkpoint(iteration, last_task_loss, previous_z)

    elapsed = time.perf_counter() - started
    best = _choose_best(checkpoints)
    result = {
        "optimization_wall_time_seconds": elapsed,
        "hyperparameters": hyperparameters_dict(hp),
        "calibration": calibration,
        "initial_parity": parity,
        "slot_token_count": len(seed_ids),
        "seed_slot_token_ids": seed_ids.detach().cpu().tolist(),
        "allowed_vocabulary_size": len(allowed_ids),
        "forward_calls": state.forward_calls,
        "backward_calls": state.backward_calls,
        "checkpoints": checkpoints,
        "best_checkpoint": best,
        "selection_rule": "highest discrete validation accuracy; mean loss tie-breaker",
        "objective_interpretation": (
            "Label loss is diagnostic because the GEPA audit found accuracy/loss ranking disagreements."
        ),
        "official_test_examples_present": False,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"DONE rho={hp.rho} best_iteration={best['iteration']} "
        f"best_accuracy={best['discrete_validation']['accuracy']:.4f}"
    )
    print(f"WROTE {result_path}")


if __name__ == "__main__":
    main()
