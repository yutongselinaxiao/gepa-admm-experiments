"""Proposal-level bench for variable-length ADMM proposers (small, controlled test).

Each method proposes one child for each of the same parent prompts, with identical randomness:
the same pretend GEPA minibatch (excluded), the same held-out selection batch S, the same
independent acceptance batch S' and the same ADMM training batches. Methods:

- fixed:    fixed-length beam ADMM (the fair beam ADMM proposer)
- ml_lm:    multi-length, offsets (0, -2, +2), insertion = the LM's most likely token (Phase 7)
- ml_grad:  multi-length, offsets (0, -2, +2), insertion chosen by the task-loss gradient
- gated:    gated variable-length ADMM (keep-gates + blank-capable beam projection)
- multistart: three fixed-length beam ADMM runs from the parent (different batch orders), pooled

For diagnosis only, parent and child are both scored on the 512 validation examples; nothing is
selected or accepted on them. Parents: the seed instruction plus accepted prompts from the
Phase 6 fair beam ADMM runs. Writes one JSON line per (parent, method) to --out.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from .admm import ADMMHyperparameters, ADMMRefiner, build_allowed_token_ids
from .admm_proposer import ADMMProposer
from .config import PilotConfig
from .data import load_examples_json
from .evaluator import CommonEvaluator
from .gated_admm import GatedConfig

TASKS = {
    "agnews": ("configs/pilot.yaml", "artifacts/data/train.json", "artifacts/data/validation.json",
               "artifacts/phase6_independent_acceptance/agnews_beam_indep_s"),
    "emotion": ("configs/emotion.yaml", "artifacts/data_emotion/train.json", "artifacts/data_emotion/validation.json",
                "artifacts/phase6_independent_acceptance/emotion_beam_indep_s"),
}
METHODS = {
    "fixed": {"length_offsets": (0,)},
    "ml_lm": {"length_offsets": (0, -2, 2), "insertion": "lm"},
    "ml_grad": {"length_offsets": (0, -2, 2), "insertion": "gradient"},
    "gated": {"gated": GatedConfig()},
    "multistart": {"length_offsets": (0,), "starts": 3},
}


def parents(prefix: str, count: int) -> list[str]:
    pool: list[str] = []
    for seed in range(10):
        result = json.loads(Path(f"{prefix}{seed}", "gepa_result.json").read_text(encoding="utf-8"))
        for candidate in result["all_accepted_candidates"]:
            if candidate["instruction"] not in pool:
                pool.append(candidate["instruction"])
    seed_instruction, rest = pool[0], pool[1:]
    random.Random(0).shuffle(rest)
    return [seed_instruction] + rest[: count - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--parents", type=int, default=6)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config_path, train_path, validation_path, prefix = TASKS[args.task]
    config = PilotConfig.from_yaml(config_path)
    evaluator = CommonEvaluator.from_pretrained(
        name_or_path=config.model.resolved_name_or_path,
        revision=None if config.model.local_snapshot_path else config.model.revision,
        local_files_only=config.model.local_files_only, dtype=config.model.dtype, device=config.model.device,
        system_prompt=config.prompt.system_prompt, label_names=config.prompt.label_names,
        enable_thinking=config.prompt.enable_thinking, input_heading=config.task.input_heading,
        answer_style=config.prompt.answer_style,
    )
    train, validation = load_examples_json(train_path), load_examples_json(validation_path)
    hp = ADMMHyperparameters(rho=0.1, outer_iterations=20, projection="context_beam", top_p=0.99,
                             beam_width=4, fluency_weight=0.3)
    allowed = build_allowed_token_ids(evaluator.tokenizer, evaluator.model.get_input_embeddings().weight.shape[0])
    refiner = ADMMRefiner(evaluator, hp, allowed)

    correctness: dict[str, list[bool]] = {}

    def validation_correct(text: str) -> list[bool]:
        if text not in correctness:
            ids = evaluator.prompt_builder.encode(validation[0].article, text).slot_ids
            scores = torch.cat([evaluator.score_fixed_slot_ids(validation[i : i + 32], text, ids)
                                for i in range(0, len(validation), 32)])
            gold = torch.tensor([evaluator.labels.index(x.gold_label) for x in validation])
            correctness[text] = scores.argmax(1).eq(gold).tolist()
        return correctness[text]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["parent_index"], r["method"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    for index, parent in enumerate(parents(prefix, args.parents)):
        minibatch = random.Random(1000 + index).sample(train, 8)  # pretend GEPA minibatch (excluded)
        records = {"instruction": [{"article": example.article} for example in minibatch]}
        for method in args.methods:
            if (index, method) in done:
                continue
            log_path = out.parent / f"{out.stem}_{method}_proposals.jsonl"
            proposer = ADMMProposer(refiner, train, selection_batch_size=32, seed=20_000 + index, log_path=log_path,
                                    acceptance_batch_size=32, **METHODS[method])
            child = proposer({"instruction": parent}, records, ["instruction"])["instruction"]
            log = json.loads(log_path.read_text().splitlines()[-1])
            p, c = validation_correct(parent), validation_correct(child)
            row = {
                "task": args.task, "parent_index": index, "method": method, "parent": parent, "child": child,
                "outcome": log["outcome"], "accepted_on_S_prime": log.get("heldout_accept"),
                "length_change": len(log["chosen"]["slot_ids"]) - len(log["parent_slot_ids"]) if log.get("chosen") else 0,
                "chosen_offset": log["chosen"].get("start_offset") if log.get("chosen") else None,
                "selection_gain_pp": 100 * (log["chosen"]["selection_accuracy"] - log["parent_selection_accuracy"])
                if log.get("chosen") else 0.0,
                "selection_loss_gain": log["parent_selection_loss"] - log["chosen"]["selection_loss"] if log.get("chosen") else 0.0,
                "nll_per_token": log["chosen"]["slot_nll"] / len(log["chosen"]["slot_ids"]) if log.get("chosen") else None,
                "parent_nll_per_token": log["parent_slot_nll"] / len(log["parent_slot_ids"]),
                "fixes": sum(x and not y for x, y in zip(c, p)), "breaks": sum(y and not x for x, y in zip(c, p)),
                "wall_time_seconds": log["wall_time_seconds"],
            }
            with out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{args.task} parent {index} {method}] {row['outcome']} len{row['length_change']:+d} "
                  f"S {row['selection_gain_pp']:+.1f}pp accept={row['accepted_on_S_prime']} "
                  f"val net {row['fixes'] - row['breaks']:+d} | {child}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
