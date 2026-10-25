"""Proposal-level bench for variable-length ADMM proposers (small, controlled test).

Each method proposes one child for each of the same parent prompts, with identical randomness:
the same pretend GEPA minibatch (excluded), the same held-out selection batch S, the same
independent acceptance batch S' and the same ADMM training batches. Methods:

- fixed:    fixed-length beam ADMM (the fair beam ADMM proposer)
- ml_lm:    multi-length, offsets (0, -2, +2), insertion = the LM's most likely token (Phase 7)
- ml_grad:  multi-length, offsets (0, -2, +2), insertion chosen by the task-loss gradient
- gated:    gated variable-length ADMM (keep-gates + blank-capable beam projection)
- multistart: three fixed-length beam ADMM runs from the parent (different batch orders), pooled

For diagnosis only, parent and child are both scored on a held-out scoring set (the 512 validation
examples, or the task's 2k report set with --score-set); nothing is selected or accepted on it.
Parents: the seed instruction plus accepted prompts from the Phase 6 fair beam ADMM runs, or a JSON
list given with --parents-file. --repeats R proposes R times per (parent, method), each repeat with
its own minibatch, S, S' and training batches. Writes one JSON line per (parent, repeat, method).
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
    # no Phase 6 runs for these, so parents come from --parents-file
    "yahoo": ("configs/yahoo.yaml", "artifacts/data_yahoo/train.json", "artifacts/data_yahoo/validation.json", None),
    "yahoo_names": ("configs/yahoo_names.yaml", "artifacts/data_yahoo/train.json",
                    "artifacts/data_yahoo/validation.json", None),
    "emotion_names": ("configs/emotion_names.yaml", "artifacts/data_emotion/train.json",
                      "artifacts/data_emotion/validation.json", None),
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
    parser.add_argument("--parents-file", help="JSON list of parent instructions (default: Phase 6 runs)")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--score-set", help="examples for the diagnostic net (default: the validation split)")
    args = parser.parse_args()
    config_path, train_path, validation_path, prefix = TASKS[args.task]
    if args.score_set:
        validation_path = args.score_set
    if prefix is None and not args.parents_file:
        parser.error(f"{args.task} needs --parents-file")
    config = PilotConfig.from_yaml(config_path)
    evaluator = CommonEvaluator.from_pretrained(
        name_or_path=config.model.resolved_name_or_path,
        revision=None if config.model.local_snapshot_path else config.model.revision,
        local_files_only=config.model.local_files_only, dtype=config.model.dtype, device=config.model.device,
        system_prompt=config.prompt.system_prompt, label_names=config.prompt.label_names,
        enable_thinking=config.prompt.enable_thinking, input_heading=config.task.input_heading,
        answer_style=config.prompt.answer_style,
        multi_label_classes=config.task.class_names if config.task.multi_label else (),
        candidate_heading=config.task.candidate_heading,
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
    done = {(r["parent_index"], r.get("repeat", 0), r["method"])
            for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    pool = (json.loads(Path(args.parents_file).read_text(encoding="utf-8"))[: args.parents] if args.parents_file
            else parents(prefix, args.parents))
    for index, parent in enumerate(pool):
        for repeat in range(args.repeats):
            # repeat 0 keeps the original seeds, so earlier bench rows stay reproducible
            minibatch = random.Random(1000 + index + 100_000 * repeat).sample(train, 8)  # pretend GEPA minibatch (excluded)
            records = {"instruction": [{"article": example.article} for example in minibatch]}
            for method in args.methods:
                if (index, repeat, method) in done:
                    continue
                log_path = out.parent / f"{out.stem}_{method}_proposals.jsonl"
                proposer = ADMMProposer(refiner, train, selection_batch_size=32, seed=20_000 + index + 100_000 * repeat,
                                        log_path=log_path, acceptance_batch_size=32, **METHODS[method])
                child = proposer({"instruction": parent}, records, ["instruction"])["instruction"]
                log = json.loads(log_path.read_text().splitlines()[-1])
                p, c = validation_correct(parent), validation_correct(child)
                row = {
                    "task": args.task, "parent_index": index, "repeat": repeat, "method": method, "parent": parent, "child": child,
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
                print(f"[{args.task} parent {index} r{repeat} {method}] {row['outcome']} len{row['length_change']:+d} "
                      f"S {row['selection_gain_pp']:+.1f}pp accept={row['accepted_on_S_prime']} "
                      f"val net {row['fixes'] - row['breaks']:+d} | {child}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
