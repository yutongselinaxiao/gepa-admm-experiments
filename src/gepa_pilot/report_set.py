"""Large report sets and re-scoring of selected prompts.

`prepare`: draw a class-stratified report set from the *unused* part of a task's official
training split (never the test split; disjoint from the train/validation subsets the runs used).
Written to <data_dir>/report.json with a manifest.

`rescore`: score every prompt the finished runs selected (plus the seed instruction) on that
report set, deduplicating identical prompts. The 256-example report half (≈3 pp standard error
per run at ~35% accuracy) is replaced by ~2,000 examples (≈1.1 pp). Selection is unchanged: runs
still chose their prompt on the 256-example selection half.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .config import PilotConfig
from .data import CLASS_ID_TO_LABEL, NewsExample, format_input, load_examples_json

TASKS = {
    "agnews": ("configs/pilot.yaml", "artifacts/data"),
    "emotion": ("configs/emotion.yaml", "artifacts/data_emotion"),
    "yahoo": ("configs/yahoo.yaml", "artifacts/data_yahoo"),
    "sst5": ("configs/sst5.yaml", "artifacts/data_sst5"),
    "dbpedia": ("configs/dbpedia.yaml", "artifacts/data_dbpedia"),
}


def prepare(task: str, per_class: int, seed: int) -> Path:
    from datasets import load_dataset

    config_path, data_dir = TASKS[task]
    config = PilotConfig.from_yaml(config_path)
    used = {
        int(example.example_id.split(":")[1])
        for split in ("train", "validation")
        for example in load_examples_json(f"{data_dir}/{split}.json")
    }
    dataset = load_dataset(config.data.dataset_name, config.task.dataset_config, split="train")
    labels = list(dataset[config.task.label_field])
    rng = random.Random(seed)
    chosen: list[int] = []
    for class_id in range(len(config.prompt.label_names)):
        pool = [i for i, label in enumerate(labels) if int(label) == class_id and i not in used]
        rng.shuffle(pool)
        if len(pool) < per_class:
            raise ValueError(f"{task}: class {class_id} has only {len(pool)} unused examples.")
        chosen.extend(pool[:per_class])
    rng.shuffle(chosen)
    examples = [
        NewsExample(
            example_id=f"official-train:{index}:report",
            article=format_input(dataset[index], config.task.text_field, config.task.field_headings),
            gold_label=CLASS_ID_TO_LABEL[int(dataset[index][config.task.label_field])],
        )
        for index in chosen
    ]
    rows = [{"example_id": e.example_id, "article": e.article, "gold_label": e.gold_label} for e in examples]
    out = Path(data_dir) / "report.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "task": task, "source_split": "train (official)", "official_test_examples_used": False,
        "excluded_indices": "all indices used by train.json and validation.json",
        "per_class": per_class, "size": len(rows), "seed": seed,
        "file_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    (Path(data_dir) / "report_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"WROTE {out} ({len(rows)} examples)")
    return out


def rescore(task: str, summary_path: str, output: str, batch_size: int, extra_runs: list[str]) -> None:
    from .evaluator import CommonEvaluator

    config_path, data_dir = TASKS[task]
    config = PilotConfig.from_yaml(config_path)
    report = load_examples_json(f"{data_dir}/report.json")
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    runs: dict[str, str] = {}  # run directory -> selected prompt
    for arm in summary["arms"].values():
        if arm["task"] == task:
            for run in arm["runs"]:
                runs[run["run"]] = run["selected_instruction"]
    for run in extra_runs:
        result = json.loads(Path(run, "gepa_result.json").read_text(encoding="utf-8"))
        runs[run] = result["best_instruction"]

    out_path = Path(output)
    cache: dict[str, dict[str, Any]] = {}
    earlier_runs: dict[str, float] = {}
    if out_path.exists():  # resume: keep prompts already scored, and runs scored by earlier calls
        previous = json.loads(out_path.read_text(encoding="utf-8"))
        cache = previous.get("by_prompt", {})
        # Runs from earlier calls (e.g. other phases' --extra-runs) stay, unless their run directory
        # no longer holds a result (moved aside after a failure); runs passed now are re-derived.
        earlier_runs = {
            run: accuracy for run, accuracy in previous.get("by_run", {}).items()
            if run not in runs and Path(run, "gepa_result.json").exists()
        }
    prompts = sorted({config.prompt.seed_instruction, *runs.values()} - set(cache))
    if prompts:
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
        for index, prompt in enumerate(prompts, start=1):
            summary_ = evaluator.evaluate(report, prompt, batch_size=batch_size)
            cache[prompt] = {
                "accuracy": summary_.accuracy,
                "mean_loss": summary_.mean_loss,
                "per_class_accuracy": summary_.per_class_accuracy,
                "correct_by_id": {r.example_id: r.correct for r in summary_.records},
            }
            print(f"SCORED {index}/{len(prompts)} acc={summary_.accuracy:.4f}")
            if index % 10 == 0:
                out_path.write_text(json.dumps({"by_prompt": cache}), encoding="utf-8")
    result = {
        "task": task,
        "report_size": len(report),
        "seed_accuracy": cache[config.prompt.seed_instruction]["accuracy"],
        "by_run": {**earlier_runs, **{run: cache[prompt]["accuracy"] for run, prompt in runs.items()}},
        "by_prompt": cache,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result), encoding="utf-8")
    print(f"WROTE {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("prepare", "rescore"))
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--summary", default="artifacts/classification_results/summary.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--extra-runs", nargs="*", default=[])
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.task, args.per_class, args.seed)
    else:
        rescore(args.task, args.summary, args.output or f"artifacts/report2k/{args.task}.json",
                args.batch_size, args.extra_runs)


if __name__ == "__main__":
    main()
