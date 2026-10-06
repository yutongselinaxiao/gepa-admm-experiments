"""Re-select each finished GEPA run's candidate on the selection half of validation and
report it on the untouched report half (see data.split_selection_report).

Uses only the per-example correctness already stored in each run's post-hoc audit,
so no model calls are needed. GEPA's own parent selection saw all 512 examples, so
this removes the bias of the final pick but not every trace of validation use.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from .config import PilotConfig
from .data import load_examples_json, split_selection_report

ARMS = {
    "reflective": [
        "artifacts/gepa_confirmation_mb8",
        "artifacts/phase1/reflective_s1",
        "artifacts/phase1/reflective_s2",
    ],
    **{
        f"admm_rho{rho}_{acceptance}": [
            f"artifacts/{folder}/admm_rho{rho}_s{seed}" for seed in range(3)
        ]
        for rho in ("0.3", "1")
        for acceptance, folder in (("minibatch", "phase1"), ("heldout", "phase1_heldout"))
    },
}


def _accuracy(correct_by_id: dict[str, bool], ids: list[str]) -> float:
    return sum(correct_by_id[example_id] for example_id in ids) / len(ids)


def rescore_run(run_dir: Path, selection_ids: list[str], report_ids: list[str]) -> dict[str, Any]:
    result = json.loads((run_dir / "gepa_result.json").read_text(encoding="utf-8"))
    audit = result["audit_by_instruction"]
    pool = [candidate["instruction"] for candidate in result["all_accepted_candidates"]]
    scored = [
        {
            "instruction": text,
            "selection_accuracy": _accuracy(audit[text]["correct_by_id"], selection_ids),
            "report_accuracy": _accuracy(audit[text]["correct_by_id"], report_ids),
            "full_accuracy": audit[text]["accuracy"],
        }
        for text in pool
    ]
    # max() keeps the earliest candidate on ties, matching GEPA's own selection.
    chosen = max(scored, key=lambda row: row["selection_accuracy"])
    gepa_full = max(scored, key=lambda row: row["full_accuracy"])
    return {
        "run": str(run_dir),
        "accepted_pool_size": len(pool),
        "chosen": chosen,
        "seed_report_accuracy": scored[0]["report_accuracy"],
        "gepa_full_selected_full_accuracy": gepa_full["full_accuracy"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--validation", default="artifacts/data/validation.json")
    parser.add_argument("--output", default="artifacts/split_rescore/rescore.json")
    args = parser.parse_args()
    config = PilotConfig.from_yaml(args.config)
    selection, report = split_selection_report(
        load_examples_json(args.validation), seed=config.data.seed
    )
    selection_ids = [example.example_id for example in selection]
    report_ids = [example.example_id for example in report]
    output: dict[str, Any] = {"selection_ids": selection_ids, "report_ids": report_ids, "arms": {}}
    for arm, runs in ARMS.items():
        rows = [rescore_run(Path(run), selection_ids, report_ids) for run in runs]
        report_scores = [100 * row["chosen"]["report_accuracy"] for row in rows]
        full_scores = [100 * row["gepa_full_selected_full_accuracy"] for row in rows]
        output["arms"][arm] = {
            "runs": rows,
            "report_mean": statistics.mean(report_scores),
            "report_sd": statistics.stdev(report_scores),
            "full_selection_mean": statistics.mean(full_scores),
        }
        print(
            f"{arm:24s} report-half {statistics.mean(report_scores):5.2f} ± "
            f"{statistics.stdev(report_scores):4.2f} {['%.2f' % s for s in report_scores]} | "
            f"pool sizes {[row['accepted_pool_size'] for row in rows]} | "
            f"full-val (biased) {statistics.mean(full_scores):5.2f}"
        )
    print(f"seed instruction report-half accuracy: {100 * rows[0]['seed_report_accuracy']:.2f}")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
