from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def compare_runs(run_specs: list[str], output_dir: str | Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in run_specs:
        if "=" not in spec:
            raise ValueError(f"Run specification must be LABEL=PATH, received {spec!r}.")
        label, directory_text = spec.split("=", 1)
        directory = Path(directory_text)
        result = json.loads((directory / "gepa_result.json").read_text(encoding="utf-8"))
        summary = json.loads((directory / "gepa_summary.json").read_text(encoding="utf-8"))
        audit = json.loads((directory / "objective_audit.json").read_text(encoding="utf-8"))
        manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "run": label,
                "directory": str(directory),
                "reflection_minibatch_size": manifest["reflection_minibatch_size"],
                "proposal_count": summary["proposal_count"],
                "accepted_proposal_count": summary["accepted_proposal_count"],
                "gepa_selected_accuracy": summary["gepa_selected"]["validation_accuracy"],
                "gepa_selected_mean_loss": summary["gepa_selected"]["validation_mean_loss"],
                "posthoc_best_accuracy": summary["posthoc_audit_best"]["validation_accuracy"],
                "posthoc_best_mean_loss": summary["posthoc_audit_best"]["validation_mean_loss"],
                "posthoc_best_was_accepted": "accepted"
                in summary["posthoc_audit_best"]["gepa_statuses"],
                "ranking_disagreements": audit["ranking_disagreement_count"],
                "pairwise_comparisons": audit["pairwise_comparison_count"],
                "pearson_loss_vs_accuracy": audit["pearson_loss_vs_accuracy"],
                "spearman_loss_rank_vs_accuracy_rank": audit[
                    "spearman_loss_rank_vs_accuracy_rank"
                ],
                "optimization_wall_time_seconds": result["optimization_wall_time_seconds"],
                "gepa_metric_calls": result["total_metric_calls"],
                "official_test_examples_present": manifest[
                    "official_test_examples_present"
                ],
                "gepa_selected_instruction": summary["gepa_selected"]["instruction"],
            }
        )

    selected = max(
        rows,
        key=lambda row: (
            row["gepa_selected_accuracy"],
            -row["gepa_selected_mean_loss"],
        ),
    )
    report = {
        "runs": rows,
        "best_fixed_seed_confirmation_run": selected["run"],
        "best_fixed_seed_confirmation_accuracy": selected["gepa_selected_accuracy"],
        "official_test_unused_in_all_runs": not any(
            row["official_test_examples_present"] for row in rows
        ),
        "interpretation_limit": (
            "Each setting has one fixed-seed run. This comparison diagnoses minibatch "
            "sensitivity but does not estimate between-seed uncertainty."
        ),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    fields = [key for key in rows[0] if key != "gepa_selected_instruction"]
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    table_lines = [
        "| Run | Minibatch | Accepted | GEPA accuracy | Post-hoc accuracy | Rank disagreements |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table_lines.append(
            f"| {row['run']} | {row['reflection_minibatch_size']} | "
            f"{row['accepted_proposal_count']} | {row['gepa_selected_accuracy']:.4f} | "
            f"{row['posthoc_best_accuracy']:.4f} | "
            f"{row['ranking_disagreements']}/{row['pairwise_comparisons']} |"
        )
    markdown = "\n".join(
        [
            "# GEPA minibatch comparison",
            "",
            *table_lines,
            "",
            f"Best fixed-seed GEPA selection: **{selected['run']}** at "
            f"{selected['gepa_selected_accuracy']:.4f} validation accuracy.",
            "",
            report["interpretation_limit"],
            "",
            f"Official test examples unused in every run: {report['official_test_unused_in_all_runs']}.",
            "",
        ]
    )
    (output_dir / "comparison.md").write_text(markdown, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare completed GEPA observation runs.")
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        help="Run specification LABEL=PATH; may be repeated.",
    )
    parser.add_argument("--output-dir", default="artifacts/gepa_comparison")
    args = parser.parse_args()
    specs = args.run or [
        "mb4=artifacts/gepa_observation",
        "mb8=artifacts/gepa_confirmation_mb8",
        "mb16=artifacts/gepa_confirmation_mb16",
    ]
    report = compare_runs(specs, args.output_dir)
    print(
        f"BEST {report['best_fixed_seed_confirmation_run']} "
        f"accuracy={report['best_fixed_seed_confirmation_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
