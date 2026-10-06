from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from .data import load_examples_json


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_scale == 0 or right_scale == 0:
        return None
    return numerator / (left_scale * right_scale)


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def _candidate_metadata(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    seed = result["all_accepted_candidates"][0]["instruction"]
    metadata: dict[str, dict[str, Any]] = {
        seed: {
            "candidate_id": "seed",
            "proposal_numbers": [],
            "gepa_statuses": ["seed"],
            "slot_token_count": None,
            "tokenizer_compositional": True,
        }
    }
    for row in result.get("candidate_table", []):
        text = row.get("instruction")
        if not text:
            continue
        if text not in metadata:
            metadata[text] = {
                "candidate_id": f"proposal_{row['proposal_number']}",
                "proposal_numbers": [],
                "gepa_statuses": [],
                "slot_token_count": row.get("slot_token_count"),
                "tokenizer_compositional": row.get("tokenizer_compositional"),
            }
        metadata[text]["proposal_numbers"].append(row["proposal_number"])
        metadata[text]["gepa_statuses"].append(row["status"])
    return metadata


def _example_transition(
    example_id: str,
    examples_by_id: dict[str, Any],
    low_loss_audit: dict[str, Any],
    high_accuracy_audit: dict[str, Any],
) -> dict[str, Any]:
    example = examples_by_id[example_id]
    return {
        "example_id": example_id,
        "gold_label": example.gold_label,
        "article_excerpt": example.article[:240],
        "lower_loss_candidate_prediction": low_loss_audit["predicted_by_id"][example_id],
        "higher_accuracy_candidate_prediction": high_accuracy_audit["predicted_by_id"][example_id],
        "lower_loss_candidate_loss": low_loss_audit["loss_by_id"][example_id],
        "higher_accuracy_candidate_loss": high_accuracy_audit["loss_by_id"][example_id],
    }


def run_objective_audit(
    result_path: str | Path,
    validation_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    result_path = Path(result_path)
    output_dir = Path(output_dir) if output_dir is not None else result_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    audits: dict[str, dict[str, Any]] = result.get("audit_by_instruction", {})
    if not audits:
        raise ValueError("The GEPA result does not contain post-hoc candidate audits.")
    examples = load_examples_json(validation_path)
    examples_by_id = {example.example_id: example for example in examples}
    metadata = _candidate_metadata(result)

    candidate_rows: list[dict[str, Any]] = []
    for instruction, audit in audits.items():
        unknown_ids = set(audit["loss_by_id"]) - set(examples_by_id)
        if unknown_ids:
            raise ValueError(f"Audit contains unknown validation IDs: {sorted(unknown_ids)[:3]}")
        losses_by_class: dict[str, list[float]] = defaultdict(list)
        for example_id, loss in audit["loss_by_id"].items():
            losses_by_class[examples_by_id[example_id].gold_label].append(float(loss))
        info = metadata.get(
            instruction,
            {
                "candidate_id": "untracked",
                "proposal_numbers": [],
                "gepa_statuses": ["untracked"],
                "slot_token_count": None,
                "tokenizer_compositional": None,
            },
        )
        candidate_rows.append(
            {
                **info,
                "instruction": instruction,
                "validation_accuracy": float(audit["accuracy"]),
                "validation_mean_loss": float(audit["mean_loss"]),
                "per_class_accuracy": audit["per_class_accuracy"],
                "per_class_mean_loss": {
                    label: mean(losses_by_class[label]) for label in sorted(losses_by_class)
                },
            }
        )
    candidate_rows.sort(
        key=lambda row: (
            row["candidate_id"] != "seed",
            row["proposal_numbers"][0] if row["proposal_numbers"] else -1,
        )
    )

    disagreements: list[dict[str, Any]] = []
    audit_by_text = audits
    for left, right in combinations(candidate_rows, 2):
        if left["validation_mean_loss"] == right["validation_mean_loss"]:
            continue
        low_loss, other = (
            (left, right)
            if left["validation_mean_loss"] < right["validation_mean_loss"]
            else (right, left)
        )
        if low_loss["validation_accuracy"] >= other["validation_accuracy"]:
            continue
        low_audit = audit_by_text[low_loss["instruction"]]
        high_audit = audit_by_text[other["instruction"]]
        high_only = [
            example_id
            for example_id in low_audit["correct_by_id"]
            if high_audit["correct_by_id"][example_id]
            and not low_audit["correct_by_id"][example_id]
        ]
        low_only = [
            example_id
            for example_id in low_audit["correct_by_id"]
            if low_audit["correct_by_id"][example_id]
            and not high_audit["correct_by_id"][example_id]
        ]
        high_only.sort(
            key=lambda example_id: abs(
                float(low_audit["loss_by_id"][example_id])
                - float(high_audit["loss_by_id"][example_id])
            ),
            reverse=True,
        )
        disagreements.append(
            {
                "lower_loss_candidate_id": low_loss["candidate_id"],
                "higher_accuracy_candidate_id": other["candidate_id"],
                "lower_loss": low_loss["validation_mean_loss"],
                "higher_loss": other["validation_mean_loss"],
                "lower_accuracy": low_loss["validation_accuracy"],
                "higher_accuracy": other["validation_accuracy"],
                "higher_accuracy_only_correct_count": len(high_only),
                "lower_loss_only_correct_count": len(low_only),
                "representative_examples": [
                    _example_transition(example_id, examples_by_id, low_audit, high_audit)
                    for example_id in high_only[:5]
                ],
            }
        )
    disagreements.sort(
        key=lambda row: row["higher_accuracy"] - row["lower_accuracy"], reverse=True
    )

    losses = [row["validation_mean_loss"] for row in candidate_rows]
    accuracies = [row["validation_accuracy"] for row in candidate_rows]
    best_accuracy = max(
        candidate_rows,
        key=lambda row: (row["validation_accuracy"], -row["validation_mean_loss"]),
    )
    best_loss = min(
        candidate_rows,
        key=lambda row: (row["validation_mean_loss"], -row["validation_accuracy"]),
    )
    report = {
        "candidate_count": len(candidate_rows),
        "candidates": candidate_rows,
        "best_accuracy_candidate_id": best_accuracy["candidate_id"],
        "lowest_loss_candidate_id": best_loss["candidate_id"],
        "best_accuracy_and_lowest_loss_agree": (
            best_accuracy["instruction"] == best_loss["instruction"]
        ),
        "pearson_loss_vs_accuracy": _pearson(losses, accuracies),
        "spearman_loss_rank_vs_accuracy_rank": _pearson(
            _ranks(losses), _ranks(accuracies)
        ),
        "pairwise_comparison_count": len(candidate_rows) * (len(candidate_rows) - 1) // 2,
        "ranking_disagreement_count": len(disagreements),
        "ranking_disagreements": disagreements,
        "decision_gate": (
            "retain_initial_label_loss_for_first_ADMM_run"
            if best_accuracy["instruction"] == best_loss["instruction"]
            else "treat_label_loss_as_diagnostic_due_to_top_candidate_mismatch"
        ),
    }
    (output_dir / "objective_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    flat_fields = [
        "candidate_id",
        "proposal_numbers",
        "gepa_statuses",
        "validation_accuracy",
        "validation_mean_loss",
        "slot_token_count",
        "tokenizer_compositional",
        "instruction",
    ]
    with (output_dir / "objective_audit_candidates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(candidate_rows)

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/gepa-pilot-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.scatter(losses, accuracies, color="#2563eb")
    for row in candidate_rows:
        axis.annotate(
            row["candidate_id"].replace("proposal_", "P"),
            (row["validation_mean_loss"], row["validation_accuracy"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Validation mean label loss (lower is better)")
    axis.set_ylabel("Validation accuracy (higher is better)")
    axis.set_title("GEPA candidate objective audit")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "objective_audit_scatter.png", dpi=180)
    plt.close(figure)

    markdown = f"""# Objective audit

- Unique candidates: {report['candidate_count']}
- Best-accuracy candidate: {report['best_accuracy_candidate_id']}
- Lowest-loss candidate: {report['lowest_loss_candidate_id']}
- Same top candidate: {report['best_accuracy_and_lowest_loss_agree']}
- Pearson correlation (loss versus accuracy): {report['pearson_loss_vs_accuracy']:.4f}
- Spearman rank correlation (loss versus accuracy): {report['spearman_loss_rank_vs_accuracy_rank']:.4f}
- Pairwise ranking disagreements: {report['ranking_disagreement_count']} / {report['pairwise_comparison_count']}
- Decision gate: `{report['decision_gate']}`

Lower loss should generally accompany higher accuracy, so the expected correlation is negative. Pairwise disagreements are retained with representative validation examples in `objective_audit.json`.
"""
    (output_dir / "objective_audit.md").write_text(markdown, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit loss/accuracy agreement in GEPA candidates.")
    parser.add_argument("--result", default="artifacts/gepa_observation/gepa_result.json")
    parser.add_argument("--validation", default="artifacts/data/validation.json")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    report = run_objective_audit(args.result, args.validation, args.output_dir)
    print(
        f"AUDITED candidates={report['candidate_count']} "
        f"disagreements={report['ranking_disagreement_count']}/"
        f"{report['pairwise_comparison_count']} decision={report['decision_gate']}"
    )


if __name__ == "__main__":
    main()
