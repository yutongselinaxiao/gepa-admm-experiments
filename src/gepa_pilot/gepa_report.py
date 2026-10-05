from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _prediction_counts(audit: dict[str, Any]) -> dict[str, int]:
    counts = Counter(audit.get("predicted_by_id", {}).values())
    return {label: int(counts.get(label, 0)) for label in ("A", "B", "C", "D")}


def write_gepa_report(result_path: str | Path) -> dict[str, Any]:
    """Write compact machine- and human-readable summaries for one GEPA run."""
    result_path = Path(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    audits: dict[str, dict[str, Any]] = result.get("audit_by_instruction", {})
    if not audits:
        raise ValueError("The GEPA result has no post-hoc audits to summarize.")

    seed_instruction = result["all_accepted_candidates"][0]["instruction"]
    gepa_instruction = result["best_instruction"]
    audit_best_instruction = max(
        audits,
        key=lambda text: (audits[text]["accuracy"], -audits[text]["mean_loss"]),
    )
    seed_audit = audits[seed_instruction]
    gepa_audit = audits[gepa_instruction]
    audit_best = audits[audit_best_instruction]
    matching_proposals = [
        row
        for row in result.get("candidate_table", [])
        if row.get("instruction") == audit_best_instruction
    ]

    summary = {
        "interpretation": (
            "GEPA-selected and post-hoc-audit-best are reported separately. The post-hoc "
            "audit is exploratory and does not retroactively change GEPA acceptance."
        ),
        "proposal_count": result["proposal_count"],
        "accepted_proposal_count": result["candidate_count_accepted_including_seed"] - 1,
        "rejected_proposal_count": sum(
            row.get("status") == "rejected" for row in result.get("candidate_table", [])
        ),
        "unique_proposal_text_count": len(
            {row.get("instruction") for row in result.get("candidate_table", [])}
        ),
        "seed": {
            "instruction": seed_instruction,
            "validation_accuracy": seed_audit["accuracy"],
            "validation_mean_loss": seed_audit["mean_loss"],
            "per_class_accuracy": seed_audit["per_class_accuracy"],
            "prediction_counts": _prediction_counts(seed_audit),
        },
        "gepa_selected": {
            "candidate_idx": result["best_candidate_idx"],
            "instruction": gepa_instruction,
            "validation_accuracy": gepa_audit["accuracy"],
            "validation_mean_loss": gepa_audit["mean_loss"],
            "is_seed": gepa_instruction == seed_instruction,
        },
        "posthoc_audit_best": {
            "instruction": audit_best_instruction,
            "validation_accuracy": audit_best["accuracy"],
            "validation_mean_loss": audit_best["mean_loss"],
            "per_class_accuracy": audit_best["per_class_accuracy"],
            "prediction_counts": _prediction_counts(audit_best),
            "accuracy_gain_over_seed": audit_best["accuracy"] - seed_audit["accuracy"],
            "mean_loss_change_from_seed": audit_best["mean_loss"] - seed_audit["mean_loss"],
            "proposal_numbers": [row["proposal_number"] for row in matching_proposals],
            "gepa_statuses": [row["status"] for row in matching_proposals],
            "errors_fixed_vs_parent": (
                matching_proposals[0]["errors_fixed_count"] if matching_proposals else 0
            ),
            "new_errors_vs_parent": (
                matching_proposals[0]["new_errors_count"] if matching_proposals else 0
            ),
        },
        "all_proposals_tokenizer_compositional": all(
            bool(row.get("tokenizer_compositional"))
            for row in result.get("candidate_table", [])
        ),
    }

    run_dir = result_path.parent
    (run_dir / "gepa_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    fieldnames = [
        "proposal_number",
        "parent_candidate_idx",
        "candidate_idx",
        "status",
        "selected_best",
        "validation_accuracy",
        "validation_mean_loss",
        "slot_token_count",
        "tokenizer_compositional",
        "errors_fixed_count",
        "new_errors_count",
        "instruction",
    ]
    with (run_dir / "gepa_candidates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result.get("candidate_table", []))

    best = summary["posthoc_audit_best"]
    accepted_count = summary["accepted_proposal_count"]
    selected_label = (
        "seed"
        if summary["gepa_selected"]["is_seed"]
        else f"candidate {summary['gepa_selected']['candidate_idx']}"
    )
    acceptance_note = (
        "GEPA accepted no proposal under its strict minibatch improvement rule."
        if accepted_count == 0
        else f"GEPA accepted {accepted_count} proposal(s) under its strict minibatch improvement rule."
    )
    markdown = f"""# GEPA observation summary

- Proposals: {summary['proposal_count']} ({summary['accepted_proposal_count']} accepted, {summary['rejected_proposal_count']} rejected)
- GEPA-selected prompt: {selected_label} (validation accuracy {summary['gepa_selected']['validation_accuracy']:.4f})
- Post-hoc audit winner: proposal {', '.join(map(str, best['proposal_numbers']))} (validation accuracy {best['validation_accuracy']:.4f})
- Accuracy change versus seed: {best['accuracy_gain_over_seed']:+.4f}
- Mean label-loss change versus seed: {best['mean_loss_change_from_seed']:+.4f}
- Error transitions versus its parent: {best['errors_fixed_vs_parent']} fixed, {best['new_errors_vs_parent']} introduced
- All proposed instruction slots tokenizer-compositional: {summary['all_proposals_tokenizer_compositional']}

{acceptance_note} The post-hoc winner is exploratory and does not retroactively change GEPA's selected candidate.

## Post-hoc audit winner

> {best['instruction']}
"""
    (run_dir / "gepa_summary.md").write_text(markdown, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a completed GEPA observation run.")
    parser.add_argument("result", help="Path to gepa_result.json")
    args = parser.parse_args()
    summary = write_gepa_report(args.result)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
