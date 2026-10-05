"""Check that two GEPA runs are identical (tiered parity gate).

Tier 1: seed validation scores per example.
Tier 2: the ordered sequence of proposed instruction texts.
Tier 3: accept/reject decisions, per-candidate validation scores, and the
        GEPA-selected candidate's accuracy and loss.
Tier 4: the full callback event stream (reflection prompts and raw outputs
        included), reflection-call records, the post-hoc candidate audit,
        per-example label logits, and the serialized GEPA state.

Used to confirm that swapping the installed GEPA wheel for the editable fork,
or any later fork change, leaves reflective-only runs unchanged.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any


def _events(run_dir: Path) -> list[dict[str, Any]]:
    with (run_dir / "callback_events.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _of(events: list[dict[str, Any]], *names: str) -> list[dict[str, Any]]:
    return [event for event in events if event["event"] in names]


def _first_difference(left: list[Any], right: list[Any]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


_OUTPUT_PATTERN = re.compile(r"label_scores=(\{[^}]*\}), loss=([0-9.eE+-]+)")


def _per_example_outputs(reference: Path, candidate: Path) -> dict[str, Any]:
    """Compare GEPA's saved per-example outputs, including the four label logits."""
    subdir = "generated_best_outputs_valset"
    ref_files = {p.relative_to(reference / subdir) for p in (reference / subdir).rglob("*.json")}
    new_files = {p.relative_to(candidate / subdir) for p in (candidate / subdir).rglob("*.json")}
    common = sorted(ref_files & new_files)
    byte_identical = 0
    max_difference = 0.0
    for relative in common:
        ref_text = (reference / subdir / relative).read_text(encoding="utf-8")
        new_text = (candidate / subdir / relative).read_text(encoding="utf-8")
        byte_identical += ref_text == new_text
        for (ref_scores, ref_loss), (new_scores, new_loss) in zip(
            _OUTPUT_PATTERN.findall(ref_text), _OUTPUT_PATTERN.findall(new_text), strict=True
        ):
            a, b = ast.literal_eval(ref_scores), ast.literal_eval(new_scores)
            max_difference = max(
                max_difference,
                max(abs(a[label] - b[label]) for label in a),
                abs(float(ref_loss) - float(new_loss)),
            )
    return {
        "per_example_output_files": [len(ref_files), len(new_files)],
        "per_example_output_files_byte_identical": byte_identical,
        "max_abs_label_score_difference": max_difference,
    }


def compare(reference: Path, candidate: Path) -> dict[str, Any]:
    ref_events, new_events = _events(reference), _events(candidate)
    report: dict[str, Any] = {}

    seed_ref = _of(ref_events, "valset_evaluated")[0]["data"]["scores_by_val_id"]
    seed_new = _of(new_events, "valset_evaluated")[0]["data"]["scores_by_val_id"]
    report["tier1_seed_scores_identical"] = seed_ref == seed_new

    texts_ref = [e["data"]["new_instructions"] for e in _of(ref_events, "proposal_end")]
    texts_new = [e["data"]["new_instructions"] for e in _of(new_events, "proposal_end")]
    divergence = _first_difference(texts_ref, texts_new)
    report["tier2_proposal_texts_identical"] = divergence is None
    report["tier2_first_divergent_proposal"] = divergence
    if divergence is not None:
        report["tier2_divergence"] = {
            "reference": texts_ref[divergence] if divergence < len(texts_ref) else None,
            "candidate": texts_new[divergence] if divergence < len(texts_new) else None,
        }

    def decisions(events: list[dict[str, Any]]) -> list[tuple[int, str]]:
        return [
            (e["data"]["iteration"], e["event"])
            for e in _of(events, "candidate_accepted", "candidate_rejected")
        ]

    def val_scores(events: list[dict[str, Any]]) -> list[dict[str, float]]:
        return [e["data"]["scores_by_val_id"] for e in _of(events, "valset_evaluated")]

    selected_ref = json.loads((reference / "gepa_summary.json").read_text())["gepa_selected"]
    selected_new = json.loads((candidate / "gepa_summary.json").read_text())["gepa_selected"]
    keys = ("candidate_idx", "instruction", "validation_accuracy", "validation_mean_loss")
    report["tier3_decisions_identical"] = decisions(ref_events) == decisions(new_events)
    report["tier3_valset_scores_identical"] = val_scores(ref_events) == val_scores(new_events)
    report["tier3_selected_identical"] = all(selected_ref[k] == selected_new[k] for k in keys)
    report["selected_reference"] = {k: selected_ref[k] for k in keys}
    report["selected_candidate"] = {k: selected_new[k] for k in keys}

    # Tier 4: everything GEPA wrote, beyond the binary scores above.
    report["tier4_event_stream_identical"] = ref_events == new_events
    report["event_count"] = [len(ref_events), len(new_events)]
    result_ref = json.loads((reference / "gepa_result.json").read_text())
    result_new = json.loads((candidate / "gepa_result.json").read_text())
    report["tier4_reflection_calls_identical"] = result_ref["reflection"] == result_new["reflection"]
    report["tier4_candidate_audit_identical"] = (
        result_ref["audit_by_instruction"] == result_new["audit_by_instruction"]
    )
    outputs = _per_example_outputs(reference, candidate)
    report.update(outputs)
    report["tier4_per_example_outputs_identical"] = (
        outputs["per_example_output_files"][0] == outputs["per_example_output_files"][1]
        == outputs["per_example_output_files_byte_identical"]
        and outputs["max_abs_label_score_difference"] == 0.0
    )
    state_ref, state_new = reference / "gepa_state.bin", candidate / "gepa_state.bin"
    if state_ref.exists() and state_new.exists():
        report["tier4_gepa_state_identical"] = state_ref.read_bytes() == state_new.read_bytes()
    report["parity"] = all(
        value for key, value in report.items() if key.startswith("tier") and isinstance(value, bool)
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", default="artifacts/gepa_confirmation_mb8")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", default=None, help="Defaults to <candidate>/parity_report.json")
    args = parser.parse_args()
    report = compare(Path(args.reference), Path(args.candidate))
    output = Path(args.output or Path(args.candidate) / "parity_report.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    sys.exit(0 if report["parity"] else 1)


if __name__ == "__main__":
    main()
