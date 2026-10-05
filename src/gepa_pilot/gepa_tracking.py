from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gepa.core.callbacks import GEPACallback


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_safe(item) for item in value]
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return repr(value)


class ObservationTracker(GEPACallback):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.proposals: list[dict[str, Any]] = []
        self.val_evaluations: list[dict[str, Any]] = []
        self.frontier_updates: list[dict[str, Any]] = []
        self._pending_parent: dict[int, dict[str, Any]] = {}
        self._selected_parent: dict[int, dict[str, Any]] = {}

    @property
    def proposal_count(self) -> int:
        return len(self.proposals)

    def _append_event(self, name: str, event: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": name, "data": _json_safe(event)}) + "\n")

    def on_candidate_selected(self, event: dict[str, Any]) -> None:
        iteration = int(event["iteration"])
        self._selected_parent[iteration] = {
            "parent_candidate_idx": int(event["candidate_idx"]),
            "parent_validation_score": float(event["score"]),
        }
        self._append_event("candidate_selected", event)

    def on_proposal_start(self, event: dict[str, Any]) -> None:
        iteration = int(event["iteration"])
        self._pending_parent[iteration] = {
            "iteration": iteration,
            "parent_candidate": _json_safe(event["parent_candidate"]),
            "components": _json_safe(event["components"]),
            "reflective_dataset": _json_safe(event["reflective_dataset"]),
            "status": "proposed",
            **self._selected_parent.pop(iteration, {}),
        }
        self._append_event("proposal_start", event)

    def on_proposal_end(self, event: dict[str, Any]) -> None:
        iteration = int(event["iteration"])
        record = self._pending_parent.pop(iteration, {"iteration": iteration})
        record.update(
            {
                "new_instructions": _json_safe(event["new_instructions"]),
                "reflection_prompts": _json_safe(event["prompts"]),
                "raw_reflection_outputs": _json_safe(event["raw_lm_outputs"]),
            }
        )
        self.proposals.append(record)
        self._append_event("proposal_end", event)

    def on_evaluation_end(self, event: dict[str, Any]) -> None:
        if event.get("candidate_idx") is None and self.proposals:
            self.proposals[-1]["minibatch_scores_after"] = _json_safe(event["scores"])
        self._append_event(
            "evaluation_end",
            {
                key: value
                for key, value in event.items()
                if key not in {"outputs", "trajectories"}
            },
        )

    def on_candidate_accepted(self, event: dict[str, Any]) -> None:
        if self.proposals:
            self.proposals[-1].update(
                {
                    "status": "accepted",
                    "candidate_idx": int(event["new_candidate_idx"]),
                    "parent_ids": list(event["parent_ids"]),
                }
            )
        self._append_event("candidate_accepted", event)

    def on_candidate_rejected(self, event: dict[str, Any]) -> None:
        if self.proposals:
            self.proposals[-1].update(
                {"status": "rejected", "rejection_reason": str(event["reason"])}
            )
        self._append_event("candidate_rejected", event)

    def on_valset_evaluated(self, event: dict[str, Any]) -> None:
        record = {
            key: _json_safe(value)
            for key, value in event.items()
            if key != "outputs_by_val_id"
        }
        self.val_evaluations.append(record)
        self._append_event("valset_evaluated", record)

    def on_pareto_front_updated(self, event: dict[str, Any]) -> None:
        record = _json_safe(event)
        self.frontier_updates.append(record)
        self._append_event("pareto_front_updated", event)

    def snapshot(self) -> dict[str, Any]:
        return {
            "proposal_count": self.proposal_count,
            "proposals": self.proposals,
            "val_evaluations": self.val_evaluations,
            "frontier_updates": self.frontier_updates,
        }


class ProposalCountStopper:
    def __init__(self, tracker: ObservationTracker, maximum: int) -> None:
        self.tracker = tracker
        self.maximum = maximum

    def __call__(self, state: Any) -> bool:
        del state
        return self.tracker.proposal_count >= self.maximum


class IterationSafetyStopper:
    def __init__(self, maximum: int) -> None:
        self.maximum = maximum

    def __call__(self, state: Any) -> bool:
        return int(state.i) >= self.maximum
