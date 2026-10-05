"""ADMM as a GEPA candidate proposer (passed to `optimize(custom_candidate_proposer=...)`).

GEPA still selects the parent, samples the reflection minibatch, accepts or rejects
the child on that minibatch, and runs full validation. This proposer only turns the
parent instruction into a child instruction:

1. Slot length is the parent's own token length.
2. ADMM trains on training batches that exclude GEPA's acceptance minibatch, so the
   child is not accepted on examples it was optimized on.
3. Every projected iterate that differs from the parent and survives an exact
   tokenizer round trip is scored on a held-out selection batch (also disjoint from
   the minibatch). The lowest-loss iterate is returned as text, but only if its
   loss is below the parent's on that batch.
4. Otherwise the parent text is returned; GEPA then rejects it as a tie.

Validation data is never touched here. Diagnostics go to a JSONL side log because
GEPA's proposal hook can only return text.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .admm import ADMMRefiner, decode_and_check_slot
from .data import NewsExample


class ADMMProposer:
    def __init__(
        self,
        refiner: ADMMRefiner,
        trainset: Sequence[NewsExample],
        *,
        selection_batch_size: int,
        seed: int,
        log_path: str | Path,
    ) -> None:
        hp = refiner.hp
        if selection_batch_size < 1:
            raise ValueError("selection_batch_size must be positive.")
        if len(trainset) < selection_batch_size + hp.train_batch_size:
            raise ValueError("Training set is too small for the selection and training batches.")
        self.refiner = refiner
        self.trainset = list(trainset)
        self.selection_batch_size = selection_batch_size
        self.seed = seed
        self.log_path = Path(log_path)
        self.reference_article = self.trainset[0].article
        self._ids_by_article: dict[str, list[str]] = defaultdict(list)
        for example in self.trainset:
            self._ids_by_article[example.article].append(example.example_id)
        self.call_count = 0
        self.forward_calls = 0
        self.backward_calls = 0
        self.outcomes: dict[str, int] = defaultdict(int)

    def _excluded_ids(self, records: Sequence[Mapping[str, Any]]) -> set[str]:
        excluded: set[str] = set()
        for record in records:
            matches = self._ids_by_article.get(record["article"])
            if not matches:
                raise KeyError("Reflective record article does not match any training example.")
            excluded.update(matches)
        return excluded

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        if set(components_to_update) != {"instruction"}:
            raise ValueError(f"Unsupported components: {sorted(components_to_update)}")
        call_index = self.call_count
        self.call_count += 1
        started = time.perf_counter()
        parent = candidate["instruction"]
        excluded = self._excluded_ids(reflective_dataset["instruction"])

        generator = torch.Generator(device="cpu").manual_seed(self.seed + call_index)
        pool = [example for example in self.trainset if example.example_id not in excluded]
        order = torch.randperm(len(pool), generator=generator).tolist()
        selection = [pool[index] for index in order[: self.selection_batch_size]]
        training = [pool[index] for index in order[self.selection_batch_size :]]

        record: dict[str, Any] = {
            "call_index": call_index,
            "parent_instruction": parent,
            "excluded_minibatch_ids": sorted(excluded),
            "selection_ids": [example.example_id for example in selection],
            "hyperparameters": {
                "rho": self.refiner.hp.rho,
                "learning_rate": self.refiner.hp.learning_rate,
                "outer_iterations": self.refiner.hp.outer_iterations,
                "e_steps": self.refiner.hp.e_steps,
                "train_batch_size": self.refiner.hp.train_batch_size,
                "selection_batch_size": self.selection_batch_size,
            },
        }
        try:
            parent_ids = self.refiner.slot_ids(parent, self.reference_article)
        except ValueError as exc:
            return self._finish(record, parent, "parent_not_compositional", started, str(exc))

        state = self.refiner.init_state(parent, parent_ids)
        record["parent_slot_ids"] = parent_ids.cpu().tolist()
        record["parent_selection_loss"] = self.refiner.discrete_loss(state, parent_ids, selection)

        cursor, training_order = 0, list(range(len(training)))
        batch_size = self.refiner.hp.train_batch_size
        scored: dict[tuple[int, ...], dict[str, Any]] = {}
        iterations: list[dict[str, Any]] = []
        for iteration in range(1, self.refiner.hp.outer_iterations + 1):
            if cursor + batch_size > len(training_order):
                training_order = torch.randperm(len(training), generator=generator).tolist()
                cursor = 0
            batch = [training[index] for index in training_order[cursor : cursor + batch_size]]
            cursor += batch_size
            task_loss, _ = self.refiner.step(state, batch)
            ids = tuple(state.z.cpu().tolist())
            ez = self.refiner.embedding_weight.index_select(0, state.z).detach().float()
            row: dict[str, Any] = {
                "iteration": iteration,
                "train_soft_task_loss": task_loss,
                "primal_residual_norm": float((state.e.detach() - ez).norm()),
                "changed_token_count_from_parent": int(state.z.ne(parent_ids).sum()),
            }
            if row["changed_token_count_from_parent"] and ids not in scored:
                roundtrip = decode_and_check_slot(self.refiner.evaluator, self.reference_article, state.z)
                entry = {
                    "iteration": iteration,
                    "slot_ids": list(ids),
                    "text": roundtrip["reconstructed_instruction"],
                    "roundtrip_exact": roundtrip["roundtrip_exact"],
                    "selection_loss": None,
                }
                if roundtrip["roundtrip_exact"] and roundtrip["reconstruction_error"] is None:
                    entry["selection_loss"] = self.refiner.discrete_loss(state, state.z, selection)
                scored[ids] = entry
                row["new_projection"] = entry
            iterations.append(row)

        record["iterations"] = iterations
        record["forward_calls"] = state.forward_calls
        record["backward_calls"] = state.backward_calls
        self.forward_calls += state.forward_calls
        self.backward_calls += state.backward_calls

        valid = [entry for entry in scored.values() if entry["selection_loss"] is not None]
        improving = [
            entry for entry in valid if entry["selection_loss"] < record["parent_selection_loss"]
        ]
        if not improving:
            if not scored:
                outcome = "no_token_change"
            elif not valid:
                outcome = "no_exact_roundtrip"
            else:
                outcome = "no_selection_improvement"
            return self._finish(record, parent, outcome, started)
        valid = improving
        best = min(valid, key=lambda entry: (entry["selection_loss"], entry["iteration"]))
        record["chosen"] = best
        return self._finish(record, best["text"], "proposed", started)

    def _finish(
        self,
        record: dict[str, Any],
        text: str,
        outcome: str,
        started: float,
        error: str | None = None,
    ) -> dict[str, str]:
        self.outcomes[outcome] += 1
        record["outcome"] = outcome
        record["error"] = error
        record["child_instruction"] = text
        record["wall_time_seconds"] = time.perf_counter() - started
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        chosen = record.get("chosen")
        changed = (
            sum(a != b for a, b in zip(chosen["slot_ids"], record["parent_slot_ids"]))
            if chosen
            else 0
        )
        print(
            f"ADMM proposal {record['call_index']}: {outcome}, "
            f"changed_tokens={changed}, {record['wall_time_seconds']:.1f}s"
        )
        return {"instruction": text}

    def manifest(self) -> dict[str, Any]:
        return {
            "proposer": "admm",
            "calls": self.call_count,
            "outcomes": dict(self.outcomes),
            "forward_calls": self.forward_calls,
            "backward_calls": self.backward_calls,
            "selection_batch_size": self.selection_batch_size,
            "seed": self.seed,
            "log_path": str(self.log_path),
        }
