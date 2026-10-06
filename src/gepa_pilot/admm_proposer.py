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


def heldout_split(
    trainset: Sequence[NewsExample], excluded: set[str], selection_batch_size: int, seed: int
) -> tuple[list[NewsExample], list[NewsExample], torch.Generator]:
    """Shuffle the training examples outside GEPA's minibatch into a held-out selection
    batch and the remainder. Returns the generator so callers can keep drawing from it."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pool = [example for example in trainset if example.example_id not in excluded]
    order = torch.randperm(len(pool), generator=generator).tolist()
    selection = [pool[index] for index in order[:selection_batch_size]]
    training = [pool[index] for index in order[selection_batch_size:]]
    return selection, training, generator


def excluded_minibatch_ids(
    ids_by_article: Mapping[str, list[str]], records: Sequence[Mapping[str, Any]]
) -> set[str]:
    """Training IDs of GEPA's reflection minibatch, recovered from the reflective records."""
    excluded: set[str] = set()
    for record in records:
        matches = ids_by_article.get(record["article"])
        if not matches:
            raise KeyError("Reflective record article does not match any training example.")
        excluded.update(matches)
    return excluded


class ADMMProposer:
    def __init__(
        self,
        refiner: ADMMRefiner,
        trainset: Sequence[NewsExample],
        *,
        selection_batch_size: int,
        seed: int,
        log_path: str | Path,
        dual_step: float = 0.0,
        dual_batch_size: int = 64,
    ) -> None:
        hp = refiner.hp
        if dual_step < 0 or dual_batch_size < 1:
            raise ValueError("dual_step must be >= 0 and dual_batch_size >= 1.")
        self.dual_step = dual_step
        self.dual_batch_size = dual_batch_size
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
        # Child text -> held-out verdict (accuracy >= parent; loss < parent is required to propose).
        self.heldout_decisions: dict[str, bool] = {}

    def _excluded_ids(self, records: Sequence[Mapping[str, Any]]) -> set[str]:
        return excluded_minibatch_ids(self._ids_by_article, records)

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

        selection, training, generator = heldout_split(
            self.trainset, excluded, self.selection_batch_size, self.seed + call_index
        )

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
                "dual_step": self.dual_step,
                "dual_batch_size": self.dual_batch_size,
                "projection": self.refiner.hp.projection,
                "top_p": self.refiner.hp.top_p,
                "beam_width": self.refiner.hp.beam_width,
                "fluency_weight": self.refiner.hp.fluency_weight,
            },
        }
        try:
            encoding = self.refiner.encode(parent, self.reference_article)
        except ValueError as exc:
            return self._finish(record, parent, "parent_not_compositional", started, str(exc))
        parent_ids = torch.tensor(
            encoding.slot_ids, dtype=torch.long, device=self.refiner.evaluator.device
        )
        prefix_ids = tuple(encoding.prefix_ids)
        state = self.refiner.init_state(parent, parent_ids, prefix_ids=prefix_ids)
        record["parent_slot_ids"] = parent_ids.cpu().tolist()
        record["parent_slot_nll"] = self.refiner.slot_nll(prefix_ids, parent_ids)
        record["parent_selection_loss"], record["parent_selection_accuracy"] = (
            self.refiner.discrete_metrics(state, parent_ids, selection)
        )

        # Reward constraints (dual_step > 0): one multiplier per class, raised for classes
        # whose discrete accuracy on a dual batch falls below the mean class accuracy.
        # The dual batch comes from the ADMM training pool, so it is disjoint from both
        # GEPA's minibatch and the selection batch that judges the child.
        use_duals = self.dual_step > 0
        dual_batch = training[: self.dual_batch_size] if use_duals else []
        duals = torch.zeros(len(self.refiner.evaluator.labels))
        dual_cache: dict[tuple[int, ...], torch.Tensor] = {}
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
            task_loss, _ = self.refiner.step(
                state, batch, class_weights=1.0 + duals if use_duals else None
            )
            ids = tuple(state.z.cpu().tolist())
            ez = self.refiner.embedding_weight.index_select(0, state.z).detach().float()
            row: dict[str, Any] = {
                "iteration": iteration,
                "train_soft_task_loss": task_loss,
                "primal_residual_norm": float((state.e.detach() - ez).norm()),
                "changed_token_count_from_parent": int(state.z.ne(parent_ids).sum()),
            }
            if use_duals:
                if ids not in dual_cache:
                    dual_cache[ids] = self.refiner.per_class_accuracy(state, state.z, dual_batch)
                class_accuracy = dual_cache[ids]
                present = ~torch.isnan(class_accuracy)
                target = class_accuracy[present].mean()
                duals[present] = (
                    duals[present] + self.dual_step * (target - class_accuracy[present])
                ).clamp_min(0.0)
                row["dual_class_accuracy"] = class_accuracy.tolist()
                row["class_duals"] = duals.tolist()
            if row["changed_token_count_from_parent"] and ids not in scored:
                roundtrip = decode_and_check_slot(self.refiner.evaluator, self.reference_article, state.z)
                entry = {
                    "iteration": iteration,
                    "slot_ids": list(ids),
                    "text": roundtrip["reconstructed_instruction"],
                    "roundtrip_exact": roundtrip["roundtrip_exact"],
                    "selection_loss": None,
                    "selection_accuracy": None,
                }
                if roundtrip["roundtrip_exact"] and roundtrip["reconstruction_error"] is None:
                    entry["selection_loss"], entry["selection_accuracy"] = (
                        self.refiner.discrete_metrics(state, state.z, selection)
                    )
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
        best["slot_nll"] = self.refiner.slot_nll(
            prefix_ids, torch.tensor(best["slot_ids"], device=self.refiner.evaluator.device)
        )
        record["chosen"] = best
        record["heldout_accept"] = (
            best["selection_accuracy"] >= record["parent_selection_accuracy"]
        )
        self.heldout_decisions[best["text"]] = record["heldout_accept"]
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


class ADMMHeldOutAcceptance:
    """GEPA acceptance criterion that judges children on a held-out batch.

    `proposer` is anything with a `heldout_decisions` dict mapping child text to a verdict:
    an ADMMProposer, or a HeldOutJudgedReflectionLM for reflective GEPA.
    A child proposed by ADMM is accepted if, on the held-out selection batch (disjoint
    from GEPA's minibatch and from ADMM's training batches), its loss is below the
    parent's and its accuracy is at least the parent's. Any other proposal falls back
    to GEPA's default strict improvement on the minibatch.
    """

    def __init__(self, proposer: ADMMProposer) -> None:
        self.proposer = proposer

    def _verdict(self, proposal: Any) -> bool | None:
        return self.proposer.heldout_decisions.get(proposal.candidate.get("instruction"))

    def should_accept(self, proposal: Any, state: Any) -> bool:
        verdict = self._verdict(proposal)
        if verdict is not None:
            return verdict
        return sum(proposal.subsample_scores_after or []) > sum(
            proposal.subsample_scores_before or []
        )

    def reject_reason(self, proposal: Any, state: Any) -> str:
        if self._verdict(proposal) is False:
            return "ADMM child lowered held-out accuracy relative to its parent"
        return "New subsample score not better than parent (minibatch strict improvement)"


class PolishingReflectionLM:
    """GEPA reflection strategy: the default reflector rewrites, then ADMM polishes the rewrite.

    `inner` is GEPA's own StatelessReflectionLM, so the reflection step is identical to a
    reflective-only run. The polish starts from the reflected text and keeps it unless an
    ADMM iterate beats it on the proposer's held-out batch. Polish diagnostics go into the
    proposal metadata (and the proposer's JSONL log).
    """

    def __init__(self, inner: Any, polisher: ADMMProposer) -> None:
        self.inner = inner
        self.polisher = polisher

    def reflect(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> tuple[Any, "PolishingReflectionLM"]:
        proposal, next_inner = self.inner.reflect(candidate, reflective_dataset, components_to_update)
        self.inner = next_inner
        reflected = proposal.new_texts.get("instruction")
        if reflected is None:
            return proposal, self
        polished = self.polisher(
            {"instruction": reflected}, reflective_dataset, ["instruction"]
        )["instruction"]
        proposal.metadata["admm_polish"] = {
            "reflected_instruction": reflected,
            "polished_instruction": polished,
            "changed": polished != reflected,
            "call_index": self.polisher.call_count - 1,
        }
        proposal.new_texts["instruction"] = polished
        return proposal, self


class HeldOutJudgedReflectionLM:
    """Reflective GEPA with the same held-out acceptance test the ADMM proposer uses.

    The default reflector proposes the child unchanged. Parent and child are then scored
    on a held-out batch drawn exactly as in ADMMProposer (training examples outside GEPA's
    minibatch, same seed per call). The verdict, child loss < parent loss and child
    accuracy >= parent accuracy, is read by ADMMHeldOutAcceptance.
    """

    def __init__(
        self,
        inner: Any,
        evaluator: Any,
        trainset: Sequence[NewsExample],
        *,
        selection_batch_size: int,
        seed: int,
        log_path: str | Path,
    ) -> None:
        self.inner = inner
        self.evaluator = evaluator
        self.trainset = list(trainset)
        self.selection_batch_size = selection_batch_size
        self.seed = seed
        self.log_path = Path(log_path)
        self._ids_by_article: dict[str, list[str]] = defaultdict(list)
        for example in self.trainset:
            self._ids_by_article[example.article].append(example.example_id)
        self.heldout_decisions: dict[str, bool] = {}
        self.call_count = 0

    def reflect(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> tuple[Any, "HeldOutJudgedReflectionLM"]:
        proposal, next_inner = self.inner.reflect(candidate, reflective_dataset, components_to_update)
        self.inner = next_inner
        child = proposal.new_texts.get("instruction")
        if child is None:
            return proposal, self
        call_index = self.call_count
        self.call_count += 1
        excluded = excluded_minibatch_ids(self._ids_by_article, reflective_dataset["instruction"])
        selection, _, _ = heldout_split(
            self.trainset, excluded, self.selection_batch_size, self.seed + call_index
        )
        parent_summary = self.evaluator.evaluate(
            selection, candidate["instruction"], batch_size=len(selection)
        )
        child_summary = self.evaluator.evaluate(selection, child, batch_size=len(selection))
        accept = (
            child_summary.mean_loss < parent_summary.mean_loss
            and child_summary.accuracy >= parent_summary.accuracy
        )
        self.heldout_decisions[child] = accept
        record = {
            "call_index": call_index,
            "parent_instruction": candidate["instruction"],
            "child_instruction": child,
            "excluded_minibatch_ids": sorted(excluded),
            "selection_ids": [example.example_id for example in selection],
            "parent_selection_loss": parent_summary.mean_loss,
            "parent_selection_accuracy": parent_summary.accuracy,
            "child_selection_loss": child_summary.mean_loss,
            "child_selection_accuracy": child_summary.accuracy,
            "heldout_accept": accept,
        }
        proposal.metadata["heldout_judge"] = record
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return proposal, self
