from __future__ import annotations

import math
import unicodedata
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .data import NewsExample
from .evaluator import CommonEvaluator


@dataclass(frozen=True)
class ADMMHyperparameters:
    rho: float
    learning_rate: float = 0.005
    outer_iterations: int = 30
    e_steps: int = 3
    train_batch_size: int = 16
    evaluation_batch_size: int = 32
    validation_frequency: int = 5
    projection_chunk_size: int = 16384

    def validate(self) -> None:
        numeric_positive = {
            "rho": self.rho,
            "learning_rate": self.learning_rate,
            "outer_iterations": self.outer_iterations,
            "e_steps": self.e_steps,
            "train_batch_size": self.train_batch_size,
            "evaluation_batch_size": self.evaluation_batch_size,
            "validation_frequency": self.validation_frequency,
            "projection_chunk_size": self.projection_chunk_size,
        }
        invalid = [name for name, value in numeric_positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"ADMM hyperparameters must be positive: {invalid}")


def build_allowed_token_ids(tokenizer: Any, embedding_rows: int) -> torch.Tensor:
    """Allow ordinary vocabulary tokens while excluding special/control tokens."""
    excluded = {int(token_id) for token_id in tokenizer.all_special_ids}
    for token_id, token in getattr(tokenizer, "added_tokens_decoder", {}).items():
        if bool(getattr(token, "special", False)):
            excluded.add(int(token_id))
    allowed: list[int] = []
    upper = min(int(embedding_rows), len(tokenizer))
    for token_id in range(upper):
        if token_id in excluded:
            continue
        token_text = str(tokenizer.convert_ids_to_tokens(token_id))
        if token_text.startswith("<|") and token_text.endswith("|>"):
            continue
        decoded = tokenizer.decode(
            [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if any(
            unicodedata.category(character) == "Cc"
            and character not in {"\n", "\r", "\t"}
            for character in decoded
        ):
            continue
        allowed.append(token_id)
    if not allowed:
        raise ValueError("No allowed vocabulary tokens remain after filtering.")
    return torch.tensor(allowed, dtype=torch.long)


def nearest_embedding_token_ids(
    targets: torch.Tensor,
    embedding_weight: torch.Tensor,
    allowed_token_ids: torch.Tensor,
    *,
    chunk_size: int = 16384,
) -> torch.Tensor:
    """Exact per-position Euclidean nearest-neighbor projection."""
    if targets.ndim != 2 or embedding_weight.ndim != 2:
        raise ValueError("targets and embedding_weight must both be rank-two tensors.")
    if targets.shape[1] != embedding_weight.shape[1]:
        raise ValueError("Target and vocabulary embedding dimensions do not match.")
    if allowed_token_ids.ndim != 1 or not len(allowed_token_ids):
        raise ValueError("allowed_token_ids must be a non-empty vector.")
    device = targets.device
    allowed = allowed_token_ids.to(device=device)
    target_float = targets.detach().float()
    best_scores = torch.full(
        (targets.shape[0],), -math.inf, dtype=torch.float32, device=device
    )
    best_ids = torch.empty(targets.shape[0], dtype=torch.long, device=device)
    for start in range(0, len(allowed), chunk_size):
        ids = allowed[start : start + chunk_size]
        vectors = embedding_weight.index_select(0, ids).detach().float()
        # Maximizing 2<t,w> - ||w||^2 is equivalent to minimizing ||t-w||^2.
        scores = 2.0 * (target_float @ vectors.T) - vectors.square().sum(dim=1)
        chunk_scores, chunk_indices = scores.max(dim=1)
        replace = chunk_scores > best_scores
        best_scores = torch.where(replace, chunk_scores, best_scores)
        best_ids = torch.where(replace, ids[chunk_indices], best_ids)
    return best_ids


def gold_indices(examples: Sequence[NewsExample], labels: Sequence[str], device: torch.device) -> torch.Tensor:
    lookup = {label: index for index, label in enumerate(labels)}
    return torch.tensor(
        [lookup[example.gold_label] for example in examples],
        dtype=torch.long,
        device=device,
    )


def summarize_scores(
    examples: Sequence[NewsExample], scores: torch.Tensor, labels: Sequence[str]
) -> dict[str, Any]:
    scores = scores.detach().float().cpu()
    targets = gold_indices(examples, labels, torch.device("cpu"))
    losses = F.cross_entropy(scores, targets, reduction="none")
    predictions = scores.argmax(dim=-1)
    correct = predictions.eq(targets)
    per_class_accuracy: dict[str, float] = {}
    per_class_loss: dict[str, float] = {}
    confusion = {gold: {predicted: 0 for predicted in labels} for gold in labels}
    for label_index, label in enumerate(labels):
        mask = targets.eq(label_index)
        per_class_accuracy[label] = float(correct[mask].float().mean())
        per_class_loss[label] = float(losses[mask].mean())
    records: dict[str, dict[str, Any]] = {}
    for row, example in enumerate(examples):
        predicted = labels[int(predictions[row])]
        confusion[example.gold_label][predicted] += 1
        records[example.example_id] = {
            "gold_label": example.gold_label,
            "predicted_label": predicted,
            "correct": bool(correct[row]),
            "loss": float(losses[row]),
        }
    return {
        "accuracy": float(correct.float().mean()),
        "mean_loss": float(losses.mean()),
        "per_class_accuracy": per_class_accuracy,
        "per_class_mean_loss": per_class_loss,
        "confusion_matrix": confusion,
        "records_by_id": records,
    }


def evaluate_soft_slot(
    evaluator: CommonEvaluator,
    examples: Sequence[NewsExample],
    reference_instruction: str,
    slot_embeddings: torch.Tensor,
    *,
    batch_size: int,
) -> dict[str, Any]:
    score_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            score_batches.append(
                evaluator.score_slot_embeddings(
                    batch, reference_instruction, slot_embeddings
                ).detach().cpu()
            )
    return summarize_scores(examples, torch.cat(score_batches), evaluator.labels)


def evaluate_discrete_slot(
    evaluator: CommonEvaluator,
    examples: Sequence[NewsExample],
    reference_instruction: str,
    slot_ids: torch.Tensor,
    *,
    batch_size: int,
) -> dict[str, Any]:
    score_batches: list[torch.Tensor] = []
    for start in range(0, len(examples), batch_size):
        score_batches.append(
            evaluator.score_fixed_slot_ids(
                examples[start : start + batch_size], reference_instruction, slot_ids
            )
        )
    return summarize_scores(examples, torch.cat(score_batches), evaluator.labels)


def decode_and_check_slot(
    evaluator: CommonEvaluator,
    reference_article: str,
    slot_ids: torch.Tensor,
) -> dict[str, Any]:
    ids = tuple(int(item) for item in slot_ids.detach().cpu().tolist())
    decoded_slot = evaluator.tokenizer.decode(
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    reconstructed_instruction = (
        decoded_slot[:-1] if decoded_slot.endswith("\n") else decoded_slot
    )
    try:
        encoding = evaluator.prompt_builder.encode(
            reference_article, reconstructed_instruction
        )
        retokenized = encoding.slot_ids
        error = None
    except Exception as exc:  # diagnostic path for pathological projected text
        retokenized = ()
        error = f"{type(exc).__name__}: {exc}"
    return {
        "decoded_slot_text": decoded_slot,
        "reconstructed_instruction": reconstructed_instruction,
        "retokenized_slot_ids": list(retokenized),
        "roundtrip_exact": tuple(retokenized) == ids,
        "reconstruction_error": error,
    }


def calibration_record(
    evaluator: CommonEvaluator,
    examples: Sequence[NewsExample],
    instruction: str,
    seed_embeddings: torch.Tensor,
    *,
    learning_rate: float,
    rho_grid: Sequence[float],
) -> dict[str, Any]:
    e = seed_embeddings.detach().float().clone().requires_grad_(True)
    scores = evaluator.score_slot_embeddings(examples, instruction, e)
    targets = gold_indices(examples, evaluator.labels, evaluator.device)
    task_loss = F.cross_entropy(scores.float(), targets)
    task_loss.backward()
    gradient = e.grad.detach()
    # One sign-normalized Adam-like displacement provides a reproducible local scale.
    displacement = -learning_rate * gradient.sign()
    squared_norm = float(displacement.square().sum())
    return {
        "batch_size": len(examples),
        "task_loss_at_seed": float(task_loss.detach()),
        "task_gradient_norm": float(gradient.norm()),
        "seed_embedding_norm": float(seed_embeddings.float().norm()),
        "probe_displacement_norm": math.sqrt(squared_norm),
        "probe_squared_residual": squared_norm,
        "rho_penalty_at_probe": {
            str(rho): 0.5 * float(rho) * squared_norm for rho in rho_grid
        },
    }


def hyperparameters_dict(config: ADMMHyperparameters) -> dict[str, Any]:
    return asdict(config)


@dataclass
class ADMMState:
    """Iterates of one fixed-length ADMM run over the slot of `reference_instruction`."""

    reference_instruction: str
    initial_ids: torch.Tensor
    e: torch.Tensor
    z: torch.Tensor
    u: torch.Tensor
    optimizer: torch.optim.Optimizer
    forward_calls: int = 0
    backward_calls: int = 0


class ADMMRefiner:
    """Scaled ADMM over one instruction slot: Adam E-steps, nearest-token Z-step, dual update.

    The slot length is the token length of whichever instruction initializes the
    state, so any GEPA candidate can serve as the starting point.
    """

    def __init__(
        self,
        evaluator: CommonEvaluator,
        hyperparameters: ADMMHyperparameters,
        allowed_token_ids: torch.Tensor,
    ) -> None:
        hyperparameters.validate()
        self.evaluator = evaluator
        self.hp = hyperparameters
        self.allowed_token_ids = allowed_token_ids
        self.embedding_weight = evaluator.model.get_input_embeddings().weight

    def slot_ids(self, instruction: str, reference_article: str) -> torch.Tensor:
        encoding = self.evaluator.prompt_builder.encode(reference_article, instruction)
        if not encoding.compositional:
            raise ValueError("Instruction does not tokenize compositionally inside the prompt.")
        return torch.tensor(encoding.slot_ids, dtype=torch.long, device=self.evaluator.device)

    def init_state(self, instruction: str, initial_ids: torch.Tensor) -> ADMMState:
        e = (
            self.embedding_weight.index_select(0, initial_ids)
            .detach()
            .float()
            .clone()
            .requires_grad_(True)
        )
        return ADMMState(
            reference_instruction=instruction,
            initial_ids=initial_ids.clone(),
            e=e,
            z=initial_ids.clone(),
            u=torch.zeros_like(e),
            optimizer=torch.optim.Adam([e], lr=self.hp.learning_rate),
        )

    def step(self, state: ADMMState, batch: Sequence[NewsExample]) -> tuple[float, torch.Tensor]:
        """One outer iteration. Returns the last E-step task loss and the previous z."""
        targets = gold_indices(batch, self.evaluator.labels, self.evaluator.device)
        ez_current = self.embedding_weight.index_select(0, state.z).detach().float()
        last_task_loss = math.nan
        for _ in range(self.hp.e_steps):
            state.optimizer.zero_grad(set_to_none=True)
            scores = self.evaluator.score_slot_embeddings(
                batch, state.reference_instruction, state.e
            )
            task_loss = F.cross_entropy(scores.float(), targets)
            penalty = 0.5 * self.hp.rho * (state.e - ez_current + state.u).square().sum()
            (task_loss + penalty).backward()
            state.optimizer.step()
            state.forward_calls += 1
            state.backward_calls += 1
            last_task_loss = float(task_loss.detach())

        previous_z = state.z.clone()
        with torch.no_grad():
            state.z = nearest_embedding_token_ids(
                state.e.detach() + state.u,
                self.embedding_weight,
                self.allowed_token_ids,
                chunk_size=self.hp.projection_chunk_size,
            )
            ez_new = self.embedding_weight.index_select(0, state.z).detach().float()
            state.u = state.u + state.e.detach() - ez_new
        return last_task_loss, previous_z

    def discrete_loss(
        self, state: ADMMState, ids: torch.Tensor, examples: Sequence[NewsExample]
    ) -> float:
        """Mean label cross-entropy of explicit slot IDs, scored in one forward batch."""
        scores = self.evaluator.score_fixed_slot_ids(
            examples, state.reference_instruction, ids
        )
        state.forward_calls += 1
        targets = gold_indices(examples, self.evaluator.labels, torch.device("cpu"))
        return float(F.cross_entropy(scores.detach().float().cpu(), targets))
