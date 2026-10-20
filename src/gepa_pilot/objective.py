"""Task loss and metric shared by every optimizer, for single-label and multi-label tasks.

Single-label (the original setup): scores are [examples, labels] label scores; the loss is
cross-entropy (optionally class-weighted) and the metric is accuracy.

Multi-label: one example is one text with a set of gold classes (gold_label = their letters, e.g.
"BG"); scores are [examples, classes] log-odds that each class applies. The metric is the text's F1
between the predicted set {k : log-odds_k > 0} and the gold set, the same example-level F1 that is
reported; the differentiable loss is 1 - soft F1, with probabilities p = sigmoid(log-odds):
soft F1 = 2 sum_k p_k y_k / (sum_k p_k + sum_k y_k), which equals F1 when p is 0/1.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .data import LABEL_TO_CLASS_ID, NewsExample


def is_multilabel(evaluator: Any) -> bool:
    return bool(getattr(evaluator, "multi_label_classes", ()))


def gold_matrix(examples: Sequence[NewsExample], classes: int, device: torch.device) -> torch.Tensor:
    y = torch.zeros(len(examples), classes, device=device)
    for row, example in enumerate(examples):
        for letter in example.gold_label:
            y[row, LABEL_TO_CLASS_ID[letter]] = 1.0
    return y


def soft_f1(log_odds: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-example soft F1 from log-odds [examples, classes] and 0/1 targets."""
    p = torch.sigmoid(log_odds.float())
    return 2 * (p * y).sum(1) / (p.sum(1) + y.sum(1)).clamp_min(1e-8)


def hard_f1(predicted: set[int], gold: set[int]) -> float:
    if not predicted and not gold:
        return 1.0
    return 2 * len(predicted & gold) / (len(predicted) + len(gold))


def task_loss(
    evaluator: Any,
    examples: Sequence[NewsExample],
    scores: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean differentiable task loss of a batch."""
    if is_multilabel(evaluator):
        if class_weights is not None:
            raise NotImplementedError("Class-weighted (dual) losses are single-label only.")
        y = gold_matrix(examples, scores.shape[1], scores.device)
        return (1 - soft_f1(scores, y)).mean()
    labels = evaluator.labels
    targets = torch.tensor([labels.index(e.gold_label) for e in examples], dtype=torch.long, device=scores.device)
    if class_weights is None:
        return F.cross_entropy(scores.float(), targets)
    weights = class_weights.to(scores.device)[targets]
    per_example = F.cross_entropy(scores.float(), targets, reduction="none")
    return (weights * per_example).sum() / weights.sum()


def metric(evaluator: Any, examples: Sequence[NewsExample], scores: torch.Tensor) -> float:
    """Mean task metric of a batch: accuracy, or example-level F1 for multi-label tasks."""
    scores = scores.detach().float().cpu()
    if is_multilabel(evaluator):
        return sum(
            hard_f1({k for k in range(scores.shape[1]) if scores[row, k] > 0},
                    {LABEL_TO_CLASS_ID[c] for c in example.gold_label})
            for row, example in enumerate(examples)
        ) / len(examples)
    labels = evaluator.labels
    return sum(labels[int(scores[row].argmax())] == e.gold_label for row, e in enumerate(examples)) / len(examples)


def backward_task_loss(
    evaluator: Any,
    examples: Sequence[NewsExample],
    scores_fn: Any,
    extra_loss: Any = None,
    class_weights: torch.Tensor | None = None,
    max_rows: int = 32,
) -> float:
    """Backpropagate mean task loss (+ extra_loss) and return the task loss.

    Single-label: one forward and backward, exactly (task loss + extra loss).backward() as before.
    Multi-label: each text is scored once per class, so the batch is split into chunks of at most
    max_rows (text, class) prompts and the gradient is accumulated chunk by chunk; the loss is a mean
    over texts, so the result is the same gradient with a fraction of the activation memory."""
    # extra_loss: a tensor, a callable returning one, or a sequence of them. Callables are evaluated after
    # the forward pass, so the autograd graph (and the gradient accumulation order) is the same as before.
    extras = [] if extra_loss is None else list(extra_loss) if isinstance(extra_loss, (list, tuple)) else [extra_loss]
    if not is_multilabel(evaluator):
        loss = task_loss(evaluator, examples, scores_fn(examples), class_weights)
        extras = [extra() if callable(extra) else extra for extra in extras]
        total = loss
        for extra in extras:  # summed in the given order, as the callers did before
            total = total + extra
        total.backward()
        return float(loss.detach())
    extras = [extra() if callable(extra) else extra for extra in extras]
    if extras:
        sum(extras[1:], extras[0]).backward()
    per = max(1, max_rows // len(evaluator.multi_label_classes))
    total = 0.0
    for start in range(0, len(examples), per):
        chunk = examples[start : start + per]
        loss = task_loss(evaluator, chunk, scores_fn(chunk), class_weights) * (len(chunk) / len(examples))
        loss.backward()
        total += float(loss.detach())
    return total
