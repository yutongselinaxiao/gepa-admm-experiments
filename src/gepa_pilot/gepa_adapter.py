from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from gepa.core.adapter import EvaluationBatch, GEPAAdapter

from .data import NewsExample
from .evaluator import CommonEvaluator


@dataclass(frozen=True)
class ClassificationOutput:
    example_id: str
    predicted_label: str
    gold_label: str
    label_scores: dict[str, float]
    loss: float
    correct: bool
    prompt_token_count: int


@dataclass(frozen=True)
class ClassificationTrajectory:
    example_id: str
    article: str
    current_instruction: str
    predicted_label: str
    gold_label: str
    label_scores: dict[str, float]
    score: float
    loss: float
    diagnostic: str


class AGNewsGEPAAdapter(
    GEPAAdapter[NewsExample, ClassificationTrajectory, ClassificationOutput]
):
    """GEPA adapter backed by the pilot's shared Qwen evaluator."""

    def __init__(
        self,
        evaluator: CommonEvaluator,
        *,
        evaluation_batch_size: int = 32,
        focus: str = "the article's central topic",
    ) -> None:
        if evaluation_batch_size < 1:
            raise ValueError("evaluation_batch_size must be positive.")
        self.evaluator = evaluator
        self.focus = focus
        self.evaluation_batch_size = evaluation_batch_size
        self.evaluation_invocations = 0
        self.examples_scored = 0

    def _diagnostic(self, predicted: str, gold: str) -> str:
        classes = self.evaluator.multi_label_classes
        if classes:  # multi-label: predicted / gold are letter sets over the candidate classes
            from .data import LABEL_TO_CLASS_ID

            name = lambda letters: ", ".join(classes[LABEL_TO_CLASS_ID[c]] for c in letters) or "none"  # noqa: E731
            missed, extra = [c for c in gold if c not in predicted], [c for c in predicted if c not in gold]
            if not missed and not extra:
                return f"Correctly identified all expressed classes ({name(gold)})."
            return (f"Predicted classes: {name(predicted)}. Gold classes: {name(gold)}. Missed: {name(missed)}. "
                    f"Wrongly added: {name(extra)}. Revise the general decision rule for {self.focus}.")
        names = self.evaluator.prompt_builder.label_names
        if predicted == gold:
            return f"Correctly classified the article as {gold} ({names[gold]})."
        return (
            f"Predicted {predicted} ({names[predicted]}) but the gold class is "
            f"{gold} ({names[gold]}). Revise the general decision rule to distinguish "
            f"{self.focus} from incidental entities and keywords."
        )

    def evaluate(
        self,
        batch: list[NewsExample],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[ClassificationTrajectory, ClassificationOutput]:
        if "instruction" not in candidate:
            raise KeyError("GEPA candidate must contain the 'instruction' component.")
        summary = self.evaluator.evaluate(
            batch,
            candidate["instruction"],
            batch_size=self.evaluation_batch_size,
        )
        self.evaluation_invocations += 1
        self.examples_scored += len(batch)

        outputs = [
            ClassificationOutput(
                example_id=record.example_id,
                predicted_label=record.predicted_label,
                gold_label=record.gold_label,
                label_scores=record.label_scores,
                loss=record.loss,
                correct=record.correct,
                prompt_token_count=record.prompt_token_count,
            )
            for record in summary.records
        ]
        scores = [float(output.correct) for output in outputs]
        trajectories = None
        if capture_traces:
            trajectories = [
                ClassificationTrajectory(
                    example_id=example.example_id,
                    article=example.article,
                    current_instruction=candidate["instruction"],
                    predicted_label=output.predicted_label,
                    gold_label=output.gold_label,
                    label_scores=output.label_scores,
                    score=float(output.correct),
                    loss=output.loss,
                    diagnostic=self._diagnostic(output.predicted_label, output.gold_label),
                )
                for example, output in zip(batch, outputs, strict=True)
            ]

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            num_metric_calls=len(batch),
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[ClassificationTrajectory, ClassificationOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        if eval_batch.trajectories is None:
            raise ValueError("Reflective dataset requires capture_traces=True.")
        unsupported = set(components_to_update) - {"instruction"}
        if unsupported:
            raise ValueError(f"Unsupported components: {sorted(unsupported)}")

        records = [
            {
                "article": trajectory.article,
                "current_instruction": trajectory.current_instruction,
                "assistant_prediction": trajectory.predicted_label,
                "gold_class": trajectory.gold_label,
                "label_scores": trajectory.label_scores,
                "score": trajectory.score,
                "label_loss": trajectory.loss,
                "feedback": trajectory.diagnostic,
            }
            for trajectory in eval_batch.trajectories
        ]
        return {component: records for component in components_to_update}

    def counters(self) -> dict[str, int]:
        return {
            "evaluation_invocations": self.evaluation_invocations,
            "examples_scored": self.examples_scored,
        }
