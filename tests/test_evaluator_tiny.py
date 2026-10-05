from __future__ import annotations

import os
import tempfile
import types
import unittest
from pathlib import Path

import torch
from transformers import AutoTokenizer

from gepa_pilot.admm import (
    build_allowed_token_ids,
    decode_and_check_slot,
    nearest_embedding_token_ids,
)
from gepa_pilot.config import PilotConfig
from gepa_pilot.data import load_examples_json
from gepa_pilot.evaluator import CommonEvaluator
from gepa_pilot.gepa_adapter import AGNewsGEPAAdapter
from gepa_pilot.gepa_tracking import ObservationTracker, ProposalCountStopper
from gepa_pilot.prompting import PromptBuilder


ROOT = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(ROOT, "configs", "pilot.yaml")
EXAMPLES_PATH = os.path.join(ROOT, "configs", "sanity_examples.json")


class TinyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 8) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.output = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.embedding

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        use_cache=False,
        logits_to_keep=0,
    ):
        del position_ids, use_cache
        hidden = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        if attention_mask is not None:
            hidden = hidden * attention_mask.unsqueeze(-1)
        # Give the toy model causal context so the final answer logit depends on
        # earlier editable-slot embeddings, as it does in a transformer.
        hidden = hidden.cumsum(dim=1)
        logits = self.output(hidden)
        if isinstance(logits_to_keep, torch.Tensor):
            logits = logits[:, logits_to_keep]
        elif logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        return types.SimpleNamespace(logits=logits)


class TinyEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = PilotConfig.from_yaml(CONFIG_PATH)
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.resolved_name_or_path,
            revision=None,
            local_files_only=True,
            use_fast=True,
        )
        builder = PromptBuilder(
            tokenizer,
            config.prompt.system_prompt,
            config.prompt.label_names,
            enable_thinking=config.prompt.enable_thinking,
        )
        model = TinyCausalLM(len(tokenizer))
        cls.config = config
        cls.examples = load_examples_json(EXAMPLES_PATH)
        cls.evaluator = CommonEvaluator(
            model, tokenizer, builder, device="cpu"
        )

    def test_records_and_summary_are_complete(self) -> None:
        result = self.evaluator.evaluate(
            self.examples, self.config.prompt.seed_instruction
        )
        self.assertEqual(len(result.records), len(self.examples))
        self.assertEqual(set(result.per_class_accuracy), {"A", "B", "C", "D"})
        self.assertEqual(set(result.confusion_matrix), {"A", "B", "C", "D"})
        self.assertTrue(all(len(record.label_scores) == 4 for record in result.records))
        self.assertTrue(all(record.raw_generation is None for record in result.records))

    def test_evaluation_is_deterministic(self) -> None:
        first = self.evaluator.evaluate(
            self.examples, self.config.prompt.seed_instruction
        )
        second = self.evaluator.evaluate(
            self.examples, self.config.prompt.seed_instruction
        )
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_batched_and_unbatched_evaluation_match(self) -> None:
        unbatched = self.evaluator.evaluate(
            self.examples, self.config.prompt.seed_instruction
        )
        batched = self.evaluator.evaluate(
            self.examples, self.config.prompt.seed_instruction, batch_size=3
        )
        self.assertEqual(unbatched.accuracy, batched.accuracy)
        self.assertEqual(
            [record.predicted_label for record in unbatched.records],
            [record.predicted_label for record in batched.records],
        )
        for left, right in zip(unbatched.records, batched.records):
            for label in ("A", "B", "C", "D"):
                self.assertAlmostEqual(
                    left.label_scores[label], right.label_scores[label], places=5
                )

    def test_input_id_and_embedding_paths_match(self) -> None:
        parity = self.evaluator.parity_check(
            self.examples, self.config.prompt.seed_instruction
        )
        self.assertEqual(parity.prediction_agreement, 1.0)
        self.assertEqual(parity.maximum_absolute_label_score_difference, 0.0)
        self.assertTrue(parity.answer_positions_identical)

    def test_fixed_slot_id_path_matches_ordinary_seed_path(self) -> None:
        ordinary_encodings, ordinary_scores = self.evaluator.score_examples(
            self.examples, self.config.prompt.seed_instruction
        )
        fixed_scores = self.evaluator.score_fixed_slot_ids(
            self.examples,
            self.config.prompt.seed_instruction,
            ordinary_encodings[0].slot_ids,
        )
        self.assertTrue(torch.equal(ordinary_scores, fixed_scores))

    def test_admm_projection_is_exact_nearest_neighbor(self) -> None:
        embeddings = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, 2.0]]
        )
        targets = torch.tensor([[0.9, 0.1], [0.1, 1.8]])
        allowed = torch.tensor([0, 1, 2])
        projected = nearest_embedding_token_ids(
            targets, embeddings, allowed, chunk_size=2
        )
        self.assertEqual(projected.tolist(), [1, 2])

    def test_allowed_vocabulary_and_seed_roundtrip(self) -> None:
        allowed = build_allowed_token_ids(
            self.evaluator.tokenizer,
            self.evaluator.model.get_input_embeddings().weight.shape[0],
        )
        self.assertTrue(set(self.evaluator.tokenizer.all_special_ids).isdisjoint(allowed.tolist()))
        encoding = self.evaluator.prompt_builder.encode(
            self.examples[0].article, self.config.prompt.seed_instruction
        )
        diagnostic = decode_and_check_slot(
            self.evaluator,
            self.examples[0].article,
            torch.tensor(encoding.slot_ids),
        )
        self.assertTrue(diagnostic["roundtrip_exact"])
        self.assertEqual(
            diagnostic["reconstructed_instruction"],
            self.config.prompt.seed_instruction,
        )

    def test_gradient_reaches_only_the_soft_slot_input(self) -> None:
        encoding = self.evaluator.prompt_builder.encode(
            self.examples[0].article, self.config.prompt.seed_instruction
        )
        slot_ids = torch.tensor(encoding.slot_ids, dtype=torch.long)
        slot_embeddings = (
            self.evaluator.model.get_input_embeddings()(slot_ids)
            .detach()
            .clone()
            .requires_grad_(True)
        )
        scores = self.evaluator.score_slot_embeddings(
            self.examples[:2], self.config.prompt.seed_instruction, slot_embeddings
        )
        scores.sum().backward()
        self.assertIsNotNone(slot_embeddings.grad)
        self.assertEqual(tuple(slot_embeddings.grad.shape), tuple(slot_embeddings.shape))
        self.assertGreater(float(slot_embeddings.grad.abs().sum()), 0.0)

    def test_gepa_adapter_returns_aligned_scores_outputs_and_traces(self) -> None:
        adapter = AGNewsGEPAAdapter(self.evaluator, evaluation_batch_size=3)
        batch = adapter.evaluate(
            self.examples,
            {"instruction": self.config.prompt.seed_instruction},
            capture_traces=True,
        )
        self.assertEqual(len(batch.outputs), len(self.examples))
        self.assertEqual(len(batch.scores), len(self.examples))
        self.assertEqual(len(batch.trajectories or []), len(self.examples))
        self.assertTrue(all(score in (0.0, 1.0) for score in batch.scores))
        self.assertEqual(batch.num_metric_calls, len(self.examples))
        self.assertEqual(adapter.counters()["examples_scored"], len(self.examples))

        reflective = adapter.make_reflective_dataset(
            {"instruction": self.config.prompt.seed_instruction},
            batch,
            ["instruction"],
        )
        self.assertEqual(set(reflective), {"instruction"})
        self.assertEqual(len(reflective["instruction"]), len(self.examples))
        required = {
            "article",
            "current_instruction",
            "assistant_prediction",
            "gold_class",
            "label_scores",
            "score",
            "label_loss",
            "feedback",
        }
        self.assertTrue(all(required <= set(row) for row in reflective["instruction"]))

    def test_gepa_tracker_records_proposal_and_stops_at_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tracker = ObservationTracker(Path(directory) / "events.jsonl")
            tracker.on_candidate_selected(
                {
                    "iteration": 1,
                    "candidate_idx": 0,
                    "candidate": {"instruction": "old"},
                    "score": 0.25,
                }
            )
            tracker.on_proposal_start(
                {
                    "iteration": 1,
                    "parent_candidate": {"instruction": "old"},
                    "components": ["instruction"],
                    "reflective_dataset": {"instruction": []},
                }
            )
            tracker.on_proposal_end(
                {
                    "iteration": 1,
                    "new_instructions": {"instruction": "new"},
                    "prompts": {"instruction": "reflect"},
                    "raw_lm_outputs": {"instruction": "new"},
                }
            )
            tracker.on_candidate_rejected(
                {"iteration": 1, "old_score": 1.0, "new_score": 1.0, "reason": "tie"}
            )
            self.assertEqual(tracker.proposal_count, 1)
            self.assertEqual(tracker.proposals[0]["status"], "rejected")
            self.assertEqual(tracker.proposals[0]["parent_candidate_idx"], 0)
            self.assertEqual(tracker.proposals[0]["parent_validation_score"], 0.25)
            self.assertTrue(ProposalCountStopper(tracker, maximum=1)(object()))
            self.assertTrue((Path(directory) / "events.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
