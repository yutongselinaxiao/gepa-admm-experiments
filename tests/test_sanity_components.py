from __future__ import annotations

import os
import unittest

from transformers import AutoTokenizer

from gepa_pilot.config import PilotConfig
from gepa_pilot.data import (
    load_ag_news_test_split,
    load_examples_json,
    stratified_split_indices,
)
from gepa_pilot.prompting import PromptBuilder


ROOT = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(ROOT, "configs", "pilot.yaml")
EXAMPLES_PATH = os.path.join(ROOT, "configs", "sanity_examples.json")


class PromptSanityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = PilotConfig.from_yaml(CONFIG_PATH)
        cls.examples = load_examples_json(EXAMPLES_PATH)
        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.config.model.resolved_name_or_path,
            revision=None,
            local_files_only=True,
            use_fast=True,
        )
        cls.builder = PromptBuilder(
            cls.tokenizer,
            cls.config.prompt.system_prompt,
            cls.config.prompt.label_names,
            enable_thinking=cls.config.prompt.enable_thinking,
        )

    def test_labels_are_single_tokens(self) -> None:
        token_ids = self.builder.label_token_ids()
        self.assertEqual(tuple(token_ids), ("A", "B", "C", "D"))
        self.assertTrue(all(len(ids) == 1 for ids in token_ids.values()))

    def test_seed_prompt_is_compositional_and_roundtrips(self) -> None:
        encodings = self.builder.assert_seed_gates(
            self.config.prompt.seed_instruction,
            [example.article for example in self.examples],
        )
        self.assertEqual(len({encoding.slot_ids for encoding in encodings}), 1)
        self.assertTrue(all(encoding.compositional for encoding in encodings))
        self.assertTrue(all(encoding.full_roundtrip_exact for encoding in encodings))
        self.assertTrue(all(encoding.slot_roundtrip_exact for encoding in encodings))

    def test_answer_position_is_last_prompt_token(self) -> None:
        for example in self.examples:
            encoding = self.builder.encode(
                example.article, self.config.prompt.seed_instruction
            )
            self.assertEqual(encoding.answer_position, len(encoding.input_ids) - 1)

    def test_official_test_split_is_locked(self) -> None:
        with self.assertRaises(PermissionError):
            load_ag_news_test_split(self.config.data.dataset_name)

    def test_development_split_is_stratified_and_disjoint(self) -> None:
        labels = [class_id for class_id in range(4) for _ in range(20)]
        train, validation = stratified_split_indices(labels, 16, 12, seed=19)
        self.assertTrue(set(train).isdisjoint(validation))
        self.assertEqual([sum(labels[index] == c for index in train) for c in range(4)], [4] * 4)
        self.assertEqual(
            [sum(labels[index] == c for index in validation) for c in range(4)],
            [3] * 4,
        )


if __name__ == "__main__":
    unittest.main()
