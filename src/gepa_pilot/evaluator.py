from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch

from .data import NewsExample
from .prompting import PromptBuilder, PromptEncoding


@dataclass(frozen=True)
class EvaluationRecord:
    example_id: str
    gold_label: str
    predicted_label: str
    label_scores: dict[str, float]
    gold_log_probability: float
    loss: float
    correct: bool
    prompt_token_count: int
    answer_position: int
    error_tag: str
    raw_generation: str | None = None


@dataclass(frozen=True)
class EvaluationSummary:
    instruction: str
    accuracy: float
    mean_loss: float
    per_class_accuracy: dict[str, float]
    confusion_matrix: dict[str, dict[str, int]]
    records: tuple[EvaluationRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["records"] = [asdict(record) for record in self.records]
        return result


@dataclass(frozen=True)
class ParityResult:
    examples_checked: int
    maximum_absolute_label_score_difference: float
    mean_absolute_label_score_difference: float
    prediction_agreement: float
    answer_positions_identical: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CommonEvaluator:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        prompt_builder: PromptBuilder,
        *,
        device: str,
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.prompt_builder = prompt_builder
        self.device = torch.device(device)
        self.labels = prompt_builder.labels
        self.label_token_ids = prompt_builder.label_token_ids()
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer defines neither a pad token nor an EOS token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @staticmethod
    def _torch_dtype(name: str) -> torch.dtype:
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        try:
            return mapping[name]
        except KeyError as exc:
            raise ValueError(f"Unsupported dtype: {name}") from exc

    @classmethod
    def from_pretrained(
        cls,
        *,
        name_or_path: str,
        revision: str | None,
        local_files_only: bool,
        dtype: str,
        device: str,
        system_prompt: str,
        label_names: dict[str, str],
        enable_thinking: bool,
        input_heading: str = "Article",
    ) -> "CommonEvaluator":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            name_or_path,
            revision=revision,
            local_files_only=local_files_only,
            use_fast=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            revision=revision,
            local_files_only=local_files_only,
            dtype=cls._torch_dtype(dtype),
        )
        model.requires_grad_(False)
        model.to(device)
        builder = PromptBuilder(
            tokenizer,
            system_prompt,
            label_names,
            enable_thinking=enable_thinking,
            input_heading=input_heading,
        )
        return cls(model, tokenizer, builder, device=device)

    def _batch_encodings(
        self, encodings: Sequence[PromptEncoding]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_length = max(len(encoding.input_ids) for encoding in encodings)
        batch_size = len(encodings)
        input_ids = torch.full(
            (batch_size, max_length),
            int(self.tokenizer.pad_token_id),
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_length), dtype=torch.long, device=self.device
        )
        last_positions = torch.empty(batch_size, dtype=torch.long, device=self.device)
        for row, encoding in enumerate(encodings):
            length = len(encoding.input_ids)
            input_ids[row, :length] = torch.tensor(
                encoding.input_ids, dtype=torch.long, device=self.device
            )
            attention_mask[row, :length] = 1
            last_positions[row] = length - 1
        return input_ids, attention_mask, last_positions

    def _single_token_label_scores(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        last_positions: torch.Tensor,
        *,
        use_embeddings: bool,
    ) -> torch.Tensor:
        label_ids = torch.tensor(
            [self.label_token_ids[label][0] for label in self.labels],
            dtype=torch.long,
            device=self.device,
        )
        unique_positions, position_lookup = torch.unique(
            last_positions, sorted=True, return_inverse=True
        )
        kwargs: dict[str, Any] = {
            "attention_mask": attention_mask,
            "use_cache": False,
            "logits_to_keep": unique_positions,
        }
        if use_embeddings:
            kwargs["inputs_embeds"] = self.model.get_input_embeddings()(input_ids)
        else:
            kwargs["input_ids"] = input_ids
        outputs = self.model(**kwargs)
        batch_indices = torch.arange(input_ids.shape[0], device=self.device)
        answer_logits = outputs.logits[batch_indices, position_lookup]
        return answer_logits.index_select(-1, label_ids).float()

    def _single_token_scores_from_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        last_positions: torch.Tensor,
    ) -> torch.Tensor:
        label_ids = torch.tensor(
            [self.label_token_ids[label][0] for label in self.labels],
            dtype=torch.long,
            device=self.device,
        )
        unique_positions, position_lookup = torch.unique(
            last_positions, sorted=True, return_inverse=True
        )
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=unique_positions,
        )
        batch_indices = torch.arange(inputs_embeds.shape[0], device=self.device)
        answer_logits = outputs.logits[batch_indices, position_lookup]
        return answer_logits.index_select(-1, label_ids).float()

    def _compose_slot_embedding_batch(
        self,
        encodings: Sequence[PromptEncoding],
        slot_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Splice a shared differentiable slot between frozen prefix/suffix embeddings."""
        if slot_embeddings.ndim != 2:
            raise ValueError("slot_embeddings must have shape [slot_tokens, hidden_size].")
        if any(not encoding.compositional for encoding in encodings):
            raise ValueError("Cannot inject embeddings into a non-compositional prompt.")
        expected_slot_ids = encodings[0].slot_ids
        if any(encoding.slot_ids != expected_slot_ids for encoding in encodings):
            raise ValueError("All examples must share the same editable slot token IDs.")
        if slot_embeddings.shape[0] != len(expected_slot_ids):
            raise ValueError(
                f"Expected {len(expected_slot_ids)} slot embeddings, "
                f"received {slot_embeddings.shape[0]}."
            )

        embedding_layer = self.model.get_input_embeddings()
        embedding_dtype = embedding_layer.weight.dtype
        slot_embeddings = slot_embeddings.to(device=self.device, dtype=embedding_dtype)
        max_length = max(len(encoding.input_ids) for encoding in encodings)
        pad_id = torch.tensor([int(self.tokenizer.pad_token_id)], device=self.device)
        pad_embedding = embedding_layer(pad_id).detach()
        rows: list[torch.Tensor] = []
        last_positions: list[int] = []
        masks: list[torch.Tensor] = []

        for encoding in encodings:
            prefix_ids = torch.tensor(encoding.prefix_ids, dtype=torch.long, device=self.device)
            suffix_ids = torch.tensor(encoding.suffix_ids, dtype=torch.long, device=self.device)
            prefix = embedding_layer(prefix_ids).detach()
            suffix = embedding_layer(suffix_ids).detach()
            row = torch.cat((prefix, slot_embeddings, suffix), dim=0)
            if row.shape[0] != len(encoding.input_ids):
                raise AssertionError("Injected embedding length does not match token-ID length.")
            padding = max_length - row.shape[0]
            if padding:
                row = torch.cat((row, pad_embedding.expand(padding, -1)), dim=0)
            rows.append(row)
            masks.append(
                torch.cat(
                    (
                        torch.ones(len(encoding.input_ids), dtype=torch.long, device=self.device),
                        torch.zeros(padding, dtype=torch.long, device=self.device),
                    )
                )
            )
            last_positions.append(len(encoding.input_ids) - 1)

        return (
            torch.stack(rows),
            torch.stack(masks),
            torch.tensor(last_positions, dtype=torch.long, device=self.device),
        )

    def score_slot_embeddings(
        self,
        examples: Sequence[NewsExample],
        instruction: str,
        slot_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Score a shared soft slot while preserving gradients to `slot_embeddings`."""
        encodings = [
            self.prompt_builder.encode(example.article, instruction) for example in examples
        ]
        inputs_embeds, attention_mask, last_positions = self._compose_slot_embedding_batch(
            encodings, slot_embeddings
        )
        if not self._all_labels_are_single_token():
            key_bias = torch.where(attention_mask.bool(), 0.0, torch.finfo(torch.float32).min)
            positions = torch.arange(inputs_embeds.shape[1], device=self.device)[None].expand(len(encodings), -1)
            return self._label_sequence_scores(inputs_embeds, key_bias, positions, last_positions)
        return self._single_token_scores_from_embeddings(
            inputs_embeds, attention_mask, last_positions
        )

    def score_gated_slot_embeddings(
        self,
        examples: Sequence[NewsExample],
        reference_instruction: str,
        slot_embeddings: torch.Tensor,
        log_gates: torch.Tensor,
    ) -> torch.Tensor:
        """Score a soft slot of any length whose positions carry keep-gates g in (0, 1].

        log g is added to every query's attention logit for that slot key, and positions advance
        by g, so g -> 0 removes the token exactly as deleting it would (its key is unattended and
        later tokens shift back by one position). Gradients reach `slot_embeddings` and, through the
        attention mask, `log_gates`. The prefix and suffix come from `reference_instruction`.
        """
        if slot_embeddings.ndim != 2 or log_gates.shape != (slot_embeddings.shape[0],):
            raise ValueError("Need slot_embeddings [n, hidden] and log_gates [n].")
        encodings = [self.prompt_builder.encode(example.article, reference_instruction) for example in examples]
        if any(not encoding.compositional for encoding in encodings):
            raise ValueError("Cannot inject embeddings into a non-compositional prompt.")
        embedding_layer = self.model.get_input_embeddings()
        dtype = embedding_layer.weight.dtype
        slot = slot_embeddings.to(device=self.device, dtype=dtype)
        log_gates = log_gates.to(self.device).float().clamp_min(-1e4)
        gates = log_gates.detach().exp()
        n = slot.shape[0]
        lengths = [len(e.prefix_ids) + n + len(e.suffix_ids) for e in encodings]
        total = max(lengths)
        pad_embedding = embedding_layer(torch.tensor([int(self.tokenizer.pad_token_id)], device=self.device)).detach()
        rows, positions, masks, last_positions, key_biases = [], [], [], [], []
        causal = torch.ones(total, total, dtype=torch.bool, device=self.device).tril()
        for encoding, length in zip(encodings, lengths):
            p, s = len(encoding.prefix_ids), len(encoding.suffix_ids)
            prefix = embedding_layer(torch.tensor(encoding.prefix_ids, device=self.device)).detach()
            suffix = embedding_layer(torch.tensor(encoding.suffix_ids, device=self.device)).detach()
            row = torch.cat((prefix, slot, suffix, pad_embedding.expand(total - length, -1)), dim=0)
            slot_positions = p + torch.cumsum(gates, 0) - gates
            position = torch.cat((
                torch.arange(p, device=self.device, dtype=torch.float32), slot_positions,
                p + gates.sum() + torch.arange(s, device=self.device, dtype=torch.float32),
                torch.zeros(total - length, device=self.device),
            ))
            allowed = causal.clone()
            allowed[:, length:] = False
            allowed[length:, :] = torch.eye(total, dtype=torch.bool, device=self.device)[length:, :]
            bias = torch.zeros(total, device=self.device)
            bias = torch.cat((bias[:p], log_gates, bias[p + n:]))
            mask = torch.where(allowed, bias.unsqueeze(0), torch.finfo(torch.float32).min)
            rows.append(row)
            positions.append(position)
            masks.append(mask)
            last_positions.append(length - 1)
            key_biases.append(torch.where(torch.arange(total, device=self.device) < length, bias,
                                          torch.finfo(torch.float32).min))
        if not self._all_labels_are_single_token():
            return self._label_sequence_scores(torch.stack(rows), torch.stack(key_biases), torch.stack(positions),
                                               torch.tensor(last_positions, device=self.device))
        inputs_embeds = torch.stack(rows)
        attention_mask = torch.stack(masks).unsqueeze(1).to(dtype)
        label_ids = torch.tensor([self.label_token_ids[label][0] for label in self.labels], device=self.device)
        unique_positions, position_lookup = torch.unique(
            torch.tensor(last_positions, device=self.device), sorted=True, return_inverse=True
        )
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                             position_ids=torch.stack(positions), use_cache=False,
                             logits_to_keep=unique_positions)
        batch_indices = torch.arange(len(rows), device=self.device)
        return outputs.logits[batch_indices, position_lookup].index_select(-1, label_ids).float()

    def score_fixed_slot_ids(
        self,
        examples: Sequence[NewsExample],
        reference_instruction: str,
        slot_ids: Sequence[int] | torch.Tensor,
    ) -> torch.Tensor:
        """Score explicit slot token IDs without decoding and re-tokenizing them."""
        if not examples:
            raise ValueError("At least one example is required.")
        if isinstance(slot_ids, torch.Tensor):
            discrete_ids = tuple(int(item) for item in slot_ids.detach().cpu().tolist())
        else:
            discrete_ids = tuple(int(item) for item in slot_ids)
        if not discrete_ids:
            raise ValueError("slot_ids must not be empty.")

        encodings = [
            self.prompt_builder.encode(example.article, reference_instruction)
            for example in examples
        ]
        if any(not encoding.compositional for encoding in encodings):
            raise ValueError("Cannot replace IDs in a non-compositional prompt.")
        expected_length = len(encodings[0].slot_ids)
        if len(discrete_ids) != expected_length:
            raise ValueError(
                f"Expected {expected_length} fixed slot IDs, received {len(discrete_ids)}."
            )
        rows = [
            encoding.prefix_ids + discrete_ids + encoding.suffix_ids
            for encoding in encodings
        ]
        if not self._all_labels_are_single_token():
            with torch.inference_mode():
                return self._ids_label_sequence_scores(rows).detach().cpu()
        max_length = max(map(len, rows))
        input_ids = torch.full(
            (len(rows), max_length),
            int(self.tokenizer.pad_token_id),
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        last_positions = torch.empty(len(rows), dtype=torch.long, device=self.device)
        for row_index, ids in enumerate(rows):
            input_ids[row_index, : len(ids)] = torch.tensor(
                ids, dtype=torch.long, device=self.device
            )
            attention_mask[row_index, : len(ids)] = 1
            last_positions[row_index] = len(ids) - 1
        with torch.inference_mode():
            scores = self._single_token_label_scores(
                input_ids,
                attention_mask,
                last_positions,
                use_embeddings=False,
            )
        return scores.detach().cpu()

    def _all_labels_are_single_token(self) -> bool:
        return all(len(ids) == 1 for ids in self.label_token_ids.values())

    def _label_sequence_scores(
        self,
        inputs_embeds: torch.Tensor,
        key_bias: torch.Tensor,
        positions: torch.Tensor,
        last_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Summed teacher-forced log-likelihood of every label's token sequence, [batch, labels].

        inputs_embeds: right-padded prompts [B, T, d] (gradients allowed). key_bias: [B, T] additive
        attention bias per key (0 = attend, very negative = padding, log g for gated slots).
        positions: [B, T] (float allowed). last_positions: [B] index of each prompt's last token.
        The prompt runs once; its KV cache is repeated per label and only the label tokens after the
        first are run on top, and logits are computed only at the positions that predict label tokens.
        """
        batch, length, _ = inputs_embeds.shape
        device, dtype = inputs_embeds.device, inputs_embeds.dtype
        negative = torch.finfo(torch.float32).min
        index = torch.arange(length, device=device)
        allowed = (index[None, :] <= index[:, None])[None] & (index[None, None, :] <= last_positions[:, None, None])
        allowed = allowed | (index[None, :, None] == index[None, None, :])  # padded queries see themselves
        mask = torch.where(allowed, key_bias.float()[:, None, :], negative)
        output = self.model.model(inputs_embeds=inputs_embeds, attention_mask=mask[:, None].to(dtype),
                                  position_ids=positions, use_cache=True)
        rows = torch.arange(batch, device=device)
        first_log_probs = self.model.lm_head(output.last_hidden_state[rows, last_positions]).float().log_softmax(-1)
        label_ids = [self.label_token_ids[label] for label in self.labels]
        scores = first_log_probs[:, [ids[0] for ids in label_ids]]
        extra = max(len(ids) for ids in label_ids) - 1
        if extra == 0:
            return scores
        labels = len(label_ids)
        continuation = torch.full((labels, extra), int(self.tokenizer.pad_token_id), device=device)
        for k, ids in enumerate(label_ids):
            if len(ids) > 1:
                continuation[k, : len(ids) - 1] = torch.tensor(ids[:-1], device=device)
        cache = output.past_key_values
        cache.reorder_cache(rows.repeat_interleave(labels))
        embeds = self.model.get_input_embeddings()(continuation).to(dtype).repeat(batch, 1, 1)
        prompt_part = torch.where(index[None, :] <= last_positions[:, None], key_bias.float(), negative)
        prompt_part = prompt_part.repeat_interleave(labels, 0)[:, None, :].expand(-1, extra, -1)
        causal = torch.where(torch.ones(extra, extra, dtype=torch.bool, device=device).tril(), 0.0, negative)
        mask2 = torch.cat((prompt_part, causal[None].expand(batch * labels, -1, -1)), dim=-1)
        start = positions[rows, last_positions].float().repeat_interleave(labels)
        positions2 = start[:, None] + 1 + torch.arange(extra, device=device, dtype=torch.float32)[None]
        if not torch.is_floating_point(positions):
            positions2 = positions2.long()
        hidden = self.model.model(inputs_embeds=embeds, attention_mask=mask2[:, None].to(dtype),
                                  position_ids=positions2, past_key_values=cache, use_cache=False).last_hidden_state
        log_probs = self.model.lm_head(hidden).float().log_softmax(-1).view(batch, labels, extra, -1)
        tail = []
        for k, ids in enumerate(label_ids):
            term = torch.zeros(batch, device=device)
            for j, token in enumerate(ids[1:]):
                term = term + log_probs[:, k, j, token]
            tail.append(term)
        return scores + torch.stack(tail, dim=1)

    def _ids_label_sequence_scores(self, rows: Sequence[Sequence[int]]) -> torch.Tensor:
        """_label_sequence_scores for explicit token-ID prompts (no gradients needed by callers)."""
        width = max(map(len, rows))
        ids = torch.full((len(rows), width), int(self.tokenizer.pad_token_id), device=self.device)
        key_bias = torch.full((len(rows), width), torch.finfo(torch.float32).min, device=self.device)
        for r, row in enumerate(rows):
            ids[r, : len(row)] = torch.tensor(row, device=self.device)
            key_bias[r, : len(row)] = 0.0
        positions = torch.arange(width, device=self.device)[None].expand(len(rows), -1)
        last = torch.tensor([len(row) - 1 for row in rows], device=self.device)
        return self._label_sequence_scores(self.model.get_input_embeddings()(ids), key_bias, positions, last)

    def _multi_token_label_scores(self, encodings: Sequence[PromptEncoding]) -> torch.Tensor:
        rows: list[tuple[int, ...]] = []
        prompt_lengths: list[int] = []
        label_lengths: list[int] = []
        for encoding in encodings:
            for label in self.labels:
                label_ids = self.label_token_ids[label]
                rows.append(encoding.input_ids + label_ids)
                prompt_lengths.append(len(encoding.input_ids))
                label_lengths.append(len(label_ids))

        max_length = max(map(len, rows))
        input_ids = torch.full(
            (len(rows), max_length),
            int(self.tokenizer.pad_token_id),
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row_index, ids in enumerate(rows):
            input_ids[row_index, : len(ids)] = torch.tensor(ids, device=self.device)
            attention_mask[row_index, : len(ids)] = 1

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        log_probs = outputs.logits.float().log_softmax(dim=-1)
        scores: list[torch.Tensor] = []
        for row_index, (prompt_length, label_length) in enumerate(
            zip(prompt_lengths, label_lengths)
        ):
            label = rows[row_index][-label_length:]
            score = torch.zeros((), dtype=torch.float32, device=self.device)
            for offset, token_id in enumerate(label):
                score = score + log_probs[row_index, prompt_length + offset - 1, token_id]
            scores.append(score)
        return torch.stack(scores).reshape(len(encodings), len(self.labels))

    def score_examples(
        self, examples: Sequence[NewsExample], instruction: str
    ) -> tuple[list[PromptEncoding], torch.Tensor]:
        if not examples:
            raise ValueError("At least one example is required.")
        encodings = [
            self.prompt_builder.encode(example.article, instruction) for example in examples
        ]
        with torch.inference_mode():
            if self._all_labels_are_single_token():
                input_ids, attention_mask, last_positions = self._batch_encodings(encodings)
                scores = self._single_token_label_scores(
                    input_ids,
                    attention_mask,
                    last_positions,
                    use_embeddings=False,
                )
            else:
                scores = self._ids_label_sequence_scores([encoding.input_ids for encoding in encodings])
        return encodings, scores.detach().cpu()

    def evaluate(
        self,
        examples: Sequence[NewsExample],
        instruction: str,
        *,
        batch_size: int | None = None,
        include_generation: bool = False,
        generation_max_new_tokens: int = 8,
    ) -> EvaluationSummary:
        if batch_size is not None:
            if batch_size < 1:
                raise ValueError("batch_size must be positive.")
            if len(examples) > batch_size:
                records: list[EvaluationRecord] = []
                for start in range(0, len(examples), batch_size):
                    chunk = examples[start : start + batch_size]
                    records.extend(
                        self.evaluate(
                            chunk,
                            instruction,
                            include_generation=include_generation,
                            generation_max_new_tokens=generation_max_new_tokens,
                        ).records
                    )
                return self._summarize_records(instruction, records)

        encodings, scores = self.score_examples(examples, instruction)
        log_probabilities = scores.log_softmax(dim=-1)
        predictions = scores.argmax(dim=-1).tolist()
        records: list[EvaluationRecord] = []
        generations = (
            self.diagnostic_generations(
                examples,
                instruction,
                max_new_tokens=generation_max_new_tokens,
            )
            if include_generation
            else {}
        )

        for row, (example, encoding, prediction_index) in enumerate(
            zip(examples, encodings, predictions)
        ):
            gold_index = self.labels.index(example.gold_label)
            predicted_label = self.labels[prediction_index]
            loss = -float(log_probabilities[row, gold_index])
            correct = predicted_label == example.gold_label
            records.append(
                EvaluationRecord(
                    example_id=example.example_id,
                    gold_label=example.gold_label,
                    predicted_label=predicted_label,
                    label_scores={
                        label: float(scores[row, index])
                        for index, label in enumerate(self.labels)
                    },
                    gold_log_probability=-loss,
                    loss=loss,
                    correct=correct,
                    prompt_token_count=len(encoding.input_ids),
                    answer_position=encoding.answer_position,
                    error_tag="correct"
                    if correct
                    else f"{example.gold_label}->{predicted_label}",
                    raw_generation=generations.get(example.example_id),
                )
            )

        return self._summarize_records(instruction, records)

    def _summarize_records(
        self, instruction: str, records: Sequence[EvaluationRecord]
    ) -> EvaluationSummary:
        if not records:
            raise ValueError("At least one evaluation record is required.")
        confusion = {
            gold: {predicted: 0 for predicted in self.labels} for gold in self.labels
        }
        per_class_accuracy: dict[str, float] = {}
        for record in records:
            confusion[record.gold_label][record.predicted_label] += 1
        for label in self.labels:
            class_records = [record for record in records if record.gold_label == label]
            per_class_accuracy[label] = (
                sum(record.correct for record in class_records) / len(class_records)
                if class_records
                else math.nan
            )

        return EvaluationSummary(
            instruction=instruction,
            accuracy=sum(record.correct for record in records) / len(records),
            mean_loss=sum(record.loss for record in records) / len(records),
            per_class_accuracy=per_class_accuracy,
            confusion_matrix=confusion,
            records=tuple(records),
        )

    def diagnostic_generations(
        self,
        examples: Sequence[NewsExample],
        instruction: str,
        *,
        max_new_tokens: int = 8,
    ) -> dict[str, str]:
        """Generate short unconstrained outputs without changing the primary score."""
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive.")
        generations: dict[str, str] = {}
        encodings = [
            self.prompt_builder.encode(example.article, instruction) for example in examples
        ]
        with torch.inference_mode():
            for example, encoding in zip(examples, encodings):
                input_ids = torch.tensor(
                    [encoding.input_ids], dtype=torch.long, device=self.device
                )
                attention_mask = torch.ones_like(input_ids)
                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=int(self.tokenizer.pad_token_id),
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                new_ids = generated[0, input_ids.shape[1] :]
                generations[example.example_id] = self.tokenizer.decode(
                    new_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
        return generations

    def parity_check(self, examples: Sequence[NewsExample], instruction: str) -> ParityResult:
        encodings = [
            self.prompt_builder.encode(example.article, instruction) for example in examples
        ]
        input_ids, attention_mask, last_positions = self._batch_encodings(encodings)
        if not self._all_labels_are_single_token():
            raise NotImplementedError("Embedding parity currently requires single-token labels.")
        with torch.inference_mode():
            id_scores = self._single_token_label_scores(
                input_ids, attention_mask, last_positions, use_embeddings=False
            )
            seed_slot_ids = torch.tensor(
                encodings[0].slot_ids, dtype=torch.long, device=self.device
            )
            seed_slot_embeddings = self.model.get_input_embeddings()(seed_slot_ids)
            embedding_scores = self.score_slot_embeddings(
                examples, instruction, seed_slot_embeddings
            )
        differences = (id_scores - embedding_scores).abs()
        agreement = (id_scores.argmax(-1) == embedding_scores.argmax(-1)).float().mean()
        return ParityResult(
            examples_checked=len(examples),
            maximum_absolute_label_score_difference=float(differences.max()),
            mean_absolute_label_score_difference=float(differences.mean()),
            prediction_agreement=float(agreement),
            answer_positions_identical=all(
                encoding.answer_position == len(encoding.input_ids) - 1
                for encoding in encodings
            ),
        )
