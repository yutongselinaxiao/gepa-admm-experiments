from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


SLOT_MARKER = "__GEPA_EDITABLE_SLOT_7F3B9D2A__"


@dataclass(frozen=True)
class PromptEncoding:
    rendered_text: str
    input_ids: tuple[int, ...]
    prefix_ids: tuple[int, ...]
    slot_ids: tuple[int, ...]
    suffix_ids: tuple[int, ...]
    slot_start: int | None
    slot_end: int | None
    answer_position: int
    compositional: bool
    full_roundtrip_exact: bool
    slot_roundtrip_exact: bool


def label_list(labels: Sequence[str]) -> str:
    """'A, B, C, or D' for four labels; 'A or B' for two."""
    if len(labels) == 2:
        return f"{labels[0]} or {labels[1]}"
    return ", ".join(labels[:-1]) + f", or {labels[-1]}"


class PromptBuilder:
    """Construct the single editable instruction slot and verify token boundaries."""

    def __init__(
        self,
        tokenizer: Any,
        system_prompt: str,
        label_names: dict[str, str],
        *,
        enable_thinking: bool = False,
        input_heading: str = "Article",
        answer_style: str = "letter",
    ) -> None:
        self.tokenizer = tokenizer
        self.input_heading = input_heading
        self.system_prompt = system_prompt
        self.label_names = label_names
        self.enable_thinking = enable_thinking
        if answer_style not in ("letter", "name"):
            raise ValueError(f"Unknown answer_style: {answer_style!r}")
        self.answer_style = answer_style

    @property
    def answers(self) -> dict[str, str]:
        """The text the model must output for each label: the letter itself, or the label name."""
        if self.answer_style == "letter":
            return {label: label for label in self.label_names}
        return dict(self.label_names)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.label_names)

    def label_token_ids(self) -> dict[str, tuple[int, ...]]:
        result: dict[str, tuple[int, ...]] = {}
        for label in self.labels:
            ids = tuple(self.tokenizer.encode(self.answers[label], add_special_tokens=False))
            if not ids:
                raise ValueError(f"Label {label!r} tokenized to an empty sequence.")
            result[label] = ids
        return result

    def _messages(self, article: str, instruction: str) -> list[dict[str, str]]:
        # article: the input text, whatever the task calls it (see input_heading).
        if self.answer_style == "letter":
            mapping = "\n".join(
                f"{label} = {class_name}" for label, class_name in self.label_names.items()
            )
            label_block = f"### Label mapping\n{mapping}\n\n"
        else:
            label_block = "### Labels\n" + "\n".join(self.label_names.values()) + "\n\n"
        user_content = (
            "### Editable instruction\n"
            f"{instruction}"
            "### End editable instruction\n\n"
            f"{label_block}"
            f"### {self.input_heading}\n"
            f"{article}\n\n"
            f"Respond with exactly one label: {label_list(list(self.answers.values()))}."
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _render(self, article: str, instruction: str) -> str:
        return str(
            self.tokenizer.apply_chat_template(
                self._messages(article, instruction),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        )

    def render_parts(self, article: str) -> tuple[str, str]:
        rendered = self._render(article, SLOT_MARKER)
        if rendered.count(SLOT_MARKER) != 1:
            raise ValueError("The chat template did not preserve the unique editable-slot marker.")
        prefix, suffix = rendered.split(SLOT_MARKER)
        return prefix, suffix

    def encode(self, article: str, instruction: str) -> PromptEncoding:
        if SLOT_MARKER in article or SLOT_MARKER in instruction:
            raise ValueError("Input contains the reserved editable-slot marker.")

        prefix_text, suffix_text = self.render_parts(article)
        # The newline belongs to the editable span. For the seed this prevents its
        # final punctuation from merging with the fixed suffix under Qwen BPE.
        slot_text = f"{instruction}\n"
        rendered_text = f"{prefix_text}{slot_text}{suffix_text}"

        encode = lambda text: tuple(self.tokenizer.encode(text, add_special_tokens=False))
        full_ids = encode(rendered_text)
        prefix_ids = encode(prefix_text)
        slot_ids = encode(slot_text)
        suffix_ids = encode(suffix_text)
        compositional = full_ids == prefix_ids + slot_ids + suffix_ids

        decoded_full = self.tokenizer.decode(
            full_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        decoded_slot = self.tokenizer.decode(
            slot_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        full_roundtrip = encode(decoded_full) == full_ids
        slot_roundtrip = encode(decoded_slot) == slot_ids

        slot_start = len(prefix_ids) if compositional else None
        slot_end = len(prefix_ids) + len(slot_ids) if compositional else None
        if not full_ids:
            raise ValueError("Rendered prompt tokenized to an empty sequence.")

        return PromptEncoding(
            rendered_text=rendered_text,
            input_ids=full_ids,
            prefix_ids=prefix_ids,
            slot_ids=slot_ids,
            suffix_ids=suffix_ids,
            slot_start=slot_start,
            slot_end=slot_end,
            answer_position=len(full_ids) - 1,
            compositional=compositional,
            full_roundtrip_exact=full_roundtrip,
            slot_roundtrip_exact=slot_roundtrip,
        )

    def assert_seed_gates(self, instruction: str, articles: Sequence[str]) -> list[PromptEncoding]:
        encodings = [self.encode(article, instruction) for article in articles]
        failures: list[str] = []
        for index, encoding in enumerate(encodings):
            if not encoding.compositional:
                failures.append(f"article {index}: prompt is not prefix+slot+suffix compositional")
            if not encoding.full_roundtrip_exact:
                failures.append(f"article {index}: full token sequence does not round-trip")
            if not encoding.slot_roundtrip_exact:
                failures.append(f"article {index}: slot token sequence does not round-trip")
        slot_ids = {encoding.slot_ids for encoding in encodings}
        if len(slot_ids) != 1:
            failures.append("editable slot IDs depend on article content")
        if failures:
            raise AssertionError("Seed prompt gate failed: " + "; ".join(failures))
        return encodings
