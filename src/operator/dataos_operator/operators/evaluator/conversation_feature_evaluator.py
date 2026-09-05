"""Extract cheap structural features from paired conversations."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

from dataos_operator.core import Dataset, EvaluatorOperator, Record
from dataos_operator.view import DatasetView

_MCQ_MARKERS = (
    "option's letter",
    "answer with the option",
    "choose the correct",
    "select the correct",
)
_OCR_HINTS = (
    "author",
    "book",
    "brand",
    "number",
    "read",
    "sign",
    "text",
    "title",
    "word",
    "written",
)
_SPATIAL_HINTS = (
    "above",
    "below",
    "behind",
    "between",
    "left of",
    "right of",
    "spatial",
    "where is",
)
_REASONING_HINTS = (
    "calculate",
    "explain",
    "how many",
    "reason",
    "solve",
    "therefore",
    "why",
)
_NEGATIVE_ANSWERS = {"no", "none", "nothing", "nobody", "never"}
_YES_NO_ANSWERS = _NEGATIVE_ANSWERS | {"yes"}
_OPTION_RE = re.compile(r"(?:^|\n)\s*(?:[A-H][.)]|option\s+[A-H][.:])\s+", re.I)
_SPACE_RE = re.compile(r"\s+")


class ConversationFeatureEvaluator(EvaluatorOperator):
    """Extract length, prompt-format, capability-tag, and grouping features from paired turns.

    Implementation: parse ``[{user, assistant}, ...]`` without loading images; classify MCQ,
    long-form, short-VQA, yes/no, multi-turn, OCR, spatial, reasoning, negative-answer, and
    coordinate-output signals; compute the bounded richness weight used by high-performing
    weighted-reservoir baselines. ``DatasetView`` input is processed in Arrow-sized batches and
    stores only derived columns; ordinary record datasets retain the existing evaluator behavior.
    Dependency: pure-rule operator, no external dependencies.
    """

    dataset_backends = ("record", "indexed")

    def __init__(
        self,
        *,
        texts_key: str = "texts",
        output_key: str = "metadata.conversation",
        long_answer_chars: int = 400,
        batch_size: int = 8192,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            texts_key=texts_key,
            output_key=output_key,
            long_answer_chars=long_answer_chars,
            batch_size=batch_size,
            **kwargs,
        )
        if long_answer_chars <= 0:
            raise ValueError("long_answer_chars must be greater than zero")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.texts_key = texts_key
        self.output_key = output_key
        self.long_answer_chars = long_answer_chars
        self.batch_size = batch_size

    def evaluate_record(self, record: Record) -> Record:
        features = self._features(self.require_field(record, self.texts_key))
        return self.set_field(record, self.output_key, features)

    def evaluate_dataset(self, data: Dataset) -> Dataset:
        if not isinstance(data, DatasetView):
            return super().evaluate_dataset(data)

        columns: dict[str, list[Any]] = {}
        for batch in data.iter_batches([self.texts_key], batch_size=self.batch_size):
            for turns in batch[self.texts_key]:
                features = self._features(turns)
                for key, value in features.items():
                    columns.setdefault(f"{self.output_key}.{key}", []).append(value)
        return data.with_columns(columns)

    def _features(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, Sequence) or isinstance(value, str | bytes):
            raise ValueError("paired conversation must be a sequence")
        turns = [turn for turn in value if isinstance(turn, Mapping)]
        users = [str(turn.get("user") or "").strip() for turn in turns]
        answers = [str(turn.get("assistant") or "").strip() for turn in turns]
        user_text = " ".join(users)
        assistant_text = " ".join(answers)
        lower_user = user_text.lower()
        normalized_answers = {answer.lower().rstrip(".!?").strip() for answer in answers if answer}
        assistant_chars = len(assistant_text)
        max_answer_chars = max((len(answer) for answer in answers), default=0)
        mean_answer_chars = assistant_chars / len(answers) if answers else 0.0

        is_mcq = any(marker in lower_user for marker in _MCQ_MARKERS) or (
            len(_OPTION_RE.findall("\n" + user_text)) >= 2
        )
        is_short = "single word or phrase" in lower_user
        is_yes_no = bool(normalized_answers) and normalized_answers <= _YES_NO_ANSWERS
        is_coordinate = any(answer.lstrip().startswith("[") for answer in answers) or any(
            marker in lower_user
            for marker in ("bounding box coordinate", "bounding-box coordinate")
        )
        if is_mcq:
            prompt_format = "mcq"
        elif is_short:
            prompt_format = "short_vqa"
        elif is_yes_no:
            prompt_format = "yes_no"
        elif len(turns) == 1 and max_answer_chars >= self.long_answer_chars:
            prompt_format = "long_form"
        elif len(turns) > 1:
            prompt_format = "multi_turn"
        else:
            prompt_format = "other"

        tags = [prompt_format]
        if any(hint in lower_user for hint in _OCR_HINTS):
            tags.append("ocr")
        if any(hint in lower_user for hint in _SPATIAL_HINTS):
            tags.append("spatial")
        if any(hint in lower_user or hint in assistant_text.lower() for hint in _REASONING_HINTS):
            tags.append("reasoning")
        if normalized_answers & _NEGATIVE_ANSWERS:
            tags.append("negative_answer")
        if is_coordinate:
            tags.append("coordinate")

        richness = 1.0
        richness += min(len(turns), 4) * 0.55
        richness += min(assistant_chars / 350.0, 3.0) * 0.75
        richness += min(len(user_text) / 300.0, 2.0) * 0.25

        question = users[0] if users else ""
        stem = _OPTION_RE.split("\n" + question, maxsplit=1)[0]
        stem = _SPACE_RE.sub(" ", stem).strip().lower()
        question_group = hashlib.sha256(stem.encode("utf-8")).hexdigest() if stem else ""
        return {
            "turn_count": len(turns),
            "user_chars": len(user_text),
            "assistant_chars": assistant_chars,
            "max_answer_chars": max_answer_chars,
            "mean_answer_chars": mean_answer_chars,
            "format": prompt_format,
            "tags": tags,
            "is_coordinate": is_coordinate,
            "question_group": question_group,
            "richness_weight": richness,
        }
