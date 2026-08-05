"""把多选题样本规范成"带字母选项的题面 + 字母答案"的评测器格式。"""

from __future__ import annotations

from typing import Any

from dataos_operator.core import MapperOperator, Record

_MISSING = object()


class McqFormatter(MapperOperator):
    """Format a multiple-choice sample into lettered-choice text with a letter answer.

    Implementation: read ``question_key`` and ``choices_key`` (a list, or a dict whose values are
    the options) and render ``question`` followed by ``A. ... B. ...`` lines using ``letters``;
    then normalize ``answer_key`` to a single letter — an integer index maps through ``letters``,
    an existing letter is upper-cased, and otherwise the answer is matched against the choice texts
    to recover its letter. Field names and the letter alphabet are parameterized and the answer
    normalization is content-based, so no dataset layout is hard-coded.
    Dependency: pure-rule operator, no external dependencies.
    """

    operator_name = "mcq_formatter"

    def __init__(
        self,
        question_key: str = "question",
        choices_key: str = "choices",
        answer_key: str = "answer",
        *,
        letters: str = "ABCD",
        output_key: str = "output",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            question_key=question_key,
            choices_key=choices_key,
            answer_key=answer_key,
            letters=letters,
            output_key=output_key,
            **kwargs,
        )
        self.question_key = question_key
        self.choices_key = choices_key
        self.answer_key = answer_key
        self.letters = letters
        self.output_key = output_key

    def map_record(self, record: Record) -> Record:
        question = _as_text(self.get_field(record, self.question_key, ""))
        choices = _as_choices(self.get_field(record, self.choices_key, []))
        raw_answer = self.get_field(record, self.answer_key, _MISSING)

        lines = [question] if question else []
        lines.extend(
            f"{self.letters[index]}. {_as_text(choice)}"
            for index, choice in enumerate(choices)
            if index < len(self.letters)
        )
        output = "\n".join(lines)
        letter = self._normalize_answer(raw_answer, choices)

        result = dict(record)
        result[self.output_key] = output
        if letter is not None:
            result[self.answer_key] = letter
        return result

    def _normalize_answer(self, raw: Any, choices: list[Any]) -> str | None:
        if raw is _MISSING or raw is None:
            return None
        if isinstance(raw, bool):
            return None
        if isinstance(raw, int):
            return self.letters[raw] if 0 <= raw < len(self.letters) else None
        text = str(raw).strip()
        if len(text) == 1 and text.upper() in self.letters:
            return text.upper()
        if text.isdigit():
            index = int(text)
            if 0 <= index < len(self.letters):
                return self.letters[index]
        for index, choice in enumerate(choices):
            if index < len(self.letters) and _as_text(choice).strip() == text:
                return self.letters[index]
        return None


def _as_choices(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return list(value.values())
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _as_text(value: Any) -> str:
    return "" if value is None else str(value)
