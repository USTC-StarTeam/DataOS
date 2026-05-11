from __future__ import annotations

import re
from typing import Any

MATH_PROMPT_TEMPLATE = (
    'Solve the following math problem step by step. The last line of your response '
    'should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the '
    "answer to the problem.\n\n"
    "{prompt}\n\n"
    'Remember to put your answer on its own line at the end in the form "ANSWER: '
    '$ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you '
    "do not need to use a \\boxed command.\n\n"
    "Reasoning:"
)


def doc_to_text(doc: dict[str, Any]) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=doc["question"])


def doc_to_target(doc: dict[str, Any]) -> str:
    return _gsm8k_target(doc["answer"])


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    completion = results[0] if results else ""
    target = doc_to_target(doc)
    return {"accuracy": float(_inspect_numeric_end_match(completion, target))}


def _gsm8k_target(answer: str) -> str:
    return answer.rsplit("####", maxsplit=1)[-1].strip()


def _inspect_numeric_end_match(value: str, target: str) -> bool:
    """Mirror Inspect AI's match(location='end', numeric=True) for GSM8K integers."""
    v = value.strip().casefold()
    t = target.strip().casefold()
    answer = v
    if _is_numeric_token(t):
        v = _strip_numeric_punctuation(v)
        t = _normalize_number(_strip_numeric_punctuation(t))
        words = re.split(r"\s+", v)
        words.reverse()
        answer = _first_number_normalized(words)
    return answer.endswith(t)


def _strip_numeric_punctuation(text: str) -> str:
    stripped = re.sub(r"[$,£,€,*_]", "", text)
    return re.sub(r"\.(?=\s|$|\D)", "", stripped)


def _first_number_normalized(words: list[str]) -> str:
    number = next((word for word in words if _is_numeric_token(word.replace(".", ""))), words[0])
    return _normalize_number(number)


def _normalize_number(number: str, precision: int = 5) -> str:
    if _is_numeric_token(number.replace(".", "")):
        try:
            return format(float(number), f".{precision}g")
        except ValueError:
            return number
    return number


def _is_numeric_token(text: str) -> bool:
    return text.isnumeric()
