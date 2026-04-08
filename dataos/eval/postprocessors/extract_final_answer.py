from __future__ import annotations

import re

from dataos.eval.postprocessors.normalize_number import normalize_number

NUMBER_PATTERN = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")
FINAL_ANSWER_PATTERN = re.compile(r"####\s*([-+]?\$?\d[\d,]*(?:\.\d+)?)")


def extract_final_answer(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    explicit = FINAL_ANSWER_PATTERN.findall(text)
    if explicit:
        return normalize_number(explicit[-1])
    candidates = NUMBER_PATTERN.findall(text)
    if not candidates:
        return None
    return normalize_number(candidates[-1])
