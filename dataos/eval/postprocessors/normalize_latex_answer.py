from __future__ import annotations

import re
from typing import Any

BOX_COMMANDS = ("boxed", "fbox")
LATEX_NUMBER_ATOM = r"[-+]?\d[\d,]*(?:\.\d+)?"
LATEX_NUMERIC_FRAC = (
    rf"\\(?:dfrac|tfrac|frac)\s*\{{\s*{LATEX_NUMBER_ATOM}\s*\}}"
    rf"\s*\{{\s*{LATEX_NUMBER_ATOM}\s*\}}"
)

ANSWER_LABEL_PATTERN = re.compile(
    r"(?:final\s+answer|answer|答案|答)\s*(?:is|=|:|：)?\s*(.+)",
    re.IGNORECASE,
)
NUMERIC_LATEX_PATTERN = re.compile(
    r"""
    [-+]?\s*
    (?:
        LATEX_NUMERIC_FRAC_PLACEHOLDER
        |
        \d[\d,]*(?:\.\d+)?\s*/\s*\d[\d,]*(?:\.\d+)?
        |
        \\sqrt\s*(?:\{[^{}\n]+\}|[-+]?\d[\d,]*(?:\.\d+)?)
        |
        \d[\d,]*(?:\.\d+)?
        (?:\s*\^\s*\{?\s*[-+]?\d+\s*\}?)?
        (?:
            \s*(?:\\times|\\cdot|\*|×|·)\s*
            10\s*(?:\^\s*\{?\s*[-+]?\d+\s*\}?)?
        )?
    )
    """.replace("LATEX_NUMERIC_FRAC_PLACEHOLDER", LATEX_NUMERIC_FRAC),
    re.VERBOSE,
)


def extract_normalized_latex_answer(value: Any) -> str | None:
    candidates = extract_final_answer_texts(value)
    if not candidates:
        return None
    normalized = _normalize_answer_texts(candidates)
    return normalized[0] if normalized else None


def extract_normalized_latex_answer_sequence(value: Any) -> list[str]:
    return _normalize_answer_texts(extract_final_answer_texts(value))


def normalize_latex_answer_candidates(value: Any) -> list[str]:
    normalized: list[str] = []
    for candidate in coerce_answer_texts(value):
        extracted_answers = _extract_final_answers_from_text(candidate) or [candidate]
        for extracted in extracted_answers:
            answer = normalize_latex_answer(extracted)
            if answer is not None and answer not in normalized:
                normalized.append(answer)
    return normalized


def extract_normalized_aime_answer(value: Any) -> str | None:
    candidate = extract_final_answer_text(value)
    if candidate is None:
        return None
    return normalize_aime_answer(candidate)


def normalize_aime_answer(value: Any) -> str | None:
    normalized = normalize_latex_answer(value)
    if normalized is None:
        return None
    if not re.fullmatch(r"\+?\d{1,3}", normalized):
        return None
    number = int(normalized)
    if number < 0 or number > 999:
        return None
    return f"{number:03d}"


def extract_final_answer_text(value: Any) -> str | None:
    extracted = extract_final_answer_texts(value)
    return extracted[0] if extracted else None


def extract_final_answer_texts(value: Any) -> list[str]:
    for candidate in reversed(coerce_answer_texts(value)):
        extracted = _extract_final_answers_from_text(candidate)
        if extracted:
            return extracted
    return []


def normalize_latex_answer(value: Any) -> str | None:
    texts = coerce_answer_texts(value)
    if not texts:
        return None
    text = texts[0].strip()
    if not text:
        return None
    return _normalize_latex_syntax(text)


def coerce_answer_texts(value: Any) -> list[str]:
    texts: list[str] = []

    def visit(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, str):
            text = item.strip()
            if text:
                texts.append(text)
            return
        if isinstance(item, bool):
            texts.append(str(item))
            return
        if isinstance(item, int | float):
            texts.append(str(item))
            return
        if isinstance(item, dict):
            for key in ("answer", "final_answer", "value", "text"):
                if key in item:
                    visit(item[key])
                    return
            return
        if isinstance(item, list | tuple):
            for child in item:
                visit(child)
            return
        texts.append(str(item).strip())

    visit(value)
    return [text for text in texts if text]


def _extract_final_answer_from_text(text: str) -> str | None:
    extracted = _extract_final_answers_from_text(text)
    return extracted[0] if extracted else None


def _extract_final_answers_from_text(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    marker_index = text.rfind("####")
    if marker_index >= 0:
        answers = _extract_all_from_answer_region(text[marker_index + 4 :])
        if answers:
            return answers

    boxed = _extract_command_args(text, BOX_COMMANDS)
    if boxed:
        return boxed

    labeled = _extract_labeled_answer(text)
    if labeled is not None:
        return [labeled]

    expression = _extract_last_numeric_latex_expression(text)
    if expression is not None:
        return [expression]
    return []


def _extract_from_answer_region(text: str) -> str | None:
    extracted = _extract_all_from_answer_region(text)
    return extracted[0] if extracted else None


def _extract_all_from_answer_region(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    boxed = _extract_command_args(text, BOX_COMMANDS)
    if boxed:
        return boxed
    labeled = _extract_labeled_answer(text)
    if labeled is not None:
        return [labeled]
    expressions = _extract_numeric_latex_expressions(text)
    if expressions:
        return expressions
    return _extract_generic_answer_region(text)


def _normalize_answer_texts(candidates: list[str]) -> list[str]:
    normalized: list[str] = []
    for candidate in candidates:
        answer = normalize_latex_answer(candidate)
        if answer is not None:
            normalized.append(answer)
    return normalized


def _extract_numeric_latex_expressions(text: str) -> list[str]:
    matches = [match.group(0) for match in NUMERIC_LATEX_PATTERN.finditer(text)]
    return [_trim_candidate(match) for match in matches]


def _extract_last_numeric_latex_expression(text: str) -> str | None:
    matches = _extract_numeric_latex_expressions(text)
    if not matches:
        return None
    return matches[-1]


def _extract_generic_answer_region(text: str) -> list[str]:
    text = _trim_candidate(text)
    if not text:
        return []
    first_line = _trim_candidate(text.splitlines()[0])
    if not first_line:
        return []
    labels = re.split(r"\s*[,，、;；]\s*", first_line)
    if len(labels) > 1 and all(re.fullmatch(r"[A-Za-z]", label) for label in labels):
        return labels
    return [first_line]


def _canonical_answer_sequence(answers: list[str]) -> str | None:
    if not answers:
        return None
    return " ; ".join(answers)


def canonical_latex_answer_sequence(value: Any) -> str | None:
    answers = extract_normalized_latex_answer_sequence(value)
    return _canonical_answer_sequence(answers)


def canonical_latex_reference_sequence(value: Any) -> str | None:
    normalized = normalize_latex_answer_candidates(value)
    return _canonical_answer_sequence(normalized)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def normalize_latex_reference_answer_sequence(value: Any) -> list[str]:
    return _dedupe_preserve_order(normalize_latex_answer_candidates(value))


def normalize_latex_prediction_answer_sequence(value: Any) -> list[str]:
    return extract_normalized_latex_answer_sequence(value)


def canonical_latex_reference_answer(value: Any) -> str | None:
    return _canonical_answer_sequence(normalize_latex_reference_answer_sequence(value))


def canonical_latex_prediction_answer(value: Any) -> str | None:
    return _canonical_answer_sequence(normalize_latex_prediction_answer_sequence(value))


def _unused_keep_extract_from_answer_region_for_compat(text: str) -> str | None:
    expression = _extract_last_numeric_latex_expression(text)
    if expression is not None:
        return expression
    return None


def _extract_labeled_answer(text: str) -> str | None:
    for line in reversed(text.splitlines()):
        match = ANSWER_LABEL_PATTERN.search(line)
        if match:
            candidate = _trim_candidate(match.group(1))
            boxed = _extract_last_command_arg(candidate, BOX_COMMANDS)
            if boxed is not None:
                return boxed
            return _extract_last_numeric_latex_expression(candidate)
    return None


def _trim_candidate(text: str) -> str:
    text = text.strip()
    while text and text[0] in "`'\"":
        text = text[1:].strip()
    while text and text[-1] in "`'\"":
        text = text[:-1].strip()
    while text and text[-1] in ".;。；":
        text = text[:-1].strip()
    return text


def _normalize_latex_syntax(text: str) -> str | None:
    text = _trim_candidate(text)
    text = _strip_wrappers(text)
    wrapped = _extract_wrapped_command_arg(text, BOX_COMMANDS)
    if wrapped is not None:
        text = _strip_wrappers(wrapped)

    text = text.replace("−", "-").replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("×", "*").replace("·", "*")
    text = re.sub(r"\\(?:left|right)\s*", "", text)
    text = re.sub(r"\\[,;:!]\s*", "", text)
    text = text.replace("~", "")
    text = _replace_one_arg_commands(text, ("mathrm", "text", "operatorname"))
    text = _replace_frac_commands(text)
    text = _replace_sqrt_commands(text)
    text = re.sub(r"\\(?:dfrac|tfrac)", r"\\frac", text)
    text = re.sub(r"\\(?:times|cdot)\b", "*", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"\^\{([^{}]+)\}", r"^\1", text)
    text = re.sub(r"\{([-+]?\d+(?:\.\d+)?)\}", r"\1", text)
    text = _strip_wrappers(text)
    return text or None


def _strip_wrappers(text: str) -> str:
    text = text.strip()
    changed = True
    while changed:
        changed = False
        for start, end in (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
            if text.startswith(start) and text.endswith(end) and len(text) >= len(start) + len(end):
                text = text[len(start) : len(text) - len(end)].strip()
                changed = True
        if text.startswith("{"):
            inner, close_index = _read_braced(text, 0)
            if inner is not None and close_index == len(text) - 1:
                text = inner.strip()
                changed = True
    return text


def _extract_last_command_arg(text: str, command_names: tuple[str, ...]) -> str | None:
    extracted = _extract_command_args(text, command_names)
    return extracted[-1] if extracted else None


def _extract_command_args(text: str, command_names: tuple[str, ...]) -> list[str]:
    pattern = _command_pattern(command_names)
    extracted: list[str] = []
    for match in pattern.finditer(text):
        open_index = _skip_spaces(text, match.end())
        if open_index >= len(text) or text[open_index] != "{":
            continue
        content, _ = _read_braced(text, open_index)
        if content is not None:
            extracted.append(content)
    return extracted


def _extract_wrapped_command_arg(text: str, command_names: tuple[str, ...]) -> str | None:
    match = _command_pattern(command_names).match(text)
    if match is None:
        return None
    open_index = _skip_spaces(text, match.end())
    if open_index >= len(text) or text[open_index] != "{":
        return None
    content, close_index = _read_braced(text, open_index)
    if content is not None and close_index == len(text) - 1:
        return content
    return None


def _replace_one_arg_commands(text: str, command_names: tuple[str, ...]) -> str:
    pattern = _command_pattern(command_names)
    cursor = 0
    chunks: list[str] = []
    for match in pattern.finditer(text):
        open_index = _skip_spaces(text, match.end())
        if open_index >= len(text) or text[open_index] != "{":
            continue
        content, close_index = _read_braced(text, open_index)
        if content is None:
            continue
        chunks.append(text[cursor : match.start()])
        chunks.append(content)
        cursor = close_index + 1
    if not chunks:
        return text
    chunks.append(text[cursor:])
    return "".join(chunks)


def _replace_frac_commands(text: str) -> str:
    pattern = _command_pattern(("frac", "dfrac", "tfrac"))
    while True:
        match = pattern.search(text)
        if match is None:
            return text
        first_open = _skip_spaces(text, match.end())
        if first_open >= len(text) or text[first_open] != "{":
            return text
        numerator, first_close = _read_braced(text, first_open)
        if numerator is None:
            return text
        second_open = _skip_spaces(text, first_close + 1)
        if second_open >= len(text) or text[second_open] != "{":
            return text
        denominator, second_close = _read_braced(text, second_open)
        if denominator is None:
            return text
        numerator = _normalize_latex_syntax(numerator) or ""
        denominator = _normalize_latex_syntax(denominator) or ""
        replacement = f"{_wrap_fraction_part(numerator)}/{_wrap_fraction_part(denominator)}"
        text = f"{text[: match.start()]}{replacement}{text[second_close + 1:]}"


def _replace_sqrt_commands(text: str) -> str:
    pattern = _command_pattern(("sqrt",))
    cursor = 0
    chunks: list[str] = []
    for match in pattern.finditer(text):
        arg_start = _skip_spaces(text, match.end())
        if arg_start >= len(text):
            continue
        content: str | None = None
        close_index = arg_start
        if text[arg_start] == "{":
            content, close_index = _read_braced(text, arg_start)
        else:
            number_match = re.match(r"[-+]?\d[\d,]*(?:\.\d+)?", text[arg_start:])
            if number_match:
                content = number_match.group(0)
                close_index = arg_start + len(content) - 1
        if content is None:
            continue
        normalized = _normalize_latex_syntax(content) or ""
        chunks.append(text[cursor : match.start()])
        chunks.append(f"sqrt({normalized})")
        cursor = close_index + 1
    if not chunks:
        return text
    chunks.append(text[cursor:])
    return "".join(chunks)


def _wrap_fraction_part(value: str) -> str:
    if re.fullmatch(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|[A-Za-z]+)", value):
        return value
    return f"({value})"


def _read_braced(text: str, open_index: int) -> tuple[str | None, int]:
    if open_index >= len(text) or text[open_index] != "{":
        return None, -1
    depth = 0
    for index in range(open_index, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_index + 1 : index], index
    return None, -1


def _skip_spaces(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _command_pattern(command_names: tuple[str, ...]) -> re.Pattern[str]:
    commands = "|".join(re.escape(command) for command in command_names)
    return re.compile(rf"\\(?:{commands})\b")
