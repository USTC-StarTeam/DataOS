from __future__ import annotations

import json
import os
import re
from pathlib import Path
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

AIME_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step.\n"
    'The last line of your response should be of the form "ANSWER: $ANSWER" '
    "(without quotes) where $ANSWER is the answer to the problem.\n\n"
    "{prompt}\n\n"
    'Remember to put your answer on its own line at the end in the form "ANSWER: '
    '$ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you '
    "do not need to use a \\boxed command."
)

HUMANEVAL_INSTRUCTION = (
    "\n"
    "Read the following function signature and docstring, and fully implement\n"
    "the function described. Your response should only contain the code for\n"
    "this function.\n\n"
)


def doc_to_text(doc: dict[str, Any]) -> str:
    return MATH_PROMPT_TEMPLATE.format(prompt=doc["question"])


def doc_to_target(doc: dict[str, Any]) -> str:
    return _gsm8k_target(doc["answer"])


def aime_doc_to_text(doc: dict[str, Any]) -> str:
    return AIME_PROMPT_TEMPLATE.format(prompt=doc["problem"])


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


def process_aime_results(doc: dict[str, Any], results: list[Any]) -> dict[str, float]:
    """AIME 2025 numeric accuracy over all retained generations."""
    target = str(doc.get("answer", "")).strip()
    completions = _flatten_generations(results)
    if not completions:
        _write_aime_extraction_event(doc, target, [])
        return {"exact_match": 0.0}
    records: list[dict[str, Any]] = []
    correct = 0.0
    normalized_target = _normalize_number(target)
    for index, completion in enumerate(completions):
        extracted = _extract_aime_answer(completion)
        normalized_prediction = _normalize_number(extracted)
        is_correct = normalized_prediction == normalized_target
        correct += float(is_correct)
        records.append(
            {
                "index": index,
                "completion_chars": len(completion),
                "extracted_answer": extracted,
                "normalized_prediction": normalized_prediction,
                "normalized_target": normalized_target,
                "correct": is_correct,
            }
        )
    _write_aime_extraction_event(doc, target, records)
    return {"exact_match": correct / len(completions)}


def _flatten_generations(results: list[Any]) -> list[str]:
    completions: list[str] = []
    for result in results:
        if isinstance(result, list):
            completions.extend(_flatten_generations(result))
        elif result is not None:
            completions.append(str(result))
    return completions


def _aime_answer_matches(completion: str, target: str) -> bool:
    answer = _extract_aime_answer(completion)
    return _normalize_number(answer) == _normalize_number(target)


def _extract_aime_answer(completion: str) -> str:
    answer = _extract_last_boxed(completion) or _extract_answer_line(completion) or completion
    numbers = re.findall(r"-?\d+(?:\.\d+)?", answer.replace(",", ""))
    if numbers:
        answer = numbers[-1]
    return answer.strip()


def _extract_answer_line(text: str) -> str | None:
    matches = re.findall(r"ANSWER\s*:\s*([^\n]+)", text, flags=re.IGNORECASE)
    return matches[-1].strip() if matches else None


def _extract_last_boxed(text: str) -> str | None:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    i = start + len(marker)
    depth = 1
    chars: list[str] = []
    while i < len(text):
        char = text[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
        chars.append(char)
        i += 1
    return None


def _write_aime_extraction_event(
    doc: dict[str, Any],
    target: str,
    records: list[dict[str, Any]],
) -> None:
    path_value = os.environ.get("POSTTRAINBENCH_AIME_EXTRACTION_LOG")
    if not path_value:
        return
    path = Path(path_value).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "id": doc.get("id"),
        "problem_prefix": str(doc.get("problem", ""))[:160],
        "target": target,
        "num_generations": len(records),
        "num_correct": sum(1 for record in records if record.get("correct")),
        "generations": records,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def process_gpqa_docs(dataset: Any) -> Any:
    """Prepare GPQA Main choices like InspectAI's ``hf_dataset(..., shuffle_choices=True)``."""

    import os
    import random

    frozen_path = os.environ.get("POSTTRAINBENCH_GPQA_FROZEN_JSONL", "").strip()
    if frozen_path:
        import datasets

        path = Path(frozen_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"POSTTRAINBENCH_GPQA_FROZEN_JSONL does not exist: {path}")
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]
        return datasets.Dataset.from_list(records)

    seed_value = os.environ.get("POSTTRAINBENCH_GPQA_SHUFFLE_SEED", "42").strip()
    seed = None if seed_value.lower() in {"", "none", "random"} else int(seed_value)
    rng = random.Random(seed)

    def _process_doc(doc: dict[str, Any]) -> dict[str, Any]:
        correct = str(doc["Correct Answer"])
        choices = [
            correct,
            str(doc["Incorrect Answer 1"]),
            str(doc["Incorrect Answer 2"]),
            str(doc["Incorrect Answer 3"]),
        ]
        rng.shuffle(choices)
        answer_index = choices.index(correct)
        return {
            **doc,
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "answer": chr(ord("A") + answer_index),
        }

    return dataset.map(_process_doc)


def gpqa_doc_to_text(doc: dict[str, Any]) -> str:
    choices = (
        f"A) {doc['choice1']}\n"
        f"B) {doc['choice2']}\n"
        f"C) {doc['choice3']}\n"
        f"D) {doc['choice4']}"
    )
    return (
        "Answer the following multiple choice question. The last line of your response "
        "should be of the following format: 'ANSWER: $LETTER' (without quotes) where "
        "LETTER is one of A,B,C,D. Think step by step before answering.\n\n"
        f"{doc['Question']}\n\n"
        f"{choices}"
    )


def process_gpqa_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    completion = results[0] if results else ""
    prediction = _extract_choice(completion)
    return {"accuracy": float(prediction == str(doc["answer"]).upper())}


def _clean_gpqa_text(text: Any) -> str:
    value = " " if text is None else str(text).strip()
    value = value.replace(" [title]", ". ")
    value = re.sub(r"\[.*?\]", "", value)
    return re.sub(r"\s+", " ", value)


def _extract_choice(text: str) -> str | None:
    match = re.search(
        r"(?im)^ANSWER\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)",
        text,
    )
    if match is None:
        match = re.search(
            r"(?i)ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)",
            text,
        )
    if match is None:
        return None
    matched = match.group(1).strip().rstrip(".").upper()
    return matched if matched in {"A", "B", "C", "D"} else None


def humaneval_doc_to_text(doc: dict[str, Any]) -> str:
    return HUMANEVAL_INSTRUCTION + doc["prompt"]


def humaneval_build_predictions(resps: list[list[str]], docs: list[dict[str, Any]]) -> list[list[str]]:
    """Build executable HumanEval candidates from lm-eval direct-vLLM completions.

    The PostTrainBench/InspectAI target is still prompt + answer + tests. DataOS
    uses robust extraction before execution because direct lm-eval/vLLM logs raw
    chat completions that may include thinking blocks, fenced code, repeated
    prompts, or tokenizer artifacts. Strict InspectAI ``find_code`` remains a
    diagnostic only; the 2026-05-17 eight-model rescore showed it collapses valid
    chat-model outputs in this chain.
    """

    predictions: list[list[str]] = []
    for resp, doc in zip(resps, docs, strict=True):
        candidates = []
        for completion in resp:
            candidates.append(doc["prompt"] + _extract_python_completion(completion, doc["entry_point"]))
        predictions.append(candidates)
    return predictions


def _inspect_humaneval_find_code(completion: str) -> str:
    """Mirror InspectAI's HumanEval code extraction used by PostTrainBench."""
    pattern_1 = re.compile(r"```python\n(.*?)```", re.DOTALL)
    pattern_2 = re.compile(r"```\n(.*?)```", re.DOTALL)
    matches = pattern_1.findall(completion) + pattern_2.findall(completion)
    if matches:
        return str(_inspect_humaneval_extract_function_body(matches[0]))
    return str(completion)


def _inspect_humaneval_extract_function_body(code: str) -> str:
    return code[code.find(":\n    ") + 2 :]


def _extract_python_completion(completion: str, entry_point: str) -> str:
    code = _strip_thinking(completion)
    fenced = _extract_fenced_code(code)
    if fenced is not None:
        code = fenced
    code = _normalize_generated_python(code)
    code = _remove_repeated_prompt(code, entry_point)
    body = _extract_function_body(code, entry_point)
    return body if body is not None else code


def _normalize_generated_python(code: str) -> str:
    return code.replace("\u2581", " ")


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "<think>" in text.lower():
        parts = re.split(r"</?think>", text, flags=re.IGNORECASE)
        text = parts[-1] if parts else text
    return text.strip()


def _extract_fenced_code(text: str) -> str | None:
    matches = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0].strip()
    if "```" not in text:
        return None
    parts = text.split("```")
    return parts[1].strip() if len(parts) > 1 else None


def _remove_repeated_prompt(code: str, entry_point: str) -> str:
    pattern = rf"(?s).*?def\s+{re.escape(entry_point)}\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:"
    match = re.search(pattern, code)
    if not match:
        return code
    return code[match.start() :]


def _extract_function_body(code: str, entry_point: str) -> str | None:
    pattern = rf"def\s+{re.escape(entry_point)}\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:"
    match = re.search(pattern, code)
    if not match:
        return None
    body = code[match.end() :]
    if not body.startswith("\n"):
        body = "\n" + body.lstrip()
    return body


def humaneval_pass_at_k(
    references: list[str],
    predictions: list[list[str]],
    k: list[int] | int | None = None,
) -> dict[str, float]:
    if k is None:
        k = [1]
    if isinstance(k, int):
        k = [k]
    requested = sorted(set(k))
    if requested != [1]:
        raise ValueError("PostTrainBench HumanEval protocol only supports pass@1.")

    total = 0
    correct = 0
    for reference, candidates in zip(references, predictions, strict=True):
        total += 1
        candidate = candidates[0] if candidates else ""
        if _humaneval_candidate_passes(candidate, reference):
            correct += 1
    return {"pass@1": correct / total if total else 0.0}


def _humaneval_candidate_passes(candidate: str, reference: str, timeout: float = 30.0) -> bool:
    import subprocess
    import sys
    import tempfile
    import textwrap
    from pathlib import Path

    program = f"{candidate.rstrip()}\n\n{reference.strip()}\n"
    with tempfile.TemporaryDirectory(prefix="ptb_humaneval_") as tmpdir:
        script = Path(tmpdir) / "check_candidate.py"
        script.write_text(
            "import faulthandler\n"
            "faulthandler.enable()\n"
            + textwrap.dedent(program),
            encoding="utf-8",
        )
        try:
            completed = subprocess.run(
                [sys.executable, str(script)],
                cwd=tmpdir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError):
            return False
    return completed.returncode == 0
