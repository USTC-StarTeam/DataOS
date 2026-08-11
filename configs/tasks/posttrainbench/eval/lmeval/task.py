from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any

import datasets
import numpy as np
from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask
from lm_eval.models.api_models import JsonChatStr

_HEALTH_JUDGE_SEMAPHORE: threading.Semaphore | None = None
_HEALTH_JUDGE_SEMAPHORE_LOCK = threading.Lock()
_ARENA_JUDGE_SEMAPHORE: threading.Semaphore | None = None
_ARENA_JUDGE_SEMAPHORE_LOCK = threading.Lock()


def _mean(items: list[float]) -> float:
    values = [float(item) for item in items if item is not None and not math.isnan(float(item))]
    return float(np.mean(values)) if values else 0.0


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _benchmarks_root() -> Path | None:
    value = os.environ.get("POSTTRAINBENCH_BENCHMARKS_DIR")
    return Path(value).expanduser() if value else None


def _eval_src_root() -> Path | None:
    value = os.environ.get("POSTTRAINBENCH_EVAL_SRC") or os.environ.get(
        "POSTTRAINBENCH_EVAL_SRC_DIR"
    )
    return Path(value).expanduser() if value else None


def _workspace_posttrainbench_root() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent.parent / "PostTrainBench"
        if (candidate / "src/eval/tasks").exists():
            return candidate
    return None


def _first_existing(candidates: list[Path | None]) -> Path:
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    rendered = "\n".join(str(candidate) for candidate in candidates if candidate is not None)
    raise FileNotFoundError(f"None of the PostTrainBench data paths exist:\n{rendered}")


def _chat_instance(
    *,
    doc: dict[str, Any],
    messages: list[dict[str, Any]],
    gen_kwargs: dict[str, Any],
    request_fields: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Instance:
    """One generate_until request.

    ``request_fields`` are extra chat-completions body fields that belong to THIS doc (bfcl's
    per-doc ``tools``). They must not travel in ``gen_kwargs``: lm-eval groups requests by
    gen_kwargs and then sends the whole group with ``all_gen_kwargs[0]``, so anything per-doc
    put there is silently replaced by the first doc's copy at any concurrency above 1. The
    context, by contrast, is carried per request all the way to ``_create_payload``, so
    per-doc fields ride inside it and ``DataOSLocalToolChatCompletion`` merges them into the
    request body. Keeping gen_kwargs identical across docs also keeps them in ONE group, which
    is what preserves concurrency.
    """
    instance_kwargs = {
        key: kwargs[key]
        for key in ("metadata", "task_name", "doc_id", "repeats")
        if key in kwargs
    }
    context: Any = messages if not request_fields else {"messages": messages, **request_fields}
    return Instance(
        request_type="generate_until",
        doc=doc,
        arguments=(JsonChatStr(json.dumps(context, ensure_ascii=False)), gen_kwargs),
        idx=0,
        **instance_kwargs,
    )


def _clean_python_task_config(config: dict[str, Any] | None, *, version: int) -> dict[str, Any]:
    cleaned = dict(config or {"metadata": {"version": version}})
    cleaned.pop("class", None)
    return cleaned


def _strip_thinking(text: str) -> str:
    if text.startswith("<think>") and "</think>" in text:
        return text.split("</think>", maxsplit=1)[-1].strip()
    return text.strip()


def _limit_repetitions(text: str, max_reps: int = 5) -> str:
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        count = 1
        j = i + 1
        while j < len(lines) and lines[j] == line:
            count += 1
            j += 1
        out.extend([line] * min(count, max_reps))
        i = j
    return "\n".join(out)


class PostTrainBenchArenaHardWriting(ConfigurableTask):
    VERSION = 1
    OUTPUT_TYPE = "generate_until"

    _JUDGE_MODEL = "gpt-5-mini"
    _REASONING_EFFORT = "medium"
    _JUDGE_MAX_COMPLETION = 49152
    _PROMPT_TEMPLATE = (
        "<|User Prompt|>\n{QUESTION}\n\n"
        "<|The Start of Assistant A's Answer|>\n{ANSWER_A}\n"
        "<|The End of Assistant A's Answer|>\n\n"
        "<|The Start of Assistant B's Answer|>\n{ANSWER_B}\n"
        "<|The End of Assistant B's Answer|>"
    )
    _REGEX_PATTERNS = [r"\[\[([AB<>=]+)\]\]", r"\[([AB<>=]+)\]"]
    _OG_ARENA_HARD_PROMPT = (
        "Please act as an impartial judge and evaluate the quality of the responses "
        "provided by two AI assistants to the user prompt displayed below. You will "
        "be given assistant A's answer and assistant B's answer. Your job is to "
        "evaluate which assistant's answer is better.\n\nBegin your evaluation by "
        "generating your own answer to the prompt. You must provide your answers "
        "before judging any answers.\n\nWhen evaluating the assistants' answers, "
        "compare both assistants' answers with your answer. You must identify and "
        "correct any mistakes or inaccurate information.\n\nThen consider if the "
        "assistant's answers are helpful, relevant, and concise. Helpful means the "
        "answer correctly responds to the prompt or follows the instructions. Note "
        "when user prompt has any ambiguity or more than one interpretation, it is "
        "more helpful and appropriate to ask for clarifications or more information "
        "from the user than providing an answer based on assumptions. Relevant means "
        "all parts of the response closely connect or are appropriate to what is "
        "being asked. Concise means the response is clear and not verbose or "
        "excessive.\n\nThen consider the creativity and novelty of the assistant's "
        "answers when needed. Finally, identify any missing important information "
        "in the assistants' answers that would be beneficial to include when "
        "responding to the user prompt.\n\nAfter providing your explanation, you "
        "must output only one of the following choices as your final verdict with a "
        "label:\n\n1. Assistant A is significantly better: [[A>>B]]\n2. Assistant A "
        "is slightly better: [[A>B]]\n3. Tie, relatively the same: [[A=B]]\n4. "
        "Assistant B is slightly better: [[B>A]]\n5. Assistant B is significantly "
        "better: [[B>>A]]\n\nExample output: \"My final verdict is tie: [[A=B]]\"."
    )
    _CREATIVE_PROMPT = (
        "Please act as an impartial judge and evaluate the quality of the responses "
        "provided by two AI assistants to the user prompt displayed below. You will "
        "be given assistant A's answer and assistant B's answer. Your job is to "
        "evaluate which assistant's answer is better.\n\nWhen evaluating the "
        "assistants' answers, compare both assistants' answers. You must identify "
        "and correct any mistakes or inaccurate information.\n\nThen consider if "
        "the assistant's answers are helpful, relevant, and concise. Helpful means "
        "the answer correctly responds to the prompt or follows the instructions. "
        "Relevant means all parts of the response closely connect or are appropriate "
        "to what is being asked. Concise means the response is clear and not verbose "
        "or excessive.\n\nThen consider the creativity and novelty of the assistant's "
        "answers when needed. Finally, identify any missing important information "
        "in the assistants' answers that would be beneficial to include when "
        "responding to the user prompt.\n\nAfter providing your explanation, you "
        "must output only one of the following choices as your final verdict with a "
        "label:\n\n1. Assistant A is significantly better: [[A>>B]]\n2. Assistant A "
        "is slightly better: [[A>B]]\n3. Tie, relatively the same: [[A=B]]\n4. "
        "Assistant B is slightly better: [[B>A]]\n5. Assistant B is significantly "
        "better: [[B>>A]]\n\nExample output: \"My final verdict is tie: [[A=B]]\"."
    )

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config=_clean_python_task_config(config, version=self.VERSION))

    def download(self, *_args: Any, **_: Any) -> None:
        root = _benchmarks_root()
        eval_src = _eval_src_root()
        workspace = _workspace_posttrainbench_root()
        question_path = _first_existing(
            [
                Path(os.environ["POSTTRAINBENCH_ARENA_QUESTIONS"]).expanduser()
                if os.environ.get("POSTTRAINBENCH_ARENA_QUESTIONS")
                else None,
                root / "local/arena-hard-v2.0/question.jsonl" if root else None,
                eval_src / "tasks/arenahardwriting/evaluation_code/data/arena-hard-v2.0/question.jsonl"
                if eval_src
                else None,
                workspace / "src/eval/tasks/arenahardwriting/evaluation_code/data/arena-hard-v2.0/question.jsonl"
                if workspace
                else None,
            ]
        )
        self._data_dir = question_path.parent
        records = _load_jsonl(question_path)
        random.Random(42).shuffle(records)
        self.dataset = datasets.DatasetDict({"test": datasets.Dataset.from_list(records)})

    def has_training_docs(self) -> bool:
        return False

    def has_validation_docs(self) -> bool:
        return False

    def has_test_docs(self) -> bool:
        return True

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc: dict[str, Any]) -> str:
        return doc["prompt"]

    def doc_to_target(self, doc: dict[str, Any]) -> str:
        return ""

    def construct_requests(self, doc, ctx, **kwargs):
        del ctx
        gen_kwargs = {"max_gen_toks": 16384, "ptb_official_chat_payload": True}
        for key, value in (self.config.generation_kwargs or {}).items():
            gen_kwargs[key] = value
        return _chat_instance(
            doc=doc,
            messages=[{"role": "user", "content": doc["prompt"]}],
            gen_kwargs=gen_kwargs,
            **kwargs,
        )

    def process_results(self, doc: dict[str, Any], results: list[str]) -> dict[str, float]:
        if os.environ.get("POSTTRAINBENCH_JUDGE_DISABLED"):
            return {"arena_winrate": float("nan")}
        answer = _limit_repetitions(_strip_thinking(results[0] if results else ""))
        baseline = self._baseline_answer(doc)
        scores: list[float] = []
        for score in self._judge_pair(doc, baseline, answer, baseline_first=True):
            scores.append(1.0 - score)
        scores.extend(self._judge_pair(doc, answer, baseline, baseline_first=False))
        return {"arena_winrate": float(np.mean(scores)) if scores else float("nan")}

    def aggregation(self):
        return {"arena_winrate": _mean}

    def higher_is_better(self):
        return {"arena_winrate": True}

    def _baseline_answer(self, doc: dict[str, Any]) -> str:
        baseline_name = self._baseline_name(doc)
        answer_path = self._data_dir / "model_answer" / f"{baseline_name}.jsonl"
        answers = getattr(self, "_arena_answers", None)
        if answers is None:
            answers = {}
            for path in (self._data_dir / "model_answer").glob("*.jsonl"):
                answers[path.stem] = {
                    item["uid"]: item
                    for item in _load_jsonl(path)
                    if "uid" in item
                }
            self._arena_answers = answers
        if baseline_name not in answers:
            raise FileNotFoundError(f"Arena baseline answers not found at {answer_path}")
        record = answers[baseline_name][doc["uid"]]
        content = record["messages"][-1]["content"]
        return content["answer"] if isinstance(content, dict) else str(content)

    @staticmethod
    def _baseline_name(doc: dict[str, Any]) -> str:
        return "Qwen3-1.7B" if doc["category"] == "creative_writing" else "o3-mini-2025-01-31"

    def _judge_pair(
        self,
        doc: dict[str, Any],
        answer_a: str,
        answer_b: str,
        *,
        baseline_first: bool,
    ) -> list[float]:
        system_prompt = (
            self._CREATIVE_PROMPT if doc["category"] == "creative_writing" else self._OG_ARENA_HARD_PROMPT
        )
        prompt = self._PROMPT_TEMPLATE.format(
            QUESTION=doc["prompt"],
            ANSWER_A=answer_a,
            ANSWER_B=answer_b,
        )
        judge_model = os.environ.get("POSTTRAINBENCH_JUDGE_MODEL", self._JUDGE_MODEL)
        try:
            judgment, attempts = _arena_judge_chat(
                model=judge_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as exc:
            if not _is_skippable_content_safety_error(exc):
                raise
            _write_arena_judge_event(
                {
                    "uid": doc.get("uid"),
                    "category": doc.get("category"),
                    "baseline_model": self._baseline_name(doc),
                    "baseline_first": baseline_first,
                    "label": None,
                    "scores": [],
                    "skipped": True,
                    "skip_reason": "judge_content_safety",
                    "judge_model": judge_model,
                    "judge_base_url": os.environ.get("OPENAI_BASE_URL") or "openai_default",
                    "attempts": None,
                    "question_chars": len(str(doc.get("prompt", ""))),
                    "answer_a_chars": len(answer_a),
                    "answer_b_chars": len(answer_b),
                    "error_type": type(exc).__name__,
                    "error": _short_judge_error(exc),
                }
            )
            return []
        label = _arena_label(judgment, self._REGEX_PATTERNS)
        score_map = {
            "A>B": [1.0],
            "A>>B": [1.0, 1.0, 1.0],
            "A=B": [0.5],
            "A<<B": [0.0, 0.0, 0.0],
            "A<B": [0.0],
            "B>A": [0.0],
            "B>>A": [0.0, 0.0, 0.0],
            "B=A": [0.5],
            "B<<A": [1.0, 1.0, 1.0],
            "B<A": [1.0],
        }
        scores = score_map.get(label, [0.5])
        _write_arena_judge_event(
            {
                "uid": doc.get("uid"),
                "category": doc.get("category"),
                "baseline_model": self._baseline_name(doc),
                "baseline_first": baseline_first,
                "label": label,
                "scores": scores,
                "judge_model": judge_model,
                "judge_base_url": os.environ.get("OPENAI_BASE_URL") or "openai_default",
                "attempts": attempts,
                "question_chars": len(str(doc.get("prompt", ""))),
                "answer_a_chars": len(answer_a),
                "answer_b_chars": len(answer_b),
                "judgment": judgment,
            }
        )
        return scores


class PostTrainBenchHealthBench(ConfigurableTask):
    VERSION = 1
    OUTPUT_TYPE = "generate_until"

    _GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
    """.strip()

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config=_clean_python_task_config(config, version=self.VERSION))

    def download(self, *_args: Any, **_: Any) -> None:
        root = _benchmarks_root()
        eval_src = _eval_src_root()
        workspace = _workspace_posttrainbench_root()
        data_path = _first_existing(
            [
                Path(os.environ["POSTTRAINBENCH_HEALTHBENCH_JSONL"]).expanduser()
                if os.environ.get("POSTTRAINBENCH_HEALTHBENCH_JSONL")
                else None,
                root / "local/healthbench/healthbench.jsonl" if root else None,
                eval_src / "tasks/healthbench/evaluation_code/data/healthbench.jsonl"
                if eval_src
                else None,
                workspace / "src/eval/tasks/healthbench/evaluation_code/data/healthbench.jsonl"
                if workspace
                else None,
            ]
        )
        records = _load_jsonl(data_path)
        random.Random(42).shuffle(records)
        self.dataset = datasets.DatasetDict({"test": datasets.Dataset.from_list(records)})

    def has_training_docs(self) -> bool:
        return False

    def has_validation_docs(self) -> bool:
        return False

    def has_test_docs(self) -> bool:
        return True

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc: dict[str, Any]) -> str:
        return doc["prompt"][-1]["content"]

    def doc_to_target(self, doc: dict[str, Any]) -> str:
        return ""

    def construct_requests(self, doc, ctx, **kwargs):
        del ctx
        gen_kwargs = {"max_gen_toks": 16384, "ptb_official_chat_payload": True}
        for key, value in (self.config.generation_kwargs or {}).items():
            gen_kwargs[key] = value
        return _chat_instance(
            doc=doc,
            messages=list(doc["prompt"]),
            gen_kwargs=gen_kwargs,
            **kwargs,
        )

    def process_results(self, doc: dict[str, Any], results: list[str]) -> dict[str, float]:
        if os.environ.get("POSTTRAINBENCH_JUDGE_DISABLED"):
            return {"healthbench_accuracy": float("nan")}
        answer = _limit_repetitions(_strip_thinking(results[0] if results else ""))
        total = 0.0
        max_possible = sum(float(item["points"]) for item in doc["rubrics"] if item["points"] > 0)
        rubrics = list(doc["rubrics"])
        workers = max(
            1,
            min(
                len(rubrics) or 1,
                int(os.environ.get("POSTTRAINBENCH_HEALTHBENCH_CRITERIA_WORKERS", "8")),
            ),
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _health_criterion_met,
                    doc["prompt"],
                    answer,
                    rubric,
                    self._GRADER_TEMPLATE,
                    prompt_id=str(doc.get("prompt_id", "")),
                    rubric_index=rubric_index,
                ): rubric
                for rubric_index, rubric in enumerate(rubrics)
            }
            for future in as_completed(futures):
                rubric = futures[future]
                if future.result():
                    total += float(rubric["points"])
        score = total / max_possible if max_possible > 0 else 0.0
        return {"healthbench_accuracy": float(np.clip(score, 0.0, 1.0))}

    def aggregation(self):
        return {"healthbench_accuracy": _mean}

    def higher_is_better(self):
        return {"healthbench_accuracy": True}


class PostTrainBenchBFCL(ConfigurableTask):
    VERSION = 1
    OUTPUT_TYPE = "generate_until"
    DATASET_PATH = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
    DATASET_NAME = "exec_simple"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config=_clean_python_task_config(config, version=self.VERSION))

    def download(self, *_args: Any, **_: Any) -> None:
        root = _benchmarks_root()
        local_path = root / "local/bfcl_exec_simple.jsonl" if root else None
        if local_path and local_path.exists():
            records = _load_jsonl(local_path)
            self.dataset = datasets.DatasetDict({"test": datasets.Dataset.from_list(records)})
            return
        ds = datasets.load_dataset(
            self.DATASET_PATH,
            name=self.DATASET_NAME,
            split="train",
            revision="1bf8bbc3c0e35d04d00339c223a3fd653aa195ac",
        )
        self.dataset = datasets.DatasetDict({"test": ds})

    def has_training_docs(self) -> bool:
        return False

    def has_validation_docs(self) -> bool:
        return False

    def has_test_docs(self) -> bool:
        return True

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc: dict[str, Any]) -> str:
        return _bfcl_question_text(doc)

    def doc_to_target(self, doc: dict[str, Any]) -> str:
        return doc["ground_truth"][0]

    def construct_requests(self, doc, ctx, **kwargs):
        del ctx
        gen_kwargs = {
            "max_gen_toks": 16000,
            "until": ["<|im_end|>", "<|eot_id|>"],
        }
        for key, value in (self.config.generation_kwargs or {}).items():
            if key not in {"tools", "tool_choice"}:
                gen_kwargs[key] = value
        # THIS doc's functions — per-doc, so they go in the request, never in gen_kwargs
        # (see _chat_instance). tool_choice is a run-level constant and stays shared.
        request_fields: dict[str, Any] = {
            "tools": [_openai_tool_spec(tool) for tool in doc["function"]]
        }
        tool_choice = os.environ.get("POSTTRAINBENCH_BFCL_TOOL_CHOICE", "auto").strip()
        if tool_choice and tool_choice.lower() != "omit":
            gen_kwargs["tool_choice"] = tool_choice
        return _chat_instance(
            doc=doc,
            messages=_bfcl_messages(doc),
            gen_kwargs=gen_kwargs,
            request_fields=request_fields,
            **kwargs,
        )

    def process_results(self, doc: dict[str, Any], results: list[str]) -> dict[str, float]:
        expected = _parse_bfcl_target(doc["ground_truth"][0])
        predicted = _parse_bfcl_prediction(results[0] if results else "")
        return {"bfcl_accuracy": float(predicted == expected)}

    def aggregation(self):
        return {"bfcl_accuracy": _mean}

    def higher_is_better(self):
        return {"bfcl_accuracy": True}


def _openai_tool_spec(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
        },
    }


def _bfcl_messages(doc: dict[str, Any]) -> list[dict[str, Any]]:
    question = doc["question"][0]
    if isinstance(question, str):
        return [{"role": "user", "content": question}]
    if isinstance(question, list):
        return question
    raise TypeError(f"Unsupported BFCL question shape: {type(question).__name__}")


def _bfcl_question_text(doc: dict[str, Any]) -> str:
    return "\n".join(str(message.get("content", "")) for message in _bfcl_messages(doc))


def _parse_bfcl_target(target: str) -> dict[str, Any] | None:
    parsed = ast.parse(target, mode="single")
    if not parsed.body or not isinstance(parsed.body[0], ast.Expr):
        return None
    call = parsed.body[0].value
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        return None
    args: dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is not None:
            args[kw.arg] = ast.literal_eval(kw.value)
    return json.loads(json.dumps({"function": call.func.id, "arguments": args}, sort_keys=True))


def _parse_bfcl_prediction(text: str) -> dict[str, Any] | None:
    loaded = _load_bfcl_prediction_payload(text)
    if loaded is None:
        return None
    payload, allow_bare_call = loaded
    try:
        tool_calls = payload.get("tool_calls")
        if tool_calls:
            call = tool_calls[0]
            function = call.get("function", {})
            name = function.get("name")
            arguments = function.get("arguments", {})
        elif isinstance(payload.get("tool_call"), dict):
            call = payload["tool_call"]
            function = call.get("function", {})
            name = call.get("name") or function.get("name")
            arguments = call.get("arguments") or function.get("arguments", {})
        elif allow_bare_call and payload.get("name") and "arguments" in payload:
            name = payload.get("name")
            arguments = payload.get("arguments", {})
        else:
            return None
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
    except (IndexError, AttributeError, TypeError, json.JSONDecodeError):
        return None
    return json.loads(json.dumps({"function": name, "arguments": arguments}, sort_keys=True))


def _load_bfcl_prediction_payload(text: str) -> tuple[dict[str, Any], bool] | None:
    text = text.strip()
    candidates: list[tuple[str, bool]] = [(text, False)]
    candidates.extend(
        (match.group(1).strip(), True)
        for match in re.finditer(
            r"```(?:tool_calls?|function_call)\s*\n(.*?)```",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
    )
    candidates.extend(
        (match.group(1).strip(), True)
        for match in re.finditer(r"<tool_calls?>\s*(.*?)\s*</tool_calls?>", text, flags=re.DOTALL)
    )

    # Also try to extract embedded JSON objects from arbitrary text (e.g. when model
    # omits <tool_call> wrapper but includes bare JSON surrounded by thinking text).
    def _extract_embedded_json(s: str):
        start = s.find("{")
        while start >= 0:
            depth = 0
            for i, ch in enumerate(s[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                if depth == 0:
                    yield s[start:i+1], True
                    start = s.find("{", i+1)
                    break
            else:
                break
    candidates.extend(_extract_embedded_json(text))

    for candidate, allow_bare_call in candidates:
        cleaned = re.sub(r"</?tool_calls?>", "", candidate.strip()).strip()
        if not cleaned:
            continue
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload, allow_bare_call
    return None


def _openai_chat(
    *,
    model: str,
    messages: list[dict[str, str]],
    max_completion_tokens: int,
    reasoning_effort: str | None = None,
) -> str:
    from openai import OpenAI

    client = OpenAI()
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
    }
    if reasoning_effort:
        request["reasoning_effort"] = reasoning_effort
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(**request)
            return completion.choices[0].message.content or ""
        except Exception:
            if attempt == 2:
                raise
            time.sleep(5 * (2**attempt))
    return ""


def _arena_judge_chat(*, model: str, messages: list[dict[str, str]]) -> tuple[str, int]:
    from openai import BadRequestError, OpenAI

    client = OpenAI()
    lowered = model.lower()
    request: dict[str, Any] = {"model": model, "messages": messages}
    if any(name in lowered for name in ("gpt-5", "o1", "o3")):
        request["max_completion_tokens"] = int(
            os.environ.get("POSTTRAINBENCH_ARENA_JUDGE_GPT_MAX_COMPLETION_TOKENS", "49152")
        )
        reasoning_effort = os.environ.get("POSTTRAINBENCH_ARENA_JUDGE_REASONING_EFFORT", "medium")
        if reasoning_effort:
            request["reasoning_effort"] = reasoning_effort
    else:
        request["temperature"] = 0
        request["max_tokens"] = int(os.environ.get("POSTTRAINBENCH_ARENA_JUDGE_MAX_TOKENS", "16000"))
        request["extra_body"] = {"enable_thinking": False}

    max_retries = int(os.environ.get("POSTTRAINBENCH_ARENA_JUDGE_MAX_RETRIES", "5"))
    retry_sleep = float(os.environ.get("POSTTRAINBENCH_ARENA_JUDGE_RETRY_SLEEP", "2"))
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            semaphore = _arena_judge_semaphore()
            with semaphore:
                completion = client.chat.completions.create(**request)
            return completion.choices[0].message.content or "", attempt
        except BadRequestError as exc:
            last_error = exc
            if _is_skippable_content_safety_error(exc):
                raise
            if "reasoning" in str(exc).lower() and "reasoning_effort" in request:
                request.pop("reasoning_effort", None)
                continue
            if _is_fatal_judge_error(exc) or attempt == max_retries:
                raise
            time.sleep(retry_sleep * attempt)
        except Exception as exc:
            last_error = exc
            if _is_skippable_content_safety_error(exc):
                raise
            if _is_fatal_judge_error(exc) or attempt == max_retries:
                raise
            time.sleep(retry_sleep * attempt)
    if last_error is not None:
        raise last_error
    return "", max_retries


def _arena_judge_semaphore() -> threading.Semaphore:
    global _ARENA_JUDGE_SEMAPHORE
    with _ARENA_JUDGE_SEMAPHORE_LOCK:
        if _ARENA_JUDGE_SEMAPHORE is None:
            limit = max(1, int(os.environ.get("POSTTRAINBENCH_ARENA_JUDGE_CONCURRENCY", "32")))
            _ARENA_JUDGE_SEMAPHORE = threading.Semaphore(limit)
        return _ARENA_JUDGE_SEMAPHORE


def _arena_label(judgment: str, patterns: list[str]) -> str | None:
    upper = judgment.upper()
    for pattern in patterns:
        matches = re.findall(pattern, upper)
        matches = [m for m in matches if isinstance(m, str) and m]
        if matches:
            return matches[-1].strip()
    return None


def _health_criterion_met(
    conversation: list[dict[str, Any]],
    model_response: str,
    rubric: dict[str, Any],
    template: str,
    *,
    prompt_id: str,
    rubric_index: int,
) -> bool:
    convo = conversation + [{"role": "assistant", "content": model_response}]
    formatted_convo = "\n\n".join(f"{turn['role']}: {turn['content']}" for turn in convo)
    rubric_text = f"[{rubric['points']}] {rubric['criterion']}"
    prompt = template.replace("<<conversation>>", formatted_convo).replace(
        "<<rubric_item>>",
        rubric_text,
    )
    response, attempts = _health_judge_chat(
        model=os.environ.get("POSTTRAINBENCH_JUDGE_MODEL", "gpt-5-mini"),
        messages=[{"role": "user", "content": prompt}],
    )
    cleaned = re.sub(r"^```json\s*|\s*```$", "", response.strip())
    criteria_met = False
    explanation = ""
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = {}
    if parsed.get("criteria_met") is True or parsed.get("criteria_met") is False:
        criteria_met = parsed["criteria_met"] is True
        explanation = str(parsed.get("explanation", ""))
    _write_health_judge_event(
        {
            "prompt_id": prompt_id,
            "rubric_index": rubric_index,
            "criterion": rubric.get("criterion"),
            "points": rubric.get("points"),
            "tags": rubric.get("tags"),
            "criteria_met": criteria_met,
            "explanation": explanation,
            "judge_model": os.environ.get("POSTTRAINBENCH_JUDGE_MODEL", "gpt-5-mini"),
            "judge_base_url": os.environ.get("OPENAI_BASE_URL") or "openai_default",
            "attempts": attempts,
        }
    )
    return criteria_met


def _health_judge_chat(*, model: str, messages: list[dict[str, str]]) -> tuple[str, int]:
    from openai import OpenAI

    client = OpenAI()
    lowered = model.lower()
    request: dict[str, Any] = {"model": model, "messages": messages}
    if any(name in lowered for name in ("gpt-5", "o1", "o3")):
        request["max_completion_tokens"] = int(
            os.environ.get("POSTTRAINBENCH_HEALTHBENCH_GPT_MAX_COMPLETION_TOKENS", "49152")
        )
    else:
        request["temperature"] = 0
        request["max_tokens"] = int(
            os.environ.get("POSTTRAINBENCH_HEALTHBENCH_JUDGE_MAX_TOKENS", "512")
        )
        request["response_format"] = {"type": "json_object"}
        request["extra_body"] = {"enable_thinking": False}

    max_retries = int(os.environ.get("POSTTRAINBENCH_HEALTHBENCH_JUDGE_MAX_RETRIES", "5"))
    retry_sleep = float(os.environ.get("POSTTRAINBENCH_HEALTHBENCH_JUDGE_RETRY_SLEEP", "2"))
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            semaphore = _health_judge_semaphore()
            with semaphore:
                completion = client.chat.completions.create(**request)
            return completion.choices[0].message.content or "", attempt
        except Exception as exc:
            last_error = exc
            if _is_fatal_judge_error(exc) or attempt == max_retries:
                raise
            time.sleep(retry_sleep * attempt)
    if last_error is not None:
        raise last_error
    return "", max_retries


def _health_judge_semaphore() -> threading.Semaphore:
    global _HEALTH_JUDGE_SEMAPHORE
    with _HEALTH_JUDGE_SEMAPHORE_LOCK:
        if _HEALTH_JUDGE_SEMAPHORE is None:
            limit = max(
                1,
                int(os.environ.get("POSTTRAINBENCH_HEALTHBENCH_JUDGE_CONCURRENCY", "32")),
            )
            _HEALTH_JUDGE_SEMAPHORE = threading.Semaphore(limit)
        return _HEALTH_JUDGE_SEMAPHORE


def _is_fatal_judge_error(exc: Exception) -> bool:
    text = str(exc).lower()
    fatal_markers = (
        "401",
        "unauthorized",
        "invalid api key",
        "quota",
        "billing",
        "insufficient",
        "too many requests",
        "429",
        "rate limit",
    )
    return any(marker in text for marker in fatal_markers)


def _is_skippable_content_safety_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "data_inspection_failed",
            "datainspectionfailed",
            "inappropriate content",
        )
    )


def _short_judge_error(exc: Exception, *, max_chars: int = 500) -> str:
    text = " ".join(str(exc).split())
    return text[:max_chars]


def _write_health_judge_event(record: dict[str, Any]) -> None:
    path_value = os.environ.get("POSTTRAINBENCH_HEALTHBENCH_JUDGE_LOG")
    if not path_value:
        return
    path = Path(path_value).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _write_arena_judge_event(record: dict[str, Any]) -> None:
    path_value = os.environ.get("POSTTRAINBENCH_ARENA_JUDGE_LOG")
    if not path_value:
        return
    path = Path(path_value).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
