from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI


EVAL_SRC = Path(os.environ.get("POSTTRAINBENCH_EVAL_SRC", "/data/hcgu/projects/PostTrainBench/src/eval"))
HEALTH_GRADER_PATH = EVAL_SRC / "tasks/healthbench/evaluation_code/grader.py"
ARENA_CONFIG_PATH = EVAL_SRC / "tasks/arenahardwriting/evaluation_code/config/arena-hard-v2.0.yaml"
ARENA_JUDGE_UTILS_PATH = EVAL_SRC / "tasks/arenahardwriting/evaluation_code/utils/judge_utils.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_health_grader = _load_module(HEALTH_GRADER_PATH, "posttrainbench_health_grader_runtime")
_arena_judge_utils = _load_module(ARENA_JUDGE_UTILS_PATH, "posttrainbench_arena_judge_utils_runtime")


@dataclass
class _Rubric:
    criterion: str
    points: int
    tags: list[str]


class PostTrainBenchHealthBench:
    _GRADER_TEMPLATE = _health_grader.GRADER_TEMPLATE


def _client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")
    base_url = os.environ.get("OPENAI_BASE_URL")
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _judge_model() -> str:
    return os.environ.get("POSTTRAINBENCH_JUDGE_MODEL", "qwen3.6-chat")


def _request_timeout() -> float:
    return float(os.environ.get("POSTTRAINBENCH_JUDGE_REQUEST_TIMEOUT", "90"))


def _rate_limit() -> None:
    interval = float(os.environ.get("POSTTRAINBENCH_JUDGE_MIN_INTERVAL_SECONDS", "6.2"))
    path = Path(os.environ.get("POSTTRAINBENCH_JUDGE_RATE_LIMIT_FILE", "/tmp/posttrainbench_judge_rate_limit.txt"))
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            last = 0.0
            if path.exists():
                raw = path.read_text(encoding="utf-8").strip()
                last = float(raw) if raw else 0.0
            delay = interval - (time.time() - last)
            if delay > 0:
                time.sleep(delay)
            path.write_text(str(time.time()), encoding="utf-8")
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _append_jsonl(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    event_path = Path(path)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _health_criterion_met(
    prompt: list[dict[str, Any]],
    answer: str,
    rubric: dict[str, Any],
    template: str,
    *,
    prompt_id: str,
    rubric_index: int,
) -> bool:
    rubric_obj = _Rubric(
        criterion=str(rubric["criterion"]),
        points=int(rubric["points"]),
        tags=list(rubric.get("tags") or []),
    )
    conversation = _health_grader.format_conversation_for_grader(prompt, answer)
    rubric_item = _health_grader.format_rubric_item(rubric_obj.criterion, rubric_obj.points)
    grader_prompt = template.replace("<<conversation>>", conversation).replace("<<rubric_item>>", rubric_item)

    max_retries = int(os.environ.get("POSTTRAINBENCH_HEALTHBENCH_JUDGE_MAX_RETRIES", "40"))
    retry_sleep = float(os.environ.get("POSTTRAINBENCH_HEALTHBENCH_JUDGE_RETRY_SLEEP", "30"))
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            _rate_limit()
            response = _client().chat.completions.create(
                model=_judge_model(),
                messages=[{"role": "user", "content": grader_prompt}],
                temperature=0,
                max_tokens=512,
                timeout=_request_timeout(),
            )
            content = response.choices[0].message.content or ""
            parsed = _parse_json(content)
            if isinstance(parsed.get("criteria_met"), bool):
                met = bool(parsed["criteria_met"])
                _append_jsonl(
                    os.environ.get("POSTTRAINBENCH_HEALTHBENCH_JUDGE_LOG"),
                    {
                        "prompt_id": prompt_id,
                        "rubric_index": rubric_index,
                        "criteria_met": met,
                        "explanation": parsed.get("explanation", ""),
                        "judge_model": _judge_model(),
                        "ts": time.time(),
                    },
                )
                return met
            raise ValueError(f"bad health judge JSON: {parsed}")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt + 1 >= max_retries:
                break
            time.sleep(retry_sleep)
    raise RuntimeError(f"health judge failed for prompt={prompt_id} rubric={rubric_index}") from last_error


def _load_arena_config() -> dict[str, Any]:
    import yaml

    return yaml.safe_load(ARENA_CONFIG_PATH.read_text(encoding="utf-8"))


def _arena_score_values(label: str) -> list[float]:
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
    key = label.upper()
    if key not in score_map:
        raise ValueError(f"unknown arena score label: {label}")
    return score_map[key]


def _arena_extract_label(text: str, patterns: list[str]) -> str:
    for pattern in patterns:
        matches = re.findall(pattern, text.upper())
        if not matches:
            continue
        last = matches[-1]
        if isinstance(last, tuple):
            for item in reversed(last):
                if item:
                    return item.strip()
        if last:
            return str(last).strip()
    raise ValueError(f"no arena score found in judge response: {text[:200]}")


class PostTrainBenchArenaHardWriting:
    def __init__(self) -> None:
        self.config = _load_arena_config()

    def _judge_pair(self, doc: dict[str, Any], answer_a: str, answer_b: str, *, baseline_first: bool) -> list[float]:
        category = str(doc.get("category") or "creative_writing")
        settings = _arena_judge_utils.JUDGE_SETTINGS[category]
        prompt_template = self.config["prompt_template"]
        user_prompt = prompt_template.format(
            QUESTION=doc.get("prompt", ""),
            ANSWER_A=answer_a,
            ANSWER_B=answer_b,
        )
        messages = [
            {"role": "system", "content": settings["system_prompt"]},
            {"role": "user", "content": user_prompt},
        ]
        max_retries = int(os.environ.get("POSTTRAINBENCH_ARENA_JUDGE_MAX_RETRIES", "40"))
        retry_sleep = float(os.environ.get("POSTTRAINBENCH_ARENA_JUDGE_RETRY_SLEEP", "30"))
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                _rate_limit()
                response = _client().chat.completions.create(
                    model=_judge_model(),
                    messages=messages,
                    temperature=0,
                    max_tokens=int(self.config.get("max_tokens", 16000)),
                    timeout=_request_timeout(),
                )
                content = response.choices[0].message.content or ""
                label = _arena_extract_label(content, list(self.config["regex_patterns"]))
                scores = _arena_score_values(label)
                _append_jsonl(
                    os.environ.get("POSTTRAINBENCH_ARENA_JUDGE_LOG"),
                    {
                        "uid": str(doc.get("uid", "")),
                        "category": category,
                        "baseline_first": bool(baseline_first),
                        "score_label": label,
                        "scores": scores,
                        "judge_model": _judge_model(),
                        "ts": time.time(),
                    },
                )
                return scores
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt + 1 >= max_retries:
                    break
                time.sleep(retry_sleep)
        raise RuntimeError(f"arena judge failed for uid={doc.get('uid')} baseline_first={baseline_first}") from last_error
