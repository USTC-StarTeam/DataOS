"""Map a free-form natural-language goal to a structured EvalIntent.

Two understanders ship out of the box:

* :func:`keyword_intent` — pure-string match over a small synonym table.
  Deterministic, no I/O, used as the default and as a deterministic test seed.
* :func:`llm_intent` — calls a user-supplied ``llm_client`` (the same shape
  used by ``dataos.core.agent``) and parses a JSON response. The keyword
  understander is used as the fallback if the LLM call fails or returns
  malformed JSON.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from dataos.llm.client import LLMClient


@dataclass(frozen=True)
class EvalIntent:
    raw_goal: str
    topics: tuple[str, ...] = ()
    skills: tuple[str, ...] = ()
    allow_unsafe_code: bool = False
    rationale: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


# Topic vocabulary — keep deliberately small so that resolver matching is
# predictable. Synonyms map free-form words to canonical topic tags used in
# the gallery.
_TOPIC_SYNONYMS: dict[str, tuple[str, ...]] = {
    "math": ("math", "mathematics", "数学", "arithmetic", "算术", "计算", "math reasoning"),
    "reasoning": ("reasoning", "推理", "逻辑", "logic", "deduction", "推断"),
    "code": ("code", "coding", "代码", "programming", "编程", "python", "function"),
    "instruction_following": (
        "instruction",
        "instructions",
        "instruction following",
        "follow instructions",
        "指令",
        "守不守指令",
        "format compliance",
        "约束",
        "constraint",
    ),
    "knowledge": ("knowledge", "知识", "general knowledge", "subject", "学科", "百科"),
    "science": ("science", "scientific", "科学", "physics", "biology", "chemistry"),
    "commonsense": ("commonsense", "common sense", "常识"),
    "factuality": (
        "factuality",
        "truthful",
        "truthfulness",
        "hallucination",
        "幻觉",
        "事实",
        "真假",
    ),
}

_UNSAFE_CODE_TRIGGERS = ("code", "coding", "代码", "programming", "编程", "humaneval", "python")


def keyword_intent(goal: str) -> EvalIntent:
    """Pure-string topic extraction; deterministic, no external calls."""
    if not isinstance(goal, str) or not goal.strip():
        return EvalIntent(raw_goal=goal or "", topics=(), skills=())
    lowered = goal.lower()
    topics: list[str] = []
    for canonical, synonyms in _TOPIC_SYNONYMS.items():
        if any(_contains(lowered, synonym) for synonym in synonyms):
            topics.append(canonical)
    allow_unsafe = any(_contains(lowered, trigger) for trigger in _UNSAFE_CODE_TRIGGERS)
    return EvalIntent(
        raw_goal=goal,
        topics=tuple(topics),
        skills=(),
        allow_unsafe_code=allow_unsafe,
        rationale="keyword match",
    )


def llm_intent(goal: str, llm_client: LLMClient) -> EvalIntent:
    """Ask an LLM to extract structured intent. Falls back to keyword on failure."""
    system_prompt = (
        "You are an evaluation planner. Given a user's natural-language goal, "
        "extract structured intent for selecting LLM evaluation benchmarks. "
        "Reply with JSON only, no prose."
    )
    user_prompt = (
        "Goal:\n"
        f"{goal}\n\n"
        "Allowed canonical topics: "
        + ", ".join(sorted(_TOPIC_SYNONYMS.keys()))
        + ".\n"
        "Return JSON with keys:\n"
        "  topics: list of canonical topic strings (subset of allowed list).\n"
        "  skills: list of free-form skill tags (lowercase, snake_case).\n"
        "  allow_unsafe_code: boolean (true if the user clearly wants code-execution evals).\n"
        "  rationale: one short sentence explaining the picks."
    )
    try:
        payload = llm_client.generate_json(system_prompt, user_prompt, temperature=0.0)
    except Exception as exc:  # noqa: BLE001 — fall back deterministically
        fallback = keyword_intent(goal)
        return EvalIntent(
            raw_goal=goal,
            topics=fallback.topics,
            skills=fallback.skills,
            allow_unsafe_code=fallback.allow_unsafe_code,
            rationale=f"llm failed: {type(exc).__name__}; keyword fallback",
        )
    return _intent_from_payload(goal, payload)


def _intent_from_payload(goal: str, payload: Any) -> EvalIntent:
    if not isinstance(payload, dict):
        return keyword_intent(goal)
    topics = tuple(_clean_string_list(payload.get("topics")))
    skills = tuple(_clean_string_list(payload.get("skills")))
    if not topics:
        topics = keyword_intent(goal).topics
    allow_unsafe = bool(payload.get("allow_unsafe_code", False))
    rationale = str(payload.get("rationale", "")).strip()
    return EvalIntent(
        raw_goal=goal,
        topics=topics,
        skills=skills,
        allow_unsafe_code=allow_unsafe,
        rationale=rationale or "llm intent",
        extras={"raw_llm_payload": payload},
    )


def _clean_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        normalized = item.strip().lower()
        if normalized and normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned


def _contains(haystack: str, needle: str) -> bool:
    """Whitespace-aware substring match for both ASCII and CJK."""
    needle = needle.lower().strip()
    if not needle:
        return False
    if needle.isascii():
        # Avoid matching "code" inside "barcode": require word boundary on ASCII.
        return re.search(rf"\b{re.escape(needle)}\b", haystack) is not None
    return needle in haystack
