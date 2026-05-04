"""Turn an EvalIntent into ranked EvalPlanItems against a Gallery.

Scoring is intentionally simple and explainable: each gallery entry gets +2
per matching topic, +1 per matching skill, with a small bonus for the user's
``raw_goal`` mentioning the entry's task name verbatim.

``only_runnable`` filtering is *capability-driven*: ``generate_until`` tasks
are always allowed; ``multiple_choice`` (and any other ``loglikelihood``-style)
tasks pass only if the configured adapter implements ``score_continuations``
(see :class:`dataos.eval.adapters.base.EvalLoglikelihoodAdapter`). When a
filter drop happens the resolver logs one line so the planner never silently
hides tasks from the user.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from dataos.eval.adapters.base import EvalLoglikelihoodAdapter
from dataos.eval.planner.gallery import Gallery, GalleryEntry, builtin_gallery
from dataos.eval.planner.intent import EvalIntent, keyword_intent

_LOGGER = logging.getLogger("dataos.eval.planner")

_LOGLIKELIHOOD_OUTPUT_TYPES = frozenset({"multiple_choice", "loglikelihood"})


@dataclass(frozen=True)
class EvalPlanItem:
    task: str
    adapter: str
    adapter_config: dict[str, Any] = field(default_factory=dict)
    limit: int | None = None
    num_fewshot: int | None = None
    batch_size: int = 8
    max_new_tokens: int = 256
    apply_chat_template: bool = True
    confirm_run_unsafe_code: bool = False
    headline_metric: str = ""
    score: float = 0.0
    rationale: str = ""


_DEFAULT_FALLBACK = ("gsm8k", "ifeval", "bbh_cot_zeroshot_logical_deduction_three_objects")


def resolve_tasks(
    intent: EvalIntent,
    *,
    gallery: Gallery | None = None,
    adapter: str | Any = "transformers",
    adapter_config: dict[str, Any] | None = None,
    max_tasks: int = 4,
    only_runnable: bool = True,
) -> list[EvalPlanItem]:
    """Score gallery entries against the intent and return ranked plan items.

    ``only_runnable`` (default True) drops tasks the configured ``adapter``
    cannot execute. Detection rule: a gallery entry whose ``output_type`` is
    ``multiple_choice`` or ``loglikelihood`` requires the adapter to implement
    :class:`EvalLoglikelihoodAdapter`. ``generate_until`` tasks are always
    allowed. When any drop happens, the resolver logs a single line listing
    the affected tasks via the ``dataos.eval.planner`` logger.

    ``adapter`` may be either an adapter name string (the planner cannot
    introspect strings, so it conservatively assumes only generate is
    supported and filters logprob tasks) or a live adapter instance (the
    planner runs ``isinstance(adapter, EvalLoglikelihoodAdapter)``).
    """
    gallery = gallery or builtin_gallery()
    adapter_name = _adapter_name(adapter)
    if only_runnable:
        gallery, dropped = _apply_runnable_filter(gallery, intent, adapter, adapter_name)
        if dropped:
            _LOGGER.info(
                "eval/planner: filtered %d tasks (%s) because adapter %r does not implement score_continuations",
                len(dropped),
                ", ".join(entry.task for entry in dropped),
                adapter_name,
            )

    scored: list[tuple[float, GalleryEntry]] = []
    for entry in gallery.entries:
        score = _score_entry(intent, entry)
        if score > 0:
            scored.append((score, entry))

    if not scored:
        scored = [
            (0.0, entry)
            for entry in (gallery.by_name(name) for name in _DEFAULT_FALLBACK)
            if entry is not None
        ]

    scored.sort(key=lambda pair: (-pair[0], pair[1].task))
    selected = scored[:max_tasks]

    plan: list[EvalPlanItem] = []
    for score, entry in selected:
        plan.append(
            EvalPlanItem(
                task=entry.task,
                adapter=adapter_name,
                adapter_config=dict(adapter_config or {}),
                limit=entry.default_limit,
                num_fewshot=entry.default_num_fewshot,
                batch_size=entry.default_batch_size,
                max_new_tokens=entry.default_max_new_tokens,
                apply_chat_template=True,
                confirm_run_unsafe_code=entry.requires_unsafe_code,
                headline_metric=entry.headline_metric,
                score=score,
                rationale=_explain(intent, entry, score),
            )
        )
    return plan


def plan_eval(
    goal: str,
    *,
    llm_client: Any | None = None,
    gallery: Gallery | None = None,
    adapter: str = "transformers",
    adapter_config: dict[str, Any] | None = None,
    max_tasks: int = 4,
    only_runnable: bool = True,
) -> list[EvalPlanItem]:
    """End-to-end: NL goal → ranked EvalPlanItems.

    ``llm_client`` accepts:

    * ``None`` (default): pure keyword extraction, no I/O.
    * an :class:`dataos.llm.LLMClient` instance (or any duck-type with
      ``generate_json(system, user, temperature) -> dict``).
    * the literal string ``"auto"`` or boolean ``True``: construct an
      :class:`OpenAICompatLLMClient` from environment variables via
      :func:`dataos.eval.planner.llm.default_llm_client`.
    """
    if llm_client in ("auto", True):
        from dataos.eval.planner.llm import default_llm_client

        llm_client = default_llm_client()
    if llm_client is not None:
        from dataos.eval.planner.intent import llm_intent

        intent = llm_intent(goal, llm_client)
    else:
        intent = keyword_intent(goal)
    return resolve_tasks(
        intent,
        gallery=gallery,
        adapter=adapter,
        adapter_config=adapter_config,
        max_tasks=max_tasks,
        only_runnable=only_runnable,
    )


def _adapter_name(adapter: str | Any) -> str:
    if isinstance(adapter, str):
        return adapter
    name = getattr(adapter, "name", None)
    return str(name) if isinstance(name, str) else type(adapter).__name__


def _adapter_supports_loglikelihood(adapter: str | Any) -> bool:
    """Conservative capability detection.

    For string adapter names we don't load the adapter class — strings can
    refer to adapters this process has never imported. Only the
    ``transformers`` built-in adapter is known a-priori to ship
    ``score_continuations`` today; everything else is assumed
    generate-only. Pass a live adapter instance for accurate detection.
    """
    if not isinstance(adapter, str):
        return isinstance(adapter, EvalLoglikelihoodAdapter)
    return adapter == "transformers"


def _apply_runnable_filter(
    gallery: Gallery,
    intent: EvalIntent,
    adapter: str | Any,
    adapter_name: str,
) -> tuple[Gallery, list[GalleryEntry]]:
    can_loglik = _adapter_supports_loglikelihood(adapter)
    kept: list[GalleryEntry] = []
    dropped: list[GalleryEntry] = []
    for entry in gallery.entries:
        if entry.requires_unsafe_code and not intent.allow_unsafe_code:
            dropped.append(entry)
            continue
        if entry.output_type in _LOGLIKELIHOOD_OUTPUT_TYPES and not can_loglik:
            dropped.append(entry)
            continue
        kept.append(entry)
    # Drops we want to surface in logs are only the capability-driven ones,
    # not unsafe_code (that's an intent toggle, not an adapter limitation).
    capability_dropped = [
        entry
        for entry in dropped
        if entry.output_type in _LOGLIKELIHOOD_OUTPUT_TYPES and not can_loglik
    ]
    return Gallery(entries=tuple(kept)), capability_dropped


def _score_entry(intent: EvalIntent, entry: GalleryEntry) -> float:
    intent_topics = set(intent.topics)
    intent_skills = set(intent.skills)
    score = 0.0
    score += 2.0 * len(intent_topics & set(entry.topics))
    score += 1.0 * len(intent_skills & set(entry.skills))
    lowered_goal = (intent.raw_goal or "").lower()
    if entry.task.lower() in lowered_goal:
        score += 3.0
    return score


def _explain(intent: EvalIntent, entry: GalleryEntry, score: float) -> str:
    matches = []
    intent_topics = set(intent.topics)
    matched_topics = sorted(intent_topics & set(entry.topics))
    if matched_topics:
        matches.append(f"topics={matched_topics}")
    intent_skills = set(intent.skills)
    matched_skills = sorted(intent_skills & set(entry.skills))
    if matched_skills:
        matches.append(f"skills={matched_skills}")
    if entry.task.lower() in (intent.raw_goal or "").lower():
        matches.append("task_name_in_goal")
    detail = "; ".join(matches) or "fallback"
    return f"score={score:.1f}; {detail}; metric={entry.headline_metric}"
