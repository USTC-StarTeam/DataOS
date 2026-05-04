"""Natural-language planner for DataOS eval.

User-facing entry point: :func:`plan_eval` takes a free-form goal string and
returns a ranked list of :class:`EvalPlanItem` objects keyed by lm-eval task
name. :func:`execute_plan` then drives those items through the lm-eval bridge.
"""

from dataos.eval.planner.gallery import Gallery, GalleryEntry, builtin_gallery
from dataos.eval.planner.intent import EvalIntent, keyword_intent, llm_intent
from dataos.eval.planner.llm import default_llm_client
from dataos.eval.planner.resolver import EvalPlanItem, plan_eval, resolve_tasks
from dataos.eval.planner.runner import PlanItemResult, PlanRunReport, execute_plan

__all__ = [
    "EvalIntent",
    "EvalPlanItem",
    "Gallery",
    "GalleryEntry",
    "PlanItemResult",
    "PlanRunReport",
    "builtin_gallery",
    "default_llm_client",
    "execute_plan",
    "keyword_intent",
    "llm_intent",
    "plan_eval",
    "resolve_tasks",
]
