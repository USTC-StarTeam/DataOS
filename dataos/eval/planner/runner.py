"""Execute a plan: dispatch each EvalPlanItem through the lm-eval bridge.

Keeps the surface tiny on purpose — the runner just iterates plan items, calls
``run_lm_eval`` for each, and collects per-task ``LmEvalRunOutput`` plus a
flat headline-score table. Persisting to sandbox artifacts is left to the
caller (or to ``run_eval_stage``) so we don't duplicate ``store.py`` here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from dataos.core.artifact import ModelArtifact
from dataos.eval.planner.resolver import EvalPlanItem


@dataclass
class PlanItemResult:
    item: EvalPlanItem
    elapsed_s: float
    headline_score: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    raw_output: Any = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class PlanRunReport:
    model_id: str
    results: list[PlanItemResult] = field(default_factory=list)

    def headline_table(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for result in self.results:
            row: dict[str, Any] = {
                "task": result.item.task,
                "score": result.headline_score,
                "metric": result.item.headline_metric,
                "elapsed_s": round(result.elapsed_s, 1),
                "rationale": result.item.rationale,
            }
            if not result.ok:
                row["error"] = result.error
            rows.append(row)
        return rows


def execute_plan(
    plan: list[EvalPlanItem],
    *,
    model_artifact: ModelArtifact,
    adapter_factory: Callable[[str], Any] | None = None,
) -> PlanRunReport:
    """Run each plan item through ``run_lm_eval`` and collect results.

    ``adapter_factory(name)`` returns the DataOS adapter instance to use; when
    omitted, the built-in transformers/openai_compatible factories are used.
    """
    from dataos.eval.backend import run_lm_eval

    factory = adapter_factory or _default_adapter_factory
    report = PlanRunReport(model_id=model_artifact.artifact_id)
    for item in plan:
        adapter = factory(item.adapter)
        start = time.time()
        try:
            output = run_lm_eval(
                model_artifact,
                adapter,
                tasks=[item.task],
                limit=item.limit,
                num_fewshot=item.num_fewshot,
                batch_size=item.batch_size,
                max_new_tokens=item.max_new_tokens,
                apply_chat_template=item.apply_chat_template,
                confirm_run_unsafe_code=item.confirm_run_unsafe_code,
                adapter_config=dict(item.adapter_config),
            )
            elapsed = time.time() - start
            metrics = _flatten_metrics(output.results)
            headline = _pick_headline(metrics, item.task, item.headline_metric)
            report.results.append(
                PlanItemResult(
                    item=item,
                    elapsed_s=elapsed,
                    headline_score=headline,
                    metrics=metrics,
                    raw_output=output,
                )
            )
        except Exception as exc:  # noqa: BLE001 — keep going across items
            report.results.append(
                PlanItemResult(
                    item=item,
                    elapsed_s=time.time() - start,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return report


def _flatten_metrics(results: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for task_name, metrics in (results or {}).items():
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and "stderr" not in key and key != "alias":
                flat[f"{task_name}::{key}"] = float(value)
    return flat


def _pick_headline(
    metrics: dict[str, float], task: str, headline_metric: str
) -> float | None:
    target = f"{task}::{headline_metric}"
    if target in metrics:
        return metrics[target]
    for key, value in metrics.items():
        if key.startswith(f"{task}::"):
            return value
    return None


def _default_adapter_factory(name: str) -> Any:
    if name == "transformers":
        from dataos.eval.adapters.transformers import TransformersAdapter

        return TransformersAdapter()
    if name == "openai_compatible":
        from dataos.eval.adapters.openai_compatible import OpenAICompatibleAdapter

        return OpenAICompatibleAdapter()
    raise ValueError(f"Unknown adapter for plan execution: {name!r}")
