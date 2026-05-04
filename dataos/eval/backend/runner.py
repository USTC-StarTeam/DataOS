from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dataos.core.artifact import ModelArtifact
from dataos.eval.backend.lm import DataOSLM, _GenerateUntilAdapter


@dataclass(slots=True)
class LmEvalRunOutput:
    tasks: list[str]
    results: dict[str, Any]
    samples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def run_lm_eval(
    model_artifact: ModelArtifact,
    adapter: _GenerateUntilAdapter,
    *,
    tasks: list[str],
    limit: int | None = None,
    num_fewshot: int | None = None,
    batch_size: int = 8,
    max_new_tokens: int = 256,
    apply_chat_template: bool = False,
    log_samples: bool = True,
    confirm_run_unsafe_code: bool = False,
    adapter_config: dict[str, Any] | None = None,
    extra_simple_evaluate_kwargs: dict[str, Any] | None = None,
) -> LmEvalRunOutput:
    """Run lm-eval ``simple_evaluate`` against a DataOS adapter.

    The lm-eval package must be importable; install via
    ``pip install lm-eval`` (with the ``api`` extra if needed).
    """
    from lm_eval import simple_evaluate

    lm = DataOSLM(
        model_artifact,
        adapter,
        adapter_config=adapter_config,
        batch_size=batch_size,
        default_max_new_tokens=max_new_tokens,
    )
    extra: dict[str, Any] = dict(extra_simple_evaluate_kwargs or {})
    raw = simple_evaluate(
        model=lm,
        tasks=list(tasks),
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
        log_samples=log_samples,
        apply_chat_template=apply_chat_template,
        confirm_run_unsafe_code=confirm_run_unsafe_code,
        bootstrap_iters=0,
        **extra,
    )
    if raw is None:
        return LmEvalRunOutput(tasks=list(tasks), results={}, samples={}, raw={})
    raw_dict = raw if isinstance(raw, dict) else dict(raw)
    return LmEvalRunOutput(
        tasks=list(tasks),
        results=dict(raw_dict.get("results", {})),
        samples=dict(raw_dict.get("samples", {})),
        raw=raw_dict,
    )
