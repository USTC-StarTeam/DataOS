"""Runtime helpers.

Imports are lazy so that loading the agent-less ``run_prepared`` entrypoint does
not eagerly pull in :mod:`dataos.runtime.runner` (and its embedded
``LLMPlanningAgent`` / LLM client stack). ``run_prepared`` must work with no LLM
API key and without importing the planning agent at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "RunPreparedResult",
    "RunTaskResult",
    "build_default_registry",
    "run_prepared",
    "run_task",
]

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from dataos.operators import build_default_registry
    from dataos.runtime.prepared import RunPreparedResult, run_prepared
    from dataos.runtime.runner import RunTaskResult, run_task


def __getattr__(name: str) -> Any:
    if name in {"RunPreparedResult", "run_prepared"}:
        from dataos.runtime import prepared

        return getattr(prepared, name)
    if name in {"RunTaskResult", "run_task"}:
        from dataos.runtime import runner

        return getattr(runner, name)
    if name == "build_default_registry":
        from dataos.operators import build_default_registry

        return build_default_registry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
