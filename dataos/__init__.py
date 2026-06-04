"""DataOS public package exports.

Top-level names are exposed lazily (PEP 562 ``__getattr__``) so that importing a
single submodule — for example the agent-less ``dataos.runtime.prepared``
entrypoint — does not eagerly pull in the planning agent (``LLMPlanningAgent``)
or the LLM client stack. The public API is unchanged: ``dataos.run_task``,
``dataos.LLMPlanningAgent`` and friends still resolve on first access; they are
just loaded on demand instead of at package import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from dataos.core.agent import LLMPlanningAgent, PlanStep
    from dataos.core.artifact import (
        Artifact,
        DatasetArtifact,
        EvalArtifact,
        ExperimentArtifact,
        ModelArtifact,
        ResultArtifact,
    )
    from dataos.core.experiment import Experiment
    from dataos.core.operator import Operator, OperatorParameter
    from dataos.core.registry import OperatorRegistry
    from dataos.core.sandbox import Sandbox
    from dataos.core.store import LocalArtifactStore
    from dataos.core.task import TaskTemplate
    from dataos.llm.client import LLMClient
    from dataos.llm.openai_compat import OpenAICompatLLMClient
    from dataos.runtime.runner import run_task

__all__ = [
    "Artifact",
    "DatasetArtifact",
    "EvalArtifact",
    "Experiment",
    "ExperimentArtifact",
    "LLMClient",
    "LLMPlanningAgent",
    "LocalArtifactStore",
    "ModelArtifact",
    "Operator",
    "OperatorParameter",
    "OperatorRegistry",
    "PlanStep",
    "ResultArtifact",
    "OpenAICompatLLMClient",
    "Sandbox",
    "TaskTemplate",
    "run_task",
]

__version__ = "0.1.0"

# Map each lazily-exported name to the module that defines it. Grouped so that
# touching one name only imports its own module (and its dependencies).
_LAZY_EXPORTS: dict[str, str] = {
    "LLMPlanningAgent": "dataos.core.agent",
    "PlanStep": "dataos.core.agent",
    "Artifact": "dataos.core.artifact",
    "DatasetArtifact": "dataos.core.artifact",
    "EvalArtifact": "dataos.core.artifact",
    "ExperimentArtifact": "dataos.core.artifact",
    "ModelArtifact": "dataos.core.artifact",
    "ResultArtifact": "dataos.core.artifact",
    "Experiment": "dataos.core.experiment",
    "Operator": "dataos.core.operator",
    "OperatorParameter": "dataos.core.operator",
    "OperatorRegistry": "dataos.core.registry",
    "Sandbox": "dataos.core.sandbox",
    "LocalArtifactStore": "dataos.core.store",
    "TaskTemplate": "dataos.core.task",
    "LLMClient": "dataos.llm.client",
    "OpenAICompatLLMClient": "dataos.llm.openai_compat",
    "run_task": "dataos.runtime.runner",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)
