"""Core DataOS abstractions.

``LLMPlanningAgent`` / ``PlanStep`` are exposed lazily (PEP 562 ``__getattr__``)
so that importing the lightweight core abstractions — or the agent-less
``dataos.runtime.prepared`` entrypoint, which depends on them — does not eagerly
load the planning agent and its LLM client stack. ``dataos.core.LLMPlanningAgent``
still resolves on first access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from dataos.core.agent import LLMPlanningAgent, PlanStep

__all__ = [
    "Artifact",
    "DatasetArtifact",
    "EvalArtifact",
    "Experiment",
    "ExperimentArtifact",
    "LLMPlanningAgent",
    "LocalArtifactStore",
    "ModelArtifact",
    "Operator",
    "OperatorParameter",
    "OperatorRegistry",
    "PlanStep",
    "ResultArtifact",
    "Sandbox",
    "TaskTemplate",
]


def __getattr__(name: str) -> Any:
    if name in {"LLMPlanningAgent", "PlanStep"}:
        from dataos.core import agent

        return getattr(agent, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
