from __future__ import annotations

from importlib import import_module

__all__ = [
    "AIME2025Task",
    "ClimaQATask",
    "GSM8KTask",
    "LiveCodeBenchTask",
    "OpenFinDataTask",
    "PhysicsTask",
    "ToxicityDetectionTask",
]

_TASK_MODULES = {
    "AIME2025Task": "dataos.eval.tasks.aime2025",
    "ClimaQATask": "dataos.eval.tasks.climaqa",
    "GSM8KTask": "dataos.eval.tasks.gsm8k",
    "LiveCodeBenchTask": "dataos.eval.tasks.livecodebench",
    "OpenFinDataTask": "dataos.eval.tasks.openfindata",
    "PhysicsTask": "dataos.eval.tasks.physics",
    "ToxicityDetectionTask": "dataos.eval.tasks.toxicity_detection",
}


def __getattr__(name: str):
    if name not in _TASK_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_TASK_MODULES[name])
    return getattr(module, name)
