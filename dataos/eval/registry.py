from __future__ import annotations

from typing import Callable

from dataos.eval.adapters.base import EvalModelAdapter
from dataos.eval.providers.base import EvalProvider
from dataos.eval.tasks.base import EvalTaskSpec

_PROVIDERS: dict[str, Callable[[], EvalProvider]] = {}
_TASKS: dict[str, Callable[[], EvalTaskSpec]] = {}
_ADAPTERS: dict[str, Callable[[], EvalModelAdapter]] = {}


def register_provider(name: str, factory: Callable[[], EvalProvider]) -> None:
    _PROVIDERS[name] = factory


def register_task(name: str, factory: Callable[[], EvalTaskSpec]) -> None:
    _TASKS[name] = factory


def register_adapter(name: str, factory: Callable[[], EvalModelAdapter]) -> None:
    _ADAPTERS[name] = factory


def get_eval_provider(name: str) -> EvalProvider:
    if name not in _PROVIDERS:
        available = ", ".join(sorted(_PROVIDERS)) or "(none)"
        raise ValueError(f"Unsupported eval provider '{name}'. Available: {available}")
    return _PROVIDERS[name]()


def get_eval_task(name: str) -> EvalTaskSpec:
    if name not in _TASKS:
        available = ", ".join(sorted(_TASKS)) or "(none)"
        raise ValueError(f"Unsupported eval task '{name}'. Available: {available}")
    return _TASKS[name]()


def get_eval_adapter(name: str) -> EvalModelAdapter:
    if name not in _ADAPTERS:
        available = ", ".join(sorted(_ADAPTERS)) or "(none)"
        raise ValueError(f"Unsupported eval adapter '{name}'. Available: {available}")
    return _ADAPTERS[name]()


# -- Built-in registrations (lazy imports to avoid heavy dependencies at import time) --


def _native_provider():
    from dataos.eval.providers.native import NativeEvalProvider
    return NativeEvalProvider()


def _stub_provider():
    from dataos.eval.providers.stub import StubEvalProvider
    return StubEvalProvider()


def _gsm8k_task():
    from dataos.eval.tasks.gsm8k import GSM8KTask
    return GSM8KTask()


def _aime2025_task():
    from dataos.eval.tasks.aime2025 import AIME2025Task
    return AIME2025Task()


def _physics_task():
    from dataos.eval.tasks.physics import PhysicsTask
    return PhysicsTask()


def _toxicity_detection_task():
    from dataos.eval.tasks.toxicity_detection import ToxicityDetectionTask
    return ToxicityDetectionTask()


def _climaqa_task():
    from dataos.eval.tasks.climaqa import ClimaQATask
    return ClimaQATask()


def _openfindata_task():
    from dataos.eval.tasks.openfindata import OpenFinDataTask
    return OpenFinDataTask()


def _livecodebench_task():
    from dataos.eval.tasks.livecodebench import LiveCodeBenchTask
    return LiveCodeBenchTask()


def _transformers_adapter():
    from dataos.eval.adapters.transformers import TransformersAdapter
    return TransformersAdapter()


def _openai_compatible_adapter():
    from dataos.eval.adapters.openai_compatible import OpenAICompatibleAdapter
    return OpenAICompatibleAdapter()


register_provider("native", _native_provider)
register_provider("stub", _stub_provider)
register_task("gsm8k", _gsm8k_task)
register_task("aime2025", _aime2025_task)
register_task("physics", _physics_task)
register_task("toxicity_detection", _toxicity_detection_task)
register_task("climaqa", _climaqa_task)
register_task("openfindata", _openfindata_task)
register_task("livecodebench", _livecodebench_task)
register_adapter("transformers", _transformers_adapter)
register_adapter("openai_compatible", _openai_compatible_adapter)
