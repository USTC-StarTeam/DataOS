"""The structural protocol a training backend satisfies.

A backend turns a ``TrainSpec`` into a launcher-agnostic ``JobSpec``; running that job is
:mod:`dataos_launch`'s concern, and the ``Launcher`` protocol lives there. Online operators are
duck-typed (no shared base class); their protocol is documented in :mod:`dataos_train.online`
and ``docs/online_ops.md``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from dataos_launch.types import JobSpec

from .spec import TrainSpec
from .types import Capabilities, TrainResult


@runtime_checkable
class Backend(Protocol):
    name: str

    def capabilities(self) -> Capabilities: ...
    def validate(self, spec: TrainSpec) -> list[str]: ...
    def build(self, spec: TrainSpec) -> JobSpec: ...
    def collect(self, run_dir: str) -> TrainResult: ...
