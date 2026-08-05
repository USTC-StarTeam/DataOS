"""Structural protocols for backends and launchers.

Online operators are duck-typed (no shared base class); their protocol is documented in
:mod:`dataos_train.online` and ``docs/online_ops.md``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .spec import TrainSpec
from .types import Capabilities, Handle, JobSpec, Status, TrainResult


@runtime_checkable
class Backend(Protocol):
    name: str

    def capabilities(self) -> Capabilities: ...
    def validate(self, spec: TrainSpec) -> list[str]: ...
    def build(self, spec: TrainSpec) -> JobSpec: ...
    def collect(self, run_dir: str) -> TrainResult: ...


@runtime_checkable
class Launcher(Protocol):
    name: str

    def submit(self, job: JobSpec) -> Handle: ...
    def status(self, handle: Handle) -> Status: ...
    def logs(self, handle: Handle, *, tail: int | None = None) -> str: ...
    def cancel(self, handle: Handle) -> None: ...
