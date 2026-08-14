"""The structural contract a launcher satisfies.

Kept as a Protocol rather than a base class so a launcher can come from anywhere — an
entry point, another package, a test double — without importing this one at definition
time. ``Backend`` lives in :mod:`dataos_train.protocols` because it is about producing a
job, not about running one.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import Handle, JobSpec, Status


@runtime_checkable
class Launcher(Protocol):
    name: str

    def submit(self, job: JobSpec) -> Handle: ...
    def status(self, handle: Handle) -> Status: ...
    def logs(self, handle: Handle, *, tail: int | None = None) -> str: ...
    def cancel(self, handle: Handle) -> None: ...
