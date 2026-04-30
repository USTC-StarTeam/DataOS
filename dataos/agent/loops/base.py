from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from dataos.agent.actions import Action
from dataos.agent.events import append_loop_tick
from dataos.agent.snapshot import StateSnapshot, build_snapshot
from dataos.agent.specialist import SpecialistPool
from dataos.research.agent import DataOSAgent
from dataos.research.store import ResearchStore


MonotonicClock = Callable[[], float]
WallClock = Callable[[], datetime]


def _default_wall_clock() -> datetime:
    return datetime.now(UTC)


@dataclass(slots=True)
class LoopContext:
    store: ResearchStore
    agent: DataOSAgent
    pool: SpecialistPool
    project_id: str
    workspace_dir: str | Path


class Loop(ABC):
    """Base loop class.

    Subclasses set ``name`` and ``DEFAULT_CADENCE_SECONDS`` as class
    variables, and implement ``policy_decide`` and ``_dispatch``. Threading
    and cadence enforcement live here. LLM calls do not.
    """

    name: str = "loop"
    DEFAULT_CADENCE_SECONDS: float = 1.0

    def __init__(
        self,
        context: LoopContext,
        *,
        cadence_seconds: float | None = None,
        monotonic_clock: MonotonicClock | None = None,
        wall_clock: WallClock | None = None,
    ) -> None:
        self.context = context
        self.cadence_seconds = (
            float(cadence_seconds) if cadence_seconds is not None else self.DEFAULT_CADENCE_SECONDS
        )
        self.monotonic_clock = monotonic_clock or time.monotonic
        self.wall_clock = wall_clock or _default_wall_clock
        self._last_tick_monotonic: float = float("-inf")

    def due(self, now_monotonic: float | None = None) -> bool:
        ts = self.monotonic_clock() if now_monotonic is None else now_monotonic
        return (ts - self._last_tick_monotonic) >= self.cadence_seconds

    def tick(self) -> list[Action]:
        now_monotonic = self.monotonic_clock()
        if not self.due(now_monotonic):
            return []
        self._last_tick_monotonic = now_monotonic
        snapshot = build_snapshot(
            self.context.store,
            self.context.project_id,
            self.wall_clock(),
        )
        actions = self.policy_decide(snapshot)
        self._dispatch(actions, snapshot)
        append_loop_tick(
            self.context.store,
            project_id=self.context.project_id,
            loop_name=self.name,
            actions=[action.to_record() for action in actions],
            queue_depth=snapshot.queue_depth,
        )
        return actions

    def run(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - daemon must keep running
                # Loop exceptions are isolated; the next tick will retry.
                pass
            stop_event.wait(self.cadence_seconds)

    @abstractmethod
    def policy_decide(self, snapshot: StateSnapshot) -> list[Action]: ...

    @abstractmethod
    def _dispatch(self, actions: list[Action], snapshot: StateSnapshot) -> None: ...
