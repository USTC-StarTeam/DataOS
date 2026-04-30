from __future__ import annotations

import signal
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataos.agent.events import (
    append_daemon_lifecycle,
    append_recovery_action,
)
from dataos.agent.loops import (
    AnalyzerLoop,
    ExecutorLoop,
    GeneratorLoop,
    Loop,
    LoopContext,
    ReporterLoop,
)
from dataos.agent.specialist import SpecialistPool
from dataos.llm.client import LLMClient
from dataos.research.agent import DataOSAgent
from dataos.research.scheduler import recover_stale_tasks
from dataos.research.store import ResearchStore


@dataclass(slots=True)
class DaemonCadences:
    generator: float = 30.0
    executor: float = 2.0
    analyzer: float = 15.0
    reporter: float = 60.0


class Daemon:
    """Long-running supervisor that drives four independent loops.

    Reuses the existing ``DataOSAgent`` for specialist work and the existing
    ``scheduler.scheduler_tick`` / ``worker.worker_run`` entrypoints. State
    is reconstructed from ``dataos.db`` on every restart — no in-memory
    persistence. Per the W1 plan, real wall-clock backpressure and token
    accounting are deferred to W2.
    """

    def __init__(
        self,
        workspace_dir: str | Path,
        project_id: str,
        *,
        llm_client: LLMClient | None = None,
        cadences: DaemonCadences | None = None,
        stale_seconds: int = 900,
        max_specialist_workers: int = 4,
        install_signal_handlers: bool = True,
    ) -> None:
        self.workspace_dir = workspace_dir
        self.project_id = project_id
        self.llm_client = llm_client
        self.cadences = cadences or DaemonCadences()
        self.stale_seconds = int(stale_seconds)
        self.max_specialist_workers = int(max_specialist_workers)
        self.install_signal_handlers = install_signal_handlers

        self.store = ResearchStore(workspace_dir)
        self.store.prepare()
        self.agent = DataOSAgent(workspace_dir, llm_client)
        self.pool = SpecialistPool(
            agent=self.agent,
            workspace_dir=workspace_dir,
            llm_client=llm_client,
            max_workers=self.max_specialist_workers,
        )
        ctx = LoopContext(
            store=self.store,
            agent=self.agent,
            pool=self.pool,
            project_id=project_id,
            workspace_dir=workspace_dir,
        )
        self.loops: list[Loop] = [
            GeneratorLoop(context=ctx, cadence_seconds=self.cadences.generator),
            ExecutorLoop(context=ctx, cadence_seconds=self.cadences.executor),
            AnalyzerLoop(context=ctx, cadence_seconds=self.cadences.analyzer),
            ReporterLoop(context=ctx, cadence_seconds=self.cadences.reporter),
        ]
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._started = False
        self._previous_signal_handlers: dict[int, Any] = {}

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._run_startup_recovery()
        if self.install_signal_handlers:
            self._install_signal_handlers()
        append_daemon_lifecycle(
            self.store,
            project_id=self.project_id,
            phase="started",
            payload={"loop_names": [loop.name for loop in self.loops]},
        )
        for loop in self.loops:
            thread = threading.Thread(
                target=loop.run,
                args=(self._stop,),
                name=f"dataos-loop-{loop.name}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def shutdown(self, *, drain_timeout: float = 30.0) -> None:
        if not self._started:
            return
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=drain_timeout)
        self._threads.clear()
        self.pool.drain(timeout=drain_timeout)
        if self.install_signal_handlers:
            self._restore_signal_handlers()
        append_daemon_lifecycle(
            self.store,
            project_id=self.project_id,
            phase="stopped",
            payload={},
        )
        self._started = False

    def run_forever(self) -> None:
        self.start()
        try:
            while not self._stop.wait(1.0):
                pass
        finally:
            self.shutdown()

    # ------------------------------------------------------------------
    def _run_startup_recovery(self) -> None:
        report = recover_stale_tasks(
            self.workspace_dir,
            older_than_seconds=self.stale_seconds,
            project_id=self.project_id,
        )
        append_recovery_action(
            self.store,
            project_id=self.project_id,
            requeued_task_ids=list(report.requeued_task_ids),
            older_than_seconds=self.stale_seconds,
        )

    def _install_signal_handlers(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                self._previous_signal_handlers[sig] = signal.signal(sig, self._handle_signal)
            except (ValueError, OSError):
                self._previous_signal_handlers.pop(sig, None)

    def _restore_signal_handlers(self) -> None:
        for sig, handler in self._previous_signal_handlers.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass
        self._previous_signal_handlers.clear()

    def _handle_signal(self, signum: int, frame: Any) -> None:
        del signum, frame
        self._stop.set()
