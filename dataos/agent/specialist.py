from __future__ import annotations

import itertools
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataos.agent.events import append_specialist_call
from dataos.llm.client import LLMClient
from dataos.research.agent import DataOSAgent
from dataos.research.models import AgentTask
from dataos.research.worker import WorkerReport, worker_run


@dataclass(slots=True)
class SpecialistSubmission:
    agent_task_id: str
    role: str
    priority: int
    context_scope: str | None
    future: Future


@dataclass(slots=True)
class WorkerSubmission:
    task_id: str
    priority: int
    future: Future


class SpecialistPool:
    """W1 pool: priority-ordered submission to a shared ThreadPoolExecutor.

    Specialist work delegates to ``DataOSAgent.run_agent_task``; worker work
    delegates to ``dataos.research.worker.worker_run``. Cross-loop priority
    is enforced **at submission time** (higher-priority submissions sort
    first within a batch). True in-flight preemption is W2.
    """

    def __init__(
        self,
        agent: DataOSAgent,
        workspace_dir: str | Path,
        *,
        llm_client: LLMClient | None = None,
        max_workers: int = 4,
    ) -> None:
        self.agent = agent
        self.workspace_dir = workspace_dir
        self.llm_client = llm_client
        self.max_workers = max(1, int(max_workers))
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="dataos-specialist",
        )
        self._lock = threading.Lock()
        self._seq = itertools.count()
        self._store = agent.store
        self._closed = False

    def submit_specialists(
        self,
        project_id: str,
        roles: tuple[str, ...] | list[str],
        context_scope: str,
        *,
        priority: int = 0,
    ) -> list[SpecialistSubmission]:
        if not roles:
            return []
        spawned: list[AgentTask] = self.agent.spawn_agent_tasks(
            project_id, list(roles), context_scope
        )
        submissions: list[SpecialistSubmission] = []
        with self._lock:
            for task in spawned:
                seq = next(self._seq)
                append_specialist_call(
                    self._store,
                    project_id=project_id,
                    agent_task_id=task.agent_task_id,
                    role=task.role,
                    phase="queued",
                    priority=priority,
                    context_scope=context_scope,
                )
                future = self._executor.submit(
                    self._run_specialist,
                    project_id,
                    task.agent_task_id,
                    task.role,
                    priority,
                    context_scope,
                    seq,
                )
                submissions.append(
                    SpecialistSubmission(
                        agent_task_id=task.agent_task_id,
                        role=task.role,
                        priority=priority,
                        context_scope=context_scope,
                        future=future,
                    )
                )
        return submissions

    def submit_worker(self, task_id: str, *, priority: int = 0) -> WorkerSubmission:
        future = self._executor.submit(self._run_worker, task_id, priority)
        return WorkerSubmission(task_id=task_id, priority=priority, future=future)

    def drain(self, timeout: float | None = None) -> None:
        del timeout  # ThreadPoolExecutor.shutdown lacks a per-task timeout in stdlib.
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)

    # ------------------------------------------------------------------
    def _run_specialist(
        self,
        project_id: str,
        agent_task_id: str,
        role: str,
        priority: int,
        context_scope: str,
        seq: int,
    ) -> Any:
        del seq
        append_specialist_call(
            self._store,
            project_id=project_id,
            agent_task_id=agent_task_id,
            role=role,
            phase="started",
            priority=priority,
            context_scope=context_scope,
        )
        try:
            proposal = self.agent.run_agent_task(agent_task_id)
            append_specialist_call(
                self._store,
                project_id=project_id,
                agent_task_id=agent_task_id,
                role=role,
                phase="completed",
                priority=priority,
                context_scope=context_scope,
            )
            return proposal
        except Exception as exc:
            append_specialist_call(
                self._store,
                project_id=project_id,
                agent_task_id=agent_task_id,
                role=role,
                phase="failed",
                priority=priority,
                context_scope=context_scope,
                error=str(exc),
            )
            raise

    def _run_worker(self, task_id: str, priority: int) -> WorkerReport:
        del priority
        return worker_run(task_id, self.workspace_dir, llm_client=self.llm_client)
