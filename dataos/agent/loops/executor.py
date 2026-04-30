from __future__ import annotations

from dataos.agent.actions import Action
from dataos.agent.loops.base import Loop
from dataos.agent.policies import executor as executor_policy
from dataos.agent.snapshot import StateSnapshot
from dataos.research.scheduler import scheduler_tick


class ExecutorLoop(Loop):
    name = "executor"
    DEFAULT_CADENCE_SECONDS = 2.0

    def policy_decide(self, snapshot: StateSnapshot) -> list[Action]:
        return executor_policy.decide(snapshot)

    def _dispatch(self, actions: list[Action], snapshot: StateSnapshot) -> None:
        del snapshot
        for action in actions:
            if action.kind != "tick_scheduler":
                continue
            report = scheduler_tick(
                self.context.workspace_dir,
                project_id=action.project_id,
            )
            for task_id in report.scheduled_task_ids:
                self.context.pool.submit_worker(task_id, priority=action.priority)
