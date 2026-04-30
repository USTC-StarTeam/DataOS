from __future__ import annotations

from dataos.agent.actions import Action
from dataos.agent.loops.base import Loop
from dataos.agent.policies import reporter as reporter_policy
from dataos.agent.snapshot import StateSnapshot


class ReporterLoop(Loop):
    name = "reporter"
    DEFAULT_CADENCE_SECONDS = 60.0

    def policy_decide(self, snapshot: StateSnapshot) -> list[Action]:
        return reporter_policy.decide(snapshot)

    def _dispatch(self, actions: list[Action], snapshot: StateSnapshot) -> None:
        del snapshot
        for action in actions:
            if action.kind == "spawn_specialists":
                self.context.pool.submit_specialists(
                    action.project_id,
                    action.roles,
                    action.context_scope or "final_report",
                    priority=action.priority,
                )
