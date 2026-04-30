from __future__ import annotations

from dataos.agent.actions import Action
from dataos.agent.loops.base import Loop
from dataos.agent.policies import analyzer as analyzer_policy
from dataos.agent.snapshot import StateSnapshot


class AnalyzerLoop(Loop):
    name = "analyzer"
    DEFAULT_CADENCE_SECONDS = 15.0

    def policy_decide(self, snapshot: StateSnapshot) -> list[Action]:
        return analyzer_policy.decide(snapshot)

    def _dispatch(self, actions: list[Action], snapshot: StateSnapshot) -> None:
        del snapshot
        for action in actions:
            if action.kind == "spawn_specialists":
                self.context.pool.submit_specialists(
                    action.project_id,
                    action.roles,
                    action.context_scope or "result_analysis",
                    priority=action.priority,
                )
