from __future__ import annotations

from dataos.agent.actions import Action
from dataos.agent.loops.base import Loop
from dataos.agent.policies import generator as generator_policy
from dataos.agent.snapshot import StateSnapshot


class GeneratorLoop(Loop):
    name = "generator"
    DEFAULT_CADENCE_SECONDS = 30.0

    def policy_decide(self, snapshot: StateSnapshot) -> list[Action]:
        return generator_policy.decide(snapshot)

    def _dispatch(self, actions: list[Action], snapshot: StateSnapshot) -> None:
        del snapshot
        for action in actions:
            if action.kind == "spawn_specialists":
                self.context.pool.submit_specialists(
                    action.project_id,
                    action.roles,
                    action.context_scope or "initial_campaign",
                    priority=action.priority,
                )
            elif action.kind == "accept_proposals":
                for proposal_id in action.proposal_ids:
                    try:
                        self.context.agent.accept_proposal(proposal_id)
                    except Exception as exc:  # noqa: BLE001
                        self.context.agent.reject_proposal(proposal_id, str(exc))
