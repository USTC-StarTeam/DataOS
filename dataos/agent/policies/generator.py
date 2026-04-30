from __future__ import annotations

from dataos.agent.actions import Action
from dataos.agent.snapshot import StateSnapshot


INITIAL_AGENT_ROLES: tuple[str, ...] = (
    "hypothesis",
    "experiment_planner",
    "operator_designer",
    "critic",
)


def decide(snapshot: StateSnapshot) -> list[Action]:
    project_id = str(snapshot.project["project_id"])
    if snapshot.project_status not in {"created", "running"}:
        return [Action(kind="skip", project_id=project_id, reason="project_inactive")]
    if snapshot.proposed_proposals:
        proposal_ids = tuple(
            str(proposal["proposal_id"]) for proposal in snapshot.proposed_proposals
        )
        return [
            Action(
                kind="accept_proposals",
                project_id=project_id,
                proposal_ids=proposal_ids,
                priority=5,
            )
        ]
    if (
        not snapshot.agent_tasks
        and not snapshot.hypotheses
        and not snapshot.all_proposals
    ):
        return [
            Action(
                kind="spawn_specialists",
                project_id=project_id,
                roles=INITIAL_AGENT_ROLES,
                context_scope="initial_campaign",
                priority=3,
            )
        ]
    return [Action(kind="skip", project_id=project_id, reason="generator_idle")]
