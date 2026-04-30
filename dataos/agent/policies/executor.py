from __future__ import annotations

from dataos.agent.actions import Action
from dataos.agent.snapshot import StateSnapshot


def decide(snapshot: StateSnapshot) -> list[Action]:
    project_id = str(snapshot.project["project_id"])
    if snapshot.project_status not in {"created", "running"}:
        return [Action(kind="skip", project_id=project_id, reason="project_inactive")]
    return [
        Action(
            kind="tick_scheduler",
            project_id=project_id,
            priority=4,
        )
    ]
