from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


ActionKind = Literal[
    "spawn_specialists",
    "accept_proposals",
    "tick_scheduler",
    "skip",
]


@dataclass(frozen=True, slots=True)
class Action:
    kind: ActionKind
    project_id: str
    roles: tuple[str, ...] = ()
    context_scope: str | None = None
    proposal_ids: tuple[str, ...] = ()
    priority: int = 0
    reason: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "project_id": self.project_id,
            "roles": list(self.roles),
            "context_scope": self.context_scope,
            "proposal_ids": list(self.proposal_ids),
            "priority": self.priority,
            "reason": self.reason,
            "payload": dict(self.payload),
        }
