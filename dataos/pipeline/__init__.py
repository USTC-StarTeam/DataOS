from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataos.core.agent import PlanStep
from dataos.core.registry import OperatorRegistry


@dataclass(slots=True)
class PipelineSpec:
    name: str
    path: Path
    steps: list[PlanStep]
    description: str | None = None

    def to_plan_record(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "pipeline_name": self.name,
            "pipeline_path": str(self.path),
            "steps": [step.to_record() for step in self.steps],
            "reasoning_summary": f"Using explicit pipeline {self.name} from {self.path}.",
            "expected_gain": "Run the selected deterministic pipeline without agent planning.",
        }
        if self.description:
            payload["description"] = self.description
        return payload


def load_pipeline(path: str | Path, registry: OperatorRegistry) -> PipelineSpec:
    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Pipeline file must contain a JSON object")
    steps_payload = payload.get("steps")
    if not isinstance(steps_payload, list) or not steps_payload:
        raise ValueError("Pipeline file requires a non-empty steps list")

    operator_lookup = {operator.name: operator for operator in registry.list()}
    steps: list[PlanStep] = []
    for item in steps_payload:
        if not isinstance(item, dict):
            raise ValueError("Each pipeline step must be an object")
        operator_name = item.get("operator_name")
        if operator_name not in operator_lookup:
            raise ValueError(f"Pipeline uses unregistered operator: {operator_name}")
        params = item.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("Pipeline step params must be an object")
        steps.append(
            PlanStep(
                operator_name=operator_name,
                params=operator_lookup[operator_name].normalize_params(params),
            )
        )

    return PipelineSpec(
        name=str(payload.get("name") or target.stem),
        path=target,
        steps=steps,
        description=str(payload["description"]) if payload.get("description") is not None else None,
    )


__all__ = ["PipelineSpec", "load_pipeline"]
