from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TaskTemplate:
    task_id: str
    name: str
    goal: str
    input_description: str
    output_description: str
    constraints: dict[str, Any]
    budget: dict[str, Any]
    max_rounds: int
    pipeline_hint: list[str]
    train_config: dict[str, Any]
    eval_config: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskTemplate":
        required = [
            "task_id",
            "name",
            "goal",
            "input_description",
            "output_description",
            "constraints",
            "budget",
            "max_rounds",
            "pipeline_hint",
            "train_config",
            "eval_config",
        ]
        missing = [field_name for field_name in required if field_name not in data]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(f"TaskTemplate missing required fields: {missing_str}")
        return cls(
            task_id=data["task_id"],
            name=data["name"],
            goal=data["goal"],
            input_description=data["input_description"],
            output_description=data["output_description"],
            constraints=dict(data["constraints"]),
            budget=dict(data["budget"]),
            max_rounds=int(data["max_rounds"]),
            pipeline_hint=list(data["pipeline_hint"]),
            train_config=dict(data["train_config"]),
            eval_config=dict(data["eval_config"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "TaskTemplate":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "goal": self.goal,
            "input_description": self.input_description,
            "output_description": self.output_description,
            "constraints": self.constraints,
            "budget": self.budget,
            "max_rounds": self.max_rounds,
            "pipeline_hint": self.pipeline_hint,
            "train_config": self.train_config,
            "eval_config": self.eval_config,
        }
