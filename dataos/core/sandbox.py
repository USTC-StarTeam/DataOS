from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Sandbox:
    root: Path | str

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    @property
    def inputs_dir(self) -> Path:
        return self.root / "inputs"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def tmp_dir(self) -> Path:
        return self.root / "tmp"

    @property
    def code_dir(self) -> Path:
        return self.root / "code"

    def prepare(self) -> None:
        for path in (
            self.inputs_dir,
            self.outputs_dir,
            self.artifacts_dir,
            self.logs_dir,
            self.tmp_dir,
            self.code_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def resolve(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def artifact_dir(self, artifact_type: str, artifact_id: str) -> Path:
        target = self.artifacts_dir / f"{artifact_type}s" / artifact_id
        target.mkdir(parents=True, exist_ok=True)
        return target

    def log(self, name: str, content: str) -> Path:
        filename = name if name.endswith(".log") else f"{name}.log"
        target = self.logs_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def write_json(self, rel_path: str, data: dict[str, Any]) -> Path:
        target = self.root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        return target

    def report_path(self) -> Path:
        return self.root / "report.json"
