"""Project — the outermost container: a research project = a sequence of studies.

A project is just a directory holding studies. The first study's plan comes from the
user's initial instruct; each later study's plan is the agent's call from the previous
study's review (the "多 study 迭代" loop). **Resume is project-scoped**: reopening a
project rehydrates every study — and through them every experiment / stage / artifact —
off disk, so re-driving the project continues where it left off (completed stages skip).

On-disk layout::

    <workspace>/<project_id>/
        project.json                 # project metadata (goal / initial instruct / timestamps)
        <study_id>/
            plan.json
            review.json
            experiments/<slug>/{data,train,eval}.json + report.json + dataset/ model/ eval/ logs/

``dataos.project(id=...)`` is open-or-create: re-opening an existing project dir loads its
``project.json`` back, so a fresh process (Claude reopening the work) sees the whole project.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dataos._ids import _utc_now_iso
from dataos._study import Study
from dataos._study import study as _open_study

_JSON_KWARGS: dict[str, Any] = {
    "indent": 2,
    "sort_keys": True,
    "ensure_ascii": False,
    "default": str,
}
_SERIAL_RE = re.compile(r"^project-(\d+)$")


@dataclass
class Project:
    """A research project: a directory of studies, resumable as one unit."""

    project_id: str
    workspace: Path | str  # the ROOT dir; project_dir = workspace / project_id
    meta: dict[str, Any] = field(default_factory=dict)  # {"goal", "instruct", ...}
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)

    @property
    def id(self) -> str:
        return self.project_id

    @property
    def project_dir(self) -> Path:
        return Path(self.workspace) / self.project_id

    @property
    def project_path(self) -> Path:
        return self.project_dir / "project.json"

    @property
    def studies(self) -> list[str]:
        """Study ids (subdirectories) that exist on disk, sorted."""
        if not self.project_dir.is_dir():
            return []
        return sorted(p.name for p in self.project_dir.iterdir() if p.is_dir())

    def set_goal(self, goal: str | None = None, instruct: str | None = None) -> "Project":
        """Record the project's goal and/or the user's initial instruct, persist ``project.json``."""
        if goal is not None:
            self.meta["goal"] = goal
        if instruct is not None:
            self.meta["instruct"] = instruct
        self._persist()
        return self

    def study(self, id: str | None = None) -> Study:
        """Create or **reopen** a study under this project (rooted at the project dir).

        ``id=None`` mints a serial ``study-0001``. Reopening reads the study's ``plan.json``
        back; its experiments rehydrate lazily via :meth:`Study.experiment` / :meth:`Study.load_experiments`.
        """
        return _open_study(id=id, workspace=self.project_dir)

    def load(self) -> list[Study]:
        """Reopen every study in the project, each with its experiments rehydrated from disk.

        This is the project-scoped resume entry: ``dataos.project(id).load()`` brings the whole
        research project back as objects (state / lineage / artifacts restored)."""
        loaded: list[Study] = []
        for study_id in self.studies:
            study_obj = self.study(study_id)
            study_obj.load_experiments()
            loaded.append(study_obj)
        return loaded

    def summary(self) -> dict[str, Any]:
        """Project-level overview: each study's deterministic ``summary`` (best / varied keys).
        Reads from disk, so it reflects whatever has been run so far."""
        studies: list[dict[str, Any]] = []
        for study_id in self.studies:
            digest = self.study(study_id).summary()
            studies.append({
                "study_id": study_id,
                "goal": digest.get("goal"),
                "n_experiments": digest.get("n_experiments"),
                "varied_keys": digest.get("varied_keys"),
                "best": digest.get("best"),
            })
        return {
            "project_id": self.project_id,
            "goal": self.meta.get("goal"),
            "instruct": self.meta.get("instruct"),
            "n_studies": len(studies),
            "studies": studies,
        }

    def _persist(self) -> None:
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.updated_at = _utc_now_iso()
        payload = {
            "project_id": self.project_id,
            "meta": self.meta,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        self.project_path.write_text(json.dumps(payload, **_JSON_KWARGS), encoding="utf-8")


def project(id: str | None = None, workspace: str | Path = "workspace") -> Project:
    """Create or **reopen** a :class:`Project`.

    ``id=None`` mints a new serial (``project-0001`` ...). A given ``id`` is open-or-create:
    an existing project dir is reopened and its ``project.json`` loaded back, so a fresh
    process can attach to the whole research project by id and resume it.
    """
    root = Path(workspace)
    project_id = _next_serial(root) if id is None else str(id)
    project_dir = root / project_id
    project_dir.mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {}
    created = _utc_now_iso()
    project_path = project_dir / "project.json"
    if project_path.is_file():
        try:
            data = json.loads(project_path.read_text(encoding="utf-8")) or {}
            meta = data.get("meta") or {}
            created = data.get("created_at") or created
        except (OSError, json.JSONDecodeError):
            pass

    proj = Project(project_id=project_id, workspace=root, meta=meta, created_at=created)
    proj._persist()
    return proj


def _next_serial(root: Path) -> str:
    highest = 0
    if root.is_dir():
        for entry in root.iterdir():
            match = _SERIAL_RE.match(entry.name)
            if entry.is_dir() and match:
                highest = max(highest, int(match.group(1)))
    return f"project-{highest + 1:04d}"
