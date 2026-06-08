"""The :class:`Study` container — a set of experiments sharing one plan.

A study is just a directory. Each child :class:`~dataos._experiment.Experiment`
writes its own ``report.json`` under ``experiments/<slug>/``; :meth:`Study.summary`
globs those reports off disk and aggregates them. Nothing here knows or cares how
the experiments were driven — one process, many processes, or many machines.
Running them and deciding whether to parallelize is the runtime's job, not
DataOS's.

``dataos.study(id=...)`` is open-or-create: re-opening an existing study dir loads
its ``plan.json`` back, so separate worker processes can attach to the same study
by id.

On-disk layout::

    <workspace>/<study_id>/
        plan.json
        review.json
        experiments/<slug>/{dataset,model,eval,logs,_lf}/ + report.json
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from dataos._experiment import Experiment
from dataos._ids import _make_id, _utc_now_iso
from dataos.workspace import Workspace

_JSON_KWARGS: dict[str, Any] = {
    "indent": 2,
    "sort_keys": True,
    "ensure_ascii": False,
    "default": str,
}
_SERIAL_RE = re.compile(r"^study-(\d+)$")


@dataclass(slots=True)
class Study:
    """A multi-experiment container that produces a review (复盘)."""

    study_id: str
    workspace: Path | str  # the ROOT workspace dir; study_dir = workspace / study_id
    plan: dict[str, Any] | None = None
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)

    # -- derived paths --------------------------------------------------------

    @property
    def study_dir(self) -> Path:
        return Path(self.workspace) / self.study_id

    @property
    def id(self) -> str:
        return self.study_id

    @property
    def dir(self) -> Path:
        return self.study_dir

    @property
    def experiments_dir(self) -> Path:
        return self.study_dir / "experiments"

    @property
    def plan_path(self) -> Path:
        return self.study_dir / "plan.json"

    @property
    def review_path(self) -> Path:
        return self.study_dir / "review.json"

    @property
    def experiments(self) -> list[str]:
        """Slugs of child experiments that have a directory on disk."""
        if not self.experiments_dir.is_dir():
            return []
        return sorted(p.name for p in self.experiments_dir.iterdir() if p.is_dir())

    # -- stages ---------------------------------------------------------------

    def set_plan(self, plan: dict[str, Any]) -> "Study":
        """Record the multi-experiment plan and persist ``plan.json``."""
        self.plan = dict(plan)
        self.updated_at = _utc_now_iso()
        self.study_dir.mkdir(parents=True, exist_ok=True)
        self.plan_path.write_text(json.dumps(self.plan, **_JSON_KWARGS), encoding="utf-8")
        return self

    def experiment(self, spec: dict[str, Any] | None = None) -> Experiment:
        """Create a child :class:`Experiment` under this study.

        The experiment's directory is ``experiments/<slug(name)>/`` where ``name``
        comes from ``spec["name"]`` (give each experiment a distinct, short name —
        the slug is the join key ``compare`` uses).
        """
        name = (spec or {}).get("name") or _make_id("exp")
        workspace = Workspace(self.experiments_dir, _slug(str(name)))
        return Experiment(
            exp_id=_make_id("exp"),
            workspace=workspace,
            name=str(name),
            spec=dict(spec) if spec is not None else None,
        )

    def summary(self) -> dict[str, Any]:
        """Aggregate every child ``report.json`` into a deterministic study summary.

        Reads from disk (not in-memory state), so it works regardless of how the
        experiments were run. Tolerates missing/partial reports.
        """
        rows: list[dict[str, Any]] = []
        for report in self._read_reports():
            evaluation = report.get("eval") or {}
            spec = report.get("spec") or {}
            rows.append(
                {
                    "name": report.get("name") or report.get("id"),
                    "id": report.get("id"),
                    "status": report.get("status"),
                    "score": evaluation.get("score"),
                    "metrics": evaluation.get("metrics") or {},
                    "train": spec.get("train") or {},
                }
            )
        rows.sort(key=lambda row: str(row["name"]))
        varied = _varied_keys([row["train"] for row in rows])

        experiments: list[dict[str, Any]] = []
        for row in rows:
            entry = {
                "name": row["name"],
                "id": row["id"],
                "status": row["status"],
                "score": row["score"],
                "metrics": row["metrics"],
            }
            for key in varied:
                entry[key] = row["train"].get(key)
            experiments.append(entry)

        ranked = sorted(
            rows,
            key=lambda row: (row["score"] is None, -(row["score"] or 0.0), str(row["name"])),
        )
        ranking = [{"name": row["name"], "score": row["score"]} for row in ranked]
        best = ranking[0] if ranking and ranking[0]["score"] is not None else None

        # Flag a metric-dependent winner: a task can emit several metrics, and the headline is
        # only one of them. If a shared metric crowns a different experiment, the conclusion
        # hinges on which metric is the headline — surface that instead of hiding it.
        per_metric: dict[str, list[tuple[str, float]]] = {}
        for row in rows:
            for key, value in (row["metrics"] or {}).items():
                if isinstance(value, (int, float)):
                    per_metric.setdefault(key, []).append((row["name"], float(value)))
        metric_winners = {
            key: max(pairs, key=lambda nv: nv[1])[0]
            for key, pairs in per_metric.items()
            if len(pairs) > 1
        }
        caution = None
        if best is not None:
            disagree = sorted(k for k, w in metric_winners.items() if w != best["name"])
            if disagree:
                caution = (
                    f"headline best is {best['name']!r}, but another experiment scores higher on "
                    f"{disagree} — the winner is metric-dependent. Pin eval config['headline_metric'] "
                    "to choose which metric ranks."
                )

        return {
            "study_id": self.study_id,
            "goal": (self.plan or {}).get("goal"),
            "n_experiments": len(rows),
            "varied_keys": varied,
            "experiments": experiments,
            "ranking": ranking,
            "best": best,
            "metric_winners": metric_winners,
            "caution": caution,
        }

    def review(self, conclusion: Any = None) -> dict[str, Any]:
        """Write ``review.json`` (= :meth:`summary` + ``conclusion``) and return it."""
        review = self.summary()
        now = _utc_now_iso()
        review["conclusion"] = conclusion
        review["status"] = "completed"
        review["created_at"] = self.created_at
        review["updated_at"] = now
        self.updated_at = now
        self.study_dir.mkdir(parents=True, exist_ok=True)
        self.review_path.write_text(json.dumps(review, **_JSON_KWARGS), encoding="utf-8")
        return review

    # -- helpers --------------------------------------------------------------

    def _read_reports(self) -> list[dict[str, Any]]:
        if not self.experiments_dir.is_dir():
            return []
        reports: list[dict[str, Any]] = []
        for exp_dir in sorted(self.experiments_dir.iterdir()):
            report_path = exp_dir / "report.json"
            if not (exp_dir.is_dir() and report_path.is_file()):
                continue
            try:
                reports.append(json.loads(report_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return reports


def study(id: str | None = None, workspace: str | Path = "workspace") -> Study:
    """Create or re-open a :class:`Study`.

    ``id=None`` mints a new serial (``study-0001`` ...). A given ``id`` is
    open-or-create: an existing study dir is re-opened and its ``plan.json`` is
    loaded back, so separate processes can attach to the same study by id.
    """
    root = Path(workspace)
    study_id = _next_serial(root) if id is None else str(id)
    study_dir = root / study_id
    study_dir.mkdir(parents=True, exist_ok=True)
    (study_dir / "experiments").mkdir(parents=True, exist_ok=True)

    plan: dict[str, Any] | None = None
    plan_path = study_dir / "plan.json"
    if plan_path.is_file():
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            plan = None
    return Study(study_id=study_id, workspace=root, plan=plan)


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or f"exp-{uuid4().hex[:8]}"


def _next_serial(root: Path) -> str:
    highest = 0
    if root.is_dir():
        for entry in root.iterdir():
            match = _SERIAL_RE.match(entry.name)
            if entry.is_dir() and match:
                highest = max(highest, int(match.group(1)))
    return f"study-{highest + 1:04d}"


def _varied_keys(configs: list[dict[str, Any]]) -> list[str]:
    """Keys whose value is not identical across all train configs (the swept dims)."""
    keys: set[str] = set()
    for config in configs:
        keys.update(config)
    varied = [
        key
        for key in keys
        if len({json.dumps(c.get(key), sort_keys=True, default=str) for c in configs}) > 1
    ]
    return sorted(varied)
