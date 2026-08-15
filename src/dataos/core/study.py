"""The :class:`Study` container — a set of experiments sharing one plan.

A study is just a directory. Each child :class:`~dataos.core.experiment.Experiment`
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
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from dataos.agent import report as report_schema
from dataos.core.experiment import Experiment
from dataos.core.ids import _make_id, _utc_now_iso
from dataos.core.io import atomic_write_json
from dataos.core.redact import redact
from dataos.core.workspace import Workspace

log = logging.getLogger(__name__)

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
        # persisted copy is scrubbed of secret-looking values; in-memory plan stays raw
        atomic_write_json(self.plan_path, redact(self.plan))
        return self

    def experiment(self, spec: dict[str, Any] | None = None) -> Experiment:
        """Open-or-create a child :class:`Experiment` under this study.

        The directory is ``experiments/<slug(name)>/``. If that experiment already ran
        (its ``report.json`` is on disk), its real ``exp_id`` is reused and every stage is
        **rehydrated** from ``{kind}.json`` — so re-driving a reopened study **resumes**:
        completed stages skip, only pending / failed ones run. Give each experiment a
        distinct short ``name`` — it is the dir slug and the ``summary`` join key.
        """
        name = (spec or {}).get("name") or _make_id("exp")
        workspace = Workspace(self.experiments_dir, _slug(str(name)))
        exp_id = _make_id("exp")
        report_path = workspace.exp_dir / "report.json"
        if report_path.is_file():
            try:
                exp_id = (json.loads(report_path.read_text(encoding="utf-8")) or {}).get("id") or exp_id
            except (OSError, json.JSONDecodeError):
                pass
        exp = Experiment(
            exp_id=exp_id,
            workspace=workspace,
            name=str(name),
            spec=dict(spec) if spec is not None else None,
        )
        return exp.load_state()

    def load_experiments(self) -> list[Experiment]:
        """Rehydrate every experiment on disk into an :class:`Experiment` (reopen the whole
        study at the object level). Each spec is read back from its ``report.json`` (falling
        back to a name-only spec); pair with :meth:`summary` for the aggregated view."""
        loaded: list[Experiment] = []
        for slug in self.experiments:
            report_path = self.experiments_dir / slug / "report.json"
            spec: dict[str, Any] | None = None
            if report_path.is_file():
                try:
                    spec = (json.loads(report_path.read_text(encoding="utf-8")) or {}).get("spec")
                except (OSError, json.JSONDecodeError):
                    spec = None
            loaded.append(self.experiment(spec if spec is not None else {"name": slug}))
        return loaded

    def summary(self) -> dict[str, Any]:
        """Aggregate every child ``report.json`` into a deterministic study summary.

        Reads from disk (not in-memory state), so it works regardless of how the
        experiments were run. Tolerates missing/partial reports.

        ``experiments`` carries each report VERBATIM rather than a hand-picked projection, and
        ranking is the only aggregate left. Both facts come from measurement, not taste.

        The projection was measured against all 1232 reports on disk and three of its columns had
        never held a value: it read ``spec.train`` (0/1232), ``dataset.fingerprint`` (0/1232), and
        ``stages.eval.config`` (0/1232), while reading ``compat_fixes`` where 700 reports wrote
        ``compatibility_fixes``. A projection whose key names can silently stop matching the
        contract is the bug; passing the report through removes the class.

        The cross-experiment aggregates went the same way, for a structural reason rather than a
        naming one: ``varied_keys``, ``metric_winners`` and the comparability / coverage /
        identical-data cautions all compared MULTIPLE experiments inside one study, and the agentic
        loop runs ``fan_out(specs=[spec])`` — exactly one experiment per study. Measured over the
        87 studies of the 2026-08-12/13 batches: ``ranking`` non-empty 87 times, ``best`` 42, and
        ``varied_keys`` / ``metric_winners`` / ``caution`` **zero**. A guard with no possible
        control group is not a weak guard, it is decoration, and decoration in this position is
        worse than absence — it reads as "someone is checking this". Cross-node comparison belongs
        to the tree, which is the thing that actually spans nodes (it already renders a subset score
        as ``74.00@200/1319`` so a caliber mismatch is visible where the nodes sit side by side).
        """
        rows: list[dict[str, Any]] = []
        for raw in self._read_reports():
            try:
                parsed = report_schema.read(raw)
            except Exception as e:  # noqa: BLE001 — not report-shaped; say so, do not rank it
                log.error("study %s: unreadable report skipped: %s", self.study_id, e)
                continue
            # an invalidated/failed/re-running eval's old numbers are forensics, not scores:
            # letting them rank (while the guards exclude them as not-completed) would crown
            # exactly the least trustworthy row. The cursor answers it directly — a report still
            # sitting on `eval` is one whose numbers, if any, belong to a superseded run.
            eval_current = parsed.status == "completed"
            rows.append(
                {
                    "report": raw,
                    "name": parsed.id,
                    "status": parsed.status,
                    "score": parsed.score if eval_current else None,
                }
            )
        rows.sort(key=lambda row: str(row["name"]))

        # Reports go to the leader whole. Sorting them is the only shaping left.
        experiments = [row["report"] for row in rows]

        ranked = sorted(
            rows,
            key=lambda row: (row["score"] is None, -(row["score"] or 0.0), str(row["name"])),
        )
        ranking = [{"name": row["name"], "score": row["score"]} for row in ranked]
        best = ranking[0] if ranking and ranking[0]["score"] is not None else None

        return {
            "study_id": self.study_id,
            "goal": (self.plan or {}).get("goal"),
            "n_experiments": len(rows),
            "experiments": experiments,
            "ranking": ranking,
            "best": best,
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
        atomic_write_json(self.review_path, review)
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


