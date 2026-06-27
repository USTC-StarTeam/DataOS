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

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from dataos.core.experiment import Experiment
from dataos.core.ids import _make_id, _utc_now_iso
from dataos.core.io import atomic_write_json
from dataos.core.redact import redact
from dataos.core.workspace import Workspace

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
        """
        rows: list[dict[str, Any]] = []
        for report in self._read_reports():
            evaluation = report.get("eval") or {}
            spec = report.get("spec") or {}
            dataset = report.get("dataset") or {}
            # an invalidated/failed/re-running eval's old numbers are forensics, not scores:
            # letting them rank (while the guards exclude them as not-completed) would crown
            # exactly the least trustworthy row
            _stages = report.get("stages")
            if isinstance(_stages, dict):
                _eval_entry = _stages.get("eval")
                if isinstance(_eval_entry, dict):
                    eval_current = _eval_entry.get("status") == "completed"
                else:
                    # sub-agent wrote bare string value e.g. "completed"
                    eval_current = _eval_entry == "completed"
            else:
                # sub-agent wrote stages as a name-list or other non-dict — fall back to score presence
                eval_current = evaluation.get("score") is not None or evaluation.get("macro_average") is not None
            rows.append(
                {
                    "name": report.get("name") or report.get("id"),
                    "id": report.get("id"),
                    "status": report.get("status"),
                    "score": (
                        evaluation.get("macro_average")
                        if evaluation.get("macro_average") is not None
                        else evaluation.get("score")
                    )
                    if eval_current
                    else None,
                    "metrics": (evaluation.get("metrics") or {}) if eval_current else {},
                    "train": spec.get("train") or {},
                    "data_rows": dataset.get("rows"),
                    "data_fingerprint": dataset.get("fingerprint"),
                    "eval_protocol": _eval_protocol_fingerprint(report),
                    "per_task": evaluation.get("per_task") or {},
                }
            )
        rows.sort(key=lambda row: str(row["name"]))
        varied = _varied_keys([row["train"] for row in rows])
        # Data-side sweeps (row counts / content) are invisible to spec["train"] diffs — e.g.
        # an injected-dataset volume sweep. Surface them as pseudo-keys next to the train dims.
        with_data = [row for row in rows if row["data_rows"] is not None or row["data_fingerprint"]]
        fingerprints = [row["data_fingerprint"] for row in with_data]
        if len({row["data_rows"] for row in with_data}) > 1:
            varied.append("data.rows")
        elif all(fingerprints) and len(set(fingerprints)) > 1:
            # only when every report records one — a mixed old/new-version study must not
            # misread a missing fingerprint as a data variation
            varied.append("data.fingerprint")

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
                if key == "data.rows":
                    entry[key] = row["data_rows"]
                elif key == "data.fingerprint":
                    entry[key] = row["data_fingerprint"]
                else:
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
                # error bars are not scores: ranking experiments by ::stderr crowns the
                # NOISIEST run (seen live with bfcl's stderr columns, incl. the mislabeled all2)
                if isinstance(value, (int, float)) and not _is_noise_metric(key):
                    per_metric.setdefault(key, []).append((row["name"], float(value)))
        metric_winners: dict[str, str] = {}
        for key, pairs in per_metric.items():
            if len(pairs) < 2:
                continue
            top = max(value for _, value in pairs)
            leaders = [name for name, value in pairs if value == top]
            if len(leaders) == 1:  # a tie crowns nobody (3-way 0.85 used to "win" arbitrarily)
                metric_winners[key] = leaders[0]
        cautions: list[str] = []
        if best is not None:
            disagree = sorted(k for k, w in metric_winners.items() if w != best["name"])
            if disagree:
                cautions.append(
                    f"headline best is {best['name']!r}, but another experiment scores higher on "
                    f"{disagree} — the winner is metric-dependent. Pin eval config['headline_metric'] "
                    "to choose which metric ranks."
                )
        # Comparability guard: ranking only means something if every experiment was scored by
        # the SAME eval implementation (tasks / limits / harness commands). Model-identity keys
        # are excluded from the fingerprint, so an eval-only baseline matches its sweep.
        protocols = {
            row["name"]: row["eval_protocol"]
            for row in rows
            if row["eval_protocol"] is not None and row["status"] == "completed"
        }
        if len(set(protocols.values())) > 1:
            cautions.append(
                f"experiments were scored under DIFFERENT eval protocols {protocols} — "
                "their scores are NOT comparable; align spec['eval'] across the study"
            )
        # Coverage guard: even under one protocol, runtime skips (gated bench, broken harness)
        # shrink an experiment's macro to FEWER benches — ranking partial against full coverage
        # is apples-to-oranges. Identical coverage (everyone skipped the same bench) stays quiet.
        coverages = {
            # only benches that actually SCORED count as covered — an in-entry lm-eval task
            # skip leaves the task in per_task with value None, which is exactly the
            # missing-coverage case this guard exists to catch
            row["name"]: frozenset(
                task
                for task, value in row["per_task"].items()
                if isinstance(value, (int, float))
            )
            for row in rows
            if row["status"] == "completed" and row["per_task"]
        }
        if len(set(coverages.values())) > 1:
            shape = {name: sorted(cov) for name, cov in sorted(coverages.items())}
            cautions.append(
                f"experiments covered DIFFERENT bench sets {shape} (skipped entries/tasks?) — "
                "their macro scores are not comparable; rerun the missing benches"
            )
        caution = " | ".join(cautions) if cautions else None

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


def _is_noise_metric(key: str) -> bool:
    """Error-bar / counter pseudo-metrics that must not participate in winner ranking."""
    lowered = key.lower()
    return (
        "stderr" in lowered
        or "_sterr" in lowered
        or lowered.endswith("::all2")  # bfcl quirk: 'all2' is a stderr despite the name
    )


_PROTOCOL_SKIP_KEYS = {"model_name_or_path", "base_model", "model_args", "provider_config", "env"}


def _strip_protocol_noise(value: Any) -> Any:
    """Drop model-identity / stub-score / env keys at EVERY level (suite entries included)."""
    if isinstance(value, dict):
        return {
            key: _strip_protocol_noise(val)
            for key, val in value.items()
            if key not in _PROTOCOL_SKIP_KEYS
        }
    if isinstance(value, list):
        return [_strip_protocol_noise(item) for item in value]
    return value


def _eval_protocol_fingerprint(report: dict[str, Any]) -> str | None:
    """A short hash of the experiment's eval *protocol* — what was evaluated and how it was
    scored (tasks / limits / fewshot / harness commands / suite shape).

    Model-identity keys are excluded (recursively, so suite entries too) so an eval-only
    baseline fingerprints the same as the trained experiments it anchors; stub
    ``provider_config`` is excluded so smoke sweeps with different fake scores don't alarm."""
    _s = report.get("stages")
    _se = _s.get("eval") if isinstance(_s, dict) else None
    eval_stage = _se if isinstance(_se, dict) else {}
    if eval_stage.get("config_drift"):
        # the persisted score came from a config that is no longer the one on record —
        # the protocol is UNKNOWN until the stage re-runs; don't group it with anything
        return None
    config = eval_stage.get("config")
    if not isinstance(config, dict):
        return None
    protocol = _strip_protocol_noise(config)
    if not protocol:
        return None
    payload = json.dumps(protocol, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:8]


def _varied_keys(configs: list[dict[str, Any]]) -> list[str]:
    """Keys whose value differs across the train configs that *have* one (the swept dims).

    Experiments with no train stage (e.g. an eval-only baseline, whose train config is ``{}``)
    are excluded — otherwise every train key looks "swept" simply because the baseline lacks it.
    """
    trained = [config for config in configs if config]
    keys: set[str] = set()
    for config in trained:
        keys.update(config)
    varied = [
        key
        for key in keys
        if len({json.dumps(c.get(key), sort_keys=True, default=str) for c in trained}) > 1
    ]
    return sorted(varied)
