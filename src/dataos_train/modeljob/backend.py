"""ModelJob orchestration: validate / build / collect around :mod:`score_entry`.

``validate`` is the pre-submit gate (fail-fast, no GPU). ``build`` renders
``run_score.sh`` + ``modeljob.json`` as JobSpec assets, so any launcher — or an external
sbatch wrapper — just runs ``bash run_score.sh`` in ``run_dir``. ``collect`` parses the
resulting ``scores.jsonl`` into a :class:`ScoreResult` with row-count/finiteness checks
and summary statistics. This module is torch-free; torch only enters at job time inside
``score_entry``.
"""

from __future__ import annotations

import json
import math
import shlex
import statistics
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..data.convert import read_records
from ..types import JobSpec, LaunchPlan, SlurmExtras, Status
from .spec import GradientInteraction, ModelJobSpec

NAME = "modeljob"  # env domain + JobSpec.backend label

SFT_FORMATS = ("auto", "alpaca", "messages", "sharegpt")

SCORES_FILE = "scores.jsonl"
CONFIG_FILE = "modeljob.json"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ScoreResult(_Model):
    """Canonical scoring outcome, produced by :func:`collect`.

    ``metrics`` carries ``n_scored`` / ``n_null`` (records whose assistant segment was
    empty after truncation) plus ``<metric>_mean`` / ``<metric>_min`` / ``<metric>_max``
    / ``<metric>_p50`` over the finite values, where ``<metric>`` is ``loss`` for the
    loss kind and ``score`` for the gradient kind.
    """

    status: Status
    scores_path: str | None = None
    key: str | None = None  # the output_key every score line carries
    rows: int = 0
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)


def validate(spec: ModelJobSpec) -> list[str]:
    """Pre-submit gate: required fields present, kind supported, shapes legal."""
    errors: list[str] = []
    if not spec.model:
        errors.append("model is required (HF model dir, injected by the runtime)")
    if not spec.dataset.path:
        errors.append("dataset.path is required (canonical SFT data)")
    elif spec.dataset.format not in SFT_FORMATS:
        errors.append(
            f"dataset.format {spec.dataset.format!r} is not scorable; use one of {SFT_FORMATS}"
        )
    inter = spec.interaction
    if isinstance(inter, GradientInteraction):
        errors.extend(_validate_gradient(inter))
    elif inter.kind != "loss":
        errors.append(
            f"unsupported interaction kind {inter.kind!r}; supported: 'loss', 'gradient'"
        )
    else:
        if inter.batch_size < 1:
            errors.append("interaction.batch_size must be >= 1")
        if inter.max_len < 1:
            errors.append("interaction.max_len must be >= 1")
        if not inter.output_key:
            errors.append("interaction.output_key must be non-empty")
    return errors


def _validate_gradient(inter: GradientInteraction) -> list[str]:
    """Gradient-kind gate: projection/LoRA shapes legal, target (if any) readable."""
    if inter.kind != "gradient":  # belt-and-braces for model_construct bypasses
        return [f"unsupported interaction kind {inter.kind!r}; supported: 'loss', 'gradient'"]
    errors: list[str] = []
    if inter.projection_dim < 1:
        errors.append("interaction.projection_dim must be >= 1")
    if inter.lora_r < 1:
        errors.append("interaction.lora_r must be >= 1")
    if inter.max_len < 1:
        errors.append("interaction.max_len must be >= 1")
    if not inter.output_key:
        errors.append("interaction.output_key must be non-empty")
    if inter.target is not None:
        if not inter.target.path:
            errors.append("interaction.target.path is required when target is given")
        else:
            try:
                read_records(inter.target.path)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append(f"interaction.target is not a readable dataset: {exc}")
    return errors


def _default_run_dir(spec: ModelJobSpec) -> str:
    """``<dataset dir>/modeljobs/<name>`` — next to the data when the dataset is a file."""
    p = Path(spec.dataset.path)
    base = p if p.suffix == "" else p.parent
    return str(base / "modeljobs" / spec.name)


def _render_run_script(run_dir: str) -> str:
    # cd first so the script is location-independent when bashed by an external wrapper.
    steps = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"cd {shlex.quote(run_dir)}",
        f"python -m dataos_train.modeljob.score_entry --config {CONFIG_FILE}",
    ]
    return "\n".join(steps) + "\n"


def build(spec: ModelJobSpec, run_dir: str | None = None) -> JobSpec:
    """Render the scoring job: ``modeljob.json`` (spec) + ``run_score.sh`` (entry call)."""
    run_dir = run_dir or _default_run_dir(spec)
    assets = {
        CONFIG_FILE: json.dumps(spec.model_dump(), indent=2, ensure_ascii=False) + "\n",
        "run_score.sh": _render_run_script(run_dir),
    }
    pythonpath = list(spec.env.pythonpath)
    if spec.env.deploy_root and not pythonpath:
        pythonpath = [f"{spec.env.deploy_root}/src", spec.env.deploy_root]
    cluster = spec.cluster or {}
    return JobSpec(
        backend=NAME,
        job_name=spec.name or NAME,
        command=["bash", "run_score.sh"],
        env=dict(spec.env.extra),
        run_dir=run_dir,
        resources=spec.resources,
        launch=LaunchPlan(kind="direct"),
        conda_sh=spec.env.conda_sh,
        conda_env=spec.env.env_for(NAME),
        pythonpath=pythonpath,
        assets=assets,
        slurm=SlurmExtras(
            partition=cluster.get("partition"),
            qos=cluster.get("qos"),
            gres=cluster.get("gres"),
            nodelist=cluster.get("nodelist"),
            cpus_per_task=cluster.get("cpus_per_task"),
            account=cluster.get("account"),
        ),
    )


def _expected_rows(run_dir: Path) -> int | None:
    """Dataset record count from the persisted job config; None when unconfirmable."""
    cfg = run_dir / CONFIG_FILE
    if not cfg.is_file():
        return None
    try:
        dataset = read_json_dict(cfg).get("dataset") or {}
        return len(read_records(dataset["path"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def read_json_dict(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _metric_name(run_dir: Path) -> str:
    """Score field carried by ``scores.jsonl``: ``score`` for gradient, else ``loss``."""
    cfg = run_dir / CONFIG_FILE
    if not cfg.is_file():
        return "loss"
    try:
        interaction = read_json_dict(cfg).get("interaction") or {}
    except (OSError, ValueError):
        return "loss"
    return "score" if interaction.get("kind") == "gradient" else "loss"


def collect(run_dir: str) -> ScoreResult:
    """Parse ``<run_dir>/scores.jsonl`` — row-count and finiteness checks + stats.

    Hard failures (status=FAILED): missing/unparseable file, malformed lines, non-finite
    values, duplicate/gapped indices, inconsistent ``key``, or a row count that disagrees
    with the dataset. A ``null`` value (record whose assistant segment truncated away)
    is tolerated but counted as ``metrics.n_null`` and excluded from the statistics.
    The value field is ``loss`` for the loss kind and ``score`` for the gradient kind
    (read from the persisted job config); stats are prefixed accordingly.
    """
    rd = Path(run_dir)
    scores = rd / SCORES_FILE
    if not scores.is_file():
        return ScoreResult(status=Status.FAILED, error=f"{SCORES_FILE} not found in {run_dir!r}")
    metric = _metric_name(rd)

    entries: list[tuple[int, str, float | None]] = []
    problems: list[str] = []
    for lineno, line in enumerate(scores.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            problems.append(f"line {lineno}: invalid JSON")
            continue
        index, key, value = rec.get("index"), rec.get("key"), rec.get(metric)
        if not isinstance(index, int) or not isinstance(key, str):
            problems.append(f"line {lineno}: expected int 'index' and str 'key'")
            continue
        if value is not None and not (isinstance(value, int | float) and math.isfinite(value)):
            problems.append(f"line {lineno}: non-finite {metric} {value!r}")
            continue
        entries.append((index, key, None if value is None else float(value)))

    result = ScoreResult(
        status=Status.COMPLETED,
        scores_path=str(scores),
        rows=len(entries),
        artifacts={"scores": str(scores)},
    )
    if problems:
        result.status = Status.FAILED
        shown = "; ".join(problems[:5])
        result.error = shown + (f"; ... ({len(problems)} problems)" if len(problems) > 5 else "")
        return result

    keys = {key for _, key, _ in entries}
    if len(keys) > 1:
        result.status = Status.FAILED
        result.error = f"inconsistent 'key' across rows: {sorted(keys)}"
        return result
    result.key = keys.pop() if keys else None

    indices = sorted(index for index, _, _ in entries)
    if indices != list(range(len(entries))):
        result.status = Status.FAILED
        result.error = f"indices must cover 0..{len(entries) - 1} uniquely"
        return result

    expected = _expected_rows(rd)
    if expected is not None and expected != len(entries):
        result.status = Status.FAILED
        result.error = (
            f"{SCORES_FILE} has {len(entries)} rows but the dataset has {expected} records"
        )
        return result

    losses = [value for _, _, value in entries if value is not None]
    result.metrics = {"n_scored": len(losses), "n_null": len(entries) - len(losses)}
    if losses:
        result.metrics.update(
            {
                f"{metric}_mean": statistics.fmean(losses),
                f"{metric}_min": min(losses),
                f"{metric}_max": max(losses),
                f"{metric}_p50": statistics.median(losses),
            }
        )
    return result
