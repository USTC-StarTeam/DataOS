"""What a training backend declares and returns.

A ``Backend`` turns a ``TrainSpec`` (see ``spec.py``) into a launcher-agnostic ``Job``, and
``Backend.collect`` parses the run's outputs into a ``TrainResult``. The job/handle half of that
contract lives in :mod:`dataos_launch.types`, because it is shared with evaluation — training is
not the only thing that renders a job and needs it run somewhere.

``Status`` is imported rather than re-declared: ``TrainResult.status`` IS the launcher's status,
and two enums that must agree are two enums that eventually will not.
"""

from __future__ import annotations

from typing import Any

from dataos_launch.model import Model
from dataos_launch.types import Status
from pydantic import Field


class TrainResult(Model):
    """Canonical training outcome, produced by ``Backend.collect``."""

    spec_id: str
    status: Status
    model_path: str | None = None  # <run_dir>/final_model
    base_model: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    rows: int | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    error: str | None = None
    log_tail: str | None = None
    # a ready-to-merge {stages.train, model, dataset} subtree matching DataOS report.json
    report_fragment: dict[str, Any] = Field(default_factory=dict)


class Capabilities(Model):
    name: str
    methods: list[str]
    finetuning_types: list[str] = Field(default_factory=list)
    supports_multi_node: bool = False
    launch_kinds: list[str] = Field(default_factory=list)
    online_hook_points: list[str] = Field(default_factory=list)
    produces_format: str = "hf"
    needs_separate_env: bool = False
    env_domains: dict[str, str] = Field(
        default_factory=dict,
        description="method -> env.backends key, for methods whose dependency domain differs "
        "from the backend default; methods not listed use the backend name as their domain",
    )
    notes: str = ""
