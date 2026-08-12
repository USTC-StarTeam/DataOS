"""The report schema, the prompt that asks for it, and every report ever written.

Three things have to agree about report.json and none of them can check the others at runtime:
``report.py`` validates, ``report_contract.yaml`` is what the experimenter is actually told, and
1232 reports already on disk are what the format has to keep being able to read. The measured cost
of letting them drift: ``study.summary`` read ``compat_fixes`` while 700 reports wrote
``compatibility_fixes``, and it read ``spec.train`` — a key no report has EVER written — so the
leader's "which knob changed between these two nodes" was structurally empty for the whole of two
batches. Nothing failed. That is the failure mode these tests exist for.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataos.agent import report as report_schema
from dataos.prompt import render

_CONTRACT = "experimenter/context/report_contract"

# Fields the prompt is not expected to spell out: `stages` and `status` are structural (the shape
# of the JSON block IS the ask), and `path`/`sha256` appear inside the `script` example.
_STRUCTURAL = {"stages", "status", "path", "sha256"}


def test_every_schema_field_is_asked_for_in_the_contract():
    """A field the experimenter is never told about is a field no report will ever carry.

    ``eval.stderr`` is the case that motivates this: adding it to the schema costs nothing and
    changes nothing until the prompt asks for it, and the run that built a five-round tree on a
    1.98-point difference whose stderr was 5.06 is what not asking looks like.
    """
    text = render(_CONTRACT)
    missing = sorted(
        name for name in report_schema.field_names() if name not in _STRUCTURAL and name not in text
    )
    assert not missing, (
        f"{missing} are declared in report.py but never mentioned in {_CONTRACT}.yaml — "
        "the experimenter cannot write a field it is not asked for"
    )


def test_the_contract_asks_for_nothing_the_schema_drops():
    """The inverse: a field the prompt asks for but the schema does not name is silently discarded.

    ``extra="allow"`` means such a field survives to the leader, so this is not about data loss —
    it is about the prompt promising a field is load-bearing when nothing validates it. Any field
    worth asking for by name belongs in the schema.
    """
    text = render(_CONTRACT)
    # keys the contract quotes inside its JSON example, i.e. what it actually promises
    quoted = {
        line.split('"')[1]
        for line in text.splitlines()
        if line.lstrip().startswith('"') and '":' in line
    }
    known = report_schema.field_names() | _STRUCTURAL | {"<your class name>", "<node_id>"}
    unknown = sorted(key for key in quoted if key not in known)
    assert not unknown, f"{unknown} are asked for in {_CONTRACT}.yaml but not declared in report.py"


def test_the_top_level_score_and_status_are_no_longer_writable():
    """The single-source rule, enforced rather than documented.

    The old shape let a score be read from three places (``score``, ``eval.macro_average``,
    ``eval.score``), so every consumer re-implemented the precedence and each place was
    authoritative somewhere. ``status`` had the same problem in the other direction: it was written
    by hand, so a report could claim "completed" with no score in it.
    """
    parsed = report_schema.Report.model_validate(
        {"id": "n0001", "stages": {"eval": {"status": "completed", "score": 42.0, "limit": -1}}}
    )
    assert parsed.score == 42.0
    assert parsed.status == "completed"
    assert "status" not in report_schema.Report.model_fields
    assert "score" not in report_schema.Report.model_fields

    unscored = report_schema.Report.model_validate(
        {"id": "n0002", "stages": {"eval": {"status": "failed", "score": None, "limit": 0}}}
    )
    assert unscored.status == "failed"


def test_a_stage_cannot_be_left_out_of_the_stage_map():
    """``Stages`` forbids extras: a payload under a non-stage name is how the old shape lost track.

    ``dataset`` / ``model`` / ``eval`` / ``analysis`` were stage outputs living outside the stage
    map, which is why nobody read the map — it was the empty half of a split.
    """
    with pytest.raises(ValueError):
        report_schema.Report.model_validate(
            {
                "id": "n0003",
                "stages": {
                    "eval": {"status": "completed", "score": 1.0, "limit": -1},
                    "dataset": {"rows": 6000},
                },
            }
        )


def test_eval_limit_is_required():
    """The caliber of a score is not optional.

    Three runs of the previous batch delivered a subset-scored model as their submission (a 54-of-164
    humaneval scoring 50.00 outranked full-benchmark nodes) because ``limit`` was a field a report
    could simply omit.
    """
    with pytest.raises(ValueError):
        report_schema.Report.model_validate(
            {"id": "n0004", "stages": {"eval": {"status": "completed", "score": 50.0}}}
        )


def _corpus() -> list[Path]:
    root = Path(__file__).resolve().parents[2] / "workspace"
    return sorted(root.glob("*/round-*/experiments/*/report.json")) if root.is_dir() else []


def test_every_report_on_disk_still_reads():
    """The whole archive, through the new schema. A shape that cannot read its own history cannot
    be regression-tested against it — and the archive is the only corpus of real experimenter
    output that exists."""
    paths = _corpus()
    if not paths:
        pytest.skip("no report corpus in this checkout")
    failures = []
    scored = 0
    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        try:
            parsed = report_schema.read(raw)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{path}: {exc}")
            continue
        scored += isinstance(parsed.score, (int, float))
    assert not failures, "reports that no longer read:\n" + "\n".join(failures[:10])
    assert scored > 0, "the corpus parsed but produced no scores — the score path is broken"


def test_a_mid_flight_report_reads_as_failed_with_its_real_state_kept():
    """11 of the 1231 archived reports were killed mid-stage, leaving ``status: "running"``.

    The schema's vocabulary is deliberately terminal — an experimenter writing a report has no
    non-terminal option to reach for — so the legacy dialect is normalised on the way in. It must
    not be normalised into silence: "the train stage never started" and "training failed" call for
    different next rounds.
    """
    parsed = report_schema.read(
        {
            "id": "n0005",
            "score": None,
            "stages": {"train": {"status": "running"}, "eval": {"status": "not_started"}},
            "model": {"path": "/runs/x/final_model"},
        }
    )
    assert parsed.status == "failed"
    assert parsed.stages.train is not None
    assert parsed.stages.train.status == "failed"
    assert "running" in (parsed.stages.train.error or "")
    assert "not_started" in (parsed.stages.eval.error or "")
