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
from dataos.agent import roles
from dataos.prompt import render

_CONTRACT = "experimenter/context/report_contract"
#: report.json has TWO producers. The experimenter writes one per node from the contract above;
#: the investigator's recon stage writes the round-0 BASELINE — the anchor every later score is
#: compared against. Only the first was updated when the shape changed, so for a whole batch every
#: baseline came back in the old shape and the leader's anchor lost its caliber fields (audit 15.7).
#: A test that asserts against one producer's prompt cannot see that; this one lists both.
_PRODUCERS = (_CONTRACT, "investigator/turn/recon")

def _render(producer: str) -> str:
    """Render a producer's prompt. ``recon`` takes render params; the contract block takes none."""
    if producer != "investigator/turn/recon":
        return render(producer)
    return render(
        producer,
        target_bench="gsm8k", base_model="/m", fixed_pool="/p", deploy_root="/d",
        eval_env_python="/e/bin/python", cluster_json="{}", profile_script="/s.py",
        lm_bridge="/b", recon_path="/r.json", baseline_run_root="/rr",
        baseline_eval_dir="/rr/eval/x", baseline_report_path="/br.json", eval_limit=-1,
        env_ops="", task_def="",
    )


def _report_keys(producer: str) -> set[str]:
    """Keys the producer's prompt quotes inside its report.json example.

    The scan is BOUNDED to that one JSON block. recon.yaml writes two files — the baseline report
    AND recon.json (candidate_sources, conversion, ...) — so an unbounded line scan would drag the
    intel file's keys in and demand the report schema declare them.
    """
    lines = _render(producer).splitlines()
    start = next(i for i, line in enumerate(lines) if '{"id"' in line)
    end = next(i for i, line in enumerate(lines[start:], start) if line.rstrip().endswith("}}}"))
    return {
        line.split('"')[1]
        for line in lines[start : end + 1]
        if line.lstrip().startswith('"') and '":' in line
    }


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
    known = report_schema.field_names() | _STRUCTURAL | {"<your class name>", "<node_id>"}
    for producer in _PRODUCERS:
        unknown = sorted(key for key in _report_keys(producer) if key not in known)
        assert not unknown, f"{unknown} are asked for in {producer}.yaml but not declared in report.py"


#: What the round-0 BASELINE report must carry, whoever writes it. recon legitimately omits most
#: of the schema (a baseline has no data and no training), so the forward check for that producer
#: is this named subset rather than "every field" — but the subset is not negotiable: it is what
#: the leader's anchor is made of.
_BASELINE_REQUIRED = {
    "limit": "the caliber of the anchor score",
    "n_samples": "the ONLY place the run learns the benchmark's true size; without it the tree "
                 "renders a subset score as a bare '@200' with no denominator (audit 15.4)",
    "stderr": "the noise floor every later comparison is judged against",
    "failure_classes": "the baseline diagnosis the run's first move is chosen from",
    "conclusion": "whether the base model's failure is protocol-side or capability-side",
    "dominant_evidence": "the one concrete mismatch that makes that verdict checkable",
}


def test_the_baseline_producer_asks_for_the_anchor_fields():
    """recon writes the baseline report, and nothing used to check what it asked for.

    The shape change updated the experimenter's contract and left this producer on the old shape,
    so for a whole batch every baseline came back without its caliber fields — and the forward test
    above could not see it, because it renders only the experimenter's contract. One producer
    covered, one not, is how the gap survived a green suite.
    """
    keys = _report_keys("investigator/turn/recon")
    missing = {name: why for name, why in _BASELINE_REQUIRED.items() if name not in keys}
    assert not missing, "the baseline report would lose:\n" + "\n".join(
        f"  {name}: {why}" for name, why in sorted(missing.items())
    )


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


def test_an_unasked_for_stage_survives_instead_of_killing_the_report():
    """``Stages`` used to forbid extras, and that cost a whole report.

    An experimenter declared an ``export`` stage for its LoRA merge — a stage that really happened,
    described in the natural place — and the report was rejected for it. Rejection here is never
    the cheap option: the only consumer of an unrecognised stage is the leader, which reads prose
    for a living, so an extra stage costs nothing to keep and a real score to drop.
    """
    parsed = report_schema.Report.model_validate(
        {
            "id": "n0003",
            "stages": {
                "eval": {"status": "completed", "score": 1.0, "limit": -1},
                "export": {"status": "completed", "note": "merged the adapter"},
            },
        }
    )
    assert parsed.score == 1.0
    assert (parsed.stages.model_extra or {}).get("export", {}).get("note")


def test_eval_limit_comes_from_the_harness_not_the_experimenter(monkeypatch):
    """The caliber of a score is not optional — it is just not the experimenter's to type.

    Three runs of an earlier batch delivered a subset-scored model as their submission (a 54-of-164
    humaneval scoring 50.00 outranked full-benchmark nodes) because ``limit`` was a field a report
    could simply omit. Requiring it turned that into a rejected report, which loses the score too.
    ``summary.json`` has carried the answer all along.
    """
    report = {"id": "n0004", "stages": {"eval": {"status": "completed", "score": 50.0}}}
    assert report_schema.read(report).stages.eval.limit is None  # the report alone cannot say

    summary = {"score": 50.0, "limit": 164, "n_samples": 164, "task": "humaneval",
               "metric_key": "pass@1,none", "metrics": {"pass@1_stderr,none": 0.039}}
    monkeypatch.setattr(
        roles.runtime, "_remote_eval_summaries",
        lambda cfg, study_id, name: [("/runs/x/eval/humaneval_limit164", summary)],
    )
    roles._stamp_measurements(cfg=None, study_id="round-0000", name="x", report=report)
    parsed = report_schema.read(report)
    assert parsed.stages.eval.limit == 164
    assert parsed.stages.eval.n_samples == 164
    # stderr arrives on the SAME scale as the score it belongs to, which the reports did not manage
    # on their own: one baseline said `score=0.89, stderr=0.0044` while its siblings said 30.36/2.13
    assert parsed.stages.eval.stderr == pytest.approx(3.9)


def _corpus() -> list[Path]:
    root = Path(__file__).resolve().parents[2] / "workspace"
    return sorted(root.glob("*/round-*/experiments/*/report.json")) if root.is_dir() else []


def test_no_report_on_disk_loses_a_score_it_declares():
    """The regression test for the bug this schema shape was rewritten around.

    ``read()`` used to catch every validation error and fall through to a reader for a report shape
    no live path produces, which turned any mismatch into ``score=None, status="failed"`` — a
    plausible-looking lie, logged nowhere. On the archive that was 33 rejected reports and 22
    destroyed scores, including a 27.44 and a 15.15 that their runs had recorded correctly at the
    time. The rule that replaces it: whatever a report declares under ``stages.eval.score``,
    reading it back yields that number or raises.
    """
    paths = _corpus()
    if not paths:
        pytest.skip("no report corpus in this checkout")
    lost, unreadable, scored = [], [], 0
    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        declared = ((raw.get("stages") or {}).get("eval") or {}).get("score") \
            if isinstance(raw.get("stages"), dict) else None
        try:
            parsed = report_schema.read(raw)
        except Exception as exc:  # noqa: BLE001
            # Allowed, and the point: a file that is not report-shaped raises where someone can
            # see it. What is NOT allowed is doing that quietly to a report carrying a score.
            unreadable.append(f"{path}: {exc}")
            if isinstance(declared, (int, float)):
                lost.append(f"{path}: declared {declared}, raised instead")
            continue
        scored += isinstance(parsed.score, (int, float))
        if isinstance(declared, (int, float)) and parsed.score != declared:
            lost.append(f"{path}: declared {declared}, read back {parsed.score}")
    assert not lost, "scores lost on read:\n" + "\n".join(lost[:10])
    assert scored > 0, "the corpus parsed but produced no scores — the score path is broken"
    assert len(unreadable) <= 1, (
        "more than the one known pre-contract file is unreadable:\n"
        + "\n".join(unreadable[:10])
    )


def test_a_mid_flight_stage_keeps_the_word_it_was_killed_on():
    """Reports killed mid-stage carry ``running`` / ``pending`` / ``not_started``.

    The vocabulary an experimenter is ASKED for is terminal, so these used to be normalised to
    ``failed`` with the original word moved into ``error``. Keeping the word where it was written
    says the same thing without a translation layer, and "the train stage never started" still
    reads differently from "training failed". The overall verdict is unchanged: no score, so the
    report is failed.
    """
    parsed = report_schema.read(
        {
            "id": "n0005",
            "stages": {
                "train": {"status": "running", "model_path": "/runs/x/final_model"},
                "eval": {"status": "not_started"},
            },
        }
    )
    assert parsed.status == "failed"
    assert parsed.score is None
    assert parsed.stages.train is not None
    assert parsed.stages.train.status == "running"
    assert parsed.stages.eval.status == "not_started"
    assert parsed.model_path == "/runs/x/final_model"
