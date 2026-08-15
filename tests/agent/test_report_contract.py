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


# Fields the prompt is not expected to spell out: `stages` is structural (the shape of the JSON
# block IS the ask), and `path`/`sha256` appear inside the `script` example. `status` used to sit
# here too, back when it was derived from the score and nobody was supposed to write it. It is now
# the one field the experimenter must write and keep advancing, so the forward check has to see it.
_STRUCTURAL = {"stages", "path", "sha256"}


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
#:
#: The measurement fields (limit / n_samples / stderr) left this list when they stopped being
#: anyone's to write: the runtime stamps them from the summary.json at ``summary_dir``, so what
#: the producer owes is the POINTER, and the caliber that used to go missing now cannot.
_BASELINE_REQUIRED = {
    "summary_dir": "the eval dir the runtime stamps the anchor's caliber and size from — without "
                   "it there is nothing to stamp, and the tree renders a subset score as a bare "
                   "'@200' with no denominator (audit 15.4)",
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


def test_the_score_has_one_source_and_the_status_has_one_writer():
    """The single-source rule for the score; the single-writer rule for the cursor.

    The old shape let a score be read from three places (``score``, ``eval.macro_average``,
    ``eval.score``), so every consumer re-implemented the precedence and each place was
    authoritative somewhere. That half is unchanged: ``stages.eval.score`` or nothing.

    ``status`` moved the other way. It used to be DERIVED from the score, which meant a report that
    had simply not got to its eval yet was indistinguishable from one that had failed — and four
    nodes died of exactly that. It is now the report's only status: a cursor its author advances,
    and the only field any reader consults to ask whether the experiment is over.
    """
    parsed = report_schema.Report.model_validate(
        {"id": "n0001", "status": "completed", "stages": {"eval": {"score": 42.0, "limit": -1}}}
    )
    assert parsed.score == 42.0
    assert parsed.status == "completed"
    assert "score" not in report_schema.Report.model_fields  # derived from the eval stage, only

    # A node three stages in is not a failed node, and says so without a score existing yet.
    mid = report_schema.Report.model_validate(
        {"id": "n0002", "status": "train", "stages": {"data": {"rows": 6000}}}
    )
    assert mid.status == "train"
    assert mid.score is None
    assert mid.stages.eval is None  # absent, not a synthesised failure


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
                "eval": {"score": 1.0, "limit": -1},
                "export": {"note": "merged the adapter"},
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
    report = {"id": "n0004", "status": "review", "stages": {"eval": {"score": 50.0}}}
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


def test_a_report_handed_back_mid_experiment_is_not_terminal():
    """The regression test for the four nodes lost on 2026-08-15.

    The experimenter hands control back whenever it stops to wait on a Slurm job, and the loop
    decides from the file on disk whether to wake it again. Under the old predicate — ``review is
    not None or eval.status == "failed"`` — the report it left behind on the way out ended it: told
    to "never exit without one", four nodes across two runs wrote an eval stanza they had
    themselves labelled a placeholder, and each was read as a finished, failed experiment. One had
    a training job 100% of the way through.

    The cursor cannot say that. While it names a stage the experiment is running, whatever any
    stage carries — and a stage carries no status at all now, only what it produced.
    """
    parsed = report_schema.read(
        {
            "id": "n0005",
            "status": "train",
            "stages": {"train": {"model_path": "/runs/x/final_model"}},
        }
    )
    assert report_schema.is_terminal(parsed.model_dump()) is False
    assert parsed.score is None
    assert parsed.model_path == "/runs/x/final_model"  # the work so far is still on the record


def test_only_two_words_end_an_experiment_and_the_rest_mean_keep_going():
    """Terminality is a closed set of two; everything else is a stage name, hence in flight.

    The open half is deliberate. ``Stages`` accepts stages nobody declared (an ``export`` for a
    LoRA merge, once), so the cursor has to be able to name them — and a value this does not
    recognise has to mean "still going", because the alternative is what the old predicate did:
    treat an unfamiliar situation as an ending and throw the node away. Ambiguity costs a wake-up
    now, and the per-experiment deadline bounds the waiting.
    """
    assert report_schema.TERMINAL == {"completed", "failed"}
    for ending in ("completed", "failed"):
        assert report_schema.is_terminal({"id": "n", "status": ending}) is True
    for going in ("data", "train", "eval", "review", "export", "", "Completed"):
        assert report_schema.is_terminal({"id": "n", "status": going}) is False


def test_a_malformed_report_is_unfinished_not_a_crash(caplog):
    """`is_terminal` is the loop's control question and must always answer it.

    Making `read()` raise (instead of laundering every failure into "failed") left three unguarded
    `is_terminal` calls in the resume path, where a report.json that is not report-shaped would
    have taken the whole experiment down instead of reading as unfinished — and the archive holds
    a real one whose `stages` is a list. Answering False is right; answering it silently is what
    this whole change was about, so it logs.
    """
    bad = {"id": "x", "stages": [{"detail": "a list, not a stage map"}]}
    with caplog.at_level("ERROR"):
        assert report_schema.is_terminal(bad) is False
    assert "not report-shaped" in caplog.text
    with pytest.raises(Exception):
        report_schema.read(bad)  # callers that want the VALUE still get the error
