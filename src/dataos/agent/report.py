"""report.json — the experimenter's one output, and the leader's evidence for the next round.

THE ADMISSION RULE for every field here: *would the leader act differently because of it?*
Paths, names, derivable counts and un-verifiable transcripts answer no, so they are not here.
That rule cut 46% of the bytes the 2026-08-12 measurement found in the 175 reports of the two
current batches — ``artifacts`` (19.3%: seven paths the leader never opens, and which the executor
guessed wrong anyway) and ``analysis.examples`` (22.6%: five clipped rollouts the leader cannot
re-parse to check). What replaced the transcripts is
:attr:`ReviewStage.dominant_evidence` — the one comparison that actually separates a protocol
loss from a capability loss, at a fraction of the size.

WHY IT IS SHAPED BY STAGE
-------------------------
data / train / eval / review is already the system's spine: the task prompt is assembled from
stage blocks, the prompts are named per stage, and the old report carried a
``stages.{data,train,eval}.status`` map. But the *payload* sat outside that map — ``dataset``,
``model``, ``eval``, ``analysis`` were stage outputs under non-stage names, while ``fixes`` and
``artifacts`` were flat. Nobody read the status map because it was the empty half of a split.

That split had a measured cost. healthbench's round-7 wrote "added max_steps=500 after the
measured full-epoch ETA left no time" into a run-level ``compatibility_fixes`` list, where it sat
among data-format casts; the review concluded "the second epoch was therefore beneficial" from a
node that had trained 0.40 epoch (audit 13.8). A train-stage fix and a data-stage cast are not the
same kind of fact, and keeping them in one list invited exactly that conflation.

WHY THERE IS NO DIGEST ANY MORE
-------------------------------
The whole report goes to the leader. ``study.py`` used to hand-pick a dozen fields, and every
mismatch between its key names and the contract's was silent: it read ``compat_fixes`` while every
report wrote ``compatibility_fixes``, and it read a ``spec`` key no report has ever had, so
``_varied_keys`` was permanently empty and the leader could not see which knob changed between two
nodes (audit 13.8). A hand-picked projection cannot drift out of sync with itself if there is no
projection. The measurement that makes this affordable: the deepest run of the previous batch was
24 reports totalling 145 KB (~37k tokens) for its ENTIRE history, and the per-round increment is
one report.

``extra="allow"`` is deliberate. Experimenters volunteered an ``eval_format_contract`` block
(parser location, format sensitivity, expected target shape, pre-check outcome) that no contract
asked for and nothing consumed — high-value information the platform was throwing away. Since the
leader now sees the whole report, volunteered fields reach it; the models here only guarantee that
the load-bearing part is present and typed.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_STAGE = Literal["completed", "failed", "skipped"]


class ScriptRef(BaseModel):
    """A script the experimenter wrote, identified so a later round can reproduce it.

    Its consumer is NOT the leader — the leader cannot read a script either. This exists for
    reproducibility and audit: the previous batch produced 4494 and 9028 rows from one nominally
    identical recipe because the real target rendering lived in per-round hand-written scripts that
    nothing pointed at, so "byte-identical recipe" was never a reproduction (audit 13.10 #10).
    A path with no hash is not identification, which is why both fields are required.
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="absolute path on the exec host")
    sha256: str = Field(description="sha256sum of the file — the recipe's real identity")


class Stage(BaseModel):
    """What every stage owes: did it finish, what did it have to work around, who ran it."""

    model_config = ConfigDict(extra="allow")

    status: _STAGE
    error: str | None = Field(
        default=None,
        description="only when status=failed. Say WHICH gate or tool refused and why — '哪个阶段挂"
        "的' must be answerable without parsing prose out of a run-level string.",
    )
    fixes: list[str] = Field(
        default_factory=list,
        description="mechanical workarounds THIS stage needed. They belong to the stage, not to a "
        "run-level list: a training-time fix (max_steps added to fit the budget) means something "
        "different from a data-format cast, and one flat list is how the platform lost that "
        "distinction and let a 0.40-epoch node be read as a completed epochs comparison.",
    )
    script: ScriptRef | None = Field(
        default=None,
        description="the script YOU wrote for this stage, or null when the tool did all of it. "
        "An empty string is not an answer — the previous batch reported 17 of them.",
    )


class DataStage(Stage):
    rows: int = Field(description="rows ACTUALLY written to data/sft")
    requested_rows: int | None = Field(
        default=None,
        description="what the recipe asked for. A selector with strict=false silently returns "
        "whatever the candidate pool had: one plan asked for 50000 and got 176, and the review "
        "still described '50,000 deduplicated examples'. If this differs from `rows` by more than "
        "10%, say so in fixes — the leader's whole update-size axis depends on it being true.",
    )
    fingerprint: str | None = Field(
        default=None,
        description="content hash of train.json. Two nodes with the same fingerprint trained on "
        "identical bytes whatever their plans said — the previous batch grew a three-deep branch "
        "off two 'different seeds' whose data was md5-identical.",
    )


class TrainStage(Stage):
    model_path: str = Field(
        description="the model dir this stage produced — the one the eval stage scores and the "
        "next node continues from. The only path in this report, because the tree consumes it."
    )
    template: str = Field(
        description="the chat template LF ACTUALLY used — read it back from train_config.yaml, do "
        "not restate the plan. A run whose online-ops entry could not register its family's "
        "template produced its best score under another family's turn framing, with the frozen "
        "scorer sending this family's, and nothing in any report said so (audit 13.2)."
    )
    actual_epochs: float | None = Field(
        default=None,
        description="epochs actually completed. If max_steps cut it short this is NOT what the "
        "plan asked for: set it honestly and record the reason in fixes.",
    )
    train_loss: float | None = Field(
        default=None,
        description="final training loss. It is the cheapest detector of a broken premise — the "
        "nodes that trained under a foreign chat template sat at 1.33-1.70 while the correct ones "
        "sat at 0.21-0.32, and no one noticed for five rounds.",
    )


class SweepPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int | Literal["final"]
    score: float


class EvalStage(Stage):
    score: float | None = Field(
        description="native % from summary.json; null when no score exists. This is the ONLY "
        "authoritative score in the report — the old shape let it be read from three different "
        "places."
    )
    limit: int = Field(
        description="items evaluated. -1 = the full frozen benchmark. REQUIRED: a subset score and "
        "a full score are not comparable, and three runs of the previous batch delivered a "
        "subset-scored model as their submission because the caliber was optional."
    )
    stderr: float | None = Field(
        default=None,
        description="the metric's stderr from summary.json when present. Without it the leader "
        "cannot tell a result from noise: one run built a five-round tree on a 1.98-point "
        "difference whose stderr was 5.06.",
    )
    elapsed_s: float | None = Field(
        default=None,
        description="wall clock of this eval. The leader prices its own next move with it — one "
        "run traded 60 seconds of eval for a stderr rise from 2.20 to 3.86 because it could not "
        "see that the full benchmark cost only 155 seconds.",
    )
    checkpoint_sweep: list[SweepPoint] | None = Field(
        default=None,
        description="every candidate you scored when the plan set save_steps, winner included. "
        "'the midpoint beat the final' is a finding for the next round even when the winner is "
        "unremarkable.",
    )


class ReviewStage(Stage):
    """The one stage that produces a judgement rather than an artifact."""

    failure_classes: dict[str, int] = Field(
        description="your own class names -> counts. The leader compares these ACROSS rounds."
    )
    sampled: bool = Field(
        description="true when the counts come from a SAMPLE, false for a full census. The "
        "previous batch mixed 5-item samples with 100-item censuses under identical key names and "
        "the leader compared the numbers anyway."
    )
    anomalies: str = Field(description="repetition / length / format drift, or 'none'")
    conclusion: str = Field(
        description="one sentence: the dominant failure mode AND where it lives — model "
        "capability / output shape / scorer parsing."
    )
    dominant_evidence: str | None = Field(
        default=None,
        description="ONE short comparison that justifies the conclusion: what the parser wanted "
        "versus what the model actually emitted, quoted. This replaces the five clipped rollouts "
        "the old shape carried (22.6% of all report bytes) because the leader cannot re-parse a "
        "transcript to check it — but it CAN weigh one concrete mismatch against your conclusion, "
        "and it should, since a conclusion has contradicted its own failure_classes before.",
    )


class Stages(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: DataStage | None = Field(default=None, description="absent on a baseline node")
    train: TrainStage | None = Field(default=None, description="absent on a baseline node")
    eval: EvalStage
    review: ReviewStage | None = Field(
        default=None,
        description="written in a SECOND pass, after the score is already on disk. A completed "
        "report without review is acceptable; a review written before the score is a violation.",
    )


class Report(BaseModel):
    """One experiment's report. The leader receives this verbatim."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(description="the tree node id from your task prompt")
    stages: Stages

    @property
    def score(self) -> float | None:
        """The node's score. Single-sourced: there is exactly one place it can come from."""
        return self.stages.eval.score

    @property
    def status(self) -> Literal["completed", "failed"]:
        """Derived, never written by hand: completed only when a real score exists."""
        return "completed" if isinstance(self.score, (int, float)) else "failed"

    @property
    def model_path(self) -> str | None:
        """What the next node continues from, when this one trained anything."""
        return self.stages.train.model_path if self.stages.train else None


def _legacy_stage_status(raw: dict, stage: str, *, default: str = "completed") -> tuple[str, str | None]:
    """``(terminal_status, note)`` for one stage of an old report.

    Old reports wrote ``stages`` as a dict of dicts, a dict of plain strings, or a bare name list,
    and 11 of the 1232 on disk are MID-FLIGHT snapshots — a stage left at ``running`` / ``pending``
    / ``not_started`` / ``timeout`` because the experiment was killed before it finished. This
    schema's vocabulary is deliberately terminal (an experimenter writing a report has no
    non-terminal option to reach for), so the non-terminal words are normalised here, in the one
    place that speaks the old dialect. The original word is returned as a note rather than dropped:
    "the train stage never started" and "training failed" are different diagnoses, and the leader
    needs the difference.
    """
    stages = raw.get("stages")
    entry = stages.get(stage) if isinstance(stages, dict) else None
    if isinstance(entry, dict):
        word = str(entry.get("status") or default)
    elif isinstance(entry, str):
        word = entry
    else:
        word = default
    if word in ("completed", "failed", "skipped"):
        return word, None
    return "failed", f"stage was still {word!r} when this report was written"


def from_legacy(raw: dict) -> Report:
    """Read a pre-stage-shaped report.

    1232 reports in the workspace still carry it, and they are the regression corpus for this
    schema — a shape change that cannot read its own history cannot be checked against it
    (``tests/agent/test_report_contract.py`` reads all of them). This is the only place that knows
    the old field names; consumers call :func:`read` and never see them.
    """
    ev = raw.get("eval") or {}
    analysis = raw.get("analysis") or {}
    dataset = raw.get("dataset") or {}
    model = raw.get("model") or {}
    # the old shape allowed the score in three places; take them in the order _report_score did
    score = raw.get("score")
    if not isinstance(score, (int, float)):
        score = ev.get("macro_average")
    if not isinstance(score, (int, float)):
        score = ev.get("score")
    limit = ev.get("limit")

    def stage(name: str, **fields) -> dict:
        status, note = _legacy_stage_status(raw, name)
        error = fields.pop("error", None) or note
        return {"status": status, "error": error, **fields}

    stages: dict = {
        "eval": stage(
            "eval",
            score=float(score) if isinstance(score, (int, float)) else None,
            # a legacy report that never recorded its caliber is the very hazard this field
            # exists for; -1 would silently claim "full", so leave it visibly wrong instead
            limit=int(limit) if isinstance(limit, (int, float)) else 0,
            checkpoint_sweep=analysis.get("checkpoint_sweep"),
            error=raw.get("error") if not isinstance(score, (int, float)) else None,
        )
    }
    if dataset:
        stages["data"] = stage(
            "data",
            rows=int(dataset.get("rows") or 0),
            fingerprint=dataset.get("fingerprint"),
        )
    if model.get("path"):
        stages["train"] = stage(
            "train",
            model_path=str(model["path"]),
            # legacy reports never recorded what LF actually used — that gap is audit 13.2
            template=str(raw.get("template") or "unrecorded"),
        )
    if analysis:
        stages["review"] = {
            "status": "completed",
            "failure_classes": analysis.get("failure_classes") or {},
            # legacy counts mixed 5-item samples with full censuses under the same key names,
            # and nothing recorded which was which — assume the cautious answer
            "sampled": True,
            "anomalies": str(analysis.get("anomalies") or "none"),
            "conclusion": str(analysis.get("conclusion") or ""),
        }
    fixes = raw.get("compatibility_fixes") or raw.get("compat_fixes") or []
    if fixes:
        stages.setdefault("data", {"status": "completed", "rows": 0})
        stages["data"].setdefault("fixes", [])
        stages["data"]["fixes"] = [str(f) for f in fixes]
    return Report.model_validate({"id": raw.get("id") or raw.get("name") or "?", "stages": stages})


def read(raw: dict) -> Report:
    """Validate a report of either shape. The only entry point consumers should use."""
    try:
        return Report.model_validate(raw)
    except Exception:
        return from_legacy(raw)


def is_terminal(raw: dict) -> bool:
    """Has the experimenter FINISHED, or is this the score-first incremental write mid-run?

    The executor's resume logic hangs off this question: answer "yes" too early and the review
    stage never gets written, answer "no" for a finished report and the experimenter is re-run
    forever. The old shape answered it with a hand-written top-level ``status``, where ``running``
    meant "scored but still working" — the one legitimate use of a field this schema drops.

    The replacement is already in the shape rather than beside it. The contract orders the two
    writes: data/train/eval first (so a score is never hostage to later steps), then a rewrite
    adding ``review``. So a report carrying ``review`` is one whose author considered itself done,
    and a scored report without one is mid-run — exactly what ``status: "running"`` encoded. A
    failed eval is terminal too: there is no review to wait for.
    """
    status = raw.get("status")
    if isinstance(status, str):  # legacy: the experimenter declared terminality by hand
        return status in ("completed", "failed")
    try:
        parsed = Report.model_validate(raw)
    except Exception:
        return False  # not a report we can reason about — treat as unfinished, never as done
    return parsed.stages.review is not None or parsed.stages.eval.status == "failed"


def field_names() -> set[str]:
    """Every field name this schema declares, at any depth — the drift test's left-hand side."""
    seen: set[str] = set()

    def walk(model: type[BaseModel]) -> None:
        for name, field in model.model_fields.items():
            seen.add(name)
            for candidate in (field.annotation, *getattr(field.annotation, "__args__", ())):
                if isinstance(candidate, type) and issubclass(candidate, BaseModel):
                    walk(candidate)

    walk(Report)
    return seen
