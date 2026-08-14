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
leader now sees the whole report, volunteered fields reach it.

WHY NOTHING HERE IS REQUIRED
----------------------------
Because a report carries two payloads with opposite natures, and validation helps neither.

The MEASUREMENTS (score, limit, n_samples, stderr, elapsed) are written by the frozen eval harness
into ``summary.json``. When the experimenter restates them here it is acting as a courier, and a
courier of numbers should be removed, not inspected: ``roles._stamp_report`` now reads the
harness's own files and overwrites this block, so what reaches the leader is machine-sourced and
the fields below are documentation of what will appear, not a demand on the writer.

The JUDGEMENT (conclusion, failure_classes, anomalies, dominant_evidence, fixes) is written by a
model and read by a model. No schema can check whether ``conclusion`` is TRUE, and the leader does
not need it well-formed — it reads prose for a living.

So validation had no true consumer, and it had a measured cost. Of the 69 stage-shaped reports on
disk, 33 failed strict validation and 22 of those lost a real score: ``read()`` treated "failed to
validate" as "must be an old-shape report", fell through to a legacy reader that looks for
top-level ``eval``/``score``/``dataset`` keys a stage-shaped report does not have, and returned
``score=None, status="failed"`` — a plausible-looking lie, logged nowhere. Scores of 27.44, 15.15
and 13.84 read back as failed experiments. Five of the 33 rejections WERE real (an ``eval`` block
with no ``limit``), and they were as silent as the 28 false ones; detection that cannot raise its
voice is worse than no detection, because it looks like a guard.

What remains is the one thing that was always paying off: this file is the field-name registry the
prompt contract is checked against (``test_every_schema_field_is_asked_for_in_the_contract``). A
field declared here and never asked for is a field no report will carry; a field asked for and not
declared is one the schema silently drops. That test needs no runtime validation at all.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

log = logging.getLogger(__name__)

# Not a Literal. The vocabulary an experimenter is ASKED for is completed/failed/skipped, but
# reports killed mid-flight carry `running` / `pending` / `not_started`, and a snapshot of a stage
# that never finished is a fact worth keeping, not a parse error. Every consumer tests for
# `== "completed"`, so an unknown word is already handled: it simply is not completed.
_STAGE = str


class ScriptRef(BaseModel):
    """A script the experimenter wrote, identified so a later round can reproduce it.

    Its consumer is NOT the leader — the leader cannot read a script either. This exists for
    reproducibility and audit: the previous batch produced 4494 and 9028 rows from one nominally
    identical recipe because the real target rendering lived in per-round hand-written scripts that
    nothing pointed at, so "byte-identical recipe" was never a reproduction (audit 13.10 #10).

    A path with no hash is weak identification — but requiring the hash is worse than accepting a
    weak one. This class was the last place in the file still able to reject, and it did: an
    arenahardwriting node wrote ``script: {"path": ...}`` with no ``sha256`` and its whole report,
    score 26.60 included, failed to parse. A provenance field that can destroy a measurement is
    not paying for itself.
    """

    model_config = ConfigDict(extra="allow")

    path: str | None = Field(default=None, description="absolute path on the exec host")
    sha256: str | None = Field(
        default=None,
        description="sha256sum of the file — the recipe's real identity. Without it the path names "
        "a file whose contents nothing pins, so two runs can cite the same script and mean "
        "different bytes.",
    )


class Stage(BaseModel):
    """What every stage owes: did it finish, what did it have to work around, who ran it."""

    model_config = ConfigDict(extra="allow")

    status: _STAGE | None = Field(
        default=None,
        description="completed / failed / skipped. Absent is not a parse error — every consumer "
        "asks `== 'completed'`, and an unstated status is already not that.",
    )
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
    rows: int | None = Field(
        default=None, description="rows ACTUALLY written to data/sft; omit when status=skipped"
    )
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
    model_path: str | None = Field(
        default=None,
        description="the model dir this stage produced — the one the eval stage scores and the "
        "next node continues from. The only path in this report, because the tree consumes it, "
        "and the only field here the runtime cannot recover from the harness's own files: which "
        "checkpoint you hand forward is a DECISION, not a measurement. Omit when status=skipped.",
    )
    template: str | None = Field(
        default=None,
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
    """One scored eval directory. Written by the runtime, never by the experimenter.

    Asking the experimenter to declare its own sweep only ever surfaced the winner: 15 of the 29
    nodes of the 2026-08-13 batch scored between 2 and 6 checkpoints on the FULL benchmark and
    reported the maximum, and nothing downstream could see that a champion's 30.36 came from
    checkpoint 12 while its final weights scored 28.12. The eval dirs were on disk the whole time.
    """

    model_config = ConfigDict(extra="allow")

    eval_dir: str | None = None
    score: float | None = None
    limit: int | None = None
    n_samples: int | None = None


class EvalStage(Stage):
    score: float | None = Field(
        default=None,
        description="native % — restate it from summary.json if you like, but the runtime "
        "overwrites this block from the harness's own files, so a typo here costs nothing.",
    )
    limit: int | None = Field(
        default=None,
        description="items evaluated. -1 = the full frozen benchmark. A subset score and a full "
        "score are not comparable, and three runs of an earlier batch delivered a subset-scored "
        "model as their submission because the caliber was missing — which is why the runtime now "
        "takes it from summary.json rather than asking you to be careful.",
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
        description="every scored eval dir under this node, winner included — stamped by the "
        "runtime from the harness's own files, so 'the midpoint beat the final' reaches the next "
        "round whether or not the experimenter thought to mention it.",
    )
    n_samples: int | None = Field(
        default=None,
        description="items the scorer ACTUALLY scored, copied from summary.json. Distinct from "
        "`limit`, which is what was REQUESTED: an AUTO run asks for -2 and a capped run can be "
        "cut short, so only this says how big the measurement really was. The round-0 baseline's "
        "value is load-bearing beyond its own report — it is the only place a run learns the "
        "benchmark's true size, which is what lets the tree render a subset score as "
        "'74.00@200/1319' instead of a bare '@200'.",
    )


class ReviewStage(Stage):
    """The one stage that produces a judgement rather than an artifact."""

    failure_classes: dict[str, int] = Field(
        default_factory=dict,
        description="your own class names -> counts. The leader compares these ACROSS rounds.",
    )
    sampled: bool | None = Field(
        default=None,
        description="true when the counts come from a SAMPLE, false for a full census. The "
        "previous batch mixed 5-item samples with 100-item censuses under identical key names and "
        "the leader compared the numbers anyway. Derived automatically when you give "
        "`sample_size` instead.",
    )
    sample_size: int | dict[str, int] | None = Field(
        default=None,
        description="how many rollouts you actually read — a count (5) or a breakdown "
        "({'correct': 5, 'wrong': 5}). Omit for a full census. Both spellings appear in real "
        "reports and both answer the question, so both are accepted: insisting on one of them is "
        "what made this the single most common validation failure on disk (21 of 33).",
    )

    @model_validator(mode="before")
    @classmethod
    def _sampled_from_sample_size(cls, data: object) -> object:
        """Let ``sample_size`` stand in for ``sampled``.

        An experimenter wrote ``sample_size: {"correct": 5, "wrong": 5}`` and no ``sampled``; the
        field the leader reads was then absent even though the report had said MORE than the
        contract asked. Deriving costs six lines and removes the whole class of "right answer,
        wrong key".
        """
        if isinstance(data, dict) and data.get("sampled") is None and data.get("sample_size"):
            return {**data, "sampled": True}
        return data
    anomalies: str | None = Field(
        default=None, description="repetition / length / format drift, or 'none'"
    )
    conclusion: str | None = Field(
        default=None,
        description="one sentence: the dominant failure mode AND where it lives — model "
        "capability / output shape / scorer parsing.",
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
    # `forbid` here rejected a whole report because an experimenter declared an `export` stage for
    # its LoRA merge — a stage that really happened, described in the natural place.
    model_config = ConfigDict(extra="allow")

    data: DataStage | None = Field(default=None, description="absent on a baseline node")
    train: TrainStage | None = Field(default=None, description="absent on a baseline node")
    eval: EvalStage = Field(
        default_factory=lambda: EvalStage(status="failed"),
        description="the scored stage. Its absence is not a parse error, it is a node with no "
        "measurement — which is what `failed` means.",
    )
    review: ReviewStage | None = Field(
        default=None,
        description="written in a SECOND pass, after the score is already on disk. A completed "
        "report without review is acceptable; a review written before the score is a violation.",
    )


class Report(BaseModel):
    """One experiment's report. The leader receives this verbatim."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(default="", description="the tree node id from your task prompt")
    stages: Stages = Field(default_factory=Stages)

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


def read(raw: dict) -> Report:
    """The one way to turn a report on disk into typed values.

    It does not validate — nothing above is required — so the only way this raises is a `raw` that
    is not report-shaped at all (``stages`` as a list, say). That is a real error and belongs to
    the caller, which is the change: the previous version caught every exception and fell through
    to a reader for a report shape no live code path ever encounters, turning any mismatch into a
    silent ``score=None, status="failed"``.
    """
    return Report.model_validate(raw)


def is_terminal(raw: dict) -> bool:
    """Has the experimenter FINISHED, or is this the score-first incremental write mid-run?

    The executor's resume logic hangs off this question: answer "yes" too early and the review
    stage never gets written, answer "no" for a finished report and the experimenter is re-run
    forever — which is exactly what happened to a node whose report was complete on disk from its
    first write: validation kept failing, this returned False, and the agent was woken four times
    over six minutes to re-assert a report it had already written correctly.

    The answer is in the shape rather than beside it. The contract orders the two writes:
    data/train/eval first (so a score is never hostage to later steps), then a rewrite adding
    ``review``. So a report carrying ``review`` is one whose author considered itself done, and a
    scored report without one is mid-run. A failed eval is terminal too: there is no review to
    wait for.

    A file that is not report-shaped at all is not a finished experiment either, so it answers
    False — but it says so first. That distinction is the whole point: the version this replaces
    swallowed EVERY parse failure into the same False, which is how a report that was complete and
    correct on disk kept reading as unfinished while the loop woke its author four times to
    re-assert it. ``read()`` still raises for callers that want the value; the loop's control
    question has to stay answerable.
    """
    try:
        parsed = read(raw)
    except Exception as e:  # noqa: BLE001 — not report-shaped; unfinished, and worth saying
        log.error("report.json is not report-shaped, treating as unfinished: %s", e)
        return False
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
