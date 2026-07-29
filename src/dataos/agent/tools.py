"""Pluggable stage tools — the three symmetric tool kinds behind the experimenter's
data → train → eval pipeline.

A tool is a PROMPT ASSEMBLY UNIT, not a runtime ``tool_use``: it declares a capability
(``name`` / ``owner`` / ``invoke(resolved)`` prose) that stages.py assembles into the
experimenter task prompt. The experimenter stays a free CLI agent that discovers and calls
the actual mechanics at execution time.

Three kinds, one shape:

- :class:`Operator`  (data)  — turns fixed-pool sources into THIS experiment's dataset.
  Registered FROM the installed ``dataos_operator`` package (:class:`OperatorToolbox`).
- :class:`Trainer`   (train) — (data, recipe) -> a complete HF model dir. Shipped:
  :class:`DataosTrainCli` (the deterministic dataos-train launcher lifecycle).
- :class:`Evaluator` (eval)  — model -> native benchmark score. Registered FROM the
  resolved runtime contract (:class:`FrozenEvaluator`, the task's frozen scorer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from dataos import prompt as _prompt

OWNER_OURS = "ours — DataOS method, improvable"
OWNER_PROVIDED = "provided — frozen scorer, do NOT modify"


class Operator(Protocol):
    """data-stage tool: turns fixed-pool sources into THIS experiment's dataset."""

    name: str
    owner: str  # OWNER_OURS

    def invoke(self, resolved: dict) -> str: ...


class Trainer(Protocol):
    """train-stage tool: (data, recipe) -> a complete HF model dir.

    Shipped implementation: :class:`DataosTrainCli` (the deterministic TrainSpec
    lifecycle). Contract: the brain code-injects the runtime-owned TRAIN SPEC fields;
    only ``method`` comes from the leader's plan.
    """

    name: str
    owner: str  # OWNER_OURS (OWNER_PROVIDED when the task ships a frozen train/ dir)

    def invoke(self, resolved: dict) -> str: ...


class Evaluator(Protocol):
    """eval-stage tool: model -> native benchmark score (the task's frozen scorer)."""

    name: str
    owner: str  # OWNER_PROVIDED

    def invoke(self, resolved: dict) -> str: ...


class OperatorToolbox:
    """The installed ``dataos_operator`` toolbox, registered as the data-stage Operator.

    The invoke prose carries the toolbox's own catalog (introspected from the installed
    package, so it always matches the pinned version) so the experimenter plans its recipe
    against the real menu instead of discovering it blind inside the Slurm job."""

    name = "dataos_operator"
    owner = OWNER_OURS

    def invoke(self, resolved: dict) -> str:  # noqa: ARG002 — catalog is install-local
        import dataos_operator

        return _prompt.render("kb/stage_data", catalog=dataos_operator.catalog())


class FrozenEvaluator:
    """The task's frozen scorer, registered as the eval-stage Evaluator from the
    resolved runtime contract (``resolved["task"]["scoring"]``)."""

    name = "frozen_evaluator"
    owner = OWNER_PROVIDED

    def invoke(self, resolved: dict) -> str:
        task = resolved.get("task", {})
        scoring = task.get("scoring", {})
        result = scoring.get("result", {})
        fc = scoring.get("format_contract")
        assets_root = task.get("assets_root", "")
        return _prompt.render(
            "kb/stage_eval",
            command=str(scoring.get("command", "")).strip(),
            result_field=result.get("field", "score"),
            result_file=result.get("file", "<result>"),
            result_range=result.get("range", [0, 100]),
            judge_command=str(scoring.get("judge_command") or "").strip(),
            format_contract_loc=(
                f"{assets_root}/{fc}" if fc and " " not in str(fc) else str(fc or "")
            ),
        )


class DataosTrainCli:
    """dataos-train, registered as the train-stage Trainer: (data, recipe) -> a complete
    HF model via the deterministic lifecycle — validate / build / submit --launcher slurm
    / status / collect. The invoke prose (kb/stage_train) points the experimenter at the
    CLI in train.cli_env (the thin orchestrator env) or, undeclared, the backend env."""

    name = "dataos_train"
    owner = OWNER_OURS

    def invoke(self, resolved: dict) -> str:
        env = resolved.get("environment", {})
        train_cfg = env.get("train", {}) or {}
        tool_key = train_cfg.get("cli_env") or (train_cfg.get("backends") or {}).get(
            train_cfg.get("backend", "llamafactory"), "sft_env"
        )
        te = env.get("tools", {}).get(tool_key, {})
        te_path = te.get("path") if isinstance(te, dict) else (te or "<train_env>")
        return _prompt.render(
            "kb/stage_train", frozen_command="", frozen_produces="",
            cli_path=f"{te_path}/bin/dataos-train",
        )


@dataclass
class ToolRegistry:
    """The three tool slots behind the three stages. Sources register; stages.py
    assembles the prompt. data takes an ordered list of Operators; train/eval are
    single slots (one training method / one frozen scorer per run)."""

    operators: list[Operator] = field(default_factory=list)
    trainer: Trainer | None = None  # single slot — one training method per run
    evaluator: Evaluator | None = None

    def register_operator(self, op: Operator) -> None:
        self.operators.append(op)

    def register_trainer(self, trainer: Trainer) -> None:
        self.trainer = trainer

    def register_evaluator(self, ev: Evaluator) -> None:
        self.evaluator = ev


def build_registry(resolved: dict) -> ToolRegistry:  # noqa: ARG001 — impls are self-sourcing
    """The default registry: the operator toolbox FROM the installed dataos_operator,
    the dataos-train Trainer, and the frozen scorer FROM the resolved runtime contract."""
    reg = ToolRegistry()
    reg.register_operator(OperatorToolbox())
    reg.register_trainer(DataosTrainCli())
    reg.register_evaluator(FrozenEvaluator())
    return reg
