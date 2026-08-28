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


def _tool_providing(env: dict, what: str) -> dict:
    """The env's venv that declares ``what`` in its ``provides``.

    The environment declares which venv ships which tool; nothing here holds a table of paths.
    A deployment that installs the instruments somewhere else changes one line of env.yaml.
    """
    for tool in (env.get("tools") or {}).values():
        if isinstance(tool, dict) and what in (tool.get("provides") or []):
            return tool
    return {}


def _placement_flags(resolved: dict) -> str:
    """The run's placement choices as CLI flags — the same four every instrument takes.

    Rendered from the run config rather than written into any prompt: a stage that could state
    its own partition or QOS would be a fourth opinion on placement, and the one furthest from
    the config that decides it. dataos-train carried exactly that, as `--launcher slurm`.
    """
    run = resolved.get("run", {}) or {}
    resource = run.get("resource", {}) or {}
    return " ".join(filter(None, [
        f"--qos {resource['qos']}" if resource.get("qos") else "",
        f"--nodelist {resource['node']}" if resource.get("node") else "",
        f"--partition {resource['partition']}" if resource.get("partition") else "",
    ]))


class OperatorToolbox:
    """The installed ``dataos_operator`` toolbox, registered as the data-stage Operator.

    The invoke prose carries the toolbox's own catalog (introspected from the installed
    package, so it always matches the pinned version) so the experimenter plans its recipe
    against the real menu instead of discovering it blind inside the Slurm job."""

    name = "dataos_operator"
    owner = OWNER_OURS

    def invoke(self, resolved: dict) -> str:
        import dataos_operator

        env = resolved.get("environment", {})
        tool = _tool_providing(env, "dataos-data")
        return _prompt.render(
            "experimenter/turn/stage_data",
            catalog=dataos_operator.catalog(),
            data_bin=f"{tool.get('path', '<sft_env>')}/bin/dataos-data",
            data_env=tool.get("path", "<sft_env>"),
            provides=", ".join(tool.get("provides") or []) or "<the env's declared packages>",
            placement=_placement_flags(resolved),
        )


class FrozenEvaluator:
    """The task's frozen scorer, registered as the eval-stage Evaluator from the
    resolved runtime contract (``resolved["task"]["scoring"]``)."""

    name = "frozen_evaluator"
    owner = OWNER_PROVIDED

    def invoke(self, resolved: dict) -> str:
        task = resolved.get("task", {})
        env = resolved.get("environment", {})
        run = resolved.get("run", {})
        scoring = task.get("scoring", {})
        result = scoring.get("result", {})
        fc = scoring.get("format_contract")
        eval_bin = f"{_tool_providing(env, 'dataos-eval').get('path', '<sft_env>')}/bin/dataos-eval"
        repo = run.get("deploy_root") or "<deploy_root>"
        task_config = run.get("task", "")
        eval_contract = (task.get("stages") or {}).get("eval") or {}
        if eval_contract.get("driver") == "external_cli":
            return _prompt.render(
                "experimenter/turn/stage_eval_external",
                command=str(scoring.get("command", "")).strip(),
                followup=str(eval_contract.get("followup", "")).strip(),
                result_field=result.get("field", "score"),
                result_file=result.get("file", "<result>"),
                result_range=result.get("range", [0, 1]),
                format_contract_loc=str(fc or ""),
            )
        return _prompt.render(
            "experimenter/turn/stage_eval",
            command=str(scoring.get("command", "")).strip(),
            eval_bin=eval_bin,
            repo=repo,
            environment=run.get("environment", ""),
            task_config=task_config,
            result_field=result.get("field", "score"),
            result_file=result.get("file", "<result>"),
            result_range=result.get("range", [0, 100]),
            format_contract_loc=(
                f"{repo}/configs/tasks/{task_config}/{fc}"
                if fc and " " not in str(fc) and not str(fc).startswith("/")
                else str(fc or "")
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
        tools = env.get("tools", {}) or {}
        te_path = next(
            (
                t.get("path") if isinstance(t, dict) else t
                for t in tools.values()
                if "dataos-train" in ((t or {}).get("provides") or [])
            ),
            None,
        ) or "<train_cli_env>"
        # The same four flags dataos-eval takes (see task.yaml scoring.command): the
        # deployment to read, and the run's own choices within it.
        run = resolved.get("run", {}) or {}
        placement = " ".join(filter(None, [
            f"--repo {run.get('deploy_root')} --environment {run.get('environment')}",
            _placement_flags(resolved),
        ]))
        return _prompt.render(
            "experimenter/turn/stage_train", frozen_command="", frozen_produces="",
            frozen_followup="",
            cli_path=f"{te_path}/bin/dataos-train", placement=placement,
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
