"""Declarative Stage abstraction — the single source of truth for the experimenter's
four-phase pipeline (data → train → eval → review).

A Stage is a CONTRACT-AND-PRESENTATION structure, not a scheduling one: execution
stays a single self-healing experimenter session (data+train deliberately share one
Slurm job), but the prompt capability blocks, the report.json ``stages`` skeleton,
and the artifact expectations all derive from this one table.

Each stage's ``invoke`` is ASSEMBLED FROM REGISTERED TOOLS (see
:mod:`dataos.agent.tools`): Operators from the installed dataos_operator, the
Trainer (dataos-train's deterministic lifecycle via :class:`~dataos.agent.tools.DataosTrainCli`),
and the Evaluator from the resolved runtime contract. No prompt prose lives in this file.

Ownership encodes lever-vs-ruler and comes from the task's explicit stage contracts.
"""

from __future__ import annotations

from dataclasses import dataclass

from dataos import prompt as _prompt
from dataos.agent import tools as _tools

OWNER_OURS = _tools.OWNER_OURS
OWNER_PROVIDED = _tools.OWNER_PROVIDED


@dataclass(frozen=True)
class Stage:
    name: str    # data | train | eval
    role: str    # what this step turns into what
    input: str
    output: str  # artifact contract (path + format)
    invoke: str  # how to discover + call — assembled from the registry's tools
    where: str   # compute job / login node
    owner: str   # OWNER_OURS | OWNER_PROVIDED


def _train_invoke(resolved: dict, registry: _tools.ToolRegistry) -> str:
    """Render the configured frozen workflow or the registered native trainer."""
    contract = resolved.get("task", {}).get("stages", {}).get("train", {})
    if contract.get("ownership") == "frozen":
        return _prompt.render(
            "experimenter/turn/stage_train",
            frozen_command=str(contract.get("command", "")).strip(),
            frozen_produces=contract.get("produces", "the harness run_dir model"),
            frozen_followup=str(contract.get("followup", "")).strip(),
            cli_path="", placement="",
        )
    if registry.trainer is None:
        raise ValueError("no Trainer registered in the tool registry (train stage)")
    return registry.trainer.invoke(resolved)


def build_stages(resolved: dict, registry: _tools.ToolRegistry | None = None) -> list[Stage]:
    """All three stages derive from the task contract, with invokes assembled from the
    tool registry (built from the same contract when not given). Owners: DATA=ours (the
    agent's lever); TRAIN/EVAL ownership is declared in the selected stage task.yaml."""
    registry = registry or _tools.build_registry(resolved)
    scoring = resolved.get("task", {}).get("scoring", {})
    result = scoring.get("result", {})
    contracts = resolved.get("task", {}).get("stages", {})
    data_contract = contracts.get("data", {})
    train_contract = contracts.get("train", {})
    eval_contract = contracts.get("eval", {})
    task_agent = resolved.get("task", {}).get("agent") or {}
    review_template = (task_agent.get("turns") or {}).get(
        "review_stage", "experimenter/turn/stage_review"
    )
    external_eval = eval_contract.get("driver") == "external_cli"
    train_frozen = train_contract.get("ownership") == "frozen"
    data_parts = []
    if data_contract.get("toolbox") == "dataos-operator":
        data_parts.extend(op.invoke(resolved) for op in registry.operators)
    data_parts.append(str(data_contract.get("invoke", "")).strip())
    data_invoke = "\n\n".join(part for part in data_parts if part)
    return [
        Stage(
            name="data",
            role="turn fixed-pool sources into THIS experiment's training dataset",
            input=data_contract.get("input", "the task's declared data pool"),
            output=data_contract.get("output", "the task's declared training dataset"),
            invoke=data_invoke,
            where="inside the Slurm GPU job (pool lives there); only bounded llm_* scoring on login",
            owner=(OWNER_OURS if data_contract.get("ownership") == "agent" else OWNER_PROVIDED),
        ),
        Stage(
            name="train",
            role="(data, recipe) -> a complete HF model",
            input="the curated data from DATA + " + (
                "the task's FROZEN recipe" if train_frozen else "the plan's SFT hyperparameters"
            ),
            output=(
                "the task's frozen-training model output" if train_frozen else
                "{ctx.run_root}/final_model — COMPLETE HF model dir (lora: adapter then export-merge)"
            ),
            invoke=_train_invoke(resolved, registry),
            where="a Slurm GPU job" if not train_frozen else "the task's frozen training harness",
            owner=OWNER_PROVIDED if train_frozen else OWNER_OURS,
        ),
        Stage(
            name="eval",
            role="model -> native benchmark score",
            input="{ctx.model_path} (the model from TRAIN; the parent model itself for a BASELINE run)",
            output=(
                f"{result.get('file', '{ctx.eval_dir}/summary.json')} — field "
                f"`{result.get('field', 'score')}` in range {result.get('range', [0, 100])}"
            ),
            invoke=(
                registry.evaluator.invoke(resolved)
                if registry.evaluator is not None
                else "run the task's evaluator per the TASK contract above"
            ),
            where="the dataos-eval CLI on the login node (the tool owns the GPU job + judge step)",
            owner=(OWNER_PROVIDED if eval_contract.get("ownership") == "frozen" else OWNER_OURS),
        ),
        Stage(
            name="review",
            role=(
                "official native benchmark artifacts -> curation evidence"
                if external_eval else "model rollouts -> failure analysis"
            ),
            input=(
                "{ctx.eval_dir}/summary.json plus the official native results and bounded logs"
                if external_eval
                else "{ctx.eval_dir}/predictions.jsonl (the model's raw answers from EVAL)"
            ),
            output=(
                "the `stages.review` curation conclusion in report.json"
                if external_eval
                else "the `analysis` block in report.json (failure classes + examples + conclusion)"
            ),
            invoke=_prompt.render(review_template),
            where=(
                "login node (bounded reads of the selected suite only)"
                if external_eval else "login node (reading jsonl is light)"
            ),
            owner=OWNER_OURS,
        ),
    ]


def render_stage_blocks(stages: list[Stage]) -> str:
    """Render the stages as uniform capability blocks for the task prompt."""
    parts = [
        "## STAGES — three capabilities, one shape (fill `{ctx.*}` at execution time)",
    ]
    for s in stages:
        parts.append(
            f"### STAGE {s.name} — owner: {s.owner}\n"
            f"- role:   {s.role}\n"
            f"- input:  {s.input}\n"
            f"- output: {s.output}\n"
            f"- invoke: {s.invoke}\n"
            f"- where:  {s.where}"
        )
    return "\n\n".join(parts)
