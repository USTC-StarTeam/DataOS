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

Ownership encodes lever-vs-ruler: DATA and TRAIN are *ours* (the agent's improvable
method); EVAL is *provided* (the task's frozen scorer — outside the method space).
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
    """The train stage's invoke: FROZEN tasks (the task ships a train/ dir) render the
    frozen branch; everything else is the registered Trainer's own contract."""
    frozen = resolved.get("task", {}).get("train")
    if frozen:
        return _prompt.render(
            "experimenter/turn/stage_train",
            frozen_command=str(frozen.get("command", "")).strip(),
            frozen_produces=frozen.get("produces", "the harness run_dir model"),
            cli_path="", placement="",
        )
    if registry.trainer is None:
        raise ValueError("no Trainer registered in the tool registry (train stage)")
    return registry.trainer.invoke(resolved)


def build_stages(resolved: dict, registry: _tools.ToolRegistry | None = None) -> list[Stage]:
    """All three stages derive from the task contract, with invokes assembled from the
    tool registry (built from the same contract when not given). Owners: DATA=ours (the
    agent's lever); TRAIN=frozen if the task ships a train/ dir (resolved.task.train set)
    else ours; EVAL is always the uniform model→score runner (provided)."""
    registry = registry or _tools.build_registry(resolved)
    scoring = resolved.get("task", {}).get("scoring", {})
    result = scoring.get("result", {})
    train_frozen = bool(resolved.get("task", {}).get("train"))
    return [
        Stage(
            name="data",
            role="turn fixed-pool sources into THIS experiment's training dataset",
            input="the pool sources named in the DATA RECIPE (dirs under the fixed pool)",
            output="{ctx.run_root}/data/sft — an HF save_to_disk dataset; the job derives LLaMA-Factory's train.json + dataset_info.json from it",
            invoke="\n\n".join(op.invoke(resolved) for op in registry.operators),
            where="inside the Slurm GPU job (pool lives there); only bounded llm_* scoring on login",
            owner=OWNER_OURS,
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
            owner=OWNER_PROVIDED,
        ),
        Stage(
            name="review",
            role="model rollouts -> failure analysis",
            input="{ctx.eval_dir}/predictions.jsonl (the model's raw answers from EVAL)",
            output="the `analysis` block in report.json (failure classes + examples + conclusion)",
            invoke=_prompt.render("experimenter/turn/stage_review"),
            where="login node (reading jsonl is light)",
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
