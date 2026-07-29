"""Declarative Stage abstraction — the single source of truth for the experimenter's
three-phase pipeline (data → train → eval).

A Stage is a CONTRACT-AND-PRESENTATION structure, not a scheduling one: execution
stays a single self-healing experimenter session (data+train deliberately share one
Slurm job), but the prompt capability blocks, the report.json ``stages`` skeleton,
and the artifact expectations all derive from this one table.

Each stage's ``invoke`` is ASSEMBLED FROM REGISTERED TOOLS (see
:mod:`dataos.agent.tools`): Operators from the installed dataos_operator, the
Evaluator from the resolved runtime contract, and — once dataos-train lands — a
Trainer from the reserved registry slot (until then the train stage renders the
legacy prose block ``stage_train_legacy``). No prompt prose lives in this file.

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
    """The train stage's invoke: the registered Trainer's own contract once one exists;
    the legacy prose block (frozen recipe / dataos-train CLI flow) while the slot is empty."""
    if registry.trainer is not None:
        return registry.trainer.invoke(resolved)
    frozen = resolved.get("task", {}).get("train")
    env = resolved.get("environment", {})
    train_cfg = env.get("train", {}) or {}
    tool_key = (train_cfg.get("backends") or {}).get(
        train_cfg.get("backend", "llamafactory"), "sft_env"
    )
    te = env.get("tools", {}).get(tool_key, {})
    te_path = te.get("path") if isinstance(te, dict) else (te or "<train_env>")
    return _prompt.render(
        "kb/stage_train_legacy",
        frozen_command=str((frozen or {}).get("command", "")).strip(),
        frozen_produces=(frozen or {}).get("produces", "the harness run_dir model"),
        cli_path=f"{te_path}/bin/dataos-train",
    )


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
            output="{ctx.run_root}/data/sft — LLaMA-Factory alpaca train.json + dataset_info.json",
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
            where="scoring command in the GPU job (vLLM); judge step on the login node",
            owner=OWNER_PROVIDED,
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
