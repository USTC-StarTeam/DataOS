"""Declarative Stage abstraction — the single source of truth for the sub-agent's
three-phase pipeline (data → train → eval).

A Stage is a CONTRACT-AND-PRESENTATION structure, not a scheduling one: execution
stays a single self-healing sub-agent session (data+train deliberately share one
Slurm job), but the prompt capability blocks, the report.json ``stages`` skeleton,
and the artifact expectations all derive from this one table.

Ownership encodes lever-vs-ruler: DATA and TRAIN are *ours* (the agent's improvable
method); EVAL is *provided* (the task's frozen scorer — outside the method space).
When DataOS-train lands, only the TRAIN stage's ``invoke`` body changes.
"""

from __future__ import annotations

from dataclasses import dataclass

OWNER_OURS = "ours — DataOS method, improvable"
OWNER_PROVIDED = "provided — frozen scorer, do NOT modify"


@dataclass(frozen=True)
class Stage:
    name: str    # data | train | eval
    role: str    # what this step turns into what
    input: str
    output: str  # artifact contract (path + format)
    invoke: str  # how to discover + call
    where: str   # compute job / login node
    owner: str   # OWNER_OURS | OWNER_PROVIDED


def _eval_invoke(scoring: dict, assets_root: str) -> str:
    lines = [f"run: `{scoring.get('command', '').strip()}`"]
    judge = scoring.get("judge_command")
    if judge:
        lines.append(
            "judge benches (healthbench/arenahardwriting): the command above generates "
            f"predictions only; then on the LOGIN node run: `{str(judge).strip()}`"
        )
    fc = scoring.get("format_contract")
    if fc:
        lines.append(
            f"how the scorer parses model output (read BEFORE building data): {assets_root}/{fc}"
        )
    return "\n  ".join(lines)


def build_stages(resolved: dict) -> list[Stage]:
    """DATA/TRAIN are brain constants; EVAL is formatted from the resolved task scoring."""
    scoring = resolved.get("task", {}).get("scoring", {})
    result = scoring.get("result", {})
    assets_root = resolved.get("task", {}).get("assets_root", "")
    return [
        Stage(
            name="data",
            role="turn fixed-pool sources into THIS experiment's training dataset",
            input="the pool sources named in the DATA RECIPE (dirs under the fixed pool)",
            output="{ctx.run_root}/data/sft — LLaMA-Factory alpaca train.json + dataset_info.json",
            invoke=(
                "dataos_operator recipe (discover: `python -c \"import dataos_operator as o; "
                "print(o.guide())\"` / `o.usage('<op>')`) + hand-written prep scripts where the "
                "toolbox lacks a step; custom ops via DATAOS_OPERATOR_PLUGINS — full mechanics "
                "in your system prompt"
            ),
            where="inside the Slurm GPU job (pool lives there); only bounded llm_* scoring on login",
            owner=OWNER_OURS,
        ),
        Stage(
            name="train",
            role="(parent model, dataset, plan.train) -> a complete HF model",
            input="parent model path + {ctx.run_root}/data/sft + the plan's SFT hyperparameters",
            output=(
                "{ctx.run_root}/final_model — COMPLETE HF model dir (lora: adapter at "
                "{ctx.run_root}/adapter, then export-merge; full/freeze: write final_model directly)"
            ),
            invoke=(
                "LLaMA-Factory in train_env: `llamafactory-cli train` + (lora) "
                "`llamafactory-cli export` — recipe details in your system prompt"
            ),
            where="same Slurm GPU job as the data stage",
            owner=OWNER_OURS,
        ),
        Stage(
            name="eval",
            role="model -> native benchmark score",
            input="{ctx.model_path} (final_model; the parent model itself for a BASELINE run)",
            output=(
                f"{result.get('file', '{ctx.eval_dir}/summary.json')} — field "
                f"`{result.get('field', 'score')}` in range {result.get('range', [0, 100])}"
            ),
            invoke=_eval_invoke(scoring, assets_root),
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


def report_stage_names(stages: list[Stage]) -> list[str]:
    return [s.name for s in stages]
