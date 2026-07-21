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


def _eval_invoke(resolved: dict) -> str:
    """UNIFORM across every task: score a model with the common eval runner."""
    env = resolved.get("environment", {})
    eval_py = f"{env.get('tools', {}).get('eval_env', {}).get('path', '<eval_env>')}/bin/python"
    runner = f"{env.get('paths', {}).get('runtime_root', '<runtime_root>')}/common/eval_runner/evaluate.py"
    task = resolved.get("task", {}).get("name", "")
    scoring = resolved.get("task", {}).get("scoring", {})
    assets_root = resolved.get("task", {}).get("assets_root", "")
    fc = scoring.get("format_contract")
    hint = (
        f"  (want a high score? read how the scorer parses output first: {assets_root}/{fc})"
        if fc and " " not in str(fc) else (f"  ({fc})" if fc else "")
    )
    return (
        "UNIFORM (same for every task) — score the model with the common eval runner:\n"
        f"  `{eval_py} {runner} --task {task} --model {{ctx.model_path}} "
        "--out {ctx.eval_dir} --run-config {ctx.run_root}/run_config.json`\n"
        "  it runs this task's frozen scorer on the model + normalizes → {ctx.eval_dir}/score.json;\n"
        "  read score.json `score` into report.json. Write run_config.json from the RUN CONFIG "
        "JSON in your task prompt before calling it." + ("\n" + hint if hint else "")
    )


def _train_invoke(resolved: dict) -> str:
    frozen = resolved.get("task", {}).get("train")
    if frozen:  # task ships a train/ dir → training is FROZEN (agent doesn't choose the recipe)
        cmd = str(frozen.get("command", "")).strip()
        produces = frozen.get("produces", "the harness run_dir model")
        return (
            "FROZEN recipe — you do NOT choose hyperparameters. Turn your curated data into a "
            f"model with the task's fixed training: `{cmd}` → model at {produces}. Fill the "
            "{ctx.*} placeholders; that model is the EVAL stage's {ctx.model_path}."
        )
    tools = resolved.get("environment", {}).get("tools", {})
    te = tools.get("train_env", {})
    te_path = te.get("path") if isinstance(te, dict) else (te or "<train_env>")
    cli = f"{te_path}/bin/dataos-train"
    return (
        "DETERMINISTIC via dataos-train (installed in train_env). Your task prompt carries a "
        "ready TRAIN SPEC JSON — runtime-owned fields are code-injected, do NOT edit them; only "
        "`method` came from the plan. Flow: "
        "1) write it to {ctx.run_root}/trainspec.json on the exec host; "
        f"2) on the LOGIN node: `{cli} validate {{ctx.run_root}}/trainspec.json` (fail-fast, no "
        f"GPU) then `{cli} build {{ctx.run_root}}/trainspec.json` — build renders AND writes "
        "run_train.sh + train config into run_root; "
        "3) submit a thin sbatch wrapper (activate op with ctx.env_path=train_env, then `bash "
        "{ctx.run_root}/run_train.sh`) via the submit op; poll to terminal; "
        f"4) `{cli} collect --run-dir {{ctx.run_root}}` → result.json; merge its "
        "report_fragment into report.json. "
        "FALLBACK only if dataos-train is broken/unavailable: hand-write the LF config per the "
        "legacy recipe in your system prompt (record the fallback in report.json)."
    )


def build_stages(resolved: dict) -> list[Stage]:
    """All three stages derive from the task contract. Owners: DATA=ours (agent's lever);
    TRAIN=frozen if the task ships a train/ dir (resolved.task.train set) else ours; EVAL is
    always the uniform model→score runner (provided)."""
    scoring = resolved.get("task", {}).get("scoring", {})
    result = scoring.get("result", {})
    train_frozen = bool(resolved.get("task", {}).get("train"))
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
            role="(data, recipe) -> a complete HF model",
            input="the curated data from DATA + " + (
                "the task's FROZEN recipe" if train_frozen else "the plan's SFT hyperparameters"
            ),
            output=(
                "the task's frozen-training model output" if train_frozen else
                "{ctx.run_root}/final_model — COMPLETE HF model dir (lora: adapter then export-merge)"
            ),
            invoke=_train_invoke(resolved),
            where="a Slurm GPU job" if not train_frozen else "the task's frozen training harness",
            owner=OWNER_PROVIDED if train_frozen else OWNER_OURS,
        ),
        Stage(
            name="eval",
            role="model -> comparable score (uniform across tasks)",
            input="{ctx.model_path} (the model from TRAIN; the parent model itself for a BASELINE run)",
            output=(
                "{ctx.eval_dir}/score.json — normalized `score` in range "
                f"{result.get('range', [0, 100])} (+ per-metric breakdown)"
            ),
            invoke=_eval_invoke(resolved),
            where="GPU job (vLLM) via the common eval runner",
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
