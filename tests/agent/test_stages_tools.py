"""Stage tool registry: invokes are assembled from registered tools, and the three
prompt templates render the same prose the hardcoded blocks used to (byte-identical)."""

import dataos_operator

from dataos import prompt
from dataos.agent import stages as _stages
from dataos.agent import tools as _tools


def _resolved(*, frozen: bool = False) -> dict:
    task = {
        "scoring": {
            "command": "bash {ctx.eval_dir}/run_eval.sh {ctx.model_path}",
            "result": {"file": "{ctx.eval_dir}/summary.json", "field": "score", "range": [0, 100]},
            "judge_command": "python judge.py {ctx.eval_dir}",
            "format_contract": "contracts/gsm8k.md",
        },
        "assets_root": "/assets",
    }
    if frozen:
        task["train"] = {"command": "bash train.sh", "produces": "out/model"}
    return {
        "task": task,
        "environment": {"tools": {"train_env": {"path": "/envs/Llama"}}},
    }


def _stage(resolved, registry=None, name="data"):
    return next(s for s in _stages.build_stages(resolved, registry) if s.name == name)


# --- default registry: tools register from their sources --------------------

def test_default_registry_slots():
    reg = _tools.build_registry(_resolved())
    assert [op.name for op in reg.operators] == ["dataos_operator"]
    assert reg.evaluator is not None and reg.evaluator.name == "frozen_evaluator"
    assert reg.trainer is None  # the reserved slot stays empty until dataos-train lands


def test_data_invoke_carries_the_installed_catalog():
    invoke = _stage(_resolved(), name="data").invoke
    assert "dataos_operator recipe (discover:" in invoke
    # the real installed toolbox catalog is injected (matches dataos_operator.catalog())
    assert dataos_operator.catalog().splitlines()[1].strip() in invoke


def test_eval_invoke_from_the_resolved_contract():
    invoke = _stage(_resolved(), name="eval").invoke
    assert "run the task's evaluator: `bash {ctx.eval_dir}/run_eval.sh {ctx.model_path}`" in invoke
    assert "then read `score` from {ctx.eval_dir}/summary.json (range [0, 100])" in invoke
    assert "on the LOGIN node run: `python judge.py {ctx.eval_dir}`" in invoke
    assert "/assets/contracts/gsm8k.md" in invoke


# --- the reserved trainer slot -----------------------------------------------

def test_train_falls_back_to_legacy_block_while_slot_empty():
    invoke = _stage(_resolved(), name="train").invoke
    assert invoke.startswith("DETERMINISTIC via dataos-train")
    assert "`/envs/Llama/bin/dataos-train validate {ctx.run_root}/trainspec.json`" in invoke


def test_train_frozen_branch_from_legacy_block():
    invoke = _stage(_resolved(frozen=True), name="train").invoke
    assert invoke.startswith("FROZEN recipe — you do NOT choose hyperparameters")
    assert "`bash train.sh` → model at out/model" in invoke


def test_registering_a_trainer_replaces_the_legacy_block():
    class DummyTrainer:
        name = "dummy_train"
        owner = _tools.OWNER_OURS

        def invoke(self, resolved):
            return "dummy trainer invoke"

    reg = _tools.build_registry(_resolved())
    reg.register_trainer(DummyTrainer())
    invoke = _stage(_resolved(), reg, name="train").invoke
    assert invoke == "dummy trainer invoke"
    assert "DETERMINISTIC" not in invoke  # the legacy block disappeared automatically


def test_multiple_operators_concatenate_in_order():
    class ExtraOp:
        name = "extra"
        owner = _tools.OWNER_OURS

        def invoke(self, resolved):
            return "extra operator invoke"

    reg = _tools.build_registry(_resolved())
    reg.register_operator(ExtraOp())
    invoke = _stage(_resolved(), reg, name="data").invoke
    assert invoke.startswith("dataos_operator recipe")
    assert invoke.endswith("\n\nextra operator invoke")  # appended after the toolbox, in order


# --- prose parity with the pre-registry hardcoded blocks ----------------------

def test_rendered_blocks_parity():
    """The assembled stage blocks keep the old prose byte-for-byte (modulo the data
    block's appended catalog). Guards the YAML move against accidental rewording."""
    blocks = _stages.render_stage_blocks(_stages.build_stages(_resolved()))
    assert "- invoke: dataos_operator recipe (discover: `python -c \"import dataos_operator as o;" in blocks
    assert "custom ops via DATAOS_OPERATOR_PLUGINS — full mechanics in your system prompt" in blocks
    assert "FALLBACK only if dataos-train is broken/unavailable: hand-write the LF config" in blocks
    assert "want a high score? read how the scorer parses output first: /assets/contracts/gsm8k.md" in blocks


# --- the three templates render standalone (StrictUndefined — no empty holes) --

def test_templates_render_with_declared_params():
    assert "o.guide()" in prompt.render("kb/stage_data", catalog="CATALOG")
    frozen = prompt.render(
        "kb/stage_train_legacy", frozen_command="cmd", frozen_produces="prod", cli_path=""
    )
    assert "`cmd` → model at prod" in frozen
    ours = prompt.render(
        "kb/stage_train_legacy", frozen_command="", frozen_produces="", cli_path="/te/bin/dataos-train"
    )
    assert ours.startswith("DETERMINISTIC via dataos-train")
    minimal = prompt.render(
        "kb/stage_eval", command="cmd", result_field="score", result_file="<result>",
        result_range=[0, 100], judge_command="", format_contract_loc="",
    )
    assert minimal == "run the task's evaluator: `cmd`\n  then read `score` from <result> (range [0, 100]) → report.json."
