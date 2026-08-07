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
            "format_contract": "contracts/gsm8k.md",
        },
        "assets_root": "/assets",
    }
    if frozen:
        task["train"] = {"command": "bash train.sh", "produces": "out/model"}
    return {
        "task": task,
        "run": {"deploy_root": "/deploy", "environment": "PTB_H100_slurm", "task": "posttrainbench"},
        "environment": {
            "tools": {
                "sft_env": {"path": "/envs/Llama",
                            "provides": ["llamafactory", "torch", "dataos-eval"]},
                "rl_env": {"path": "/envs/verl", "provides": ["verl", "torch"]},
                "train_cli_env": {"path": "/envs/train-cli", "provides": ["dataos-train"]},
            },
        },
    }


def _stage(resolved, registry=None, name="data"):
    return next(s for s in _stages.build_stages(resolved, registry) if s.name == name)


# --- default registry: tools register from their sources --------------------

def test_default_registry_slots():
    reg = _tools.build_registry(_resolved())
    assert [op.name for op in reg.operators] == ["dataos_operator"]
    assert reg.trainer is not None and reg.trainer.name == "dataos_train"
    assert reg.evaluator is not None and reg.evaluator.name == "frozen_evaluator"


def test_data_invoke_carries_the_installed_catalog():
    invoke = _stage(_resolved(), name="data").invoke
    assert "dataos_operator recipe (discover:" in invoke
    # the real installed toolbox catalog is injected (matches dataos_operator.catalog())
    assert dataos_operator.catalog().splitlines()[1].strip() in invoke


def test_eval_invoke_from_the_resolved_contract():
    invoke = _stage(_resolved(), name="eval").invoke
    assert "1. submit: `bash {ctx.eval_dir}/run_eval.sh {ctx.model_path}`" in invoke
    assert "`/envs/Llama/bin/dataos-eval status --repo /deploy <handle>`" in invoke
    assert "collect --repo /deploy --environment PTB_H100_slurm --task-config posttrainbench" in invoke
    assert "`score` (range [0, 100])" in invoke
    assert "NEVER hand-write sbatch for eval" in invoke
    assert "/deploy/configs/tasks/posttrainbench/contracts/gsm8k.md" in invoke


# --- the reserved trainer slot -----------------------------------------------

def test_train_invoke_is_the_registered_trainer():
    invoke = _stage(_resolved(), name="train").invoke
    assert invoke.startswith("DETERMINISTIC via dataos-train")
    # the launcher lifecycle, NOT a hand-written sbatch wrapper; cli from provides
    assert "`/envs/train-cli/bin/dataos-train validate {ctx.run_root}/trainspec.json`" in invoke
    assert "submit {ctx.run_root}/spec.json --launcher slurm" in invoke
    assert "handle.json" in invoke


def test_train_frozen_branch_wins_over_the_trainer():
    invoke = _stage(_resolved(frozen=True), name="train").invoke
    assert invoke.startswith("FROZEN recipe — you do NOT choose hyperparameters")
    assert "`bash train.sh` → model at out/model" in invoke


def test_registering_a_trainer_replaces_the_default():
    class DummyTrainer:
        name = "dummy_train"
        owner = _tools.OWNER_OURS

        def invoke(self, resolved):
            return "dummy trainer invoke"

    reg = _tools.build_registry(_resolved())
    reg.register_trainer(DummyTrainer())
    invoke = _stage(_resolved(), reg, name="train").invoke
    assert invoke == "dummy trainer invoke"
    assert "DETERMINISTIC" not in invoke  # the default Trainer is replaced


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
    assert "submit {ctx.run_root}/spec.json --launcher slurm" in blocks
    assert "want a high score? read how the scorer parses output first: /deploy/configs/tasks/posttrainbench/contracts/gsm8k.md" in blocks


# --- the three templates render standalone (StrictUndefined — no empty holes) --

def test_train_spec_injects_the_full_env_menu():
    from dataos.agent.train_spec import build_train_spec

    spec = build_train_spec(
        _resolved(), name="e0", plan_train={"finetuning_type": "lora"},
        parent_model="/m", run_root="/rb/e0", time_h=2.0,
    )
    assert spec["backend"] == "llamafactory"
    # the FULL menu is injected; dataos-train's env_for() resolves method→env at build
    assert spec["env"]["backends"] == {"llamafactory": "/envs/Llama", "verl": "/envs/verl"}


def test_templates_render_with_declared_params():
    assert "o.guide()" in prompt.render("kb/stage_data", catalog="CATALOG")
    frozen = prompt.render(
        "kb/stage_train", frozen_command="cmd", frozen_produces="prod", cli_path=""
    )
    assert "`cmd` → model at prod" in frozen
    ours = prompt.render(
        "kb/stage_train", frozen_command="", frozen_produces="", cli_path="/te/bin/dataos-train"
    )
    assert ours.startswith("DETERMINISTIC via dataos-train")
    minimal = prompt.render(
        "kb/stage_eval", command="cmd", eval_bin="/env/bin/dataos-eval", repo="/deploy",
        environment="env", task_config="task", result_field="score", result_file="<result>",
        result_range=[0, 100], format_contract_loc="",
    )
    assert minimal.startswith("run the eval through the `dataos-eval` CLI")
    assert "1. submit: `cmd`" in minimal
    assert "`/env/bin/dataos-eval status --repo /deploy <handle>`" in minimal
    assert "model family name" in minimal  # the label->template selection rule
