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
        "run": {"deploy_root": "/deploy", "environment": "PTB_H100_slurm", "task": "posttrainbench",
                "resource": {"qos": "gpu_priv", "node": "gpu04", "gpus": 1, "cpus": 8}},
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
    assert "`/envs/Llama/bin/dataos-eval status <handle>`" in invoke
    assert "collect --repo /deploy --environment PTB_H100_slurm --task-config posttrainbench" in invoke
    assert '{"score": ...} (range [0, 100])' in invoke  # the task's result field + range reach it
    # ...but what it WRITES is the directory, not the number: the runtime stamps the numbers from
    # that dir's summary.json, so a stage prompt telling it to record the score would contradict
    # the contract block in its own system prompt — which is what this line exists to catch.
    assert "write `stages.eval.summary_dir`" in invoke
    assert "NEVER hand-write sbatch for eval" in invoke
    assert "/deploy/configs/tasks/posttrainbench/contracts/gsm8k.md" in invoke


# --- the reserved trainer slot -----------------------------------------------

def test_train_invoke_is_the_registered_trainer():
    invoke = _stage(_resolved(), name="train").invoke
    assert invoke.startswith("DETERMINISTIC via dataos-train")
    # the launcher lifecycle, NOT a hand-written sbatch wrapper; cli from provides
    assert "`/envs/train-cli/bin/dataos-train validate {ctx.run_root}/trainspec.json`" in invoke
    # placement comes from the run config, the same four flags dataos-eval takes — never from
    # a `--launcher` on the command line, and never from a cluster block on the spec
    assert "submit {ctx.run_root}/trainspec.json --repo /deploy --environment PTB_H100_slurm" in invoke
    assert "--qos gpu_priv" in invoke
    assert "--launcher" not in invoke
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
    assert "submit {ctx.run_root}/trainspec.json --repo /deploy" in blocks
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


def test_train_spec_scopes_the_menu_by_declared_model_family():
    """An env that lists `model_families` claims ONLY those families' domains. Nothing here
    knows what a family is — it expands the environment's declaration, and dataos-train
    matches it against the model's own model_type. An undeclared family then fails at build
    instead of at model load, which is where the 2026-08-11 SmolLM3 batch lost 146 jobs."""
    from dataos.agent.train_spec import build_train_spec

    res = _resolved()
    res["environment"]["tools"]["sft_env"]["model_families"] = ["qwen3", "smollm3"]
    spec = build_train_spec(
        res, name="e0", plan_train={}, parent_model="/m", run_root="/rb/e0", time_h=2.0,
    )
    assert spec["env"]["backends"] == {
        "llamafactory@qwen3": "/envs/Llama",
        "llamafactory@smollm3": "/envs/Llama",
        "verl": "/envs/verl",  # declares no families -> unscoped, as before
    }


def test_train_spec_clamps_time_to_the_partition_max():
    from dataos.agent.train_spec import build_train_spec

    res = _resolved()
    res["environment"]["slurm"] = {"partition": "gpu", "qos": "q", "max_time": "3-00:00:00"}
    spec = build_train_spec(
        res, name="e0", plan_train={}, parent_model="/m", run_root="/rb/e0", time_h=200.0,
    )
    assert spec["resources"]["time_h"] == 72.0


def test_templates_render_with_declared_params():
    data = prompt.render("experimenter/turn/stage_data", catalog="CATALOG",
                         data_bin="/envs/sft/bin/dataos-data", data_env="/envs/sft",
                         provides="dataos-operator, dataos-data", placement="--qos gpu_priv")
    assert "o.guide()" in data
    # the venv and its contents are the ENVIRONMENT's to state; this file used to name one
    # deployment's directory in a template rendered for every deployment
    assert "runs in `/envs/sft`, which ships: dataos-operator, dataos-data" in data
    assert "/envs/sft/bin/dataos-data submit" in data and "--qos gpu_priv" in data
    frozen = prompt.render(
        "experimenter/turn/stage_train", frozen_command="cmd", frozen_produces="prod",
        cli_path="", placement="",
    )
    assert "`cmd` → model at prod" in frozen
    ours = prompt.render(
        "experimenter/turn/stage_train", frozen_command="", frozen_produces="",
        cli_path="/te/bin/dataos-train", placement="--repo /deploy --environment E --qos gpu_priv",
    )
    assert ours.startswith("DETERMINISTIC via dataos-train")
    minimal = prompt.render(
        "experimenter/turn/stage_eval", command="cmd", eval_bin="/env/bin/dataos-eval", repo="/deploy",
        environment="env", task_config="task", result_field="score", result_file="<result>",
        result_range=[0, 100], format_contract_loc="",
    )
    assert minimal.startswith("run the eval through the `dataos-eval` CLI")
    assert "1. submit: `cmd`" in minimal
    # status takes only the handle: polling needs no deployment config, and the parser it used
    # to be given did not even declare --repo.
    assert "`/env/bin/dataos-eval status <handle>`" in minimal
    # the label no longer selects the template (the runner reads the model's config.json), and the
    # prompt must say so: the old rule told the experimenter its label WAS the protocol
    assert "the eval label is yours to name freely" in minimal
    assert "stages.train.template" in minimal
