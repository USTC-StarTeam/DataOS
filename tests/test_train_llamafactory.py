from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

from dataos.core.artifact import DatasetArtifact
from dataos.core.sandbox import Sandbox
from dataos.core.task import TaskTemplate
from dataos.train import run_train_stage


def _task(train_config: dict) -> TaskTemplate:
    return TaskTemplate.from_dict(
        {
            "task_id": "task-train",
            "name": "train",
            "goal": "run training",
            "input_description": "hf dataset directory",
            "output_description": "artifacts",
            "constraints": {"dataset_format": "hf_datasets"},
            "budget": {"max_experiments": 1},
            "max_rounds": 1,
            "pipeline_hint": ["dedup", "sample"],
            "train_config": train_config,
            "eval_config": {"provider": "stub"},
        }
    )


def _dataset_artifact(sandbox: Sandbox, dataset_obj: Dataset | DatasetDict) -> DatasetArtifact:
    artifact = DatasetArtifact(
        artifact_id="dataset-1",
        name="dataset",
        path=sandbox.artifact_dir("dataset", "dataset-1"),
        producer="test",
    )
    artifact.save(dataset_obj)
    return artifact


def test_train_llamafactory_exports_dataset_and_records_metadata(tmp_path, monkeypatch):
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()
    dataset_artifact = _dataset_artifact(
        sandbox,
        DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "instruction": ["Say hi"],
                        "input": [""],
                        "response": ["Hi there"],
                        "system": ["You are helpful"],
                        "history": [[["Previous", "Answer"]]],
                    }
                ),
                "validation": Dataset.from_dict(
                    {
                        "instruction": ["Say bye"],
                        "input": [""],
                        "response": ["Bye"],
                        "system": ["You are helpful"],
                        "history": [[["Previous", "Answer"]]],
                    }
                ),
            }
        ),
    )
    task = _task(
        {
            "mode": "llamafactory",
            "model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
            "template": "qwen",
            "finetuning_type": "lora",
            "extra_args": {
                "cutoff_len": 128,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-4,
                "logging_steps": 1,
                "max_samples": 4,
                "num_train_epochs": 1.0,
                "per_device_train_batch_size": 1,
                "save_steps": 5,
            },
        }
    )

    monkeypatch.setattr(
        "dataos.train.llamafactory.shutil.which",
        lambda _: "/usr/local/bin/llamafactory-cli",
    )
    monkeypatch.setattr(
        "dataos.train.llamafactory.importlib.util.find_spec",
        lambda _: object(),
    )

    def _fake_run(command, cwd, stdout, stderr, check):
        del cwd, stderr, check
        stdout.write("training\n")
        config = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        (output_dir / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("dataos.train.llamafactory.subprocess.run", _fake_run)

    artifact = run_train_stage(
        [dataset_artifact],
        sandbox=sandbox,
        task=task,
    )

    assert artifact.meta["mode"] == "llamafactory"
    assert artifact.meta["base_model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert artifact.meta["template"] == "qwen"
    assert artifact.meta["finetuning_type"] == "lora"
    assert artifact.meta["train_split"] == "train"
    assert artifact.meta["eval_split"] == "validation"
    assert artifact.meta["selected_rows"] == 1
    assert Path(artifact.meta["log_path"]).exists()
    assert (artifact.path / "adapter_model.safetensors").exists()

    manifest = json.loads((artifact.path / "training_manifest.json").read_text(encoding="utf-8"))
    assert manifest["command"] == ["/usr/local/bin/llamafactory-cli", "train", manifest["config_path"]]
    assert manifest["train_config"]["dataset"].endswith("_train")
    assert manifest["train_config"]["eval_dataset"].endswith("_validation")
    assert manifest["train_config"]["lora_target"] == "all"

    dataset_info = json.loads(
        Path(manifest["exported_data"]["dataset_info_path"]).read_text(encoding="utf-8")
    )
    train_name = manifest["exported_data"]["train_dataset_name"]
    assert dataset_info[train_name]["columns"] == {
        "prompt": "instruction",
        "query": "input",
        "response": "output",
        "system": "system",
        "history": "history",
    }

    exported_train = json.loads(Path(manifest["exported_data"]["train_path"]).read_text(encoding="utf-8"))
    assert exported_train == [
        {
            "instruction": "Say hi",
            "input": "",
            "output": "Hi there",
            "system": "You are helpful",
            "history": [["Previous", "Answer"]],
        }
    ]
    assert "COMMAND: /usr/local/bin/llamafactory-cli train" in Path(artifact.meta["log_path"]).read_text(
        encoding="utf-8"
    )


def test_train_llamafactory_requires_instruction_and_response_columns(tmp_path):
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()
    dataset_artifact = _dataset_artifact(
        sandbox,
        Dataset.from_dict(
            {
                "instruction": ["Say hi"],
                "output": ["Hi there"],
            }
        ),
    )
    task = _task(
        {
            "mode": "llamafactory",
            "model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
            "template": "qwen",
        }
    )

    with pytest.raises(ValueError, match="instruction' and 'response"):
        run_train_stage([dataset_artifact], sandbox=sandbox, task=task)


def test_train_llamafactory_rejects_unknown_step_param(tmp_path):
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()
    dataset_artifact = _dataset_artifact(
        sandbox,
        Dataset.from_dict(
            {
                "instruction": ["Say hi"],
                "response": ["Hi there"],
            }
        ),
    )
    task = _task(
        {
            "mode": "llamafactory",
            "model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
            "template": "qwen",
        }
    )

    with pytest.raises(ValueError, match="Unsupported train_llamafactory step params: unknown_flag"):
        run_train_stage(
            [dataset_artifact],
            sandbox=sandbox,
            task=task,
            unknown_flag=True,
        )


# ── Phase 2 ⑧ — run_task with LESS step (stub mode) ──────────────────────────


class _LessPlannerStub:
    """Stub planning + reviewing LLM that wires a LESS step into run_task.

    The first ``generate_json`` call is the plan request — we read the
    initial-artifact id out of the user_prompt JSON and use it as
    ``target_artifact_id`` so the LESS step can find it back at execution
    time. Subsequent calls return a stub review payload.
    """

    def __init__(self, pool_path: Path, top_k: int = 5):
        self.pool_path = pool_path
        self.top_k = top_k

    def generate_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> dict:
        del temperature
        if "review" in system_prompt.lower():
            return {
                "summary": "stub review for less smoke",
                "issues": [],
                "next_hypothesis": "n/a",
                "recommended_changes": [],
            }
        payload = json.loads(user_prompt)
        initial_id = payload["initial_artifacts"][0]["artifact_id"]
        return {
            "steps": [
                {
                    "operator_name": "import_dataset",
                    "params": {
                        "source": "hf_local",
                        "path": str(self.pool_path),
                        "name": "alpaca-pool-stub",
                    },
                },
                {
                    "operator_name": "less",
                    "params": {
                        "target_artifact_id": initial_id,
                        "base_model": "stub-model",
                        "top_k": self.top_k,
                        "mode": "stub",
                        "seed": 7,
                    },
                },
            ],
            "reasoning_summary": "import alpaca pool then run LESS against the initial ifeval target",
            "expected_gain": "select top-k high-influence rows for downstream training",
        }


def _hf_save(path: Path, *, n: int, prefix: str) -> Path:
    Dataset.from_dict(
        {
            "instruction": [f"{prefix}-instruction-{i}" for i in range(n)],
            "response": [f"{prefix}-response-{i}" for i in range(n)],
        }
    ).save_to_disk(str(path))
    return path


def test_run_task_with_less_step_stub(tmp_path):
    from dataos.runtime.runner import run_task

    target_dir = _hf_save(tmp_path / "target", n=4, prefix="target")
    pool_dir = _hf_save(tmp_path / "pool", n=20, prefix="pool")

    workspace = tmp_path / "workspace"
    (workspace / "sandboxes").mkdir(parents=True, exist_ok=True)

    template = {
        "task_id": "sft-less-smoke",
        "name": "less smoke",
        "goal": "select via LESS then stub-train + stub-eval",
        "input_description": "ifeval target subset; alpaca pool imported by plan",
        "output_description": "stub model + stub eval",
        "constraints": {"dataset_format": "hf_datasets"},
        "budget": {"max_experiments": 1},
        "max_rounds": 1,
        "pipeline_hint": ["import_dataset", "less"],
        "train_config": {"mode": "stub"},
        "eval_config": {"provider": "stub", "provider_config": {"score": 0.5}},
    }

    llm = _LessPlannerStub(pool_path=pool_dir, top_k=5)
    result = run_task(
        task_template=template,
        dataset_path=target_dir,
        workspace_dir=workspace,
        llm_client=llm,
    )

    operator_steps = [
        step for step in result.experiment.steps if step.get("step_type") == "operator"
    ]
    step_names = [step["operator_name"] for step in operator_steps]
    assert step_names == ["import_dataset", "less"], step_names

    less_step = next(step for step in operator_steps if step["operator_name"] == "less")
    assert less_step["status"] == "completed"
    assert len(less_step["output_artifact_ids"]) == 1

    less_output_id = less_step["output_artifact_ids"][0]
    less_artifact = next(a for a in result.experiment.artifacts if a.artifact_id == less_output_id)
    assert less_artifact.producer == "less"
    assert less_artifact.meta["less_mode"] == "stub"
    assert less_artifact.meta["less_selected_count"] == 5
    assert less_artifact.meta["less_pool_count"] == 20

    manifest = json.loads((less_artifact.path / "selection_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["selected_indices"]) == 5
    assert manifest["target_artifact_id"] == result.initial_artifact.artifact_id
    assert manifest["pool_artifact_id"] != result.initial_artifact.artifact_id
