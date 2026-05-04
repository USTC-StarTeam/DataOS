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
