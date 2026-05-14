from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

from dataos.core.artifact import Artifact, DatasetArtifact, ModelArtifact, make_artifact_id
from dataos.core.sandbox import Sandbox

RESERVED_CONFIG_KEYS = {
    "stage",
    "do_train",
    "dataset",
    "eval_dataset",
    "dataset_dir",
    "output_dir",
    "overwrite_output_dir",
}
SUPPORTED_FINETUNING_TYPES = {"lora", "full", "freeze"}
ALLOWED_EXTRA_ARGS = {
    "bf16",
    "cutoff_len",
    "fp16",
    "gradient_accumulation_steps",
    "learning_rate",
    "logging_steps",
    "lr_scheduler_type",
    "max_samples",
    "num_train_epochs",
    "per_device_train_batch_size",
    "preprocessing_num_workers",
    "save_steps",
    "val_size",
    "warmup_ratio",
}
ALLOWED_STEP_OVERRIDE_KEYS = ALLOWED_EXTRA_ARGS | {"finetuning_type", "lora_target", "mode"}
EVAL_SPLIT_CANDIDATES = ("validation", "eval", "dev")


def run_train_stage(inputs: list[Artifact], sandbox: Sandbox, **kwargs) -> ModelArtifact:
    task = kwargs.get("task")
    dataset_artifact = _find_latest_dataset(inputs)
    dataset_obj = dataset_artifact.load()
    effective_config = _resolve_train_config(getattr(task, "train_config", {}), kwargs)

    if effective_config["mode"] == "stub":
        return _run_stub_training(
            dataset_artifact=dataset_artifact,
            dataset_obj=dataset_obj,
            sandbox=sandbox,
            task=task,
            mode=effective_config["mode"],
        )

    artifact_id = make_artifact_id("model")
    model_dir = sandbox.artifact_dir("model", artifact_id)
    train_dataset, train_split, eval_dataset, eval_split = _select_dataset_splits(dataset_obj)
    exported = _export_datasets_for_llamafactory(
        dataset_artifact=dataset_artifact,
        train_dataset=train_dataset,
        train_split=train_split,
        eval_dataset=eval_dataset,
        eval_split=eval_split,
        sandbox=sandbox,
    )
    training_payload = _build_training_payload(
        effective_config=effective_config,
        exported=exported,
        model_dir=model_dir,
    )
    config_path = sandbox.code_dir / "llamafactory" / artifact_id / "train.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(training_payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    command = _run_llamafactory_train(config_path=config_path, sandbox=sandbox)
    _validate_training_outputs(model_dir)

    training_config = {
        "mode": effective_config["mode"],
        "dataset_artifact_id": dataset_artifact.artifact_id,
        "dataset_format": exported["dataset_format"],
        "train_split": train_split,
        "selected_rows": train_dataset.num_rows,
        "eval_split": eval_split,
        "eval_rows": eval_dataset.num_rows if eval_dataset is not None else 0,
        "task_id": getattr(task, "task_id", None),
        "base_model": effective_config["model_name_or_path"],
        "template": effective_config["template"],
        "finetuning_type": effective_config["finetuning_type"],
        "lora_target": effective_config.get("lora_target"),
        "extra_args": effective_config["extra_args"],
    }
    (model_dir / "training_config.json").write_text(
        json.dumps(training_config, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest = {
        "command": command,
        "config_path": str(config_path),
        "dataset_artifact_id": dataset_artifact.artifact_id,
        "exported_data": exported,
        "train_config": training_payload,
        "log_path": str(sandbox.logs_dir / "train_llamafactory.log"),
    }
    (model_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )

    model_artifact = ModelArtifact(
        artifact_id=artifact_id,
        name=f"model-{dataset_artifact.artifact_id}",
        path=model_dir,
        parent_ids=[dataset_artifact.artifact_id],
        producer="train_llamafactory",
        meta={
            "mode": effective_config["mode"],
            "source_dataset_id": dataset_artifact.artifact_id,
            "dataset_format": exported["dataset_format"],
            "base_model": effective_config["model_name_or_path"],
            "template": effective_config["template"],
            "finetuning_type": effective_config["finetuning_type"],
            "lora_target": effective_config.get("lora_target"),
            "train_split": train_split,
            "selected_rows": train_dataset.num_rows,
            "eval_split": eval_split,
            "eval_rows": eval_dataset.num_rows if eval_dataset is not None else 0,
            "task_id": getattr(task, "task_id", None),
            "config_path": str(config_path),
            "log_path": str(sandbox.logs_dir / "train_llamafactory.log"),
        },
    )
    sandbox.log(
        "train_llamafactory_summary",
        (
            f"dataset={dataset_artifact.artifact_id}\n"
            f"model={model_artifact.artifact_id}\n"
            f"mode={effective_config['mode']}\n"
            f"train_split={train_split}\n"
            f"eval_split={eval_split or ''}\n"
            f"config={config_path}\n"
        ),
    )
    return model_artifact


def _find_latest_dataset(inputs: list[Artifact]) -> DatasetArtifact:
    for artifact in reversed(inputs):
        if isinstance(artifact, DatasetArtifact):
            return artifact
    raise ValueError("train_llamafactory requires at least one DatasetArtifact input")


def _run_stub_training(
    dataset_artifact: DatasetArtifact,
    dataset_obj: Dataset | DatasetDict,
    sandbox: Sandbox,
    task: Any,
    mode: str,
) -> ModelArtifact:
    train_dataset, train_split, _, _ = _select_dataset_splits(dataset_obj)
    artifact_id = make_artifact_id("model")
    model_dir = sandbox.artifact_dir("model", artifact_id)
    config = {
        "mode": mode,
        "dataset_artifact_id": dataset_artifact.artifact_id,
        "train_split": train_split,
        "selected_rows": train_dataset.num_rows,
        "task_id": getattr(task, "task_id", None),
    }
    (model_dir / "training_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    (model_dir / "weights.stub").write_text("stub model binary placeholder\n", encoding="utf-8")
    model_artifact = ModelArtifact(
        artifact_id=artifact_id,
        name=f"model-{dataset_artifact.artifact_id}",
        path=model_dir,
        parent_ids=[dataset_artifact.artifact_id],
        producer="train_llamafactory",
        meta={
            "mode": mode,
            "source_dataset_id": dataset_artifact.artifact_id,
            "train_split": train_split,
            "selected_rows": train_dataset.num_rows,
            "task_id": getattr(task, "task_id", None),
        },
    )
    sandbox.log(
        "train_llamafactory",
        f"dataset={dataset_artifact.artifact_id}\nmodel={model_artifact.artifact_id}\nmode={mode}\n",
    )
    return model_artifact


def _resolve_train_config(task_train_config: dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    step_params = {key: value for key, value in kwargs.items() if key not in {"task", "experiment"}}
    mode = str(step_params.get("mode", task_train_config.get("mode", "stub")))
    if mode == "stub":
        return {"mode": "stub"}
    if mode != "llamafactory":
        raise ValueError(f"Unsupported train_llamafactory mode: {mode}")

    if not isinstance(task_train_config, dict):
        raise ValueError("task.train_config must be a dict")

    base_config = dict(task_train_config)
    base_config.pop("mode", None)
    extra_args = base_config.pop("extra_args", {})
    if not isinstance(extra_args, dict):
        raise ValueError("task.train_config.extra_args must be a dict")

    unknown_task_keys = set(base_config) - {"model_name_or_path", "template", "finetuning_type", "lora_target"}
    if unknown_task_keys:
        keys = ", ".join(sorted(unknown_task_keys))
        raise ValueError(f"Unsupported task.train_config keys for llamafactory mode: {keys}")

    unknown_step_keys = set(step_params) - ALLOWED_STEP_OVERRIDE_KEYS
    if unknown_step_keys:
        keys = ", ".join(sorted(unknown_step_keys))
        raise ValueError(f"Unsupported train_llamafactory step params: {keys}")

    effective = {
        "mode": mode,
        "model_name_or_path": step_params.get("model_name_or_path", base_config.get("model_name_or_path")),
        "template": step_params.get("template", base_config.get("template")),
        "finetuning_type": str(step_params.get("finetuning_type", base_config.get("finetuning_type", "lora"))),
        "extra_args": dict(extra_args),
    }
    if not effective["model_name_or_path"]:
        raise ValueError("train_config.model_name_or_path is required for llamafactory mode")
    if not effective["template"]:
        raise ValueError("train_config.template is required for llamafactory mode")
    if effective["finetuning_type"] not in SUPPORTED_FINETUNING_TYPES:
        allowed = ", ".join(sorted(SUPPORTED_FINETUNING_TYPES))
        raise ValueError(f"train_config.finetuning_type must be one of: {allowed}")

    if "lora_target" in base_config:
        effective["lora_target"] = base_config["lora_target"]
    if "lora_target" in step_params:
        effective["lora_target"] = step_params["lora_target"]
    if effective["finetuning_type"] == "lora":
        effective.setdefault("lora_target", "all")
    elif "lora_target" in effective:
        raise ValueError("lora_target can only be set when finetuning_type='lora'")

    for key, value in step_params.items():
        if key in {"mode", "finetuning_type", "lora_target"}:
            continue
        effective["extra_args"][key] = value

    unknown_extra_args = set(effective["extra_args"]) - ALLOWED_EXTRA_ARGS
    if unknown_extra_args:
        keys = ", ".join(sorted(unknown_extra_args))
        raise ValueError(f"Unsupported train_config.extra_args keys: {keys}")

    reserved_keys = set(effective["extra_args"]) & RESERVED_CONFIG_KEYS
    if reserved_keys:
        keys = ", ".join(sorted(reserved_keys))
        raise ValueError(f"Reserved LLaMA-Factory keys cannot be overridden: {keys}")

    return effective


def _select_dataset_splits(
    dataset_obj: Dataset | DatasetDict,
) -> tuple[Dataset, str, Dataset | None, str | None]:
    if isinstance(dataset_obj, DatasetDict):
        train_split = "train" if "train" in dataset_obj else next(iter(dataset_obj.keys()))
        eval_split = next(
            (candidate for candidate in EVAL_SPLIT_CANDIDATES if candidate in dataset_obj and candidate != train_split),
            None,
        )
        eval_dataset = dataset_obj[eval_split] if eval_split else None
        return dataset_obj[train_split], train_split, eval_dataset, eval_split
    return dataset_obj, "train", None, None


def _export_datasets_for_llamafactory(
    dataset_artifact: DatasetArtifact,
    train_dataset: Dataset,
    train_split: str,
    eval_dataset: Dataset | None,
    eval_split: str | None,
    sandbox: Sandbox,
) -> dict[str, Any]:
    export_root = sandbox.tmp_dir / "llamafactory" / dataset_artifact.artifact_id / "data"
    export_root.mkdir(parents=True, exist_ok=True)
    dataset_prefix = dataset_artifact.artifact_id.replace("-", "_")
    dataset_format = _detect_training_dataset_format(dataset_artifact, train_dataset)

    train_name = f"{dataset_prefix}_{train_split}"
    train_file = export_root / f"{train_name}.json"
    train_entry = _write_dataset_for_llamafactory(train_dataset, train_file, dataset_format)
    dataset_info: dict[str, Any] = {train_name: train_entry}
    exported: dict[str, Any] = {
        "dataset_dir": str(export_root),
        "dataset_format": dataset_format,
        "train_dataset_name": train_name,
        "train_path": str(train_file),
        "train_split": train_split,
    }

    if eval_dataset is not None and eval_split is not None:
        eval_name = f"{dataset_prefix}_{eval_split}"
        eval_file = export_root / f"{eval_name}.json"
        dataset_info[eval_name] = _write_dataset_for_llamafactory(
            eval_dataset,
            eval_file,
            dataset_format,
        )
        exported["eval_dataset_name"] = eval_name
        exported["eval_path"] = str(eval_file)
        exported["eval_split"] = eval_split

    dataset_info_path = export_root / "dataset_info.json"
    dataset_info_path.write_text(
        json.dumps(dataset_info, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    exported["dataset_info_path"] = str(dataset_info_path)
    return exported


def _detect_training_dataset_format(dataset_artifact: DatasetArtifact, dataset: Dataset) -> str:
    explicit_format = dataset_artifact.meta.get("dataset_format")
    if explicit_format in {"alpaca", "sharegpt"}:
        return str(explicit_format)
    if "conversations" in dataset.column_names:
        return "sharegpt"
    return "alpaca"


def _write_dataset_for_llamafactory(dataset: Dataset, target: Path, dataset_format: str) -> dict[str, Any]:
    if dataset_format == "sharegpt":
        columns, tags = _write_sharegpt_dataset(dataset, target)
        return {
            "file_name": target.name,
            "formatting": "sharegpt",
            "columns": columns,
            "tags": tags,
        }
    columns = _write_alpaca_dataset(dataset, target)
    return {
        "file_name": target.name,
        "columns": columns,
    }


def _write_alpaca_dataset(dataset: Dataset, target: Path) -> dict[str, str]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(dataset):
        instruction, user_input, response, system = _extract_training_pair(row, dataset.column_names, index)
        record: dict[str, Any] = {
            "instruction": instruction,
            "output": response,
        }
        if user_input:
            record["input"] = user_input
        if system:
            record["system"] = system
        if "history" in dataset.column_names:
            record["history"] = _normalize_history(row.get("history"), index)
        records.append(record)

    target.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    column_mapping = {
        "prompt": "instruction",
        "response": "output",
    }
    if any("input" in record for record in records):
        column_mapping["query"] = "input"
    if any("system" in record for record in records):
        column_mapping["system"] = "system"
    if "history" in dataset.column_names:
        column_mapping["history"] = "history"
    return column_mapping


def _write_sharegpt_dataset(dataset: Dataset, target: Path) -> tuple[dict[str, str], dict[str, str]]:
    records: list[dict[str, Any]] = []
    has_system = False
    has_tools = False
    for index, row in enumerate(dataset):
        conversations = _normalize_sharegpt_conversations(row.get("conversations"), index)
        record: dict[str, Any] = {"conversations": conversations}
        system = _coerce_optional_string(row.get("system"), "system", index)
        if system:
            record["system"] = system
            has_system = True
        if "tools" in dataset.column_names:
            tools = row.get("tools")
            if tools not in (None, "", []):
                record["tools"] = tools
                has_tools = True
        records.append(record)

    target.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    column_mapping = {"messages": "conversations"}
    if has_system:
        column_mapping["system"] = "system"
    if has_tools:
        column_mapping["tools"] = "tools"
    tags = {
        "role_tag": "from",
        "content_tag": "value",
        "user_tag": "human",
        "assistant_tag": "gpt",
        "observation_tag": "observation",
        "function_tag": "function_call",
    }
    return column_mapping, tags


def _normalize_sharegpt_conversations(value: Any, index: int) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Row {index} field 'conversations' must be a non-empty list")
    normalized: list[dict[str, str]] = []
    for turn_index, message in enumerate(value):
        if not isinstance(message, dict):
            raise ValueError(f"Row {index} conversation turn {turn_index} must be a dict")
        raw_role = str(message.get("from") or message.get("role") or "").lower()
        role = _normalize_sharegpt_role(raw_role, index, turn_index)
        content = _coerce_optional_string(
            message.get("value") or message.get("content"),
            "conversation_content",
            index,
            required=True,
        )
        normalized.append({"from": role, "value": content})
    if not any(message["from"] == "human" for message in normalized):
        raise ValueError(f"Row {index} ShareGPT conversations must include a human turn")
    if not any(message["from"] == "gpt" for message in normalized):
        raise ValueError(f"Row {index} ShareGPT conversations must include a gpt turn")
    return normalized


def _normalize_sharegpt_role(role: str, row_index: int, turn_index: int) -> str:
    if role in {"human", "user"}:
        return "human"
    if role in {"gpt", "assistant"}:
        return "gpt"
    if role in {"observation", "function_call", "system"}:
        return role
    raise ValueError(
        f"Row {row_index} conversation turn {turn_index} has unsupported ShareGPT role: {role!r}"
    )


def _extract_training_pair(
    row: dict[str, Any],
    columns: list[str],
    index: int,
) -> tuple[str, str, str, str]:
    instruction_column = "instruction" if "instruction" in columns else "prompt" if "prompt" in columns else None
    response_column = "response" if "response" in columns else "output" if "output" in columns else None
    if instruction_column is None or response_column is None:
        joined = ", ".join(columns)
        raise ValueError(
            "llamafactory mode requires Alpaca-like columns produced by data operators; "
            f"found: {joined}"
        )
    instruction = _coerce_optional_string(row.get(instruction_column), instruction_column, index, required=True)
    response = _coerce_optional_string(row.get(response_column), response_column, index, required=True)
    user_input = _coerce_optional_string(row.get("input"), "input", index)
    system = _coerce_optional_string(row.get("system"), "system", index)
    return instruction, user_input, response, system


def _coerce_optional_string(value: Any, field_name: str, index: int, required: bool = False) -> str:
    if value is None:
        if required:
            raise ValueError(f"Row {index} is missing required field '{field_name}'")
        return ""
    if not isinstance(value, str):
        raise ValueError(f"Row {index} field '{field_name}' must be a string")
    return value


def _normalize_history(value: Any, index: int) -> list[list[str]]:
    if value in (None, []):
        return []
    if not isinstance(value, list):
        raise ValueError(f"Row {index} field 'history' must be a list of two-item turns")
    normalized: list[list[str]] = []
    for turn in value:
        if not isinstance(turn, (list, tuple)) or len(turn) != 2:
            raise ValueError(f"Row {index} field 'history' must contain two-item turns")
        first, second = turn
        if not isinstance(first, str) or not isinstance(second, str):
            raise ValueError(f"Row {index} field 'history' turns must contain strings")
        normalized.append([first, second])
    return normalized


def _build_training_payload(
    effective_config: dict[str, Any],
    exported: dict[str, Any],
    model_dir: Path,
) -> dict[str, Any]:
    payload = {
        "model_name_or_path": effective_config["model_name_or_path"],
        "stage": "sft",
        "do_train": True,
        "finetuning_type": effective_config["finetuning_type"],
        "dataset": exported["train_dataset_name"],
        "dataset_dir": exported["dataset_dir"],
        "template": effective_config["template"],
        "output_dir": str(model_dir),
        "overwrite_output_dir": True,
    }
    if "lora_target" in effective_config:
        payload["lora_target"] = effective_config["lora_target"]
    if "eval_dataset_name" in exported:
        payload["eval_dataset"] = exported["eval_dataset_name"]
    payload.update(effective_config["extra_args"])
    if "eval_dataset" in payload and "val_size" in payload:
        raise ValueError("train_config.extra_args.val_size cannot be combined with an explicit eval split")
    return payload


def _run_llamafactory_train(config_path: Path, sandbox: Sandbox) -> list[str]:
    cli_path = shutil.which("llamafactory-cli")
    if cli_path is None:
        raise ValueError(
            "llamafactory-cli was not found on PATH. Install the train extra and verify the CLI "
            "on the target machine."
        )
    if importlib.util.find_spec("torch") is None:
        raise ValueError(
            "PyTorch is not installed in the active environment. Install a torch build that "
            "matches the target CPU/CUDA setup before running llamafactory mode."
        )

    command = [cli_path, "train", str(config_path)]
    log_path = sandbox.logs_dir / "train_llamafactory.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("COMMAND: " + " ".join(command) + "\n\n")
        log_file.flush()
        completed = subprocess.run(
            command,
            cwd=sandbox.root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            "LLaMA-Factory training failed with exit code "
            f"{completed.returncode}. See log: {log_path}"
        )
    return command


def _validate_training_outputs(model_dir: Path) -> None:
    ignored = {"meta.json", "training_config.json", "training_manifest.json"}
    produced = [
        path
        for path in model_dir.rglob("*")
        if path.is_file() and path.name not in ignored
    ]
    if not produced:
        raise RuntimeError(
            f"LLaMA-Factory training completed but no model files were produced in {model_dir}"
        )
