"""Open-config LLaMA-Factory training executor.

``run_train`` is the single entry point used by ``Experiment.train``. It is a
deterministic glue layer: it materializes the external
``dataos_operator.Dataset``, exports it to LLaMA-Factory's on-disk format,
assembles an *open* YAML payload (any LLaMA-Factory key the caller passes
through ``config`` is forwarded verbatim — there is no allowlist), runs
``llamafactory-cli train``, validates the outputs, and returns a :class:`Model`.

The export / write / validate helpers are ported from the previous runtime
unchanged in body; only their parameters changed from ``Sandbox`` /
``DatasetArtifact`` to plain ``Path`` / explicit args. Heavy dependencies
(torch, llamafactory-cli) are guarded and only touched on the real training
path, so the stub path and ``import dataos`` need only ``datasets`` +
``dataos_operator``.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from datasets import Dataset

from dataos.core.artifact import Model
from dataos.core.ids import _make_id

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dataos_operator import Dataset as OperatorDataset

    from dataos.core.workspace import Workspace

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


def run_train(
    dataset: "OperatorDataset",
    config: dict[str, Any] | None,
    *,
    workspace: "Workspace",
) -> Model:
    """Train a model from a ``dataos_operator.Dataset`` and return a :class:`Model`."""
    if config is None or config.get("mode") == "stub":
        return _run_stub_training(dataset, workspace=workspace)

    # Remote executor: ship training to an H100 Slurm cluster, keep data/eval local.
    if config.get("executor") == "slurm_h100":
        from dataos.integrations.slurm.h100_train import run_train_slurm_h100

        return run_train_slurm_h100(dataset, config, workspace=workspace)

    # The prepared dataset is already materialized to workspace.dataset_dir by the
    # data stage (it IS the Data artifact); re-saving here would fail because the
    # dataset passed in is memory-mapped from that same dir ("can't overwrite
    # itself"). Operate on the flat single-split HF Dataset directly (no
    # DatasetDict, so no split selection).
    hf_train: Dataset = dataset.hf_dataset

    model_id = _make_id("model")
    model_dir = workspace.artifact_dir("model", model_id)
    explicit_format = dict(getattr(dataset, "metadata", {}) or {}).get("dataset_format")
    exported = _export_datasets_for_llamafactory(
        export_root=workspace.tmp_dir / "data",
        train_dataset=hf_train,
        train_split="train",
        eval_dataset=None,
        eval_split=None,
        explicit_format=explicit_format,
    )
    training_payload = _build_open_payload(config, exported, model_dir)
    config_path = workspace.tmp_dir / "train.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(training_payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    log_path = workspace.logs_dir / "train_llamafactory.log"
    command = _run_llamafactory_train(
        config_path=config_path,
        cwd=workspace.exp_dir,
        log_path=log_path,
    )
    _validate_training_outputs(model_dir)

    base_model = training_payload["model_name_or_path"]
    template = training_payload["template"]
    finetuning_type = training_payload["finetuning_type"]
    meta = {
        "mode": "llamafactory",
        "dataset_format": exported["dataset_format"],
        "base_model": base_model,
        "template": template,
        "finetuning_type": finetuning_type,
        "lora_target": training_payload.get("lora_target"),
        "train_split": "train",
        "selected_rows": hf_train.num_rows,
        "config_path": str(config_path),
        "log_path": str(log_path),
    }
    (model_dir / "training_config.json").write_text(
        json.dumps(training_payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest = {
        "command": command,
        "config_path": str(config_path),
        "exported_data": exported,
        "train_config": training_payload,
        "log_path": str(log_path),
    }
    (model_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )

    model = Model(model_id=model_id, path=model_dir, meta=meta, producer="train_llamafactory")
    workspace.log(
        "train_llamafactory_summary",
        (
            f"model={model.model_id}\n"
            f"mode=llamafactory\n"
            f"base_model={base_model}\n"
            f"template={template}\n"
            f"finetuning_type={finetuning_type}\n"
            f"rows={hf_train.num_rows}\n"
            f"config={config_path}\n"
        ),
    )
    return model


def _run_stub_training(
    dataset: "OperatorDataset",
    *,
    workspace: "Workspace",
) -> Model:
    hf_train: Dataset = dataset.hf_dataset
    model_id = _make_id("model")
    model_dir = workspace.artifact_dir("model", model_id)
    meta = {
        "mode": "stub",
        "train_split": "train",
        "selected_rows": hf_train.num_rows,
    }
    (model_dir / "training_config.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    (model_dir / "weights.stub").write_text("stub model binary placeholder\n", encoding="utf-8")
    model = Model(model_id=model_id, path=model_dir, meta=meta, producer="train_llamafactory")
    workspace.log(
        "train_llamafactory",
        f"model={model.model_id}\nmode=stub\nrows={hf_train.num_rows}\n",
    )
    return model


def _build_open_payload(
    config: dict[str, Any],
    exported: dict[str, Any],
    model_dir: Path,
) -> dict[str, Any]:
    """Build the LLaMA-Factory YAML payload, passing through every non-reserved key.

    The caller owns every knob. We only: reject reserved keys (they are forced
    from the export/output dir), validate ``finetuning_type``, require
    ``model_name_or_path`` + ``template``, and forward everything else verbatim.
    """
    if not isinstance(config, dict):
        raise ValueError("train config must be a dict")
    payload = dict(config)
    payload.pop("mode", None)

    reserved = set(payload) & RESERVED_CONFIG_KEYS
    if reserved:
        keys = ", ".join(sorted(reserved))
        raise ValueError(
            f"Reserved LLaMA-Factory keys cannot be set via config (DataOS owns them): {keys}"
        )

    model_name_or_path = payload.get("model_name_or_path")
    if not model_name_or_path:
        raise ValueError("train config requires 'model_name_or_path'")
    template = payload.get("template")
    if not template:
        raise ValueError("train config requires 'template'")

    finetuning_type = str(payload.get("finetuning_type", "lora"))
    if finetuning_type not in SUPPORTED_FINETUNING_TYPES:
        allowed = ", ".join(sorted(SUPPORTED_FINETUNING_TYPES))
        raise ValueError(f"train config finetuning_type must be one of: {allowed}")
    payload["finetuning_type"] = finetuning_type

    # Force the reserved keys from the export + output dir.
    payload["stage"] = "sft"
    payload["do_train"] = True
    payload["dataset"] = exported["train_dataset_name"]
    payload["dataset_dir"] = exported["dataset_dir"]
    payload["output_dir"] = str(model_dir)
    payload["overwrite_output_dir"] = True
    if "eval_dataset_name" in exported:
        payload["eval_dataset"] = exported["eval_dataset_name"]
    if "eval_dataset" in payload and "val_size" in payload:
        raise ValueError("config.val_size cannot be combined with an explicit eval split")
    return payload


def _export_datasets_for_llamafactory(
    export_root: Path,
    train_dataset: Dataset,
    train_split: str,
    eval_dataset: Dataset | None,
    eval_split: str | None,
    explicit_format: str | None = None,
) -> dict[str, Any]:
    export_root.mkdir(parents=True, exist_ok=True)
    dataset_format = _detect_training_dataset_format(explicit_format, train_dataset)

    train_name = f"dataos_{train_split}"
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
        eval_name = f"dataos_{eval_split}"
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


def _detect_training_dataset_format(explicit_format: str | None, dataset: Dataset) -> str:
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


def _run_llamafactory_train(config_path: Path, cwd: Path, log_path: Path) -> list[str]:
    cli_path = shutil.which("llamafactory-cli")
    if cli_path is None:
        raise ValueError(
            "llamafactory-cli was not found on PATH. Install the train extra and verify the CLI "
            "on the target machine."
        )
    # Absolutize: a relative PATH entry (e.g. PATH=.venv/bin:$PATH) makes which() return a relative
    # path that fails once the subprocess runs with cwd=exp_dir. Use an absolute CLI path.
    cli_path = str(Path(cli_path).absolute())
    if importlib.util.find_spec("torch") is None:
        raise ValueError(
            "PyTorch is not installed in the active environment. Install a torch build that "
            "matches the target CPU/CUDA setup before running llamafactory mode."
        )

    command = [cli_path, "train", str(config_path)]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("COMMAND: " + " ".join(command) + "\n\n")
        log_file.flush()
        completed = subprocess.run(
            command,
            cwd=cwd,
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
