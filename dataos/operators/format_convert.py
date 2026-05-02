"""Convert SFT rows between alpaca / sharegpt / messages schemas.

Spec fields (all optional unless noted):

- ``target`` (default ``"messages"``): one of ``"messages"``, ``"alpaca"``,
  ``"sharegpt"``.
- ``source`` (optional): one of the same set; auto-detected from the input
  columns when omitted.
- ``system_prompt`` (optional): default system message content prepended when
  emitting ``messages`` / ``sharegpt`` from a non-conversational source.
- Column-name overrides for the alpaca schema:
  ``instruction_column`` (default ``"instruction"``),
  ``input_column`` (default ``"input"``),
  ``output_column`` (default ``"output"`` then ``"response"``).

The output schema only carries the canonical columns of the target format;
upstream extras are dropped to keep downstream training schemas predictable.
"""

from __future__ import annotations

from typing import Any

from datasets import Dataset, DatasetDict

from dataos.core.artifact import Artifact, DatasetArtifact, make_artifact_id
from dataos.core.operator import Operator
from dataos.core.sandbox import Sandbox


_VALID_FORMATS = {"messages", "alpaca", "sharegpt"}
_SHAREGPT_TO_ROLE = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
}
_ROLE_TO_SHAREGPT = {"user": "human", "assistant": "gpt", "system": "system"}


def _find_latest_dataset(inputs: list[Artifact]) -> DatasetArtifact:
    for artifact in reversed(inputs):
        if isinstance(artifact, DatasetArtifact):
            return artifact
    raise ValueError("format_convert requires at least one DatasetArtifact input")


def _detect_source(columns: list[str]) -> str:
    column_set = set(columns)
    if "messages" in column_set:
        return "messages"
    if "conversations" in column_set:
        return "sharegpt"
    if "instruction" in column_set and ("output" in column_set or "response" in column_set):
        return "alpaca"
    raise ValueError(
        f"format_convert: cannot auto-detect source format from columns={columns}; "
        "set 'source' explicitly"
    )


def _resolve_alpaca_columns(spec: dict[str, Any], columns: list[str]) -> dict[str, str]:
    instruction_col = str(spec.get("instruction_column", "instruction"))
    input_col = str(spec.get("input_column", "input"))
    output_col = spec.get("output_column")
    if output_col is None:
        if "output" in columns:
            output_col = "output"
        elif "response" in columns:
            output_col = "response"
        else:
            raise ValueError(
                "format_convert: alpaca source needs 'output' or 'response' column "
                "(or set 'output_column')"
            )
    return {
        "instruction": instruction_col,
        "input": input_col,
        "output": str(output_col),
    }


def _alpaca_user_text(instruction: str, user_input: str) -> str:
    instruction = (instruction or "").strip()
    user_input = (user_input or "").strip()
    if user_input:
        return f"{instruction}\n\n{user_input}" if instruction else user_input
    return instruction


def _alpaca_to_messages(
    row: dict[str, Any],
    cols: dict[str, str],
    system_prompt: str | None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    user_text = _alpaca_user_text(row.get(cols["instruction"], ""), row.get(cols["input"], ""))
    messages.append({"role": "user", "content": user_text})
    messages.append({"role": "assistant", "content": str(row.get(cols["output"], ""))})
    return messages


def _sharegpt_to_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    conversations = row.get("conversations") or []
    messages: list[dict[str, str]] = []
    for turn in conversations:
        role_raw = str(turn.get("from", "")).lower()
        role = _SHAREGPT_TO_ROLE.get(role_raw)
        if role is None:
            raise ValueError(f"format_convert: unknown sharegpt 'from' value: {role_raw}")
        messages.append({"role": role, "content": str(turn.get("value", ""))})
    return messages


def _messages_to_alpaca(messages: list[dict[str, Any]]) -> dict[str, str]:
    system_text = ""
    user_text = ""
    assistant_text = ""
    seen_user = False
    for msg in messages:
        role = str(msg.get("role", "")).lower()
        content = str(msg.get("content", ""))
        if role == "system" and not system_text:
            system_text = content
        elif role == "user" and not seen_user:
            user_text = content
            seen_user = True
        elif role == "assistant":
            assistant_text = content  # last assistant wins
    if not seen_user or not assistant_text:
        raise ValueError("format_convert: messages → alpaca needs a user and assistant turn")
    instruction = f"{system_text}\n\n{user_text}".strip() if system_text else user_text
    return {"instruction": instruction, "input": "", "output": assistant_text}


def _messages_to_sharegpt(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    converted: list[dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role", "")).lower()
        from_value = _ROLE_TO_SHAREGPT.get(role)
        if from_value is None:
            raise ValueError(f"format_convert: unknown role '{role}' for sharegpt target")
        converted.append({"from": from_value, "value": str(msg.get("content", ""))})
    return converted


def _convert_split(
    dataset: Dataset,
    source: str,
    target: str,
    spec: dict[str, Any],
) -> Dataset:
    if source == target:
        return dataset
    columns = list(dataset.column_names)
    system_prompt = spec.get("system_prompt")

    if source == "alpaca":
        cols = _resolve_alpaca_columns(spec, columns)
        if target == "messages":
            rows = [{"messages": _alpaca_to_messages(row, cols, system_prompt)} for row in dataset]
            return Dataset.from_list(rows)
        if target == "sharegpt":
            rows = [
                {"conversations": _messages_to_sharegpt(_alpaca_to_messages(row, cols, system_prompt))}
                for row in dataset
            ]
            return Dataset.from_list(rows)

    if source == "sharegpt":
        if target == "messages":
            rows = [{"messages": _sharegpt_to_messages(row)} for row in dataset]
            return Dataset.from_list(rows)
        if target == "alpaca":
            rows = [_messages_to_alpaca(_sharegpt_to_messages(row)) for row in dataset]
            return Dataset.from_list(rows)

    if source == "messages":
        if target == "alpaca":
            rows = [_messages_to_alpaca(row.get("messages") or []) for row in dataset]
            return Dataset.from_list(rows)
        if target == "sharegpt":
            rows = [
                {"conversations": _messages_to_sharegpt(row.get("messages") or [])}
                for row in dataset
            ]
            return Dataset.from_list(rows)

    raise ValueError(f"format_convert: unsupported conversion {source} -> {target}")


class FormatConvertOperator(Operator):
    name = "format_convert"
    description = "Convert SFT rows between alpaca, sharegpt, and messages schemas."

    def run(self, inputs: list[Artifact], sandbox: Sandbox, **kwargs) -> DatasetArtifact:
        source_artifact = _find_latest_dataset(inputs)
        target = str(kwargs.get("target", "messages"))
        if target not in _VALID_FORMATS:
            raise ValueError(f"format_convert: target must be one of {sorted(_VALID_FORMATS)}")
        explicit_source = kwargs.get("source")
        if explicit_source is not None and explicit_source not in _VALID_FORMATS:
            raise ValueError(f"format_convert: source must be one of {sorted(_VALID_FORMATS)}")

        dataset_obj = source_artifact.load()

        if isinstance(dataset_obj, DatasetDict):
            converted_splits: dict[str, Dataset] = {}
            detected: dict[str, str] = {}
            for split, split_dataset in dataset_obj.items():
                source_format = str(
                    explicit_source or _detect_source(list(split_dataset.column_names))
                )
                detected[split] = source_format
                converted_splits[split] = _convert_split(
                    split_dataset, source_format, target, kwargs
                )
            converted: Dataset | DatasetDict = DatasetDict(converted_splits)
            source_meta = detected
        else:
            source_format = str(
                explicit_source or _detect_source(list(dataset_obj.column_names))
            )
            converted = _convert_split(dataset_obj, source_format, target, kwargs)
            source_meta = source_format

        artifact_id = make_artifact_id("dataset")
        target_artifact = DatasetArtifact(
            artifact_id=artifact_id,
            name=f"{source_artifact.name}-{target}",
            path=sandbox.artifact_dir("dataset", artifact_id),
            parent_ids=[source_artifact.artifact_id],
            producer=self.name,
            meta={
                "format_target": target,
                "format_source": source_meta,
                "format_system_prompt": kwargs.get("system_prompt"),
            },
        )
        target_artifact.save(converted)
        sandbox.log(
            "format_convert",
            f"source={source_artifact.artifact_id}\noutput={target_artifact.artifact_id}\n"
            f"source_format={source_meta}\ntarget={target}\n",
        )
        return target_artifact
