from __future__ import annotations

from typing import Any

from datasets import Dataset, DatasetDict

from dataos.core.artifact import Artifact, DatasetArtifact, make_artifact_id
from dataos.core.operator import Operator
from dataos.core.sandbox import Sandbox
from dataos.operators._common import ensure_dataset_dict, find_latest_dataset, normalize_text, restore_dataset_type


def _resolve_column(columns: list[str], explicit: str | None, aliases: list[str], required: bool) -> str | None:
    if explicit is not None:
        if explicit not in columns:
            raise ValueError(f"schema_guard: column '{explicit}' not found in dataset")
        return explicit
    for alias in aliases:
        if alias in columns:
            return alias
    if required:
        raise ValueError(f"schema_guard: could not find any of required columns {aliases}")
    return None


class SchemaGuardOperator(Operator):
    name = "schema_guard"
    description = "Normalize heterogeneous SFT rows into DataOS canonical instruction/input/response columns."

    def run(self, inputs: list[Artifact], sandbox: Sandbox, **kwargs) -> DatasetArtifact:
        source = find_latest_dataset(inputs, self.name)
        dataset_obj = source.load()
        dataset_dict, was_dict = ensure_dataset_dict(dataset_obj)
        mode = str(kwargs.get("mode", "repair"))
        if mode not in {"repair", "strict"}:
            raise ValueError("schema_guard: mode must be 'repair' or 'strict'")
        drop_empty = bool(kwargs.get("drop_empty", True))
        keep_extra_columns = bool(kwargs.get("keep_extra_columns", True))
        strip_whitespace = bool(kwargs.get("strip_whitespace", True))
        collapse_whitespace = bool(kwargs.get("collapse_whitespace", True))

        transformed: dict[str, Dataset] = {}
        dropped_counts: dict[str, int] = {}
        kept_counts: dict[str, int] = {}
        resolved_columns: dict[str, dict[str, str | None]] = {}

        for split, dataset in dataset_dict.items():
            columns = list(dataset.column_names)
            instruction_col = _resolve_column(columns, kwargs.get("instruction_column"), ["instruction", "prompt", "question"], True)
            response_col = _resolve_column(columns, kwargs.get("response_column"), ["response", "output", "answer"], True)
            input_col = _resolve_column(columns, kwargs.get("input_column"), ["input", "context", "query"], False)
            resolved_columns[split] = {
                "instruction": instruction_col,
                "response": response_col,
                "input": input_col,
            }

            rows: list[dict[str, Any]] = []
            dropped = 0
            for row in dataset:
                instruction = normalize_text(row.get(instruction_col), strip=strip_whitespace, collapse_whitespace=collapse_whitespace)
                response = normalize_text(row.get(response_col), strip=strip_whitespace, collapse_whitespace=collapse_whitespace)
                user_input = normalize_text(row.get(input_col), strip=strip_whitespace, collapse_whitespace=collapse_whitespace) if input_col is not None else ""

                if mode == "strict" and (not instruction or not response):
                    raise ValueError("schema_guard: strict mode found empty instruction/response row")
                if drop_empty and (not instruction or not response):
                    dropped += 1
                    continue

                if keep_extra_columns:
                    new_row = {key: value for key, value in row.items() if key not in {instruction_col, response_col, input_col}}
                else:
                    new_row = {}
                new_row["instruction"] = instruction
                new_row["response"] = response
                new_row["input"] = user_input
                rows.append(new_row)

            transformed[split] = Dataset.from_list(rows)
            dropped_counts[split] = dropped
            kept_counts[split] = len(rows)

        output_obj = restore_dataset_type(DatasetDict(transformed), was_dict)
        artifact_id = make_artifact_id("dataset")
        artifact = DatasetArtifact(
            artifact_id=artifact_id,
            name=f"{source.name}-schema-guard",
            path=sandbox.artifact_dir("dataset", artifact_id),
            parent_ids=[source.artifact_id],
            producer=self.name,
            meta={
                "schema_guard_mode": mode,
                "schema_guard_drop_empty": drop_empty,
                "schema_guard_keep_extra_columns": keep_extra_columns,
                "schema_guard_resolved_columns": resolved_columns,
                "schema_guard_dropped_counts": dropped_counts,
                "schema_guard_kept_counts": kept_counts,
            },
        )
        artifact.save(output_obj)
        sandbox.log(
            self.name,
            f"source={source.artifact_id}\noutput={artifact.artifact_id}\nmode={mode}\nkept={kept_counts}\ndropped={dropped_counts}\n",
        )
        return artifact
