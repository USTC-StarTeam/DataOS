from __future__ import annotations

import re
from typing import Any

from datasets import Dataset, DatasetDict

from dataos.core.artifact import Artifact, DatasetArtifact, make_artifact_id
from dataos.core.operator import Operator
from dataos.core.sandbox import Sandbox
from dataos.operators._common import ensure_dataset_dict, find_latest_dataset, infer_text_columns, restore_dataset_type

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\-\s]{7,}\d)(?!\w)")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")


class PiiRedactOperator(Operator):
    name = "pii_redact"
    description = "Redact obvious emails, phones, and URLs from text fields while preserving dataset shape."

    def run(self, inputs: list[Artifact], sandbox: Sandbox, **kwargs) -> DatasetArtifact:
        source = find_latest_dataset(inputs, self.name)
        dataset_obj = source.load()
        dataset_dict, was_dict = ensure_dataset_dict(dataset_obj)
        drop_rows_with_pii = bool(kwargs.get("drop_rows_with_pii", False))
        redact_email = bool(kwargs.get("redact_email", True))
        redact_phone = bool(kwargs.get("redact_phone", True))
        redact_url = bool(kwargs.get("redact_url", True))

        transformed: dict[str, Dataset] = {}
        redaction_counts: dict[str, dict[str, int]] = {}
        dropped_counts: dict[str, int] = {}
        column_meta: dict[str, list[str]] = {}

        for split, dataset in dataset_dict.items():
            columns = list(kwargs.get("columns") or infer_text_columns(dataset, ["instruction", "input", "response", "output"]))
            column_meta[split] = columns
            rows: list[dict[str, Any]] = []
            split_counts = {"email": 0, "phone": 0, "url": 0}
            dropped = 0
            for row in dataset:
                new_row = dict(row)
                row_has_pii = False
                for column in columns:
                    value = new_row.get(column)
                    if not isinstance(value, str):
                        continue
                    text = value
                    if redact_email:
                        text, count = _EMAIL_RE.subn("[EMAIL]", text)
                        split_counts["email"] += count
                        row_has_pii = row_has_pii or count > 0
                    if redact_phone:
                        text, count = _PHONE_RE.subn("[PHONE]", text)
                        split_counts["phone"] += count
                        row_has_pii = row_has_pii or count > 0
                    if redact_url:
                        text, count = _URL_RE.subn("[URL]", text)
                        split_counts["url"] += count
                        row_has_pii = row_has_pii or count > 0
                    new_row[column] = text
                if drop_rows_with_pii and row_has_pii:
                    dropped += 1
                    continue
                rows.append(new_row)
            transformed[split] = Dataset.from_list(rows)
            redaction_counts[split] = split_counts
            dropped_counts[split] = dropped

        output_obj = restore_dataset_type(DatasetDict(transformed), was_dict)
        artifact_id = make_artifact_id("dataset")
        artifact = DatasetArtifact(
            artifact_id=artifact_id,
            name=f"{source.name}-pii-redact",
            path=sandbox.artifact_dir("dataset", artifact_id),
            parent_ids=[source.artifact_id],
            producer=self.name,
            meta={
                "pii_redact_columns": column_meta,
                "pii_redact_counts": redaction_counts,
                "pii_redact_drop_rows_with_pii": drop_rows_with_pii,
                "pii_redact_dropped_counts": dropped_counts,
            },
        )
        artifact.save(output_obj)
        sandbox.log(
            self.name,
            f"source={source.artifact_id}\noutput={artifact.artifact_id}\nredactions={redaction_counts}\ndropped={dropped_counts}\n",
        )
        return artifact
