from __future__ import annotations

import json

from datasets import Dataset, DatasetDict

from dataos.core.artifact import Artifact, DatasetArtifact
from dataos.core.operator import Operator, OperatorParameter
from dataos.core.sandbox import Sandbox
from dataos.operators._dataset_utils import (
    append_jsonl,
    build_row_for_format,
    copy_row,
    detect_dataset_format,
    ensure_dataset_dict,
    extract_text_pair,
    find_latest_dataset,
    preserve_context_fields,
    restore_dataset_type,
    save_dataset_artifact,
)
from dataos.operators._llm_utils import build_operator_llm_client


class GenerateRewrittenQueryOperator(Operator):
    name = "generate_rewritten_query"
    category = "generator"
    description = "Rewrite user queries with an LLM and append the new rows back into the dataset."
    chinese_description = "利用大模型改写用户 query，并把新样本追加回原数据集。"
    parameter_specs = [
        OperatorParameter(
            name="max_rows",
            type="int",
            description="Maximum number of source rows to augment per split.",
            default=16,
        ),
        OperatorParameter(
            name="model",
            type="str",
            description="Optional OpenAI-compatible model override.",
            default="",
        ),
    ]

    def run(self, inputs: list[Artifact], sandbox: Sandbox, **kwargs) -> DatasetArtifact:
        source = find_latest_dataset(inputs, self.name)
        max_rows = int(kwargs.get("max_rows", 16))
        model = str(kwargs.get("model", "")).strip() or None
        if max_rows < 0:
            raise ValueError("generate_rewritten_query: max_rows must be >= 0")

        client = build_operator_llm_client(model=model)
        dataset_obj = source.load()
        dataset_dict, was_dict = ensure_dataset_dict(dataset_obj)
        dataset_format = detect_dataset_format(
            list(dataset_dict[next(iter(dataset_dict))].column_names) if dataset_dict else [],
            source.meta,
        )
        transformed: dict[str, Dataset] = {}
        generated_rows_log: list[dict[str, object]] = []
        generated_counts: dict[str, int] = {}

        for split, dataset in dataset_dict.items():
            rows = [copy_row(dict(row)) for row in dataset]
            generated = 0
            for index, row in enumerate(dataset):
                if max_rows and index >= max_rows:
                    break
                source_row = dict(row)
                prompt, answer, system = extract_text_pair(source_row, list(dataset.column_names), source.meta)
                payload = client.generate_json(
                    system_prompt=(
                        "Rewrite the user prompt for SFT augmentation while preserving task intent. "
                        "Return JSON with key rewritten_prompt."
                    ),
                    user_prompt=json.dumps(
                        {"prompt": prompt, "answer": answer},
                        ensure_ascii=False,
                    ),
                    temperature=0.0,
                )
                rewritten_prompt = str(payload.get("rewritten_prompt", "")).strip()
                if not rewritten_prompt:
                    continue
                new_row = build_row_for_format(dataset_format, rewritten_prompt, answer, system=system)
                preserve_context_fields(source_row, new_row, keep_other_columns=False)
                new_row["dataos_generated"] = True
                new_row["dataos_generator"] = self.name
                new_row["dataos_source_row"] = index
                rows.append(new_row)
                generated += 1
                generated_rows_log.append(
                    {
                        "split": split,
                        "source_row": index,
                        "rewritten_prompt": rewritten_prompt,
                    }
                )
            transformed[split] = Dataset.from_list(rows)
            generated_counts[split] = generated

        output_obj = restore_dataset_type(DatasetDict(transformed), was_dict)
        artifact = save_dataset_artifact(
            source=source,
            sandbox=sandbox,
            operator_name=self.name,
            suffix="rewrite-query",
            dataset_obj=output_obj,
            meta={
                "generator_name": self.name,
                "generator_model": client.model,
                "generator_max_rows": max_rows,
                "generator_generated_counts": generated_counts,
                "dataset_format": dataset_format,
            },
        )
        append_jsonl(artifact.path / "generated_rows.jsonl", generated_rows_log)
        sandbox.log(
            self.name,
            f"source={source.artifact_id}\noutput={artifact.artifact_id}\ngenerated={generated_counts}\n",
        )
        return artifact
