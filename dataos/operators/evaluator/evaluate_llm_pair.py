from __future__ import annotations

import json

from datasets import Dataset, DatasetDict

from dataos.core.artifact import Artifact, DatasetArtifact
from dataos.core.operator import Operator, OperatorParameter
from dataos.core.sandbox import Sandbox
from dataos.operators._dataset_utils import (
    append_jsonl,
    ensure_dataset_dict,
    extract_text_pair,
    find_latest_dataset,
    restore_dataset_type,
    save_dataset_artifact,
)
from dataos.operators._llm_utils import build_operator_llm_client


class EvaluateLLMPairOperator(Operator):
    name = "evaluate_llm_pair"
    category = "evaluator"
    description = "Use an OpenAI-compatible LLM to score prompt-answer pairs and attach the results."
    chinese_description = "调用 OpenAI-compatible 大模型为问答对打分，并把分数写回数据集。"
    parameter_specs = [
        OperatorParameter(
            name="score_column",
            type="str",
            description="Column used to store the numeric score.",
            default="llm_quality_score",
        ),
        OperatorParameter(
            name="reason_column",
            type="str",
            description="Column used to store the LLM explanation.",
            default="llm_quality_reason",
        ),
        OperatorParameter(
            name="max_rows",
            type="int",
            description="Maximum number of rows to evaluate per split. Remaining rows receive a default score.",
            default=32,
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
        score_column = str(kwargs.get("score_column", "llm_quality_score"))
        reason_column = str(kwargs.get("reason_column", "llm_quality_reason"))
        max_rows = int(kwargs.get("max_rows", 32))
        model = str(kwargs.get("model", "")).strip() or None
        if max_rows < 0:
            raise ValueError("evaluate_llm_pair: max_rows must be >= 0")

        client = build_operator_llm_client(model=model)
        dataset_obj = source.load()
        dataset_dict, was_dict = ensure_dataset_dict(dataset_obj)
        transformed: dict[str, Dataset] = {}
        judgments: list[dict[str, object]] = []

        for split, dataset in dataset_dict.items():
            rows: list[dict] = []
            for index, row in enumerate(dataset):
                new_row = dict(row)
                if max_rows and index >= max_rows:
                    new_row[score_column] = 0.0
                    new_row[reason_column] = "not_evaluated_due_to_max_rows"
                    rows.append(new_row)
                    continue
                prompt, answer, _ = extract_text_pair(new_row, list(dataset.column_names), source.meta)
                payload = client.generate_json(
                    system_prompt=(
                        "You score SFT prompt-answer pairs. Return JSON with keys "
                        "score (0 to 1 float) and reason (short string)."
                    ),
                    user_prompt=json.dumps(
                        {"prompt": prompt, "answer": answer},
                        ensure_ascii=False,
                    ),
                    temperature=0.0,
                )
                score = round(float(payload.get("score", 0.0)), 4)
                reason = str(payload.get("reason", ""))
                new_row[score_column] = score
                new_row[reason_column] = reason
                rows.append(new_row)
                judgments.append(
                    {
                        "split": split,
                        "index": index,
                        "score": score,
                        "reason": reason,
                    }
                )
            transformed[split] = Dataset.from_list(rows)

        output_obj = restore_dataset_type(DatasetDict(transformed), was_dict)
        artifact = save_dataset_artifact(
            source=source,
            sandbox=sandbox,
            operator_name=self.name,
            suffix="llm-score",
            dataset_obj=output_obj,
            meta={
                "evaluator_name": self.name,
                "evaluator_score_column": score_column,
                "evaluator_reason_column": reason_column,
                "evaluator_max_rows": max_rows,
                "evaluator_model": client.model,
                "dataset_format": source.meta.get("dataset_format", "alpaca"),
            },
        )
        append_jsonl(artifact.path / "llm_judgments.jsonl", judgments)
        sandbox.log(
            self.name,
            f"source={source.artifact_id}\noutput={artifact.artifact_id}\njudged_rows={len(judgments)}\n",
        )
        return artifact
