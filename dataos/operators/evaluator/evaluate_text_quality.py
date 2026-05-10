from __future__ import annotations

from datasets import Dataset, DatasetDict

from dataos.core.artifact import Artifact, DatasetArtifact
from dataos.core.operator import Operator, OperatorParameter
from dataos.core.sandbox import Sandbox
from dataos.operators._dataset_utils import (
    count_words,
    ensure_dataset_dict,
    extract_text_pair,
    find_latest_dataset,
    restore_dataset_type,
    save_dataset_artifact,
)


def _heuristic_quality_score(prompt: str, answer: str) -> tuple[float, str]:
    score = 0.0
    reasons: list[str] = []
    if prompt.strip():
        score += 0.25
    else:
        reasons.append("missing_prompt")
    if answer.strip():
        score += 0.25
    else:
        reasons.append("missing_answer")
    if count_words(prompt) >= 3:
        score += 0.2
    else:
        reasons.append("short_prompt")
    if count_words(answer) >= 8:
        score += 0.2
    else:
        reasons.append("short_answer")
    if prompt.strip().lower() != answer.strip().lower():
        score += 0.1
    else:
        reasons.append("echo_answer")
    return round(score, 4), ",".join(reasons)


class EvaluateTextQualityOperator(Operator):
    name = "evaluate_text_quality"
    category = "evaluator"
    description = "Attach a deterministic heuristic quality score to each row."
    chinese_description = "给每条样本附加一个启发式质量分数和简单原因。"
    parameter_specs = [
        OperatorParameter(
            name="score_column",
            type="str",
            description="Column used to store the numeric score.",
            default="quality_score",
        ),
        OperatorParameter(
            name="reason_column",
            type="str",
            description="Column used to store scoring notes.",
            default="quality_reason",
        ),
    ]

    def run(self, inputs: list[Artifact], sandbox: Sandbox, **kwargs) -> DatasetArtifact:
        source = find_latest_dataset(inputs, self.name)
        score_column = str(kwargs.get("score_column", "quality_score"))
        reason_column = str(kwargs.get("reason_column", "quality_reason"))
        dataset_obj = source.load()
        dataset_dict, was_dict = ensure_dataset_dict(dataset_obj)

        transformed: dict[str, Dataset] = {}
        score_stats: dict[str, dict[str, float]] = {}
        for split, dataset in dataset_dict.items():
            rows: list[dict] = []
            scores: list[float] = []
            for row in dataset:
                prompt, answer, _ = extract_text_pair(dict(row), list(dataset.column_names), source.meta)
                score, reason = _heuristic_quality_score(prompt, answer)
                new_row = dict(row)
                new_row[score_column] = score
                new_row[reason_column] = reason
                rows.append(new_row)
                scores.append(score)
            transformed[split] = Dataset.from_list(rows)
            score_stats[split] = {
                "min": min(scores) if scores else 0.0,
                "max": max(scores) if scores else 0.0,
                "avg": round(sum(scores) / len(scores), 4) if scores else 0.0,
            }

        output_obj = restore_dataset_type(DatasetDict(transformed), was_dict)
        artifact = save_dataset_artifact(
            source=source,
            sandbox=sandbox,
            operator_name=self.name,
            suffix="quality-score",
            dataset_obj=output_obj,
            meta={
                "evaluator_name": self.name,
                "evaluator_score_column": score_column,
                "evaluator_reason_column": reason_column,
                "evaluator_stats": score_stats,
                "dataset_format": source.meta.get("dataset_format", "alpaca"),
            },
        )
        sandbox.log(
            self.name,
            f"source={source.artifact_id}\noutput={artifact.artifact_id}\nstats={score_stats}\n",
        )
        return artifact
