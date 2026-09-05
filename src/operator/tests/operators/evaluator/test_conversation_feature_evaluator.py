from __future__ import annotations

from datasets import Dataset as HFDataset

from dataos_operator import DatasetView
from dataos_operator.operators.evaluator import ConversationFeatureEvaluator


def test_conversation_features_recognize_mcq_group_and_coordinate_rows() -> None:
    operator = ConversationFeatureEvaluator()
    mcq = operator.run(
        {
            "texts": [
                {
                    "user": "Pick one.\nA. cat\nB. dog\nAnswer with the option's letter.",
                    "assistant": "B",
                }
            ]
        }
    )[0]["metadata"]["conversation"]
    coordinate = operator.run(
        {"texts": [{"user": "Locate it.", "assistant": "[0.1, 0.2, 0.3, 0.4]"}]}
    )[0]["metadata"]["conversation"]

    assert mcq["format"] == "mcq"
    assert mcq["question_group"]
    assert mcq["richness_weight"] > 1
    assert coordinate["is_coordinate"] is True
    assert "coordinate" in coordinate["tags"]


def test_conversation_features_stay_derived_on_dataset_view() -> None:
    source = HFDataset.from_list(
        [
            {"id": "a", "texts": [{"user": "Why?", "assistant": "Because."}]},
            {
                "id": "b",
                "texts": [
                    {"user": "Answer with the option's letter.\nA. x\nB. y", "assistant": "A"}
                ],
            },
        ]
    )

    result = ConversationFeatureEvaluator(batch_size=1).run(DatasetView(source))

    assert isinstance(result, DatasetView)
    assert result.derived_columns["metadata.conversation.format"] == ("other", "mcq")
    assert "metadata" not in source.column_names
    assert result.metadata["operator"] == "conversation_feature_evaluator"
