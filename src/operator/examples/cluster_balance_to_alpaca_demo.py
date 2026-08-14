"""Cluster merged message samples, balance by cluster, and export Alpaca JSONL."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dataos_operator import Dataset
from dataos_operator.operators.evaluator.embedding_cluster_evaluator import EmbeddingClusterEvaluator
from dataos_operator.operators.mapper.alpaca_mapper import AlpacaMapper
from dataos_operator.operators.mapper.field_project_mapper import FieldProjectMapper
from dataos_operator.operators.mapper.openai_message_mapper import OpenAIMessageMapper
from dataos_operator.operators.mixer.concat_mixer import ConcatMixer
from dataos_operator.operators.selector.cluster_balance_selector import ClusterBalanceSelector

ROOT = Path(__file__).resolve().parents[1]
GSM8K_PATH = ROOT / "dataset" / "gsm8k" / "gsm8k_20.jsonl"
ALPACA_PATH = ROOT / "dataset" / "alpaca" / "alpaca_20.jsonl"
OUTPUT_PATH = ROOT / "examples" / "outputs" / "cluster_balance_to_alpaca_output.jsonl"
MODEL = "Qwen/Qwen2.5-0.5B"
TARGET_SIZE = 20
TARGET_CLUSTER_SIZE = 8
PCA_DIM = 32

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


if __name__ == "__main__":
    gsm8k_dataset = Dataset.from_local(GSM8K_PATH, source_name="gsm8k_20", limit=20)
    gsm8k_messages = OpenAIMessageMapper(
        source_format="qa",
        query_key="question",
        answer_key="answer",
    ).run(gsm8k_dataset)

    alpaca_dataset = Dataset.from_local(ALPACA_PATH, source_name="alpaca_20", limit=20)
    alpaca_messages = OpenAIMessageMapper(source_format="alpaca").run(alpaca_dataset)

    merged_dataset = ConcatMixer().run([gsm8k_messages, alpaca_messages])
    clustered_dataset = EmbeddingClusterEvaluator(
        model=MODEL,
        model_params={"local_files_only": True},
        cluster_method="kmeans",
        pca_dim=PCA_DIM,
        target_cluster_size=TARGET_CLUSTER_SIZE,
        random_state=42,
        max_iters=30,
    ).run(merged_dataset)
    selected_dataset = ClusterBalanceSelector(
        target_size=TARGET_SIZE,
        quota_strategy="sqrt",
        intra_cluster_strategy="centroid_first",
        random_state=42,
    ).run(clustered_dataset)
    alpaca_output = AlpacaMapper(
        source_format="messages",
        always_include_input=True,
    ).run(selected_dataset)
    export_dataset = FieldProjectMapper(
        keep_fields=("instruction", "input", "output", "metadata"),
        strict=False,
    ).run(alpaca_output)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    export_dataset.save_local(OUTPUT_PATH)
    print(
        json.dumps(
            {
                "output_path": str(OUTPUT_PATH),
                "records": len(export_dataset),
                "metadata": dict(export_dataset.metadata),
                "first_record": export_dataset[0] if len(export_dataset) > 0 else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
