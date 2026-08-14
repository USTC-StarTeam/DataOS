"""Mix gsm8k/alpaca-eval source subsets toward benchmark target datasets and export Alpaca JSONL."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dataos_operator import Dataset
from dataos_operator.operators.mapper.alpaca_mapper import AlpacaMapper
from dataos_operator.operators.mapper.field_project_mapper import FieldProjectMapper
from dataos_operator.operators.mixer.domain2vec_mixer import Domain2VecMixer

ROOT = Path(__file__).resolve().parents[1]
SOURCE_GSM8K_PATH = ROOT / "dataset" / "gsm8k" / "gsm8k_domain2vec_source_20.jsonl"
SOURCE_ALPACA_EVAL_PATH = ROOT / "dataset" / "alpaca_eval" / "alpaca_eval_domain2vec_source_20.jsonl"
TARGET_GSM8K_PATH = ROOT / "dataset" / "gsm8k" / "gsm8k_domain2vec_test.jsonl"
TARGET_ARENAHARD_PATH = ROOT / "dataset" / "arenahard" / "arenahard_domain2vec_test.jsonl"
TARGET_HUMANEVAL_PATH = ROOT / "dataset" / "humaneval" / "humaneval_domain2vec_test.jsonl"
OUTPUT_PATH = ROOT / "examples" / "outputs" / "domain2vec_to_alpaca_output.jsonl"
MODEL = "Qwen/Qwen2.5-0.5B"
OUTPUT_SIZE = 30
SOURCE_SAMPLE_SIZE = 20
TARGET_SAMPLE_SIZE = 60

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


if __name__ == "__main__":
    source_gsm8k = Dataset.from_local(SOURCE_GSM8K_PATH, source_name="gsm8k_source", limit=None)
    source_alpaca_eval = Dataset.from_local(SOURCE_ALPACA_EVAL_PATH, source_name="alpaca_eval_source", limit=None)
    target_gsm8k = Dataset.from_local(TARGET_GSM8K_PATH, source_name="gsm8k_target", limit=None)
    target_arenahard = Dataset.from_local(TARGET_ARENAHARD_PATH, source_name="arenahard_target", limit=None)
    target_humaneval = Dataset.from_local(TARGET_HUMANEVAL_PATH, source_name="humaneval_target", limit=None)

    mixed_dataset = Domain2VecMixer(
        target_datasets=[target_gsm8k, target_arenahard, target_humaneval],
        domain_key="metadata.source",
        model=MODEL,
        model_params={"local_files_only": True},
        output_size=OUTPUT_SIZE,
        source_sample_size=SOURCE_SAMPLE_SIZE,
        target_sample_size=TARGET_SAMPLE_SIZE,
        temperature=0.7,
        random_state=42,
    ).run([source_gsm8k, source_alpaca_eval])
    alpaca_output = AlpacaMapper(
        source_format="fielded",
        instruction_key="query",
        output_key="answer",
        always_include_input=True,
    ).run(mixed_dataset)
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
