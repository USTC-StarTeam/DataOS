"""Map two datasets to messages, merge them, filter, sample, and export LlamaFactory Alpaca JSONL."""

from __future__ import annotations

import json
import random
from pathlib import Path

from dataos_operator import Dataset
from dataos_operator.operators.mapper.alpaca_mapper import AlpacaMapper
from dataos_operator.operators.mapper.field_project_mapper import FieldProjectMapper
from dataos_operator.operators.mapper.openai_message_mapper import OpenAIMessageMapper
from dataos_operator.operators.mixer.concat_mixer import ConcatMixer
from dataos_operator.operators.selector.length_filter import LengthFilter
from dataos_operator.operators.selector.random_sample_selector import RandomSampleSelector

ROOT = Path(__file__).resolve().parents[1]
GSM8K_PATH = ROOT / "dataset" / "gsm8k" / "gsm8k_20.jsonl"
ALPACA_PATH = ROOT / "dataset" / "alpaca" / "alpaca_20.jsonl"
OUTPUT_PATH = ROOT / "examples" / "outputs" / "merge_to_alpaca_output.jsonl"
MIN_USER_LENGTH = 20
MAX_USER_LENGTH = 600
SAMPLE_SIZE = 20


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
    filtered_dataset = LengthFilter(
        text_key="messages.0.content",
        min_length=MIN_USER_LENGTH,
        max_length=MAX_USER_LENGTH,
    ).run(merged_dataset)

    random.seed(7)
    sampled_dataset = RandomSampleSelector(sample_size=SAMPLE_SIZE, strict=False).run(filtered_dataset)
    alpaca_dataset = AlpacaMapper(
        source_format="messages",
        always_include_input=True,
    ).run(sampled_dataset)
    export_dataset = FieldProjectMapper(
        keep_fields=("instruction", "input", "output", "metadata"),
        strict=False,
    ).run(alpaca_dataset)

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
