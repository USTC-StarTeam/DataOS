"""Score merged message samples with IFD, select top-k, and export LlamaFactory Alpaca JSONL."""

from __future__ import annotations

import json
from pathlib import Path

from dataos_operator import Dataset
from dataos_operator.operators.evaluator.ifd_evaluator import IFDEvaluator
from dataos_operator.operators.mapper.alpaca_mapper import AlpacaMapper
from dataos_operator.operators.mapper.field_project_mapper import FieldProjectMapper
from dataos_operator.operators.mapper.openai_message_mapper import OpenAIMessageMapper
from dataos_operator.operators.mixer.concat_mixer import ConcatMixer
from dataos_operator.operators.selector.ranking_selector import RankingSelector

ROOT = Path(__file__).resolve().parents[1]
GSM8K_PATH = ROOT / "dataset" / "gsm8k" / "gsm8k_20.jsonl"
ALPACA_PATH = ROOT / "dataset" / "alpaca" / "alpaca_20.jsonl"
OUTPUT_PATH = ROOT / "examples" / "outputs" / "ifd_topk_to_alpaca_output.jsonl"
QWEN_SNAPSHOT = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen2.5-0.5B"
    / "snapshots"
    / "060db6499f32faf8b98477b0a26969ef7d8b9987"
)
TOP_K = 20


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
    scored_dataset = IFDEvaluator(
        hf_model=str(QWEN_SNAPSHOT),
        model_params={"local_files_only": True},
    ).run(merged_dataset)
    selected_dataset = RankingSelector(
        sort_keys="metadata.ifd",
        top_k=TOP_K,
        reverse=True,
    ).run(scored_dataset)
    alpaca_dataset = AlpacaMapper(
        source_format="messages",
        always_include_input=True,
    ).run(selected_dataset)
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
