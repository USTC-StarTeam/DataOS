"""Score merged message samples for task relevance with an LLM, keep top-k, and export Alpaca JSONL."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dataos_operator import Dataset
from dataos_operator.operators.evaluator.llm_task_relevance_evaluator import LLMTaskRelevanceEvaluator
from dataos_operator.operators.mapper.alpaca_mapper import AlpacaMapper
from dataos_operator.operators.mapper.field_project_mapper import FieldProjectMapper
from dataos_operator.operators.mapper.openai_message_mapper import OpenAIMessageMapper
from dataos_operator.operators.mixer.concat_mixer import ConcatMixer
from dataos_operator.operators.selector.ranking_selector import RankingSelector

ROOT = Path(__file__).resolve().parents[1]
GSM8K_PATH = ROOT / "dataset" / "gsm8k" / "gsm8k_20.jsonl"
ALPACA_PATH = ROOT / "dataset" / "alpaca" / "alpaca_20.jsonl"
OUTPUT_PATH = ROOT / "examples" / "outputs" / "llm_task_relevance_topk_to_alpaca_output.jsonl"
MODEL = os.environ.get("DATAOS_OPERATOR_MODEL", "deepseek-v4-flash-ascend")
BASE_URL = os.environ.get("DATAOS_OPERATOR_BASE_URL", "https://api2.starteam.wang")
API_KEY = os.environ.get("DATAOS_OPERATOR_API_KEY")
TASK_DESC = "Select samples that are useful for math reasoning supervised fine-tuning, especially multi-step arithmetic and problems with verifiable answers."
TOP_K = 20
CONCURRENCY = 1


if __name__ == "__main__":
    if not API_KEY:
        raise RuntimeError("Set DATAOS_OPERATOR_API_KEY before running this demo")

    gsm8k_dataset = Dataset.from_local(GSM8K_PATH, source_name="gsm8k_20", limit=20)
    gsm8k_messages = OpenAIMessageMapper(
        source_format="qa",
        query_key="question",
        answer_key="answer",
    ).run(gsm8k_dataset)

    alpaca_dataset = Dataset.from_local(ALPACA_PATH, source_name="alpaca_20", limit=20)
    alpaca_messages = OpenAIMessageMapper(source_format="alpaca").run(alpaca_dataset)

    merged_dataset = ConcatMixer().run([gsm8k_messages, alpaca_messages])
    scored_dataset = LLMTaskRelevanceEvaluator(
        model=MODEL,
        task_desc=TASK_DESC,
        backend="openai_compatible",
        api_key=API_KEY,
        base_url=BASE_URL,
        model_params={"temperature": 0, "max_tokens": 512},
        concurrency=CONCURRENCY,
    ).run(merged_dataset)
    selected_dataset = RankingSelector(
        sort_keys="metadata.llm_task_relevance.score",
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
