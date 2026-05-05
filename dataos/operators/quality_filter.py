from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import httpx
from datasets import Dataset, DatasetDict

from dataos.core.artifact import Artifact, DatasetArtifact, make_artifact_id
from dataos.core.operator import Operator
from dataos.core.sandbox import Sandbox
from dataos.operators._common import count_words, ensure_dataset_dict, find_latest_dataset, infer_text_columns, join_text_columns, restore_dataset_type

_PLACEHOLDER_PATTERNS = ("todo", "tbd", "lorem ipsum", "n/a")


def _heuristic_quality_score(instruction: str, response: str) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    instruction_text = instruction.strip()
    response_text = response.strip()
    if instruction_text:
        score += 0.2
    else:
        reasons.append("missing_instruction")
    if response_text:
        score += 0.2
    else:
        reasons.append("missing_response")
    if count_words(instruction_text) >= 3:
        score += 0.15
    else:
        reasons.append("short_instruction")
    if count_words(response_text) >= 2:
        score += 0.15
    else:
        reasons.append("short_response")
    if instruction_text and response_text and instruction_text.lower() != response_text.lower():
        score += 0.15
    else:
        reasons.append("echo_response")
    weird_ratio = 0.0
    if response_text:
        weird_ratio = sum(1 for ch in response_text if not (ch.isalnum() or ch.isspace() or ch in ",.!?;:'\"-()[]{}")) / max(1, len(response_text))
    if weird_ratio <= 0.2:
        score += 0.1
    else:
        reasons.append("noisy_symbols")
    lowered = f"{instruction_text} {response_text}".lower()
    if not any(pattern in lowered for pattern in _PLACEHOLDER_PATTERNS):
        score += 0.05
    else:
        reasons.append("placeholder_text")
    unique_words = set(re.findall(r"\b\w+\b", response_text.lower()))
    response_word_count = max(1, count_words(response_text))
    if len(unique_words) / response_word_count >= 0.5:
        score += 0.05
    else:
        reasons.append("high_repetition")
    return round(min(score, 1.0), 4), reasons


def _extract_json_payload(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


def _judge_with_openai(*, base_url: str, api_key: str, model: str, instruction: str, response: str, timeout: float) -> dict[str, Any]:
    prompt = (
        "You are evaluating whether an instruction-response training pair is suitable for SFT. "
        "Return JSON only with keys keep (bool), score (0-1 number), and reason (short string)."
    )
    user_content = json.dumps({"instruction": instruction, "response": response}, ensure_ascii=False)
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json=body, headers=headers)
        response.raise_for_status()
    payload = response.json()
    content = payload["choices"][0]["message"]["content"]
    result = _extract_json_payload(content)
    return {
        "keep": bool(result.get("keep", False)),
        "score": float(result.get("score", 0.0)),
        "reason": str(result.get("reason", "")),
    }


class QualityFilterOperator(Operator):
    name = "quality_filter"
    description = "Filter low-quality SFT pairs with heuristic scoring or an OpenAI-compatible LLM judge."

    def run(self, inputs: list[Artifact], sandbox: Sandbox, **kwargs) -> DatasetArtifact:
        source = find_latest_dataset(inputs, self.name)
        dataset_obj = source.load()
        dataset_dict, was_dict = ensure_dataset_dict(dataset_obj)
        mode = str(kwargs.get("mode", "heuristic"))
        if mode not in {"heuristic", "llm_judge"}:
            raise ValueError("quality_filter: mode must be 'heuristic' or 'llm_judge'")
        min_score = float(kwargs.get("min_score", 0.55))
        attach_score_column = kwargs.get("attach_score_column")
        attach_reason_column = kwargs.get("attach_reason_column")

        base_url = str(kwargs.get("base_url") or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE") or "https://api.openai.com/v1")
        api_key = str(kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY") or "")
        model = str(kwargs.get("model") or os.environ.get("OPENAI_MODEL") or "")
        timeout = float(kwargs.get("timeout", 120))
        if mode == "llm_judge" and (not api_key or not model):
            raise ValueError("quality_filter: llm_judge mode requires api_key and model (kwargs or env)")

        transformed: dict[str, Dataset] = {}
        kept_counts: dict[str, int] = {}
        removed_counts: dict[str, int] = {}
        judgment_path_rows: list[dict[str, Any]] = []

        for split, dataset in dataset_dict.items():
            columns = list(kwargs.get("columns") or infer_text_columns(dataset, ["instruction", "input", "response", "output"]))
            rows: list[dict[str, Any]] = []
            removed = 0
            for index, row in enumerate(dataset):
                instruction = str(row.get("instruction") or row.get("prompt") or join_text_columns(row, columns))
                response = str(row.get("response") or row.get("output") or row.get("answer") or "")
                if mode == "heuristic":
                    score, reasons = _heuristic_quality_score(instruction, response)
                    keep = score >= min_score
                    reason = ",".join(reasons)
                else:
                    judgment = _judge_with_openai(
                        base_url=base_url,
                        api_key=api_key,
                        model=model,
                        instruction=instruction,
                        response=response,
                        timeout=timeout,
                    )
                    keep = bool(judgment["keep"]) and float(judgment["score"]) >= min_score
                    score = float(judgment["score"])
                    reason = str(judgment["reason"])
                judgment_path_rows.append({
                    "split": split,
                    "index": index,
                    "score": score,
                    "keep": keep,
                    "reason": reason,
                })
                if not keep:
                    removed += 1
                    continue
                new_row = dict(row)
                if attach_score_column:
                    new_row[str(attach_score_column)] = score
                if attach_reason_column:
                    new_row[str(attach_reason_column)] = reason
                rows.append(new_row)
            transformed[split] = Dataset.from_list(rows)
            kept_counts[split] = len(rows)
            removed_counts[split] = removed

        output_obj = restore_dataset_type(DatasetDict(transformed), was_dict)
        artifact_id = make_artifact_id("dataset")
        artifact = DatasetArtifact(
            artifact_id=artifact_id,
            name=f"{source.name}-quality-filter",
            path=sandbox.artifact_dir("dataset", artifact_id),
            parent_ids=[source.artifact_id],
            producer=self.name,
            meta={
                "quality_filter_mode": mode,
                "quality_filter_min_score": min_score,
                "quality_filter_kept_counts": kept_counts,
                "quality_filter_removed_counts": removed_counts,
                "quality_filter_attach_score_column": attach_score_column,
                "quality_filter_attach_reason_column": attach_reason_column,
            },
        )
        artifact.save(output_obj)
        (artifact.path / "quality_judgments.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in judgment_path_rows) + ("\n" if judgment_path_rows else ""),
            encoding="utf-8",
        )
        sandbox.log(
            self.name,
            f"source={source.artifact_id}\noutput={artifact.artifact_id}\nmode={mode}\nkept={kept_counts}\nremoved={removed_counts}\n",
        )
        return artifact
