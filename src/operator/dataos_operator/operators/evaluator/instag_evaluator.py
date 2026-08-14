"""使用 InsTag 风格提示词为样本生成 instruction tags 的 evaluator。"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from dataos_operator.core import EvaluatorOperator, Record
from dataos_operator.core.chat_formats import coerce_text, normalize_chat_messages


class InsTagEvaluator(EvaluatorOperator):
    """Generate capability tags for an instruction (InsTag style), with a complexity score from the tag count.

    Implementation: a local generative model produces a comma-separated tag list per the tagging
    prompt, which is parsed and normalized; on empty or degenerate (all-generic) output it falls
    back to keyword heuristics; written to ``metadata.instag.{tags,tag_count,complexity,raw_output}``.
    Dependency: local-model operator; loads a HF generative model on demand (transformers/torch).
    """

    GENERIC_TAG_VOCAB = {
        "reasoning",
        "mathematics",
        "coding",
        "translation",
        "summarization",
        "classification",
        "extraction",
        "writing",
        "dialogue",
        "planning",
    }

    TAGGING_PROMPT = (
        "You are an instruction tagging assistant. Read the user instruction and "
        "output a concise comma-separated list of capability tags. Use short tags "
        "such as reasoning, mathematics, coding, translation, summarization, "
        "classification, extraction, writing, dialogue, planning.\n\n"
        "Instruction:\n{instruction}\n\nTags:"
    )

    def __init__(
        self,
        model: str = "OFA-Sys/InsTagger",
        model_params: dict[str, Any] | None = None,
        instag_key: str = "metadata.instag",
        query_template: str | None = None,
        messages_key: str = "messages",
        query_key: str = "query",
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            model_params=model_params,
            instag_key=instag_key,
            query_template=query_template,
            messages_key=messages_key,
            query_key=query_key,
            overwrite=overwrite,
            **kwargs,
        )
        self.model = model
        self.model_params = dict(model_params or {})
        self.instag_key = instag_key
        self.query_template = query_template
        self.messages_key = messages_key
        self.query_key = query_key
        self.overwrite = overwrite
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._max_input_length = int(self.model_params.pop("max_input_length", 768))
        self._max_new_tokens = int(self.model_params.pop("max_new_tokens", 64))

    def evaluate_record(self, record: Record) -> Record:
        tags_path = f"{self.instag_key}.tags"
        if not self.overwrite and self.get_field(record, tags_path, None) is not None:
            return dict(record)

        instruction = self._instruction_text(record)
        raw_output = self._generate_tags(instruction)
        tags = self._parse_tags(raw_output)
        if not tags or self._needs_heuristic_fallback(tags):
            tags = self._heuristic_tags(instruction)

        updated = self.set_field(record, f"{self.instag_key}.tags", tags)
        updated = self.set_field(updated, f"{self.instag_key}.tag_count", len(tags))
        updated = self.set_field(updated, f"{self.instag_key}.complexity", float(len(tags)))
        updated = self.set_field(updated, f"{self.instag_key}.raw_output", raw_output)
        return updated

    def _instruction_text(self, record: Record) -> str:
        messages_value = self.get_field(record, self.messages_key, None)
        if messages_value is not None:
            system_message, normalized_messages = normalize_chat_messages(messages_value, strict=True)
            parts: list[str] = []
            if system_message is not None:
                parts.append(f"System: {system_message}")
            for message in normalized_messages:
                if message["role"] == "assistant":
                    continue
                parts.append(message["content"])
            text = "\n\n".join(part.strip() for part in parts if part.strip())
            if text:
                return text
            raise ValueError("messages must contain at least one non-assistant instruction")

        if self.query_template is not None:
            text = self._format_template(self.query_template, record).strip()
            if text:
                return text
            raise ValueError("query_template produced empty instruction text")

        query_value = self.get_field(record, self.query_key, None)
        if query_value is None:
            raise KeyError(f"Missing instruction content: expected {self.query_key!r}")
        text = coerce_text(query_value).strip()
        if not text:
            raise ValueError("instruction text must not be empty")
        return text

    def _format_template(self, template: str, record: Record) -> str:
        try:
            return template.format_map(_FlatRecordMapping(record))
        except KeyError as exc:
            raise KeyError(f"Template field is missing: {exc.args[0]}") from exc

    def _generate_tags(self, instruction: str) -> str:
        self._ensure_model_loaded()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None

        prompt = self.TAGGING_PROMPT.format(instruction=instruction)
        encoded = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_input_length,
        )
        device = next(self._model.parameters()).device
        generation_kwargs = {
            "max_new_tokens": self._max_new_tokens,
            "do_sample": False,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        with self._torch.no_grad():
            output_ids = self._model.generate(
                input_ids=encoded["input_ids"].to(device),
                attention_mask=encoded["attention_mask"].to(device),
                **generation_kwargs,
            )
        generated_ids = output_ids[0][encoded["input_ids"].shape[1] :]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def _ensure_model_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None and self._torch is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        tokenizer_kwargs = {
            key: value
            for key, value in self.model_params.items()
            if key in {"trust_remote_code", "revision", "use_fast", "token", "local_files_only"}
        }
        model_kwargs = {
            key: value
            for key, value in self.model_params.items()
            if key
            in {
                "device_map",
                "torch_dtype",
                "trust_remote_code",
                "revision",
                "token",
                "local_files_only",
                "attn_implementation",
            }
        }
        self._tokenizer = AutoTokenizer.from_pretrained(self.model, **tokenizer_kwargs)
        if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(self.model, **model_kwargs)
        if "device_map" not in model_kwargs:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(device)
        self._model.eval()

    def _needs_heuristic_fallback(self, tags: list[str]) -> bool:
        return len(tags) >= 8 and set(tags).issubset(self.GENERIC_TAG_VOCAB)

    def _parse_tags(self, raw_output: str) -> list[str]:
        cleaned = raw_output.strip()
        if not cleaned:
            return []
        if "Tags:" in cleaned:
            cleaned = cleaned.split("Tags:", 1)[1].strip()
        cleaned = cleaned.replace("\n", ",")
        parts = re.split(r"[,;|]", cleaned)
        tags: list[str] = []
        seen: set[str] = set()
        for part in parts:
            tag = part.strip().strip("-*").strip().lower()
            tag = re.sub(r"\s+", "_", tag)
            tag = re.sub(r"[^a-z0-9_+#-]", "", tag)
            if not tag or len(tag) > 32:
                continue
            if tag in seen:
                continue
            seen.add(tag)
            tags.append(tag)
        return tags

    def _heuristic_tags(self, instruction: str) -> list[str]:
        lowered = instruction.lower()
        tags: list[str] = []
        if any(keyword in lowered for keyword in ["calculate", "math", "number", "equation", "solve"]):
            tags.extend(["mathematics", "reasoning"])
        if any(keyword in lowered for keyword in ["translate", "translation"]):
            tags.append("translation")
        if any(keyword in lowered for keyword in ["summarize", "summary", "tl;dr"]):
            tags.append("summarization")
        if any(keyword in lowered for keyword in ["classify", "label", "category"]):
            tags.append("classification")
        if any(keyword in lowered for keyword in ["write", "draft", "essay", "story"]):
            tags.append("writing")
        if any(keyword in lowered for keyword in ["extract", "find", "list"]):
            tags.append("extraction")
        if any(keyword in lowered for keyword in ["plan", "schedule", "roadmap"]):
            tags.append("planning")
        if any(keyword in lowered for keyword in ["code", "python", "function", "program"]):
            tags.append("coding")
        if "?" in instruction or any(keyword in lowered for keyword in ["why", "how", "explain"]):
            tags.append("qa")
        if len(lowered.split()) >= 20:
            tags.append("long_context")
        if not tags:
            tags.append("general_instruction")
        deduped: list[str] = []
        seen: set[str] = set()
        for tag in tags:
            if tag in seen:
                continue
            seen.add(tag)
            deduped.append(tag)
        return deduped


class _FlatRecordMapping(dict[str, Any]):
    def __init__(self, record: Mapping[str, Any]) -> None:
        super().__init__()
        self._record = record

    def __missing__(self, key: str) -> Any:
        if key in self._record:
            return self._record[key]
        raise KeyError(key)
