"""使用本地模型或 API 为样本计算文本 embedding 的 evaluator。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dataos_operator.core import EvaluatorOperator, Record
from dataos_operator.core.chat_formats import (
    coerce_text,
    format_llamafactory_alpaca,
    normalize_chat_messages,
)


class EmbeddingEvaluator(EvaluatorOperator):
    """Compute a text embedding vector for each sample, written to metadata.embedding.

    Implementation: prefer ``messages``, otherwise build text from ``text_template`` or
    query+answer; a local HF encoder does attention-mask-weighted mean pooling into a vector,
    written as vector/dim/model/text; reuse an existing vector when ``overwrite=False``.
    Dependency: local-model operator; loads a HF encoder on demand (transformers/torch);
    ``backend="api"`` is not implemented yet.
    """

    def __init__(
        self,
        backend: str = "local",
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_params: dict[str, Any] | None = None,
        embedding_key: str = "metadata.embedding",
        text_template: str | None = None,
        messages_key: str = "messages",
        query_key: str = "query",
        answer_key: str = "answer",
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            backend=backend,
            model=model,
            model_params=model_params,
            embedding_key=embedding_key,
            text_template=text_template,
            messages_key=messages_key,
            query_key=query_key,
            answer_key=answer_key,
            overwrite=overwrite,
            **kwargs,
        )
        if backend not in {"local", "api"}:
            raise ValueError("backend must be one of: local, api")

        self.backend = backend
        self.model = model
        self.model_params = dict(model_params or {})
        self.embedding_key = embedding_key
        self.text_template = text_template
        self.messages_key = messages_key
        self.query_key = query_key
        self.answer_key = answer_key
        self.overwrite = overwrite
        self._torch: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._max_length = int(self.model_params.pop("max_length", 512))

    def evaluate_record(self, record: Record) -> Record:
        vector_key = f"{self.embedding_key}.vector"
        if not self.overwrite and self.get_field(record, vector_key, None) is not None:
            return dict(record)

        text = self._embedding_text(record)
        vector = self._compute_embedding(text)
        payload = {
            "model": self.model,
            "text": text,
            "vector": vector,
            "dim": len(vector),
        }
        return self.set_field(record, self.embedding_key, payload)

    def _embedding_text(self, record: Record) -> str:
        messages_value = self.get_field(record, self.messages_key, None)
        if messages_value is not None:
            system_message, normalized_messages = normalize_chat_messages(messages_value, strict=True)
            parts: list[str] = []
            if system_message is not None:
                parts.append(f"System: {system_message}")
            for message in normalized_messages:
                role = "User" if message["role"] == "user" else "Assistant"
                parts.append(f"{role}: {message['content']}")
            text = "\n\n".join(parts)
            if text:
                return text
            raise ValueError("messages must contain at least one usable message")

        if self.text_template is not None:
            text = self._format_template(self.text_template, record).strip()
            if text:
                return text
            raise ValueError("text_template produced empty text")

        query_value = self.get_field(record, self.query_key, None)
        answer_value = self.get_field(record, self.answer_key, None)
        query_text = coerce_text(query_value).strip() if query_value is not None else ""
        answer_text = coerce_text(answer_value).strip() if answer_value is not None else ""
        if not query_text and not answer_text:
            raise KeyError(
                f"Missing embedding content: expected {self.messages_key!r}, text_template, "
                f"or {self.query_key!r}/{self.answer_key!r}"
            )
        if query_text:
            return format_llamafactory_alpaca(query_text, response=answer_text or None)
        return answer_text

    def _format_template(self, template: str, record: Record) -> str:
        try:
            return template.format_map(_FlatRecordMapping(record))
        except KeyError as exc:
            raise KeyError(f"Template field is missing: {exc.args[0]}") from exc

    def _compute_embedding(self, text: str) -> list[float]:
        if self.backend == "local":
            return self._compute_local_embedding(text)
        raise NotImplementedError("API embedding backend is not implemented yet")

    def _ensure_model_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None and self._torch is not None:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

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
        self._model = AutoModel.from_pretrained(self.model, **model_kwargs)
        if "device_map" not in model_kwargs:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(device)
        self._model.eval()

    def _compute_local_embedding(self, text: str) -> list[float]:
        if not text:
            raise ValueError("embedding text must not be empty")

        self._ensure_model_loaded()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None

        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
        )
        device = next(self._model.parameters()).device
        with self._torch.no_grad():
            outputs = self._model(
                input_ids=encoded["input_ids"].to(device),
                attention_mask=encoded["attention_mask"].to(device),
            )

        hidden_state = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].to(device).unsqueeze(-1)
        masked_hidden = hidden_state * attention_mask
        token_count = attention_mask.sum(dim=1).clamp(min=1)
        pooled = masked_hidden.sum(dim=1) / token_count
        return [float(value) for value in pooled[0].detach().cpu().tolist()]


class _FlatRecordMapping(dict[str, Any]):
    def __init__(self, record: Mapping[str, Any]) -> None:
        super().__init__()
        self._record = record

    def __missing__(self, key: str) -> Any:
        if key in self._record:
            return self._record[key]
        raise KeyError(key)
