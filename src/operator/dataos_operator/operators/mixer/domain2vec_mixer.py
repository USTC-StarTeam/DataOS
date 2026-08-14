"""Mix source datasets to match a target domain distribution."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from dataos_operator import Dataset
from dataos_operator.core import MixerOperator, Record
from dataos_operator.core.chat_formats import coerce_text, normalize_chat_messages


class Domain2VecMixer(MixerOperator):
    """Estimate source mixture weights from a target dataset's domain distribution and resample (a similarity-based approximation of the Domain2Vec idea).

    Implementation: group source samples by ``domain_key`` and compute a prototype vector per
    domain with a local embedding model; for each target sample compute softmax similarity to
    the domain prototypes and average to get the target distribution, then allocate per-domain
    quotas and resample the sources; dataset-level metadata records mixture_weights, etc.
    Dependency: local-model operator; loads a HF encoder on demand (transformers/torch) + numpy.
    """

    def __init__(
        self,
        target_datasets: Dataset | Sequence[Dataset],
        domain_key: str = "metadata.source",
        model: str = "Qwen/Qwen2.5-0.5B",
        messages_key: str = "messages",
        query_key: str = "query",
        answer_key: str = "answer",
        text_key: str = "text",
        output_size: int | None = None,
        source_sample_size: int = 500,
        target_sample_size: int = 500,
        temperature: float = 1.0,
        random_state: int = 42,
        model_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            target_datasets=target_datasets,
            domain_key=domain_key,
            model=model,
            messages_key=messages_key,
            query_key=query_key,
            answer_key=answer_key,
            text_key=text_key,
            output_size=output_size,
            source_sample_size=source_sample_size,
            target_sample_size=target_sample_size,
            temperature=temperature,
            random_state=random_state,
            model_params=model_params,
            **kwargs,
        )
        if source_sample_size <= 0:
            raise ValueError("source_sample_size must be greater than zero")
        if target_sample_size <= 0:
            raise ValueError("target_sample_size must be greater than zero")
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if output_size is not None and output_size <= 0:
            raise ValueError("output_size must be greater than zero")

        self.target_datasets = self._normalize_target_datasets(target_datasets)
        self.domain_key = domain_key
        self.model = model
        self.messages_key = messages_key
        self.query_key = query_key
        self.answer_key = answer_key
        self.text_key = text_key
        self.output_size = output_size
        self.source_sample_size = source_sample_size
        self.target_sample_size = target_sample_size
        self.temperature = temperature
        self.random_state = random_state
        self.model_params = dict(model_params or {})
        self._torch: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._max_length = int(self.model_params.pop("max_length", 512))
        self._rng = random.Random(self.random_state)

    def _normalize_target_datasets(self, target_datasets: Dataset | Sequence[Dataset]) -> list[Dataset]:
        if isinstance(target_datasets, (list, tuple)):
            datasets = list(target_datasets)
        else:
            datasets = [target_datasets]
        normalized = [dataset for dataset in datasets if len(dataset) > 0]
        if not normalized:
            raise ValueError("target_datasets must not be empty")
        return normalized

    def mix(self, data: Sequence[Dataset]) -> Dataset:
        source_records = [record for dataset in data for record in dataset]
        if not source_records:
            return Dataset.from_records([], metadata={})

        grouped_source_records = self._group_by_domain(source_records)
        domain_order = sorted(grouped_source_records, key=lambda item: str(item))
        domain_prototypes = self._build_domain_prototypes(grouped_source_records, domain_order)
        target_distribution = self._estimate_target_distribution(domain_prototypes, domain_order)
        output_size = self.output_size or len(source_records)
        quotas = _compute_quotas(target_distribution, grouped_source_records, domain_order, output_size)
        sampled_records = self._sample_records(grouped_source_records, domain_order, quotas)

        metadata = {
            "domain2vec": {
                "domain_key": self.domain_key,
                "source_domains": [str(domain) for domain in domain_order],
                "target_distribution": {
                    str(domain): float(target_distribution[index])
                    for index, domain in enumerate(domain_order)
                },
                "mixture_weights": {
                    str(domain): float(target_distribution[index])
                    for index, domain in enumerate(domain_order)
                },
                "model": self.model,
                "target_dataset_count": len(self.target_datasets),
                "source_sample_size": self.source_sample_size,
                "target_sample_size": self.target_sample_size,
                "temperature": self.temperature,
            }
        }
        return Dataset.from_records(sampled_records, metadata=metadata)

    def _group_by_domain(self, records: Sequence[Record]) -> dict[Any, list[Record]]:
        grouped: dict[Any, list[Record]] = defaultdict(list)
        for record in records:
            domain_value = self.get_field(record, self.domain_key, None)
            if domain_value is None:
                raise KeyError(f"Missing domain label at {self.domain_key!r}")
            grouped[domain_value].append(record)
        return grouped

    def _build_domain_prototypes(
        self,
        grouped_records: Mapping[Any, Sequence[Record]],
        domain_order: Sequence[Any],
    ) -> dict[Any, np.ndarray]:
        prototypes: dict[Any, np.ndarray] = {}
        for domain in domain_order:
            sampled_records = self._sample_subset(grouped_records[domain], self.source_sample_size)
            vectors = [self._embed_record(record) for record in sampled_records]
            matrix = np.asarray(vectors, dtype=np.float32)
            prototype = matrix.mean(axis=0)
            prototypes[domain] = _normalize_vector(prototype)
        return prototypes

    def _estimate_target_distribution(
        self,
        prototypes: Mapping[Any, np.ndarray],
        domain_order: Sequence[Any],
    ) -> np.ndarray:
        target_records = [record for dataset in self.target_datasets for record in dataset]
        sampled_records = self._sample_subset(target_records, self.target_sample_size)
        if not sampled_records:
            raise ValueError("target_datasets must not be empty")
        probabilities: list[np.ndarray] = []
        prototype_matrix = np.stack([prototypes[domain] for domain in domain_order], axis=0)
        for record in sampled_records:
            vector = self._embed_record(record)
            scores = prototype_matrix @ vector
            probabilities.append(_softmax(scores / self.temperature))
        distribution = np.mean(np.stack(probabilities, axis=0), axis=0)
        total = float(distribution.sum())
        if total <= 0:
            raise ValueError("target distribution sum must be positive")
        return distribution / total

    def _sample_records(
        self,
        grouped_records: Mapping[Any, Sequence[Record]],
        domain_order: Sequence[Any],
        quotas: Mapping[Any, int],
    ) -> list[Record]:
        selected: list[Record] = []
        leftovers: list[Record] = []
        for domain in domain_order:
            records = list(grouped_records[domain])
            shuffled = list(records)
            self._rng.shuffle(shuffled)
            quota = quotas.get(domain, 0)
            selected.extend(shuffled[:quota])
            leftovers.extend(shuffled[quota:])

        target_size = sum(quotas.values())
        if len(selected) < target_size:
            self._rng.shuffle(leftovers)
            selected.extend(leftovers[: target_size - len(selected)])
        self._rng.shuffle(selected)
        return selected[:target_size]

    def _sample_subset(self, records: Sequence[Record], limit: int) -> list[Record]:
        records_list = list(records)
        if len(records_list) <= limit:
            return records_list
        return self._rng.sample(records_list, limit)

    def _embed_record(self, record: Record) -> np.ndarray:
        text = self._record_text(record)
        vector = np.asarray(self._compute_embedding(text), dtype=np.float32)
        return _normalize_vector(vector)

    def _record_text(self, record: Record) -> str:
        messages_value = self.get_field(record, self.messages_key, None)
        if messages_value is not None:
            system_message, normalized_messages = normalize_chat_messages(messages_value, strict=True)
            parts: list[str] = []
            if system_message is not None:
                parts.append(f"System: {system_message}")
            for message in normalized_messages:
                role = "User" if message["role"] == "user" else "Assistant"
                parts.append(f"{role}: {message['content']}")
            text = "\n\n".join(parts).strip()
            if text:
                return text
            raise ValueError("messages must contain at least one usable message")

        query_value = self.get_field(record, self.query_key, None)
        answer_value = self.get_field(record, self.answer_key, None)
        query_text = coerce_text(query_value).strip() if query_value is not None else ""
        answer_text = coerce_text(answer_value).strip() if answer_value is not None else ""
        if query_text or answer_text:
            if query_text and answer_text:
                return f"User: {query_text}\n\nAssistant: {answer_text}"
            return query_text or answer_text

        text_value = self.get_field(record, self.text_key, None)
        if text_value is not None:
            text = coerce_text(text_value).strip()
            if text:
                return text
        raise KeyError(
            f"Missing text content: expected {self.messages_key!r}, {self.query_key!r}/{self.answer_key!r}, or {self.text_key!r}"
        )

    def _compute_embedding(self, text: str) -> list[float]:
        if not text:
            raise ValueError("text must not be empty")
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


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("vector norm must not be zero")
    return vector / norm


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    total = float(exp_values.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("softmax sum must be positive")
    return exp_values / total


def _compute_quotas(
    distribution: np.ndarray,
    grouped_records: Mapping[Any, Sequence[Record]],
    domain_order: Sequence[Any],
    output_size: int,
) -> dict[Any, int]:
    raw = {
        domain: float(distribution[index]) * output_size
        for index, domain in enumerate(domain_order)
    }
    quotas = {
        domain: min(len(grouped_records[domain]), int(raw_quota))
        for domain, raw_quota in raw.items()
    }
    assigned = sum(quotas.values())
    if assigned >= output_size:
        return quotas

    remainders = sorted(
        (
            raw[domain] - int(raw[domain]),
            len(grouped_records[domain]),
            domain,
        )
        for domain in domain_order
    )
    remainders.reverse()
    while assigned < output_size:
        progressed = False
        for _, _, domain in remainders:
            if quotas[domain] >= len(grouped_records[domain]):
                continue
            quotas[domain] += 1
            assigned += 1
            progressed = True
            if assigned >= output_size:
                break
        if not progressed:
            break
    return quotas
