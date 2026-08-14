"""Lightweight text-model backend for loss, embedding, gradient, and reward signals."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from dataos_operator.core import Record
from dataos_operator.core.chat_formats import (
    coerce_text,
    format_llamafactory_alpaca,
    normalize_chat_messages,
)


@dataclass
class PromptTarget:
    """Normalized prompt/target pair extracted from a record."""

    prompt: str
    target: str


class TextModelBackend:
    """Small reusable backend for text-centric policy experiments."""

    def __init__(
        self,
        model_name: str,
        *,
        max_length: int = 256,
        max_new_tokens: int = 48,
        parameter_filters: Sequence[str] = ("lm_head",),
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        torch_dtype: str | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.parameter_filters = tuple(parameter_filters)
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only
        self.torch_dtype = torch_dtype
        self.requested_device = device
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    @property
    def tokenizer(self) -> Any:
        self._ensure_model_loaded()
        assert self._tokenizer is not None
        return self._tokenizer

    @property
    def model(self) -> Any:
        self._ensure_model_loaded()
        assert self._model is not None
        return self._model

    @property
    def device(self) -> torch.device:
        model = self.model
        return next(model.parameters()).device

    def prompt_target(self, record: Record) -> PromptTarget:
        messages_value = record.get("messages")
        if messages_value is not None:
            return self._prompt_target_from_messages(messages_value)

        if "instruction" in record and "output" in record:
            prompt = format_llamafactory_alpaca(
                coerce_text(record.get("instruction")),
                extra_input=coerce_text(record.get("input")),
            ).rstrip()
            return PromptTarget(prompt=prompt, target=coerce_text(record.get("output")).strip())

        if "question" in record and "answer" in record:
            prompt = format_llamafactory_alpaca(coerce_text(record.get("question"))).rstrip()
            return PromptTarget(prompt=prompt, target=coerce_text(record.get("answer")).strip())

        if "query" in record and "answer" in record:
            prompt = format_llamafactory_alpaca(coerce_text(record.get("query"))).rstrip()
            return PromptTarget(prompt=prompt, target=coerce_text(record.get("answer")).strip())

        if "text" in record:
            return PromptTarget(prompt="", target=coerce_text(record.get("text")).strip())

        raise KeyError("Record cannot be converted into a prompt/target pair")

    def render_text(self, record: Record) -> str:
        prompt_target = self.prompt_target(record)
        if prompt_target.prompt:
            return f"{prompt_target.prompt}\n{prompt_target.target}".strip()
        return prompt_target.target

    def compute_losses(self, records: Sequence[Record]) -> list[float]:
        losses: list[float] = []
        for record in records:
            losses.append(self._single_loss(record))
        return losses

    def compute_embeddings(self, records: Sequence[Record]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        model = self.model
        tokenizer = self.tokenizer
        model.eval()
        for record in records:
            text = self.render_text(record)
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = model(**encoded, output_hidden_states=True, return_dict=True)
            hidden = outputs.hidden_states[-1]
            attention = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * attention).sum(dim=1) / attention.sum(dim=1).clamp(min=1)
            vector = pooled[0].detach().float().cpu().tolist()
            embeddings.append(_l2_normalize(vector))
        return embeddings

    def gradient_vector(
        self,
        record: Record,
        *,
        projection_dim: int | None = None,
        projection_seed: int = 42,
    ) -> list[float]:
        model = self.model
        model.train()
        encoded = self._encoded_pair(record)
        outputs = model(**encoded, return_dict=True)
        loss = outputs.loss
        loss.backward()

        pieces: list[torch.Tensor] = []
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if self.parameter_filters and not any(token in name for token in self.parameter_filters):
                continue
            pieces.append(param.grad.detach().float().view(-1).cpu())
        if not pieces:
            raise ValueError("No gradients were collected for the configured parameter filters")
        vector = torch.cat(pieces, dim=0)
        model.zero_grad(set_to_none=True)

        if projection_dim is not None and projection_dim < vector.numel():
            vector = _rademacher_project(vector, projection_dim, projection_seed)
        return vector.tolist()

    def mean_gradient(
        self,
        records: Sequence[Record],
        *,
        projection_dim: int | None = None,
        projection_seed: int = 42,
    ) -> list[float]:
        gradients = [
            torch.tensor(
                self.gradient_vector(
                    record,
                    projection_dim=projection_dim,
                    projection_seed=projection_seed,
                ),
                dtype=torch.float32,
            )
            for record in records
        ]
        if not gradients:
            raise ValueError("mean_gradient requires at least one record")
        stacked = torch.stack(gradients, dim=0)
        mean_vector = stacked.mean(dim=0)
        return mean_vector.tolist()

    def generate_candidates(
        self,
        record: Record,
        *,
        num_samples: int = 4,
        temperature: float = 0.7,
    ) -> list[str]:
        model = self.model
        tokenizer = self.tokenizer
        model.eval()
        prompt = self.prompt_target(record).prompt or self.render_text(record)
        encoded = tokenizer(prompt, truncation=True, max_length=self.max_length, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=num_samples,
            )
        candidates: list[str] = []
        for row in output_ids:
            generated_ids = row[encoded["input_ids"].shape[1] :]
            candidates.append(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())
        return candidates

    def reward_score(self, record: Record, candidate: str) -> float:
        reference = self.prompt_target(record).target
        if not reference:
            return 0.0
        reference_answer = _extract_terminal_answer(reference)
        candidate_answer = _extract_terminal_answer(candidate)
        if reference_answer and candidate_answer:
            return 1.0 if reference_answer == candidate_answer else 0.0
        return _token_f1(reference, candidate)

    def _single_loss(self, record: Record) -> float:
        model = self.model
        model.eval()
        encoded = self._encoded_pair(record)
        with torch.no_grad():
            outputs = model(**encoded, return_dict=True)
        return float(outputs.loss.detach().cpu().item())

    def _encoded_pair(self, record: Record) -> dict[str, torch.Tensor]:
        prompt_target = self.prompt_target(record)
        tokenizer = self.tokenizer
        prompt_ids = tokenizer(
            prompt_target.prompt,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]
        target_ids = tokenizer(
            prompt_target.target,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]
        if target_ids.shape[1] == 0:
            raise ValueError("Target text produced no tokens")

        input_ids = torch.cat([prompt_ids, target_ids], dim=1)[:, : self.max_length]
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        prompt_length = min(prompt_ids.shape[1], labels.shape[1])
        labels[:, :prompt_length] = -100
        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "labels": labels.to(self.device),
        }

    def _prompt_target_from_messages(self, messages_value: Any) -> PromptTarget:
        system_message, normalized_messages = normalize_chat_messages(messages_value, strict=True)
        if not normalized_messages:
            raise ValueError("messages must contain at least one non-system message")
        last = normalized_messages[-1]
        if last["role"] != "assistant":
            raise ValueError("messages must end with an assistant response")
        prompt_parts: list[str] = []
        if system_message is not None:
            prompt_parts.append(f"System: {system_message}")
        for message in normalized_messages[:-1]:
            role = "User" if message["role"] == "user" else "Assistant"
            prompt_parts.append(f"{role}: {message['content']}")
        prompt_parts.append("Assistant:")
        return PromptTarget(prompt="\n\n".join(prompt_parts), target=last["content"])

    def _ensure_model_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype: Any | None = None
        if self.torch_dtype is not None:
            dtype = getattr(torch, self.torch_dtype)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
            torch_dtype=dtype,
        )
        if self.requested_device is not None:
            device = torch.device(self.requested_device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.zero_grad(set_to_none=True)
        self._tokenizer = tokenizer
        self._model = model


def _extract_terminal_answer(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    marker_match = re.search(r"####\s*(.+)$", stripped, flags=re.MULTILINE)
    if marker_match:
        return marker_match.group(1).strip()
    last_line = stripped.splitlines()[-1].strip()
    return last_line


def _token_f1(reference: str, candidate: str) -> float:
    reference_tokens = _normalize_tokens(reference)
    candidate_tokens = _normalize_tokens(candidate)
    if not reference_tokens or not candidate_tokens:
        return 0.0
    common = 0
    candidate_bag = candidate_tokens.copy()
    for token in reference_tokens:
        if token in candidate_bag:
            candidate_bag.remove(token)
            common += 1
    if common == 0:
        return 0.0
    precision = common / len(candidate_tokens)
    recall = common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _normalize_tokens(text: str) -> list[str]:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower())
    return [token for token in cleaned.split() if token]


def _l2_normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


def _rademacher_project(vector: torch.Tensor, dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    projection = torch.randint(0, 2, (dim, vector.numel()), generator=generator, dtype=torch.int8)
    projection = projection.float().mul_(2).sub_(1)
    return projection @ vector / math.sqrt(dim)
