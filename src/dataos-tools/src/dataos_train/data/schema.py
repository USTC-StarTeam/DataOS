"""Canonical record shapes (documentation + light typing).

Offline operators run upstream and emit one of these; backend adapters convert to native.
- SFT canonical: alpaca ``{instruction,input,output[,system,history]}`` or OpenAI ``{messages}``
  or sharegpt ``{conversations}``.
- RL canonical: ``{prompt, ground_truth, data_source}`` (consumed by the verl backend).
"""

from __future__ import annotations

from typing import Any, TypedDict


class AlpacaRecord(TypedDict, total=False):
    instruction: str
    input: str
    output: str
    system: str
    history: list[list[str]]


class MessagesRecord(TypedDict, total=False):
    messages: list[dict[str, Any]]


class RLRecord(TypedDict, total=False):
    prompt: str
    ground_truth: str
    data_source: str


SFT_FORMATS = ("alpaca", "messages", "sharegpt", "prompt")
