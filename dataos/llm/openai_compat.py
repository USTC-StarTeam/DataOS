from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

from dataos.llm.client import LLMClient


DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "deepseek-v4-flash"


def _extract_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(match.group(0))


@dataclass(slots=True)
class OpenAICompatLLMClient(LLMClient):
    """Direct HTTP client for an OpenAI-compatible chat completions endpoint.

    Defaults target Alibaba Cloud DashScope's compatible-mode endpoint
    (Bailian); set ``base_url`` to point at any other OpenAI-compatible
    backend.
    """

    api_key: str
    base_url: str = DASHSCOPE_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: float = 120.0

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Return exactly one JSON object and nothing else.\n\n" + user_prompt
                ),
            },
        ]
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "stream": False,
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, headers=headers, json=body)
        if response.status_code >= 400:
            raise RuntimeError(
                f"LLM API error {response.status_code}: {response.text or '<empty body>'}"
            )
        data = response.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LLM response shape: {data!r}") from exc
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("LLM API returned empty assistant text")
        return _extract_json_object(text)
