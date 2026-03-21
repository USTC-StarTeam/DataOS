from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any
from urllib import error, request

from dataos.llm.client import LLMClient


@dataclass(slots=True)
class OpenAICompatibleClient(LLMClient):
    base_url: str
    api_key: str
    model: str
    timeout: float = 60.0
    max_retries: int = 2

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                req = request.Request(endpoint, data=body, headers=headers, method="POST")
                with request.urlopen(req, timeout=self.timeout) as response:
                    parsed = json.loads(response.read().decode("utf-8"))
                content = self._extract_content(parsed)
                return json.loads(content)
            except (error.HTTPError, error.URLError, json.JSONDecodeError, KeyError, IndexError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2**attempt, 4))
        raise RuntimeError("OpenAI-compatible JSON generation failed") from last_error

    @staticmethod
    def _extract_content(response_json: dict[str, Any]) -> str:
        message = response_json["choices"][0]["message"]["content"]
        if isinstance(message, str):
            return message
        if isinstance(message, list):
            parts: list[str] = []
            for item in message:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "".join(parts)
        raise ValueError("Unsupported OpenAI-compatible response content format")
