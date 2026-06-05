from __future__ import annotations

import httpx

from dataos.eval.types import (
    AdapterCallContext,
    EvalRequest,
    ModelLike as ModelArtifact,
    ModelResponse,
)


class OpenAICompatibleAdapter:
    name = "openai_compatible"

    def generate_until(
        self,
        model_artifact: ModelArtifact,
        requests: list[EvalRequest],
        config: AdapterCallContext,
    ) -> list[ModelResponse]:
        adapter_config = config.adapter.config if config.adapter is not None else {}
        base_url = str(
            adapter_config.get("base_url")
            or model_artifact.meta.get("api_base_url", "")
        ).rstrip("/")
        if not base_url:
            raise ValueError(
                "openai_compatible adapter requires 'base_url' in adapter config "
                "or 'api_base_url' in model artifact meta"
            )
        model_name = str(
            adapter_config.get("model")
            or model_artifact.meta.get("model_name")
            or model_artifact.meta.get("base_model", "")
        )
        if not model_name:
            raise ValueError(
                "openai_compatible adapter requires 'model' in adapter config "
                "or 'model_name'/'base_model' in model artifact meta"
            )
        api_key = str(adapter_config.get("api_key", "EMPTY"))
        request_config = config.request
        max_tokens = request_config.max_new_tokens if request_config is not None else 256
        temperature = 0.0 if (request_config is None or not request_config.do_sample) else 1.0
        batch_size = request_config.batch_size if request_config is not None else 8

        url = f"{base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        timeout = float(adapter_config.get("timeout", 120))

        responses: list[ModelResponse] = []
        with httpx.Client(timeout=timeout) as client:
            for start in range(0, len(requests), batch_size):
                batch = requests[start : start + batch_size]
                for request in batch:
                    body = {
                        "model": model_name,
                        "messages": [{"role": "user", "content": request.prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    }
                    resp = client.post(url, json=body, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    choice = data["choices"][0]
                    text = choice.get("message", {}).get("content", "")
                    responses.append(
                        ModelResponse(
                            sample_id=request.sample_id,
                            index=request.index,
                            request_type=request.request_type,
                            text=text.strip(),
                            finish_reason=choice.get("finish_reason"),
                            raw=data,
                        )
                    )
        return responses
