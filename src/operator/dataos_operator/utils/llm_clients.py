"""Common LLM client helpers for evaluator operators."""

from __future__ import annotations

import json
from datetime import datetime, timezone
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any


def call_llm_json(
    *,
    backend: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model_params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Call an LLM backend and parse the JSON object from its text output."""

    params = dict(model_params or {})
    if backend == "openai":
        raw_output = _call_openai_http(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
            model_params=params,
        )
    elif backend == "openai_compatible":
        resolved_base_url = base_url or os.environ.get("DATAOS_OPERATOR_BASE_URL")
        if not resolved_base_url:
            raise ValueError("openai_compatible backend requires base_url or DATAOS_OPERATOR_BASE_URL")
        raw_output = _call_openai_http(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_key=api_key,
            base_url=resolved_base_url,
            model_params=params,
        )
    elif backend == "hf":
        raw_output = _call_hf_local(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_params=params,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend!r}")

    return _extract_json_object(raw_output), raw_output


def _call_openai_http(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str | None,
    base_url: str,
    model_params: dict[str, Any],
) -> str:
    resolved_api_key = (
        api_key or os.environ.get("DATAOS_OPERATOR_API_KEY") or os.environ.get("OPENAI_API_KEY")
    )
    if not resolved_api_key:
        raise ValueError("Missing API key for OpenAI-compatible backend")

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
    }
    payload.update(model_params)

    endpoint = _chat_completions_endpoint(base_url)
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {resolved_api_key}",
        },
        method="POST",
    )
    last_error: Exception | None = None
    body = ""
    for attempt in range(12):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read().decode("utf-8")
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"LLM request failed with HTTP {exc.code}: {detail}")
            if attempt < 11 and exc.code in {429, 500, 502, 503, 504}:
                if exc.code == 429:
                    time.sleep(_retry_delay_from_detail(detail))
                else:
                    time.sleep(min(2 * (attempt + 1), 20))
                continue
            raise last_error from exc
        except urllib.error.URLError as exc:
            last_error = RuntimeError(f"LLM request failed: {exc.reason}")
            if attempt < 11:
                time.sleep(2)
                continue
            raise last_error from exc
    if not body and last_error is not None:
        raise last_error

    parsed = json.loads(body)
    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM response is missing choices")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("LLM response choice is not an object")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("LLM response choice is missing message")
    content = _coerce_message_content(message)
    if not content:
        raise ValueError("LLM response message content is empty")
    return content


def _call_hf_local(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    model_params: dict[str, Any],
) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_kwargs = {
        key: value
        for key, value in model_params.items()
        if key in {"trust_remote_code", "revision", "use_fast", "token", "local_files_only"}
    }
    generation_kwargs = {
        "max_new_tokens": int(model_params.get("max_new_tokens", 256)),
        "do_sample": bool(model_params.get("do_sample", False)),
        "temperature": float(model_params.get("temperature", 0.0)),
    }
    model_kwargs = {
        key: value
        for key, value in model_params.items()
        if key in {
            "device_map",
            "torch_dtype",
            "trust_remote_code",
            "revision",
            "token",
            "local_files_only",
            "attn_implementation",
        }
    }

    tokenizer = AutoTokenizer.from_pretrained(model, **tokenizer_kwargs)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
    if "device_map" not in model_kwargs:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        llm.to(device)
    llm.eval()

    prompt = f"{system_prompt}\n\n{user_prompt}"
    encoded = tokenizer(prompt, return_tensors="pt")
    device = next(llm.parameters()).device
    with torch.no_grad():
        output_ids = llm.generate(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=(
                encoded["attention_mask"].to(device)
                if encoded.get("attention_mask") is not None
                else None
            ),
            pad_token_id=tokenizer.eos_token_id,
            **generation_kwargs,
        )
    generated_ids = output_ids[0][encoded["input_ids"].shape[1] :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if not text:
        raise ValueError("Local HF model produced empty output")
    return text


def _chat_completions_endpoint(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def _retry_delay_from_detail(detail: str) -> float:
    match = re.search(r"Limit resets at:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC", detail)
    if not match:
        return 2.0
    try:
        reset_at = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return 2.0
    delay = (reset_at - datetime.now(timezone.utc)).total_seconds()
    return max(2.0, min(delay + 1.0, 120.0))


def _coerce_message_content(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"].strip())
        joined = "\n".join(part for part in parts if part)
        if joined:
            return joined
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    provider_fields = message.get("provider_specific_fields")
    if isinstance(provider_fields, dict):
        provider_reasoning = provider_fields.get("reasoning_content")
        if isinstance(provider_reasoning, str) and provider_reasoning.strip():
            return provider_reasoning.strip()
    return ""


def _extract_json_object(raw_output: str) -> dict[str, Any]:
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    candidates = [cleaned]
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        candidates.append(cleaned[first_brace : last_brace + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            raise ValueError("LLM JSON output must be an object")
        return parsed
    raise ValueError(f"Failed to parse JSON object from LLM output: {raw_output!r}")
