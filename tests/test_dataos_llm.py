from __future__ import annotations

import unittest

from dataos.llm import (
    GenerateRequest,
    GenerateResponse,
    LLM,
    LLMError,
    LLMMessage,
    OpenAICompatibleLLM,
    TokenUsage,
    VLLMOpenAIAdapter,
    build_messages,
    create_llm,
)


class RewriteOperator:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def rewrite(self, text: str) -> GenerateResponse:
        request = GenerateRequest(
            model="demo-model",
            messages=build_messages(f"Rewrite: {text}", system_prompt="You are a careful editor."),
            metadata={"operator": "rewrite_with_llm"},
        )
        return self.llm.generate(request)


class DataOSLLMTests(unittest.TestCase):
    def test_generate_request_copies_mutable_fields(self) -> None:
        messages = [LLMMessage(role="user", content="hello")]
        stop = ["END"]
        metadata = {"task": "rewrite"}
        extra = {"seed": 7}

        request = GenerateRequest(
            model="demo-model",
            messages=messages,
            stop=stop,
            metadata=metadata,
            extra=extra,
        )

        messages.append(LLMMessage(role="assistant", content="mutated"))
        stop.append("STOP")
        metadata["task"] = "changed"
        extra["seed"] = 9

        self.assertEqual(len(request.messages), 1)
        self.assertEqual(request.stop, ["END"])
        self.assertEqual(request.metadata, {"task": "rewrite"})
        self.assertEqual(request.extra, {"seed": 7})

    def test_openai_adapter_parses_response_usage_and_metadata(self) -> None:
        seen: dict[str, object] = {}

        def fake_transport(url: str, payload: dict, headers: dict, timeout_seconds: float) -> dict:
            seen["url"] = url
            seen["payload"] = payload
            seen["headers"] = headers
            seen["timeout_seconds"] = timeout_seconds
            return {
                "id": "resp_123",
                "model": "demo-model-resolved",
                "system_fingerprint": "fp_1",
                "choices": [
                    {
                        "message": {
                            "content": [{"type": "text", "text": "rewritten output"}],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
            }

        llm = OpenAICompatibleLLM(
            base_url="https://example.com",
            api_key="token",
            transport=fake_transport,
            timeout_seconds=12.5,
        )
        response = llm.generate(
            GenerateRequest(
                model="demo-model",
                messages=build_messages("hello"),
                temperature=0.2,
                max_tokens=128,
                stop=["END"],
                metadata={"trace_id": "trace-1"},
                extra={"seed": 42},
            )
        )

        self.assertIsInstance(llm, LLM)
        self.assertEqual(seen["url"], "https://example.com/v1/chat/completions")
        self.assertEqual(
            seen["headers"],
            {"Content-Type": "application/json", "Authorization": "Bearer token"},
        )
        self.assertEqual(seen["timeout_seconds"], 12.5)
        self.assertEqual(
            seen["payload"],
            {
                "model": "demo-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
                "temperature": 0.2,
                "max_tokens": 128,
                "stop": ["END"],
                "seed": 42,
            },
        )
        self.assertEqual(response.provider, "openai-compatible")
        self.assertEqual(response.model, "demo-model-resolved")
        self.assertEqual(response.text, "rewritten output")
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(
            response.usage,
            TokenUsage(prompt_tokens=10, completion_tokens=6, total_tokens=16),
        )
        self.assertEqual(
            response.metadata,
            {
                "base_url": "https://example.com",
                "response_id": "resp_123",
                "system_fingerprint": "fp_1",
                "request_metadata": {"trace_id": "trace-1"},
            },
        )
        self.assertGreaterEqual(response.latency_ms or 0.0, 0.0)

    def test_openai_adapter_allows_missing_api_key_for_local_compatible_endpoints(self) -> None:
        def fake_transport(url: str, payload: dict, headers: dict, timeout_seconds: float) -> dict:
            self.assertEqual(headers, {"Content-Type": "application/json"})
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

        llm = OpenAICompatibleLLM(
            base_url="https://localhost:8000",
            api_key=None,
            transport=fake_transport,
        )
        response = llm.generate(GenerateRequest(model="demo", messages=build_messages("ping")))
        self.assertEqual(response.text, "ok")

    def test_factory_and_operator_can_swap_openai_and_vllm_adapters(self) -> None:
        def fake_transport(url: str, payload: dict, headers: dict, timeout_seconds: float) -> dict:
            return {
                "choices": [
                    {
                        "message": {"content": f"{payload['messages'][-1]['content']} -> done"},
                        "finish_reason": "stop",
                    }
                ]
            }

        openai_operator = RewriteOperator(
            create_llm("openai", base_url="https://openai.example.com", transport=fake_transport)
        )
        vllm_operator = RewriteOperator(
            create_llm("vllm", base_url="https://vllm.example.com", transport=fake_transport)
        )

        openai_response = openai_operator.rewrite("sample")
        vllm_response = vllm_operator.rewrite("sample")

        self.assertEqual(openai_response.provider, "openai-compatible")
        self.assertEqual(vllm_response.provider, "vllm")
        self.assertEqual(openai_response.text, "Rewrite: sample -> done")
        self.assertEqual(vllm_response.text, "Rewrite: sample -> done")

    def test_factory_rejects_unknown_provider(self) -> None:
        with self.assertRaises(ValueError):
            create_llm("anthropic")

    def test_adapter_requires_base_url(self) -> None:
        with self.assertRaises(LLMError):
            OpenAICompatibleLLM(base_url=None, api_key="token")

    def test_vllm_adapter_is_openai_compatible_wrapper(self) -> None:
        llm = VLLMOpenAIAdapter(base_url="https://vllm.example.com", api_key="")
        self.assertIsInstance(llm, OpenAICompatibleLLM)
        self.assertEqual(llm.provider_name, "vllm")
