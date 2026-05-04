"""LLM client integration tests for the planner.

Covers:

* :func:`default_llm_client` env-var resolution and missing-key error.
* ``plan_eval(llm_client="auto")`` constructs the default client.
* Wiring with the real :class:`OpenAICompatLLMClient` via mocked HTTP.
* Optional live integration with DashScope (skipped unless
  ``DASHSCOPE_API_KEY`` is set), marked ``task_eval`` so CI can opt out.
"""
from __future__ import annotations

import json
import os
from typing import Any

import pytest

from dataos.eval.planner import (
    EvalPlanItem,
    default_llm_client,
    llm_intent,
    plan_eval,
)
from dataos.llm import OpenAICompatLLMClient


class TestDefaultLLMClientFactory:
    def test_constructs_openai_compat_with_dashscope_env(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test-1")
        monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://example.invalid/v1")
        monkeypatch.setenv("DASHSCOPE_MODEL", "deepseek-test")
        for stale in ("STARTEAM_API_KEY", "STARTEAM_BASE_URL", "STARTEAM_MODEL"):
            monkeypatch.delenv(stale, raising=False)

        client = default_llm_client()

        assert isinstance(client, OpenAICompatLLMClient)
        assert client.api_key == "sk-test-1"
        assert client.base_url == "https://example.invalid/v1"
        assert client.model == "deepseek-test"

    def test_falls_back_to_legacy_starteam_env(self, monkeypatch):
        for var in ("DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL", "DASHSCOPE_MODEL"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("STARTEAM_API_KEY", "legacy-key")
        monkeypatch.setenv("STARTEAM_BASE_URL", "https://legacy.invalid/v1")
        monkeypatch.setenv("STARTEAM_MODEL", "glm-test")

        client = default_llm_client()

        assert client.api_key == "legacy-key"
        assert client.base_url == "https://legacy.invalid/v1"
        assert client.model == "glm-test"

    def test_uses_dashscope_defaults_when_only_api_key_set(self, monkeypatch):
        for var in (
            "DASHSCOPE_BASE_URL",
            "DASHSCOPE_MODEL",
            "STARTEAM_API_KEY",
            "STARTEAM_BASE_URL",
            "STARTEAM_MODEL",
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-1")

        client = default_llm_client()

        assert client.api_key == "sk-1"
        assert "dashscope" in client.base_url.lower()
        assert client.model

    def test_missing_api_key_raises(self, monkeypatch):
        for var in ("DASHSCOPE_API_KEY", "STARTEAM_API_KEY"):
            monkeypatch.delenv(var, raising=False)

        with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
            default_llm_client()


class TestPlanEvalAutoLLM:
    def test_auto_flag_invokes_default_factory(self, monkeypatch):
        constructed: list[Any] = []

        class _RecordingLLM:
            def generate_json(self, system, user, temperature=0.0):  # noqa: D401
                constructed.append((system, user, temperature))
                return {
                    "topics": ["math"],
                    "skills": [],
                    "allow_unsafe_code": False,
                    "rationale": "auto-built client used",
                }

        monkeypatch.setattr(
            "dataos.eval.planner.llm.default_llm_client",
            lambda: _RecordingLLM(),
        )

        plan = plan_eval("evaluate math reasoning", llm_client="auto", max_tasks=2)

        assert constructed, "default_llm_client should have been called"
        assert any(item.task == "gsm8k" for item in plan)

    def test_true_flag_also_invokes_default_factory(self, monkeypatch):
        called = {"n": 0}

        class _DummyLLM:
            def generate_json(self, system, user, temperature=0.0):
                called["n"] += 1
                return {"topics": ["instruction_following"], "skills": [], "allow_unsafe_code": False}

        monkeypatch.setattr(
            "dataos.eval.planner.llm.default_llm_client",
            lambda: _DummyLLM(),
        )

        plan = plan_eval("anything", llm_client=True, max_tasks=1)

        assert called["n"] == 1
        assert plan[0].task == "ifeval"


class TestOpenAICompatLLMClientWiring:
    """End-to-end wiring with the real client class but mocked HTTP."""

    def test_plan_eval_with_real_client_shape(self, monkeypatch):
        captured_requests: list[dict[str, Any]] = []
        payload = {
            "topics": ["math", "code"],
            "skills": ["multi_step", "python"],
            "allow_unsafe_code": True,
            "rationale": "math and python from real client",
        }

        class _StubResponse:
            status_code = 200
            text = ""

            def json(self):
                return {
                    "choices": [
                        {"message": {"content": json.dumps(payload)}}
                    ]
                }

        class _StubHttpxClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def post(self, url, headers=None, json=None):  # noqa: A002
                captured_requests.append(
                    {"url": url, "headers": headers, "json": json}
                )
                return _StubResponse()

        monkeypatch.setattr(
            "dataos.llm.openai_compat.httpx.Client",
            _StubHttpxClient,
        )

        client = OpenAICompatLLMClient(
            api_key="sk-real-test",
            base_url="https://api.example.invalid/v1",
            model="qwen-test",
        )
        plan = plan_eval("write Python code and reason about math", llm_client=client, max_tasks=4)

        assert captured_requests, "httpx.Client.post should have been called"
        sent = captured_requests[0]
        assert sent["url"].endswith("/chat/completions")
        assert sent["headers"]["Authorization"] == "Bearer sk-real-test"
        assert sent["json"]["model"] == "qwen-test"
        sent_messages = sent["json"]["messages"]
        assert sent_messages[0]["role"] == "system"
        assert "evaluation planner" in sent_messages[0]["content"].lower()
        assert sent_messages[1]["role"] == "user"
        assert "Allowed canonical topics" in sent_messages[1]["content"]

        tasks = {item.task for item in plan}
        assert "gsm8k" in tasks  # math came back from LLM
        assert "humaneval" in tasks  # allow_unsafe_code propagated through resolver

        humaneval = next(item for item in plan if item.task == "humaneval")
        assert humaneval.confirm_run_unsafe_code is True


@pytest.mark.task_eval
@pytest.mark.skipif(
    not (os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("STARTEAM_API_KEY")),
    reason="no DASHSCOPE/STARTEAM API key in env; skipping live LLM call",
)
def test_plan_eval_live_llm_dashscope():
    """Optional live test against the real DashScope endpoint.

    Run manually with:
        DASHSCOPE_API_KEY=... uv run pytest -m task_eval tests/test_eval_planner_llm.py::test_plan_eval_live_llm_dashscope -v
    """
    plan = plan_eval(
        "评估这个模型的数学推理和编程能力",
        llm_client="auto",
        max_tasks=4,
    )
    assert plan, "live planner should return at least one task"
    tasks = {item.task for item in plan}
    # The model should at least pick up on the math signal; humaneval depends on
    # the LLM returning allow_unsafe_code=true.
    assert tasks & {"gsm8k", "bbh_cot_zeroshot_logical_deduction_three_objects"}
    for item in plan:
        assert isinstance(item, EvalPlanItem)
        assert item.headline_metric
