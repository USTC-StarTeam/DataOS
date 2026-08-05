import pytest
from pydantic import BaseModel

from dataos_agent import OutputParseError, Result, UnknownProvider, list_providers, query


class Plan(BaseModel):
    name: str
    steps: int


def test_fake_is_registered():
    assert "fake" in list_providers()


def test_query_returns_result():
    r = query("hi", provider="fake")
    assert isinstance(r, Result)
    assert str(r) == "hello from fake"
    assert r.status == "done"
    assert r.usage.total_tokens == 8
    assert r.session_id == "fake-session"


def test_tools_off_no_steps():
    r = query("hi", provider="fake", tools=False)
    assert r.steps == []


def test_tools_on_has_step():
    r = query("hi", provider="fake", tools=True)
    assert [s.tool for s in r.steps] == ["Bash"]


def test_schema_gives_validated_data():
    r = query("plan it", provider="fake", schema=Plan,
              extra={"emit": '{"name": "exp", "steps": 4}'})
    assert isinstance(r.data, Plan)
    assert r.data.name == "exp" and r.data.steps == 4


def test_schema_parse_failure_raises():
    with pytest.raises(OutputParseError):
        query("x", provider="fake", schema=Plan, extra={"emit": "not json"})


def test_forced_disconnect_status():
    # self-healing is on by default; net_retries=0 surfaces the fault immediately
    r = query("x", provider="fake", net_retries=0, extra={"status": "disconnected"})
    assert r.status == "disconnected"


def test_unknown_provider():
    with pytest.raises(UnknownProvider):
        query("x", provider="nope")
