import pytest
from pydantic import ValidationError

from dataos_agent.types import AgentConfig, Event, Result, Step, Usage


def test_result_str_is_text():
    r = Result(text="hello")
    assert str(r) == "hello"
    assert f"{r}" == "hello"


def test_result_defaults():
    r = Result()
    assert r.text == ""
    assert r.status == "done"
    assert r.files == {} and r.steps == []
    assert r.data is None and r.error is None
    assert r.usage.total_tokens == 0


def test_usage_add_accumulates_and_derives_total():
    u = Usage()
    u.add(input=10, output=5)
    u.add(input=1, output=2, cost=0.01)
    assert u.input_tokens == 11
    assert u.output_tokens == 7
    assert u.total_tokens == 18          # derived when total not given
    assert u.cost_usd == pytest.approx(0.01)


def test_usage_add_explicit_total():
    u = Usage()
    u.add(input=3, output=4, total=100)
    assert u.total_tokens == 100


def test_event_shape():
    e = Event(type="tool_use", name="Bash", args={"cmd": "ls"})
    assert e.type == "tool_use" and e.name == "Bash" and e.args == {"cmd": "ls"}


def test_config_requires_provider():
    with pytest.raises(ValidationError):
        AgentConfig(model="x")  # provider missing


def test_config_forbids_unknown_field():
    with pytest.raises(ValidationError):
        AgentConfig(provider="claude", typo=1)


def test_config_tools_bool_or_list():
    assert AgentConfig(provider="claude", tools=False).tools is False
    assert AgentConfig(provider="claude", tools=["Bash", "Read"]).tools == ["Bash", "Read"]
    assert AgentConfig(provider="claude").tools is True  # default


def test_config_defaults():
    c = AgentConfig(provider="kimi")
    assert c.model is None and c.max_steps is None
    assert c.permission == "auto"
    assert c.extra == {}
