from pydantic import BaseModel

from dataos_agent import Client


class Plan(BaseModel):
    name: str
    steps: int


async def test_client_streams_events_ending_in_done():
    async with Client(provider="fake") as c:
        types = [ev.type async for ev in c.stream("hi")]
    assert types[-1] == "done"
    assert "text" in types
    assert c.result is not None and c.result.text == "hello from fake"
    assert c.session_id == "fake-session"


async def test_client_run_returns_result():
    async with Client(provider="fake", tools=False) as c:
        r = await c.run("hi")
    assert r.status == "done" and r.steps == []


async def test_client_multiturn_keeps_session():
    async with Client(provider="fake") as c:
        await c.run("turn 1")
        first = c.session_id
        await c.run("turn 2")
    assert first == "fake-session" and c.session_id == "fake-session"


async def test_client_schema():
    async with Client(provider="fake", schema=Plan,
                      extra={"emit": '{"name": "z", "steps": 9}'}) as c:
        r = await c.run("plan")
    assert isinstance(r.data, Plan) and r.data.steps == 9


async def test_client_interrupt_is_safe():
    c = Client(provider="fake")
    await c.interrupt()  # nothing in flight — must not raise
