"""Real CLI smoke — needs the user logged into claude / kimi / qwen.

Run with:  pytest -m integration -s
Not run by default (marked integration)."""

import pytest
from pydantic import BaseModel

from dataos_agent import Result, query

PROVIDERS = ["claude", "kimi", "qwen"]
_AUTH_HINTS = ("401", "invalid access token", "token expired", "unauthorized", "please run")


@pytest.mark.integration
@pytest.mark.parametrize("provider", PROVIDERS)
def test_reply_ok(provider):
    r = query(
        "Reply with exactly the two letters: OK. Nothing else.",
        provider=provider, tools=False, extra={"timeout": 180},
    )
    print(f"\n[{provider}] status={r.status} text={r.text!r} "
          f"tokens={r.usage.total_tokens} cost=${r.usage.cost_usd:.4f} session={r.session_id}")
    assert isinstance(r, Result)
    assert r.status in ("done", "max_steps", "timeout", "disconnected", "error", "refused")
    if any(h in r.text.lower() for h in _AUTH_HINTS):
        pytest.skip(f"{provider}: not authenticated ({r.text[:80]})")
    assert r.status == "done"
    assert r.text.strip() != ""


class OkPayload(BaseModel):
    ok: bool
    n: int


@pytest.mark.integration
def test_structured_output_claude():
    r = query(
        'Output ONLY this JSON and nothing else: {"ok": true, "n": 42}',
        provider="claude", tools=False, schema=OkPayload, extra={"timeout": 180},
    )
    print(f"\n[claude structured] data={r.data}")
    assert isinstance(r.data, OkPayload)
    assert r.data.ok is True and r.data.n == 42
