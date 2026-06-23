"""Redact secret-looking values before anything is persisted.

Hard rule: tokens / API keys / private keys must never land in ``plan.json`` /
``report.json`` / ``manifest.json`` / ``{kind}.json`` / ``project.json`` or any other
on-disk receipt. The proper channel for secrets is the **process env** of the command
that needs them (e.g. ``HF_TOKEN=... python run_one.py ...``) — this module is the
safety net for configs that violate that rule, not an endorsement of putting secrets
in configs. In-memory objects keep their raw values; only persisted copies are scrubbed.
"""

from __future__ import annotations

import re
from typing import Any

REDACTED = "<redacted>"

# Single words that mark a key as secret ("hub_token", "use_auth_token", "client_secret" ...)
_SECRET_WORDS = {
    "token",
    "secret",
    "secrets",
    "password",
    "passwd",
    "credential",
    "credentials",
    "apikey",
    "authorization",
}
# Adjacent word pairs that mark a key as secret ("api_key", "private_key" ... but NOT
# "query_key" / "answer_key" / "lora_key", and "max_tokens" stays untouched).
_SECRET_PAIRS = {("api", "key"), ("private", "key"), ("access", "key"), ("secret", "key")}

# Literal token shapes scrubbed inside any string (HF / OpenAI-style keys pasted into prose).
_LITERAL_RES = (
    re.compile(r"\bhf_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
)
_KV_RE = re.compile(r"(?i)\b([a-z0-9_]+)\s*=\s*([^,;\s'\"]+)")


def _is_secret_key(key: str) -> bool:
    words = [w for w in re.split(r"[^0-9a-zA-Z]+", str(key).lower()) if w]
    if any(word in _SECRET_WORDS for word in words):
        return True
    return any(pair in _SECRET_PAIRS for pair in zip(words, words[1:]))


def _redact_string(value: str) -> str:
    def _sub(match: re.Match[str]) -> str:
        return f"{match.group(1)}={REDACTED}" if _is_secret_key(match.group(1)) else match.group(0)

    value = _KV_RE.sub(_sub, value)  # "pretrained=X,token=hf_abc" -> "pretrained=X,token=<redacted>"
    for literal in _LITERAL_RES:
        value = literal.sub(REDACTED, value)
    return value


def redact(value: Any) -> Any:
    """Deep-copy ``value`` with secret-looking parts replaced by ``<redacted>``."""
    if isinstance(value, dict):
        return {
            key: (REDACTED if _is_secret_key(key) else redact(val)) for key, val in value.items()
        }
    if isinstance(value, (list, tuple)):
        items = [redact(item) for item in value]
        return items if isinstance(value, list) else tuple(items)
    if isinstance(value, str):
        return _redact_string(value)
    return value
