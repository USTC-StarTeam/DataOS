"""Provider = a thin spec that turns a run into a stream of neutral Events.

Two shapes:
  - ``CliSpec``: the common case — a code-agent CLI driven via ``_cli.run_cli``.
    A concrete provider supplies ``bin`` + ``build_argv`` + ``parse_line``.
  - ``ProviderSpec``: the bare interface (override ``stream`` directly, e.g. the
    fake test provider).
"""

from __future__ import annotations

import os
from typing import AsyncIterator, Iterable

from .._cli import run_cli
from .._home import home_env
from ..errors import UnknownProvider
from ..types import AgentConfig, Event, Result


class ProviderSpec:
    name: str = ""

    async def stream(self, config: AgentConfig, prompt: str) -> AsyncIterator[Event]:
        raise NotImplementedError
        yield  # pragma: no cover  (makes this an async generator)


class CliSpec(ProviderSpec):
    """A code-agent CLI backend. Subclasses set ``bin`` and implement the two hooks."""

    bin: str = ""

    # The egress this CLI's endpoint is reachable through. Declared PER PROVIDER —
    # claude and codex talk to different vendors and need not share an egress.
    # None -> leave the inherited proxy env alone.
    proxy: str | None = None

    def build_argv(self, config: AgentConfig, prompt: str) -> list[str]:
        raise NotImplementedError

    def parse_line(self, raw: dict, acc: Result) -> Iterable[Event]:
        raise NotImplementedError

    def finalize(self, acc: Result, stderr: str) -> None:
        """Optional: enrich the Result from stderr after the process exits
        (e.g. a session id some CLIs print there). Default: no-op."""

    def env(self, config: AgentConfig) -> dict | None:
        """The subprocess env. Precedence, lowest first: inherited ``os.environ`` ->
        this provider's ``proxy`` -> the isolated config root (``config.home``) ->
        ``extra["env"]`` (the caller always wins, e.g. a different egress per host).

        The proxy OVERRIDES an inherited value on purpose: a stale or wrong ``*_PROXY``
        exported in the shell would otherwise silently make the CLI unreachable."""
        overrides: dict[str, str] = {}
        if self.proxy:
            overrides.update(
                dict.fromkeys(("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"), self.proxy)
            )
        if config.home:
            overrides.update(home_env(self.name, config.home))
        overrides.update(config.extra.get("env") or {})
        return {**os.environ, **overrides} if overrides else None

    async def stream(self, config: AgentConfig, prompt: str) -> AsyncIterator[Event]:
        async for ev in run_cli(
            self.bin,
            self.build_argv(config, prompt),
            cwd=config.workspace,
            env=self.env(config),
            parse_line=self.parse_line,
            finalize=self.finalize,
            idle_timeout=config.idle_timeout,  # stall watchdog (not the wall clock)
        ):
            yield ev


_REGISTRY: dict[str, ProviderSpec] = {}


def register(spec: ProviderSpec) -> None:
    _REGISTRY[spec.name] = spec


def get(name: str) -> ProviderSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise UnknownProvider(f"unknown provider {name!r}; registered: {sorted(_REGISTRY)}")


def providers() -> list[str]:
    return sorted(_REGISTRY)
