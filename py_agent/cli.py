"""CLI execution helpers."""

from __future__ import annotations

import os
from pathlib import Path
import sys

from .args import parse_args
from .config import VERSION
from .mcp.registry import McpRegistry, McpRegistryError
from .modes.print_mode import run_print_mode
from .providers import DEFAULT_MODEL, OpenAICompatibleProvider
from .providers.base import ProviderError
from .session import SessionStore, default_session_path


def run(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.version:
        print(VERSION)
        return 0

    if args.print_mode:
        if not args.messages:
            print("error: missing prompt for --print mode", file=sys.stderr)
            return 2
        if args.mcp_servers and not args.mcp_config:
            print("error: --mcp-server requires --mcp-config", file=sys.stderr)
            return 2

        session_store = _resolve_session_store(
            cwd=os.getcwd(),
            no_session=args.no_session,
            session_path_arg=args.session,
        )
        initial_messages = session_store.load_messages() if (session_store and args.continue_session) else None

        prompt = " ".join(args.messages).strip()
        model = args.model or DEFAULT_MODEL
        mcp_registry: McpRegistry | None = None
        try:
            extra_tools = {}
            if args.mcp_config:
                config_path = Path(args.mcp_config)
                if not config_path.is_absolute():
                    config_path = Path(os.getcwd()) / config_path
                mcp_registry = McpRegistry(config_path=config_path.resolve(), enabled_servers=args.mcp_servers)
                mcp_registry.start()
                extra_tools.update(mcp_registry.build_tools())
            provider = OpenAICompatibleProvider()
            result = run_print_mode(
                provider=provider,
                prompt=prompt,
                model=model,
                mode=args.mode,
                cwd=os.getcwd(),
                initial_messages=initial_messages,
                extra_tools=extra_tools,
            )
        except (ProviderError, McpRegistryError) as exc:
            print(f"provider error: {exc}", file=sys.stderr)
            return 1
        finally:
            if mcp_registry is not None:
                mcp_registry.close()

        if session_store:
            session_store.append_turn(messages=result.turn_messages, events=result.events)

        print(result.output)
        return 0

    return 0


def _resolve_session_store(cwd: str, no_session: bool, session_path_arg: str | None) -> SessionStore | None:
    if no_session:
        return None
    if session_path_arg:
        path = Path(session_path_arg)
        if not path.is_absolute():
            path = Path(cwd) / path
        return SessionStore(path.resolve())
    return SessionStore(default_session_path(cwd))
