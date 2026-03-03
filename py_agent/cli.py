"""CLI execution helpers."""

from __future__ import annotations

import os
from pathlib import Path
import sys

from .args import parse_args
from .config import VERSION
from .debug_log import DebugLogger, resolve_debug_log_path
from .mcp.registry import McpRegistry, McpRegistryError
from .modes.print_mode import run_print_mode
from .modes.tui_mode import run_tui_mode
from .providers import DEFAULT_MODEL, OpenAICompatibleProvider
from .providers.base import ProviderError
from .session import SessionStore, default_session_path


def run(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    logger: DebugLogger | None = None
    if args.debug:
        log_path = resolve_debug_log_path(cwd=os.getcwd(), log_path_arg=args.log_path)
        logger = DebugLogger(log_path, level=args.debug)
        logger.info(f"debug logging enabled path={log_path}")
    elif args.log_path:
        print("error: --log-path requires --debug", file=sys.stderr)
        return 2

    if logger:
        logger.info(f"args={args}")

    if args.version:
        print(VERSION)
        if logger:
            logger.info("exited with --version")
            logger.close()
        return 0

    if args.mode == "tui":
        if args.mcp_servers and not args.mcp_config:
            print("error: --mcp-server requires --mcp-config", file=sys.stderr)
            if logger:
                logger.error("--mcp-server used without --mcp-config")
                logger.close()
            return 2

        session_store = _resolve_session_store(
            cwd=os.getcwd(),
            no_session=args.no_session,
            session_path_arg=args.session,
        )
        initial_messages = session_store.load_messages() if (session_store and args.continue_session) else None
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
            provider = OpenAICompatibleProvider(debug_logger=logger, debug_level=args.debug)
            rc = run_tui_mode(
                provider=provider,
                model=model,
                cwd=os.getcwd(),
                session_store=session_store,
                initial_messages=initial_messages,
                extra_tools=extra_tools,
            )
        except (ProviderError, McpRegistryError) as exc:
            print(f"provider error: {exc}", file=sys.stderr)
            if logger:
                logger.error(f"provider error: {exc}")
            return 1
        finally:
            if mcp_registry is not None:
                mcp_registry.close()
            if logger:
                logger.close()
        return rc

    if args.print_mode:
        if not args.messages:
            print("error: missing prompt for --print mode", file=sys.stderr)
            if logger:
                logger.error("missing prompt for --print mode")
                logger.close()
            return 2
        if args.mcp_servers and not args.mcp_config:
            print("error: --mcp-server requires --mcp-config", file=sys.stderr)
            if logger:
                logger.error("--mcp-server used without --mcp-config")
                logger.close()
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
            if logger:
                logger.info("starting print mode")
            extra_tools = {}
            if args.mcp_config:
                config_path = Path(args.mcp_config)
                if not config_path.is_absolute():
                    config_path = Path(os.getcwd()) / config_path
                mcp_registry = McpRegistry(config_path=config_path.resolve(), enabled_servers=args.mcp_servers)
                mcp_registry.start()
                extra_tools.update(mcp_registry.build_tools())
            provider = OpenAICompatibleProvider(debug_logger=logger, debug_level=args.debug)
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
            if logger:
                logger.error(f"provider error: {exc}")
            return 1
        finally:
            if mcp_registry is not None:
                mcp_registry.close()

        if session_store:
            session_store.append_turn(messages=result.turn_messages, events=result.events)

        if logger:
            for event in result.events:
                logger.event(event)
            logger.info(f"print mode completed model={result.model}")
            logger.close()

        print(result.output)
        return 0

    if logger:
        logger.info("no mode selected; exiting 0")
        logger.close()
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
