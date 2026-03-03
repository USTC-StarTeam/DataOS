"""CLI execution helpers."""

from __future__ import annotations

import sys

from .args import parse_args
from .config import VERSION
from .modes.print_mode import run_print_mode
from .providers import DEFAULT_MODEL, OpenAICompatibleProvider
from .providers.base import ProviderError


def run(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.version:
        print(VERSION)
        return 0

    if args.print_mode:
        if not args.messages:
            print("error: missing prompt for --print mode", file=sys.stderr)
            return 2

        prompt = " ".join(args.messages).strip()
        model = args.model or DEFAULT_MODEL
        try:
            provider = OpenAICompatibleProvider()
            result = run_print_mode(provider=provider, prompt=prompt, model=model, mode=args.mode)
        except ProviderError as exc:
            print(f"provider error: {exc}", file=sys.stderr)
            return 1

        print(result.output)
        return 0

    return 0
