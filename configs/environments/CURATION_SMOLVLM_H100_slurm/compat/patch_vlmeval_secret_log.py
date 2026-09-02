#!/usr/bin/env python3
"""Redact API credentials from the pinned VLMEvalKit INFO log."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ORIGINAL_SHA256 = "fc17db161eeeb9c4c7adf3a67874202bf3443403460d42c33a75ce02fe764561"
MARKER = "API Key: ***"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patch(project: Path) -> Path:
    target = project / "vlmeval" / "api" / "gpt.py"
    text = target.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"[ok] VLMEvalKit API key log already redacted: {sha256(target)}")
        return target
    actual = sha256(target)
    if actual != ORIGINAL_SHA256:
        raise RuntimeError(f"refusing to patch unexpected vlmeval/api/gpt.py: {actual}")
    needle = "self.logger.info(f'Using API Base: {self.api_base}; API Key: {self.key}')"
    replacement = "self.logger.info(f'Using API Base: {self.api_base}; API Key: ***')"
    if text.count(needle) != 1:
        raise RuntimeError(f"expected exactly one credential log, found {text.count(needle)}")
    text = text.replace(needle, replacement)
    compile(text, str(target), "exec")
    target.write_text(text, encoding="utf-8")
    print(f"[ok] redacted VLMEvalKit API key log: {sha256(target)}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    patch(args.project)


if __name__ == "__main__":
    main()
