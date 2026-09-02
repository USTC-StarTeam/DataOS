#!/usr/bin/env python3
"""Backport the Qwen3.5 GDN backend switch to the pinned vLLM 0.17.1 source."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ORIGINAL_SHA256 = "32b4f5fa8727dcf0fd7988ae575bfeb795917777a792d415779dc93d0568e8cb"
MARKER = "DATAOS_VLLM_GDN_PREFILL_BACKEND"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patch(venv: Path) -> Path:
    target = (
        venv
        / "lib"
        / "python3.12"
        / "site-packages"
        / "vllm"
        / "model_executor"
        / "models"
        / "qwen3_next.py"
    )
    text = target.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"[ok] vLLM Qwen3.5 GDN switch already installed: {sha256(target)}")
        return target
    actual = sha256(target)
    if actual != ORIGINAL_SHA256:
        raise RuntimeError(f"refusing to patch unexpected qwen3_next.py: {actual}")
    text = text.replace("from itertools import islice\n", "from itertools import islice\nimport os\n", 1)
    needle = "if current_platform.is_cuda() and current_platform.is_device_capability(90):"
    replacement = (
        "if (\n"
        "            current_platform.is_cuda()\n"
        "            and current_platform.is_device_capability(90)\n"
        f"            and os.environ.get(\"{MARKER}\") != \"triton\"\n"
        "        ):"
    )
    if text.count(needle) != 1:
        raise RuntimeError(f"expected exactly one GDN dispatch site, found {text.count(needle)}")
    text = text.replace(needle, replacement)
    compile(text, str(target), "exec")
    target.write_text(text, encoding="utf-8")
    print(f"[ok] installed vLLM Qwen3.5 GDN switch: {sha256(target)}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venv", type=Path, required=True)
    args = parser.parse_args()
    patch(args.venv)


if __name__ == "__main__":
    main()
