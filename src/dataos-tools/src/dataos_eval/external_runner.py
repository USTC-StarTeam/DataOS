"""Execute an external evaluator and its task-local result collector."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    for index in reversed([i for i, char in enumerate(text) if char == "{"]):
        try:
            value = json.loads(text[index:].strip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("external CLI did not print a JSON object")


def run(manifest_path: str | Path) -> int:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    output_dir = Path(manifest["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [manifest["executable"], *manifest["args"]],
        cwd=manifest["cwd"],
        capture_output=True,
        text=True,
        check=False,
    )
    (output_dir / "external_eval_stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (output_dir / "external_eval_stderr.txt").write_text(proc.stderr, encoding="utf-8")
    try:
        native = _json_object(proc.stdout)
    except ValueError as exc:
        native = {"status": "failed", "error": str(exc)}
    native_path = output_dir / "external_eval_native.json"
    native_path.write_text(json.dumps(native, indent=2) + "\n", encoding="utf-8")
    accepted = manifest.get("accept_status") or []
    if proc.returncode != 0 or (accepted and native.get("status") not in accepted):
        print(json.dumps(native, ensure_ascii=False))
        return 1

    collected = subprocess.run(
        [
            sys.executable,
            manifest["collector"],
            "--suite-dir",
            manifest["suite_dir"],
            "--output-dir",
            manifest["output_dir"],
            "--config-json",
            json.dumps(manifest.get("collector_config") or {}),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    (output_dir / "collector_stdout.txt").write_text(collected.stdout, encoding="utf-8")
    (output_dir / "collector_stderr.txt").write_text(collected.stderr, encoding="utf-8")
    if collected.returncode != 0:
        print(collected.stderr.strip(), file=sys.stderr)
        return collected.returncode
    print(collected.stdout.strip())
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m dataos_eval.external_runner MANIFEST", file=sys.stderr)
        return 2
    return run(sys.argv[1])


if __name__ == "__main__":
    raise SystemExit(main())
