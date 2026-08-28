"""Execute one config-authored external training workflow inside a launcher job."""

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
    starts = [index for index, char in enumerate(text) if char == "{"]
    for index in reversed(starts):
        try:
            value = json.loads(text[index:].strip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("external CLI did not print a JSON object")


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(manifest_path: str | Path) -> int:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    output = Path(manifest["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    for step in manifest["steps"]:
        name = step["name"]
        proc = subprocess.run(
            [manifest["executable"], *step["args"]],
            cwd=manifest["cwd"],
            capture_output=True,
            text=True,
            check=False,
        )
        Path(output.parent, f"{name}_stdout.txt").write_text(proc.stdout, encoding="utf-8")
        Path(output.parent, f"{name}_stderr.txt").write_text(proc.stderr, encoding="utf-8")
        try:
            native = _json_object(proc.stdout)
        except ValueError as exc:
            native = {"status": "failed", "error": str(exc)}
        records[name] = native
        accepted = step.get("accept_status") or []
        if proc.returncode != 0 or (accepted and native.get("status") not in accepted):
            result = {
                "status": "FAILED",
                "error": native.get("error") or (
                    f"step {name} returned status {native.get('status')!r} "
                    f"(exit {proc.returncode})"
                ),
                "model_path": None,
                "steps": records,
            }
            _write(output, result)
            print(json.dumps(result, ensure_ascii=False))
            return 1

    result_cfg = manifest.get("result") or {}
    native = records.get(result_cfg.get("step"), {})
    model_path = native.get(result_cfg.get("field"))
    result = {
        "status": "COMPLETED" if isinstance(model_path, str) and model_path else "FAILED",
        "error": None if isinstance(model_path, str) and model_path else "missing model path",
        "model_path": model_path,
        "steps": records,
        "report_fragment": {"train": {"model_path": model_path}},
    }
    _write(output, result)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "COMPLETED" else 1


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m dataos_train.external_runner MANIFEST", file=sys.stderr)
        return 2
    return run(sys.argv[1])


if __name__ == "__main__":
    raise SystemExit(main())
