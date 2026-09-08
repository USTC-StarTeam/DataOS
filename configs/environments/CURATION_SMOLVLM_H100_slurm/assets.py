#!/usr/bin/env python3
"""Provision and verify the machine-local Curation-Bench assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "assets.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - matches the checksum frozen by upstream VLMEvalKit
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, output: Path, expected_md5: str) -> None:
    if output.is_file() and _md5(output) == expected_md5:
        print(f"[ok] {output.name}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    part = output.with_suffix(output.suffix + ".part")
    part.unlink(missing_ok=True)
    print(f"[download] {url} -> {output}", flush=True)
    # openxlab's served chain currently contains an expired certificate. The bytes remain
    # authenticated against checksums frozen in the pinned VLMEvalKit source tree.
    context = ssl._create_unverified_context()  # noqa: SLF001, S323
    with urllib.request.urlopen(url, context=context) as response, part.open("wb") as stream:
        shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
    actual = _md5(part)
    if actual != expected_md5:
        part.unlink(missing_ok=True)
        raise ValueError(f"checksum mismatch for {output.name}: {actual} != {expected_md5}")
    part.replace(output)
    print(f"[ok] {output.name} md5={actual}")


def prefetch_eval(asset_root: Path) -> None:
    manifest = _read_yaml(MANIFEST)
    eval_spec = manifest["eval_data"]
    output_dir = asset_root / eval_spec["relative_path"]
    for name, spec in eval_spec["files"].items():
        _download(str(spec["url"]), output_dir / f"{name}.tsv", str(spec["md5"]))


def _check_command(label: str, command: list[str], errors: list[str]) -> None:
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        errors.append(f"{label}: {detail[-1] if detail else 'command failed'}")
    else:
        print(f"[ok] {label}: {proc.stdout.strip()}")


def _verify_hf_revision(path: Path, expected: str, label: str) -> None:
    metadata_dir = path / ".cache" / "huggingface" / "download"
    revisions = set()
    for metadata in metadata_dir.glob("*.metadata"):
        lines = metadata.read_text(encoding="utf-8").splitlines()
        if lines:
            revisions.add(lines[0])
    if revisions != {expected}:
        raise ValueError(f"{label} revision metadata {sorted(revisions)} != {expected}: {path}")
    print(f"[ok] {label} revision: {expected}")


def _task_spec(harness_root: Path, task: str) -> dict[str, Any]:
    task_path = harness_root / "src" / "benchmark" / "tasks" / f"{task}.yaml"
    value = _read_yaml(task_path).get("task")
    if not isinstance(value, dict):
        raise ValueError(f"invalid official task: {task_path}")
    return value


def _verify_dataset(path: Path, task: dict[str, Any], manifest: dict[str, Any]) -> None:
    from datasets import Dataset, DatasetDict, load_from_disk

    value = load_from_disk(str(path))
    if isinstance(value, DatasetDict):
        split_name = task["dataset"].get("split")
        if not split_name or split_name not in value:
            raise ValueError(f"dataset is split, but task does not select a valid split: {path}")
        value = value[split_name]
    if not isinstance(value, Dataset):
        raise ValueError(f"not a Hugging Face Dataset: {path}")
    required = set(manifest["columns"])
    missing = sorted(required - set(value.column_names))
    if missing:
        raise ValueError(f"dataset is missing columns {missing}: {path}")
    target_rows = int(task["target_rows"])
    if len(value) < target_rows:
        raise ValueError(f"dataset has {len(value)} rows, task requires {target_rows}: {path}")
    expected_rows = int(manifest["rows"])
    if len(value) != expected_rows:
        raise ValueError(f"dataset row count {len(value)} != frozen {expected_rows}: {path}")
    print(f"[ok] dataset: {len(value)} rows, columns={value.column_names}")


def verify(
    harness_root: Path, profile_path: Path, task_name: str, manifest_path: Path = MANIFEST
) -> None:
    manifest = _read_yaml(manifest_path)
    profile = _read_yaml(profile_path)
    errors: list[str] = []
    if task_name not in manifest["scope"]["tasks"]:
        errors.append(f"task {task_name!r} is outside this deployment's frozen scope")

    revision = subprocess.run(
        ["git", "-C", str(harness_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    expected_revision = str(manifest["harness"]["revision"])
    if revision != expected_revision:
        errors.append(f"harness revision {revision!r} != {expected_revision!r}")
    else:
        print(f"[ok] harness revision: {revision}")

    compat = manifest["compat"]["vlmeval_result_transfer"]
    compat_path = harness_root / str(compat["install_path"])
    expected_compat = str(compat["sha256"])
    if not compat_path.is_file():
        errors.append(f"missing VLMEvalKit compatibility file: {compat_path}")
    elif (actual_compat := _sha256(compat_path)) != expected_compat:
        errors.append(
            f"VLMEvalKit compatibility checksum {actual_compat} != {expected_compat}: "
            f"{compat_path}"
        )
    else:
        print(f"[ok] VLMEvalKit compatibility file: {actual_compat}")
    ninja = harness_root / "vendor" / "VLMEvalKit" / ".venv" / "bin" / "ninja"
    if not ninja.is_file():
        errors.append(f"evaluation venv is missing the FlashInfer JIT builder: {ninja}")
    else:
        print(f"[ok] evaluation JIT builder: {ninja}")
    secret_patch = manifest["compat"]["vlmeval_secret_log"]
    secret_path = harness_root / str(secret_patch["install_path"])
    expected_secret = str(secret_patch["patched_sha256"])
    if not secret_path.is_file():
        errors.append(f"missing VLMEvalKit credential-log patch: {secret_path}")
    elif (actual_secret := _sha256(secret_path)) != expected_secret:
        errors.append(
            f"VLMEvalKit credential-log checksum {actual_secret} != {expected_secret}: "
            f"{secret_path}"
        )
    else:
        print(f"[ok] VLMEvalKit credential-log patch: {actual_secret}")

    _check_command(
        "benchmark venv",
        [str(harness_root / ".venv" / "bin" / "python"), "-c", "import benchmark; print('ok')"],
        errors,
    )
    _check_command(
        "training venv",
        [
            str(harness_root / "vendor" / "curation-train" / ".venv" / "bin" / "python"),
            "-c",
            "import curation_train, flash_attn; print('ok')",
        ],
        errors,
    )
    _check_command(
        "evaluation venv",
        [
            str(harness_root / "vendor" / "VLMEvalKit" / ".venv" / "bin" / "python"),
            "-c",
            "import vlmeval, vllm; print('ok')",
        ],
        errors,
    )

    try:
        task = _task_spec(harness_root, task_name)
        dataset_id = str(task["dataset"]["dataset_id"])
        model_key = str(task["model_key"])
        qwen_compat = manifest["compat"].get("qwen_text_only")
        if model_key == "qwen2.5-vl-3b-instruct" and qwen_compat:
            qwen_path = harness_root / qwen_compat["install_path"]
            if _sha256(qwen_path) != qwen_compat["patched_sha256"]:
                errors.append(f"Qwen text-only compatibility checksum mismatch: {qwen_path}")
            else:
                print("[ok] Qwen text-only/DDP compatibility patch")
        dataset_path = Path(profile["dataset_path_map"][dataset_id])
        model_path = Path(profile["model_path_map"][model_key])
        dataset_manifest = manifest["datasets"][dataset_id]
        model_manifest = manifest["models"][model_key]
        _verify_hf_revision(
            dataset_path, str(dataset_manifest["revision"]), f"dataset {dataset_id}"
        )
        _verify_dataset(dataset_path, task, dataset_manifest)
        _verify_hf_revision(model_path, str(model_manifest["revision"]), f"model {model_key}")
        config_path = model_path / "config.json"
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
        expected_type = str(model_manifest["model_type"])
        if model_config.get("model_type") != expected_type:
            errors.append(
                f"model_type {model_config.get('model_type')!r} != {expected_type!r}: {model_path}"
            )
        else:
            print(f"[ok] model: {model_path} ({expected_type})")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        errors.append(str(exc))

    eval_dir = Path(str(profile.get("eval_data_dir") or ""))
    for name, spec in manifest["eval_data"]["files"].items():
        path = eval_dir / f"{name}.tsv"
        if not path.is_file():
            errors.append(f"missing eval data: {path}")
            continue
        actual = _md5(path)
        if actual != str(spec["md5"]):
            errors.append(f"eval checksum {path}: {actual} != {spec['md5']}")
        else:
            print(f"[ok] eval data: {name}")

    if errors:
        print(json.dumps({"status": "failed", "errors": errors}, indent=2), file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps({"status": "ready", "task": task_name}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prefetch = commands.add_parser("prefetch-eval")
    prefetch.add_argument("--asset-root", type=Path, required=True)
    check = commands.add_parser("verify")
    check.add_argument("--harness-root", type=Path, required=True)
    check.add_argument("--profile", type=Path, required=True)
    check.add_argument("--task", required=True)
    check.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    if args.command == "prefetch-eval":
        prefetch_eval(args.asset_root)
    else:
        verify(args.harness_root, args.profile, args.task, args.manifest)


if __name__ == "__main__":
    main()
