from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest
import yaml


REPO = Path(__file__).resolve().parents[2]
ENV_DIR = REPO / "configs" / "environments" / "CURATION_SMOLVLM_H100_slurm"
TASK_DIR = REPO / "configs" / "tasks" / "curationbench"


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _assets_module():
    spec = importlib.util.spec_from_file_location("curation_assets", ENV_DIR / "assets.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_asset_scope_is_a_declared_curation_task() -> None:
    assets = _yaml(ENV_DIR / "assets.yaml")
    task = _yaml(TASK_DIR / "task.yaml")
    declared = task["inputs"]["bench_task_id"]["one_of"]

    assert assets["scope"]["tasks"] == ["llava665k_smolvlm_8bench_10k_unlimited"]
    assert set(assets["scope"]["tasks"]) <= set(declared)
    assert set(assets["datasets"]) == {"llava665k"}
    assert set(assets["models"]) == {"smolvlm"}
    compat = assets["compat"]["vlmeval_result_transfer"]
    assert re.fullmatch(r"[0-9a-f]{40}", compat["revision"])
    assert re.fullmatch(r"[0-9a-f]{64}", compat["sha256"])
    compat_path = ENV_DIR / "compat" / "result_transfer.py"
    assert compat_path.is_file()
    assert _assets_module()._sha256(compat_path) == compat["sha256"]
    secret_patch = assets["compat"]["vlmeval_secret_log"]
    assert re.fullmatch(r"[0-9a-f]{64}", secret_patch["original_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", secret_patch["patched_sha256"])
    assert (ENV_DIR / "compat" / "patch_vlmeval_secret_log.py").is_file()


def test_official_curation_run_reserves_two_h100s_and_uses_remote_judge() -> None:
    run = _yaml(REPO / "configs" / "runs" / "example-curationbench.json")
    task = _yaml(TASK_DIR / "task.yaml")
    eval_contract = _yaml(TASK_DIR / "eval" / "task.yaml")

    assert run["exec"]["resource"]["gpus"] == 2
    assert task["scoring"]["judge"] == {"model": "qwen3.5-plus", "workers": 128}
    assert "compute" in eval_contract and "remote_judge" in eval_contract
    assert eval_contract["remote_judge"]["args"][-4:] == [
        "--timeout-seconds",
        "120",
        "--request-retries",
        "1",
    ]


def test_eval_asset_manifest_matches_frozen_collector() -> None:
    assets = _yaml(ENV_DIR / "assets.yaml")
    eval_contract = _yaml(TASK_DIR / "eval" / "task.yaml")
    files = assets["eval_data"]["files"]

    assert set(files) == set(eval_contract["native"]["selectors"])
    assert all(re.fullmatch(r"[0-9a-f]{32}", spec["md5"]) for spec in files.values())


def test_example_profile_covers_exact_deployment_assets() -> None:
    assets = _yaml(ENV_DIR / "assets.yaml")
    profile = _yaml(ENV_DIR / "profile.example.yaml")

    assert set(profile["dataset_path_map"]) == set(assets["datasets"])
    assert set(profile["model_path_map"]) == set(assets["models"])
    assert profile["env"]["BENCHMARK_VLMEVAL_TRUST_LOCAL_TSV"] == "1"


def test_hf_revision_check_reads_download_provenance(tmp_path: Path) -> None:
    metadata = tmp_path / ".cache" / "huggingface" / "download" / "config.json.metadata"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("abc123\netag\ntimestamp\n", encoding="utf-8")
    assets = _assets_module()

    assets._verify_hf_revision(tmp_path, "abc123", "model")
    with pytest.raises(ValueError, match="revision metadata"):
        assets._verify_hf_revision(tmp_path, "wrong", "model")
