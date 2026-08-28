from __future__ import annotations

import re
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
ENV_DIR = REPO / "configs" / "environments" / "CURATION_SMOLVLM_H100_slurm"
TASK_DIR = REPO / "configs" / "tasks" / "curationbench"


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_asset_scope_is_a_declared_curation_task() -> None:
    assets = _yaml(ENV_DIR / "assets.yaml")
    task = _yaml(TASK_DIR / "task.yaml")
    declared = task["inputs"]["bench_task_id"]["one_of"]

    assert assets["scope"]["tasks"] == ["llava665k_smolvlm_8bench_10k_unlimited"]
    assert set(assets["scope"]["tasks"]) <= set(declared)
    assert set(assets["datasets"]) == {"llava665k"}
    assert set(assets["models"]) == {"smolvlm"}


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
