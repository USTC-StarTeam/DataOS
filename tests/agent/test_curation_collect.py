from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml


REPO = Path(__file__).parents[2]
COLLECT_PATH = REPO / "configs" / "tasks" / "curationbench" / "eval" / "collect.py"
TASK_PATH = REPO / "configs" / "tasks" / "curationbench" / "eval" / "task.yaml"


def _load_collector():
    spec = importlib.util.spec_from_file_location("curation_collect", COLLECT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_accepts_both_vlmevalkit_dataframe_orientations(tmp_path: Path) -> None:
    collector = _load_collector()
    suite_dir = tmp_path / "suite"
    run_dir = suite_dir / "target_iter1"
    results_path = run_dir / "eval" / "results" / "results.json"
    output_dir = tmp_path / "output"
    results_path.parent.mkdir(parents=True)

    task_path = run_dir / "task.resolved.yaml"
    task_path.write_text(
        yaml.safe_dump(
            {
                "task": {
                    "target_evals": [
                        "MMVet",
                        "LLaVABench",
                        "OCRBench",
                        "HallusionBench",
                        "MMMU_DEV_VAL",
                        "MathVista_MINI",
                        "MMStar",
                        "MMBench",
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (suite_dir / "suite_state.json").write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "state": "completed",
                        "run_dir": str(run_dir),
                        "task_path": str(task_path),
                        "task_id": "fixture",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    results_path.write_text(
        json.dumps(
            {
                "SmolVLM": {
                    "MMVet": {"Category": {"0": "Overall"}, "acc": {"0": 50}},
                    "LLaVABench": {
                        "split": {"0": "overall"},
                        "Relative Score (main)": {"0": 60},
                    },
                    "OCRBench": {"Final Score": 700},
                    "HallusionBench": {
                        "split": {"0": "Overall"},
                        "aAcc": {"0": 40},
                        "fAcc": {"0": 50},
                        "qAcc": {"0": 60},
                    },
                    "MMMU_DEV_VAL": {
                        "0": {"split": "dev", "Overall": 0.9},
                        "1": {"split": "validation", "Overall": 0.4},
                    },
                    "MathVista_MINI": {"Task&Skill": {"0": "Overall"}, "acc": {"0": 30}},
                    "MMStar": {"0": {"split": "none", "Overall": 0.2}},
                    "MMBench": {
                        "0": {"split": "test", "Overall": 0.8},
                        "1": {"split": "dev", "Overall": 0.1},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    config = yaml.safe_load(TASK_PATH.read_text(encoding="utf-8"))["native"]
    summary = collector.collect(suite_dir, output_dir, config)

    assert summary["metrics"] == {
        "MMVet": 50.0,
        "LLaVABench": 60.0,
        "OCRBench": 700.0,
        "HallusionBench": 50.0,
        "MMMU_DEV_VAL": 40.0,
        "MathVista_MINI": 30.0,
        "MMStar": 20.0,
        "MMBench": 10.0,
    }
    assert summary["score"] == pytest.approx(0.4125)
    assert json.loads((output_dir / "summary.json").read_text()) == summary
