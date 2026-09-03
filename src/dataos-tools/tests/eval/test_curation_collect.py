import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import yaml

from dataos_eval.external_runner import run as run_external_eval
from dataos_eval.stage import EvalStage, ExternalEvalStage
from dataos_launch import Resources


REPO = Path(__file__).resolve().parents[4]


def _collector():
    path = REPO / "configs" / "tasks" / "curationbench" / "eval" / "collect.py"
    spec = importlib.util.spec_from_file_location("curation_collect", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_collect_normalizes_all_eight_vlmevalkit_scores(tmp_path):
    suite = tmp_path / "suite"
    target = suite / "task_iter1"
    results_dir = target / "eval" / "results"
    results_dir.mkdir(parents=True)
    task_path = target / "task.yaml"
    benches = [
        "MMVet",
        "LLaVABench",
        "OCRBench",
        "HallusionBench",
        "MMMU_DEV_VAL",
        "MathVista_MINI",
        "MMStar",
        "MMBench",
    ]
    task_path.write_text(
        "task:\n  target_evals:\n" + "".join(f"    - {name}\n" for name in benches),
        encoding="utf-8",
    )
    scores = {
        "MMVet": {"Category": {"0": "Overall"}, "acc": {"0": 50}},
        "LLaVABench": {
            "split": {"0": "overall"},
            "Relative Score (main)": {"0": 50},
        },
        "OCRBench": {"Final Score": 500},
        "HallusionBench": {
            "split": {"0": "Overall"},
            "aAcc": {"0": 40},
            "fAcc": {"0": 50},
            "qAcc": {"0": 60},
        },
        "MMMU_DEV_VAL": {
            "0": {"split": "dev", "Overall": 0.25},
            "1": {"split": "validation", "Overall": 0.5},
        },
        "MathVista_MINI": {"Task&Skill": {"0": "Overall"}, "acc": {"0": 50}},
        "MMStar": {"0": {"split": "none", "Overall": 0.5}},
        "MMBench": {"0": {"split": "dev", "Overall": 0.5}},
    }
    (results_dir / "results.json").write_text(
        json.dumps({"SmolVLM": scores}), encoding="utf-8"
    )
    (suite / "suite_state.json").write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "state": "completed",
                        "run_dir": str(target),
                        "task_path": str(task_path),
                        "task_id": "llava665k_smolvlm_8bench_10k_unlimited",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    eval_task = Path(__file__).resolve().parents[4] / "configs/tasks/curationbench/eval/task.yaml"
    config = yaml.safe_load(eval_task.read_text(encoding="utf-8"))["native"]

    summary = _collector().collect(suite, tmp_path / "out", config)

    assert summary["score"] == 0.5
    assert summary["metrics"]["MMMU_DEV_VAL"] == 50
    assert summary["metrics"]["HallusionBench"] == 50
    assert summary["normalized"] == {name: 0.5 for name in benches}
    assert json.loads((tmp_path / "out" / "summary.json").read_text())["score"] == 0.5


def test_eval_stage_splits_offline_inference_from_remote_judge(tmp_path):
    task_dir = tmp_path / "configs" / "tasks" / "external"
    (task_dir / "train").mkdir(parents=True)
    (task_dir / "eval").mkdir()
    (task_dir / "task.yaml").write_text(
        "name: external\nstages:\n"
        "  data: {stage: data, driver: external_cli, ownership: agent, tool: official-cli}\n"
        "  train: {contract: train/task.yaml}\n"
        "  eval: {contract: eval/task.yaml}\n"
        "scoring:\n  judge: {model: qwen3.5-plus, workers: 32}\n",
        encoding="utf-8",
    )
    (task_dir / "train" / "task.yaml").write_text(
        "stage: train\ndriver: external_cli\nownership: frozen\ntool: official-cli\n",
        encoding="utf-8",
    )
    (task_dir / "eval" / "collect.py").write_text("print('{}')\n", encoding="utf-8")
    (task_dir / "eval" / "run.py").write_text("print('{}')\n", encoding="utf-8")
    (task_dir / "eval" / "task.yaml").write_text(
        "stage: eval\ndriver: external_cli\nownership: frozen\ntool: official-cli\n"
        "compute:\n  script: run.py\n"
        "  args: [infer, --root, '{tool_root}', --suite, '{suite_dir}']\n"
        "remote_judge:\n  script: run.py\n"
        "  args: [judge, --model, '{judge_model}', --workers, '{judge_workers}']\n"
        "accept_status: [completed]\n"
        "collector: collect.py\n",
        encoding="utf-8",
    )
    env_dir = tmp_path / "configs" / "environments" / "local"
    env_dir.mkdir(parents=True)
    tool_env = tmp_path / "official"
    executable = tool_env / "bin" / "official-cli"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    harness_root = REPO
    harness_revision = subprocess.check_output(
        ["git", "-C", str(harness_root), "rev-parse", "HEAD"], text=True
    ).strip()
    key_file = tmp_path / "judge.key"
    key_file.write_text("secret", encoding="utf-8")
    python_env = Path(sys.executable).parents[1]
    (env_dir / "env.yaml").write_text(
        "name: local\nkind: local\nlauncher: local\ntools:\n"
        f"  dataos: {{path: {python_env}, provides: [dataos-eval]}}\n"
        "  official:\n"
        f"    path: {tool_env}\n"
        f"    root: {harness_root}\n"
        f"    revision: {harness_revision}\n"
        "    provides: [official-cli]\n"
        "    env: {VLLM_GPU_MEMORY_UTILIZATION: '0.9'}\n"
        "gateway:\n"
        "  base_url: https://judge.example/v1/chat/completions\n"
        f"  key_file: {key_file}\n",
        encoding="utf-8",
    )
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "suite_state.json").write_text("{}", encoding="utf-8")
    stage = EvalStage.from_repo(tmp_path, "local", "external")
    assert isinstance(stage, ExternalEvalStage)
    spec = stage.spec_for(
        task="vlm",
        model_path="/model",
        model_name="m",
        output_dir=str(tmp_path / "out"),
        run_root=str(tmp_path / "run"),
        limit=-1,
        resources=Resources(gpus=2, cpus=2),
        suite_dir=str(suite),
    )

    job = stage.preview(spec)["job"]

    assert job["backend"] == "external_cli"
    assert "dataos_eval.external_runner" in job["command"]
    assert "run_lmeval_eval.py" not in json.dumps(job)
    assert job["resources"]["gpus"] == 2
    assert job["env"]["VLLM_GPU_MEMORY_UTILIZATION"] == "0.9"
    assert job["pre_cmds"] == []
    manifest = json.loads(next(iter(job["assets"].values())))
    assert manifest["collect_after_run"] is False
    assert manifest["executable"] == str(python_env / "bin" / "python")
    assert manifest["args"] == [
        str(task_dir / "eval" / "run.py"),
        "infer",
        "--root",
        str(harness_root),
        "--suite",
        str(suite),
    ]
    assert spec.needs_judge is True

    command = stage.judge_command(
        {
            "task": "vlm",
            "model_path": "/model",
            "model_name": "m",
            "output_dir": str(tmp_path / "out"),
            "suite_dir": str(suite),
        },
        tmp_path / "out" / "summary.json",
    )
    assert f"OPENAI_API_KEY=$(cat {key_file})" in command
    assert "OPENAI_API_BASE=https://judge.example/v1/chat/completions" in command
    assert "judge --model qwen3.5-plus --workers 32" in command
    assert str(task_dir / "eval" / "collect.py") in command


def test_external_runner_defers_collection_until_remote_judge(tmp_path: Path) -> None:
    compute = tmp_path / "compute.py"
    compute.write_text("import json\nprint(json.dumps({'status': 'completed'}))\n")
    collector = tmp_path / "collector.py"
    marker = tmp_path / "collector-ran"
    collector.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    output_dir = tmp_path / "output"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "executable": sys.executable,
                "cwd": str(tmp_path),
                "args": [str(compute)],
                "accept_status": ["completed"],
                "suite_dir": str(tmp_path / "suite"),
                "output_dir": str(output_dir),
                "collector": str(collector),
                "collect_after_run": False,
            }
        )
    )

    assert run_external_eval(manifest) == 0
    assert not marker.exists()
    assert json.loads((output_dir / "external_eval_native.json").read_text()) == {
        "status": "completed"
    }
