import importlib.util
import json
import sys
from pathlib import Path

import yaml

from dataos_eval.stage import EvalStage, ExternalEvalStage
from dataos_launch import Resources


def _collector():
    path = (
        Path(__file__).resolve().parents[4]
        / "configs"
        / "tasks"
        / "curationbench"
        / "eval"
        / "collect.py"
    )
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
            "split": {"0": "dev", "1": "val"},
            "Overall": {"0": 0.25, "1": 0.5},
        },
        "MathVista_MINI": {"Task&Skill": {"0": "Overall"}, "acc": {"0": 50}},
        "MMStar": {"split": {"0": "none"}, "Overall": {"0": 0.5}},
        "MMBench": {"split": {"0": "dev"}, "Overall": {"0": 0.5}},
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


def test_eval_stage_dispatches_external_cli_without_ptb_bridge(tmp_path):
    task_dir = tmp_path / "configs" / "tasks" / "external"
    (task_dir / "train").mkdir(parents=True)
    (task_dir / "eval").mkdir()
    (task_dir / "task.yaml").write_text(
        "name: external\nstages:\n"
        "  data: {stage: data, driver: external_cli, ownership: agent, tool: official-cli}\n"
        "  train: {contract: train/task.yaml}\n"
        "  eval: {contract: eval/task.yaml}\n",
        encoding="utf-8",
    )
    (task_dir / "train" / "task.yaml").write_text(
        "stage: train\ndriver: external_cli\nownership: frozen\ntool: official-cli\n",
        encoding="utf-8",
    )
    (task_dir / "eval" / "collect.py").write_text("print('{}')\n", encoding="utf-8")
    (task_dir / "eval" / "task.yaml").write_text(
        "stage: eval\ndriver: external_cli\nownership: frozen\ntool: official-cli\n"
        "args: [eval, --suite-dir, '{suite_dir}']\naccept_status: [completed]\n"
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
    judge_executable = tmp_path / "eval" / "bin" / "vllm"
    judge_executable.parent.mkdir(parents=True)
    judge_executable.write_text("#!/bin/sh\n", encoding="utf-8")
    judge_executable.chmod(0o755)
    judge_model = tmp_path / "judge-model"
    judge_model.mkdir()
    (judge_model / "config.json").write_text("{}", encoding="utf-8")
    python_env = Path(sys.executable).parents[1]
    (env_dir / "env.yaml").write_text(
        "name: local\nkind: local\nlauncher: local\ntools:\n"
        f"  dataos: {{path: {python_env}, provides: [dataos-eval]}}\n"
        "  official:\n"
        f"    path: {tool_env}\n"
        "    provides: [official-cli]\n"
        "    env: {VLLM_GPU_MEMORY_UTILIZATION: '0.9'}\n"
        "    local_judge:\n"
        f"      executable: {judge_executable}\n"
        f"      model_path: {judge_model}\n"
        "      served_model_name: Qwen3.5-27B\n"
        "      total_gpus: 2\n"
        "      enforce_eager: true\n",
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
        resources=Resources(gpus=1, cpus=2),
        suite_dir=str(suite),
    )

    job = stage.preview(spec)["job"]

    assert job["backend"] == "external_cli"
    assert "dataos_eval.external_runner" in job["command"]
    assert "run_lmeval_eval.py" not in json.dumps(job)
    assert job["resources"]["gpus"] == 2
    assert job["env"]["VLLM_GPU_MEMORY_UTILIZATION"] == "0.9"
    preamble = "\n".join(job["pre_cmds"])
    assert str(judge_executable) in preamble
    assert "BENCHMARK_EVAL_GPUS" in preamble
    assert "OPENAI_API_BASE" in preamble
    assert f"export PATH={judge_executable.parent}" in preamble
    assert "--enforce-eager" in preamble
    assert "seq 1 600" in preamble
