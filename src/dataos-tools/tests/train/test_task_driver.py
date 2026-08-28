import json
import sys
from pathlib import Path

from dataos_launch import Resources
from dataos_train.external_runner import run
from dataos_train.task_stage import TaskTrainSpec, TaskTrainStage


def _fake_cli(path: Path) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "cmd = sys.argv[1]\n"
        "out = {\n"
        " 'audit': {'status': 'clean'},\n"
        " 'submit': {'status': 'submitted', 'rows': 10000},\n"
        " 'finetune': {'status': 'completed', 'model_path': '/suite/finetune/ft-1'},\n"
        "}[cmd]\n"
        "print(json.dumps(out, indent=2))\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_external_runner_uses_official_steps_and_writes_model_result(tmp_path):
    executable = tmp_path / "harness" / "bin" / "datacuration-bench"
    _fake_cli(executable)
    output = tmp_path / "run" / "external_train_result.json"
    manifest = {
        "executable": str(executable),
        "cwd": str(tmp_path),
        "steps": [
            {"name": "audit", "args": ["audit"], "accept_status": ["clean"]},
            {"name": "submit", "args": ["submit"], "accept_status": ["submitted"]},
            {"name": "finetune", "args": ["finetune"], "accept_status": ["completed"]},
        ],
        "result": {"step": "finetune", "field": "model_path"},
        "output": str(output),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert run(manifest_path) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "COMPLETED"
    assert result["model_path"] == "/suite/finetune/ft-1"
    assert list(result["steps"]) == ["audit", "submit", "finetune"]


def test_external_runner_rejects_failed_json_even_when_cli_exits_zero(tmp_path):
    executable = tmp_path / "fake"
    executable.write_text(
        "#!/usr/bin/env python3\nimport json\nprint(json.dumps({'status': 'failed', "
        "'error': 'bad rows'}))\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    output = tmp_path / "run" / "result.json"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "executable": str(executable),
                "cwd": str(tmp_path),
                "steps": [
                    {"name": "submit", "args": [], "accept_status": ["submitted"]}
                ],
                "result": {"step": "submit", "field": "model_path"},
                "output": str(output),
            }
        ),
        encoding="utf-8",
    )

    assert run(manifest_path) == 1
    assert json.loads(output.read_text(encoding="utf-8"))["error"] == "bad rows"


def test_task_stage_builds_only_the_configured_external_cli(tmp_path):
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
        "stage: train\ndriver: external_cli\nownership: frozen\n"
        "tool: official-cli\nworkflow:\n"
        "  - {name: submit, args: [submit], accept_status: [submitted]}\n"
        "result: {step: submit, field: model_path}\n",
        encoding="utf-8",
    )
    (task_dir / "eval" / "task.yaml").write_text(
        "stage: eval\ndriver: external_cli\nownership: frozen\ntool: official-cli\n",
        encoding="utf-8",
    )
    env_dir = tmp_path / "configs" / "environments" / "local"
    env_dir.mkdir(parents=True)
    tool_env = tmp_path / "official"
    _fake_cli(tool_env / "bin" / "official-cli")
    python_env = Path(sys.executable).parents[1]
    (env_dir / "env.yaml").write_text(
        "name: local\nkind: local\nlauncher: local\ntools:\n"
        f"  dataos: {{path: {python_env}, provides: [dataos-train]}}\n"
        f"  official: {{path: {tool_env}, provides: [official-cli]}}\n",
        encoding="utf-8",
    )
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "suite_state.json").write_text("{}", encoding="utf-8")
    submission = tmp_path / "submission"
    submission.mkdir()
    run_dir = tmp_path / "run"
    spec = TaskTrainSpec(
        name="x",
        run_dir=str(run_dir),
        resources=Resources(gpus=1, cpus=2),
        task_config="external",
        suite_dir=str(suite),
        submission_path=str(submission),
    )

    preview = TaskTrainStage.from_repo(tmp_path, "local", "external").preview(spec)

    job = preview["job"]
    assert job["backend"] == "external_cli"
    assert "dataos_train.external_runner" in job["command"]
    assert "llamafactory" not in json.dumps(job).lower()
