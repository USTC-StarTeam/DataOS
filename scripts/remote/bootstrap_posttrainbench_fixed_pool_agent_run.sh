#!/usr/bin/env bash
set -euo pipefail

RUN_CONFIG="${1:?usage: bootstrap_posttrainbench_fixed_pool_agent_run.sh RUN_CONFIG_JSON}"

python3 - "$RUN_CONFIG" <<'PY'
import json
import shlex
import subprocess
import sys
from pathlib import Path

config_path = Path(sys.argv[1]).resolve()
config = json.loads(config_path.read_text())

host = config["remote"]["host"]
run_root = config["remote"]["run_root"]
repo_root = config_path.parents[6]
prompt_path = config_path.with_name("agent_prompt.md")

remote_layout = [
    run_root,
    f"{run_root}/code",
    f"{run_root}/agent",
    f"{run_root}/fixed_pool",
    f"{run_root}/work",
    f"{run_root}/final_eval",
    f"{run_root}/slurm",
    f"{run_root}/logs",
]

mkdir_cmd = "mkdir -p " + " ".join(shlex.quote(path) for path in remote_layout)
subprocess.run(["ssh", host, mkdir_cmd], check=True)

rsync_cmd = [
    "rsync",
    "-a",
    "--delete",
    "--exclude",
    ".git/",
    "--exclude",
    ".venv/",
    "--exclude",
    ".ruff_cache/",
    "--exclude",
    "workspace/",
    f"{repo_root}/",
    f"{host}:{run_root}/code/DataOS/",
]
subprocess.run(rsync_cmd, check=True)

subprocess.run(
    ["scp", str(config_path), f"{host}:{run_root}/fixed_pool/run_config.json"],
    check=True,
)
subprocess.run(
    ["scp", str(prompt_path), f"{host}:{run_root}/agent/prompt.md"],
    check=True,
)

print(json.dumps({"host": host, "run_root": run_root}, indent=2))
PY
