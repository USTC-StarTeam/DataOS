"""Remote H100 Slurm train backend (agent-version-ptb).

Runs ON the local box (dazu5090) inside ``run_train`` when the train config carries
``executor: "slurm_h100"``. Keeps data + eval local; ships only the *training* compute to
the H100 cluster:

  export dataset -> jsonl  →  rsync (train_sft.py + jsonl + sbatch) to H100 run dir
  →  sbatch (gpu partition) + poll sacct  →  rsync the LoRA adapter back  →  return Model

The returned :class:`Model` has ``finetuning_type="lora"`` and the adapter files in its
dir, so the downstream (local) eval treats it exactly like a locally-trained LoRA. Needs a
working ``ssh h100`` on the local box (dedicated key) and the prebuilt env + base model on
H100 (see run-dataos/harnesses/slurm_h100/).
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dataos._artifact import Model
from dataos._ids import _make_id

if TYPE_CHECKING:  # pragma: no cover
    from dataos.workspace import Workspace

_HARNESS = Path(__file__).resolve().parent.parent.parent / "run-dataos" / "harnesses" / "slurm_h100"
_TEMPLATE = _HARNESS / "run_train.sbatch.tmpl"
_TRAIN_SCRIPT = _HARNESS / "train_sft.py"

_TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"}


def _sh(cmd: list[str], *, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def _hyperparams(config: dict[str, Any]) -> dict[str, Any]:
    """Map DataOS/LLaMA-Factory-style train keys to train_sft.py args (with defaults)."""
    return {
        "max_len": int(config.get("cutoff_len", 1024)),
        "bs": int(config.get("per_device_train_batch_size", 2)),
        "grad_accum": int(config.get("gradient_accumulation_steps", 8)),
        "lr": float(config.get("learning_rate", 1e-4)),
        "epochs": float(config.get("num_train_epochs", 1.0)),
        "lora_r": int(config.get("lora_rank", config.get("lora_r", 16))),
        "lora_alpha": int(config.get("lora_alpha", 32)),
        "max_samples": config.get("max_samples"),
    }


def run_train_slurm_h100(
    dataset: Any, config: dict[str, Any], *, workspace: "Workspace"
) -> Model:
    slurm = dict(config.get("slurm") or {})
    host = slurm.get("host", "h100")
    remote_root = slurm.get("remote_root", "/data/tangjiahui/dataos_remote")
    venv = slurm.get("venv", "/data/tangjiahui/dataos_train/.venv")
    hf_home = slurm.get("hf_home", "/data/tangjiahui/hf_cache")
    partition = slurm.get("partition", "gpu")
    qos = slurm.get("qos", "gpu_normal")
    walltime = slurm.get("time", "06:00:00")
    poll_interval = float(slurm.get("poll_interval_s", 30))
    timeout_s = float(slurm.get("timeout_s", 6 * 3600))

    base_model = config.get("model_name_or_path")
    if not base_model:
        raise ValueError("slurm_h100 train requires 'model_name_or_path' (an H100-resolvable base)")
    base_id = config.get("base_model_id", base_model)  # what downstream local eval will load

    model_id = _make_id("model")
    model_dir = workspace.artifact_dir("model", model_id)
    run_root = f"{remote_root}/{model_id}"
    hp = _hyperparams(config)

    # 1) export the prepared dataset to a jsonl locally
    local_jsonl = workspace.tmp_dir / "train.jsonl"
    local_jsonl.parent.mkdir(parents=True, exist_ok=True)
    dataset.hf_dataset.to_json(str(local_jsonl), lines=True, force_ascii=False)

    # 2) render the sbatch script
    max_samples_arg = f"--max-samples {hp['max_samples']}" if hp["max_samples"] else ""
    # Literal .replace (not str.format): the template contains shell brace groups ({ ...; }).
    repl = {
        "jobname": f"dataos_{model_id}", "partition": partition, "qos": qos, "time": walltime,
        "run_root": run_root, "hf_home": hf_home, "venv": venv, "base_model": base_model,
        "max_len": hp["max_len"], "bs": hp["bs"], "grad_accum": hp["grad_accum"],
        "lr": hp["lr"], "epochs": hp["epochs"], "lora_r": hp["lora_r"],
        "lora_alpha": hp["lora_alpha"], "max_samples_arg": max_samples_arg,
    }
    sbatch_text = _TEMPLATE.read_text(encoding="utf-8")
    for _k, _v in repl.items():
        sbatch_text = sbatch_text.replace("{" + _k + "}", str(_v))
    local_sbatch = workspace.tmp_dir / "run_train.sbatch"
    local_sbatch.write_text(sbatch_text, encoding="utf-8")

    # 3) stage to H100: run dir + (train_sft.py, train.jsonl, sbatch)
    mk = _sh(["ssh", host, f"mkdir -p {run_root}"])
    if mk.returncode != 0:
        raise RuntimeError(f"ssh {host} mkdir failed: {mk.stderr.strip()}")
    for src in (str(_TRAIN_SCRIPT), str(local_jsonl), str(local_sbatch)):
        rs = _sh(["rsync", "-az", "-e", "ssh", src, f"{host}:{run_root}/"], timeout=600)
        if rs.returncode != 0:
            raise RuntimeError(f"rsync {src} -> {host} failed: {rs.stderr.strip()}")

    # 4) submit
    sub = _sh(["ssh", host, f"sbatch --parsable {run_root}/run_train.sbatch"])
    if sub.returncode != 0:
        raise RuntimeError(f"sbatch failed: {sub.stderr.strip() or sub.stdout.strip()}")
    jobid = sub.stdout.strip().split(";")[0]
    workspace.log("slurm_h100_submit", f"jobid={jobid}\nrun_root={run_root}\nhost={host}\n")

    # 5) poll sacct until terminal (or timeout)
    started = time.monotonic()
    state = "PENDING"
    while True:
        q = _sh(["ssh", host, f"sacct -j {jobid} --format=State --noheader -P"])
        state = (q.stdout.strip().splitlines() or ["PENDING"])[0].strip().split()[0] if q.stdout.strip() else state
        if state in _TERMINAL:
            break
        if time.monotonic() - started > timeout_s:
            _sh(["ssh", host, f"scancel {jobid}"])
            raise RuntimeError(f"slurm job {jobid} timed out after {timeout_s}s (last state {state})")
        time.sleep(poll_interval)
    if state != "COMPLETED":
        tail = _sh(["ssh", host, f"tail -30 {run_root}/train_{jobid}.err {run_root}/train_{jobid}.out 2>/dev/null"])
        raise RuntimeError(f"slurm job {jobid} ended {state}.\n{tail.stdout[-1500:]}")

    # 6) pull the adapter back into the local model dir
    pull = _sh(["rsync", "-az", "-e", "ssh", f"{host}:{run_root}/adapter/", f"{model_dir}/"], timeout=1200)
    if pull.returncode != 0:
        raise RuntimeError(f"rsync adapter back failed: {pull.stderr.strip()}")
    if not (model_dir / "adapter_config.json").is_file():
        raise RuntimeError(f"no adapter_config.json pulled into {model_dir}")

    meta = {
        "mode": "slurm_h100_lora",
        "base_model": base_id,
        "finetuning_type": "lora",
        "executor": "slurm_h100",
        "slurm": {"host": host, "jobid": jobid, "partition": partition, "qos": qos,
                  "run_root": run_root},
        "hyperparams": hp,
        "train_base_path": base_model,
        "selected_rows": dataset.hf_dataset.num_rows,
    }
    (model_dir / "training_config.json").write_text(
        json.dumps({**(config or {}), "_resolved_hyperparams": hp}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (model_dir / "training_manifest.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    model = Model(model_id=model_id, path=model_dir, meta=meta, producer="train_slurm_h100")
    workspace.log(
        "slurm_h100_summary",
        f"model={model_id}\njobid={jobid}\nbase={base_id}\nrows={dataset.hf_dataset.num_rows}\n",
    )
    return model
