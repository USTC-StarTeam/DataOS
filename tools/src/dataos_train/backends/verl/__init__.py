"""verl GRPO backend — RL post-training with reward == eval.

build() renders a ``run_train.sh`` that (1) converts canonical RL data to verl parquet,
(2) runs ``verl.trainer.main_ppo`` with rendered overrides, and (3) merges the verl
checkpoint into an HF ``final_model`` (so eval/child-training load it like any model).
collect() parses the outcome. verl needs its own conda env; real runs are gated on it.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from dataos_launch.io import read_json
from dataos_launch.types import Job, LaunchPlan, SlurmExtras, Status

from ...spec import GRPO, TrainSpec, VerlPPO
from ...types import Capabilities, TrainResult
from ..base import is_hf_model_dir, tail_text
from .collect import parse_metrics
from .config import render_overrides

SUPPORTED_METHODS = ("grpo", "verl_ppo")


class VerlBackend:
    name = "verl"

    def capabilities(self) -> Capabilities:
        return Capabilities(
            name=self.name,
            methods=list(SUPPORTED_METHODS),
            finetuning_types=["full"],  # LoRA-merge path deferred
            supports_multi_node=False,  # single-node Ray (in-process) for v1
            launch_kinds=["direct"],
            online_hook_points=["reward"],  # RL reward/dynamic-sampling is the online seam
            produces_format="hf",
            needs_separate_env=True,
            notes="GRPO and classic PPO (GAE + critic, method 'verl_ppo') via "
            "verl.trainer.main_ppo; reward is an INJECTED file path (method.reward.path, "
            "runtime-owned — typically agent-written from the task's format contract); "
            "RL prompts should arrive already eval-template-aligned (a data-stage job; "
            "optional wrapping via the data converter's --template-file). Checkpoint "
            "merged to HF final_model in-job. Requires a separate 'verl' conda env.",
        )

    def validate(self, spec: TrainSpec) -> list[str]:
        errors: list[str] = []
        m = spec.method
        if getattr(m, "kind", None) not in SUPPORTED_METHODS:
            errors.append(f"verl backend supports method.kind in {SUPPORTED_METHODS}")
            return errors
        rl: GRPO | VerlPPO = m  # type: ignore[assignment]
        if not spec.base_model:
            errors.append("base_model is required")
        if spec.dataset is None:
            errors.append("dataset is required (path to canonical RL data)")
        if not rl.reward.path:
            errors.append(
                "method.reward.path is required (runtime-owned): absolute path to a reward "
                "file defining compute_score(data_source, solution_str, ground_truth, ...) -> "
                "float; verl loads it by path (custom_reward_function). This library ships no "
                "task rewards — the driving agent writes one from the task's format contract."
            )
        elif not str(rl.reward.path).endswith(".py"):
            errors.append("method.reward.path must point to a .py file")
        if isinstance(rl, GRPO) and rl.group_size != rl.rollout.n:
            errors.append(
                f"GRPO group_size ({rl.group_size}) must equal rollout.n ({rl.rollout.n})"
            )
        if rl.train_batch % rl.ppo_mini_batch != 0:
            errors.append("train_batch must be divisible by ppo_mini_batch")
        per_step = spec.resources.gpus * rl.ppo_micro_batch_per_gpu
        if per_step and rl.ppo_mini_batch % per_step != 0:
            errors.append("ppo_mini_batch must be divisible by gpus * ppo_micro_batch_per_gpu")
        if spec.resources.gpus < rl.rollout.tensor_parallel_size:
            errors.append("resources.gpus must be >= rollout.tensor_parallel_size")
        if (
            rl.rollout.tensor_parallel_size > 0
            and spec.resources.gpus % rl.rollout.tensor_parallel_size != 0
        ):
            errors.append("resources.gpus must be divisible by rollout.tensor_parallel_size")
        attention_heads = _local_attention_heads(spec.base_model)
        if (
            attention_heads is not None
            and rl.rollout.tensor_parallel_size > 0
            and attention_heads % rl.rollout.tensor_parallel_size != 0
        ):
            errors.append(
                "base_model num_attention_heads "
                f"({attention_heads}) must be divisible by rollout.tensor_parallel_size "
                f"({rl.rollout.tensor_parallel_size})"
            )
        if isinstance(rl, VerlPPO):
            critic = rl.critic_model or spec.base_model
            critic_heads = _local_attention_heads(critic)
            if (
                critic_heads is not None
                and rl.rollout.tensor_parallel_size > 0
                and critic_heads % rl.rollout.tensor_parallel_size != 0
            ):
                errors.append(
                    f"critic_model num_attention_heads ({critic_heads}) must be divisible "
                    f"by rollout.tensor_parallel_size ({rl.rollout.tensor_parallel_size})"
                )
        if spec.resources.nodes > 1:
            errors.append("verl multi-node not supported in this version (set resources.nodes=1)")
        if not spec.env.env_for(self.name, rl.kind):
            errors.append("verl conda env required at env.backends['verl'] (or 'verl:<method>')")
        return errors

    def build(self, spec: TrainSpec) -> Job:
        rl: GRPO | VerlPPO = spec.method  # type: ignore[assignment]  # validated
        run_dir = spec.run_dir
        data_dir = f"{run_dir}/data/rl"
        reward_path = rl.reward.path or ""

        overrides = render_overrides(
            spec, rl, spec.base_model, spec.resources, reward_path, data_dir
        )
        verl_cmd = "python -m verl.trainer.main_ppo " + " ".join(shlex.quote(o) for o in overrides)

        pythonpath = list(spec.env.pythonpath)
        if not pythonpath and spec.env.deploy_root:
            pythonpath = [f"{spec.env.deploy_root}/src", spec.env.deploy_root]
        env = {
            "VLLM_ATTENTION_BACKEND": "XFORMERS",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "RAY_DEDUP_LOGS": "0",
        }
        env.update(spec.env.extra)

        steps = [
            "#!/bin/bash",
            "set -euo pipefail",
            "python -m dataos_train.backends.verl.data "
            f"--in {shlex.quote(spec.dataset.path)} --informat {shlex.quote(spec.dataset.format)} "
            f"--out {shlex.quote(data_dir)} --bench {shlex.quote(rl.reward.ref)}",
            verl_cmd,
            "python -m dataos_train.backends.verl.collect "
            f"--merge --run-dir {shlex.quote(run_dir)}",
        ]
        meta = {
            "base_model": spec.base_model,
            "bench": rl.reward.ref,
            "finetuning_type": rl.finetuning_type,
            "mode": rl.kind,
        }
        assets = {
            "run_train.sh": "\n".join(steps) + "\n",
            "dataos_meta.json": json.dumps(meta, indent=2),
        }
        return Job(
            backend=self.name,
            job_name=spec.name or rl.kind,
            command=["bash", "run_train.sh"],
            env=env,
            run_dir=run_dir,
            resources=spec.resources,
            launch=LaunchPlan(kind="direct"),  # single-node: Ray runs in-process inside main_ppo
            conda_sh=spec.env.conda_sh,
            conda_env=spec.env.env_for(self.name, rl.kind),
            pythonpath=pythonpath,
            assets=assets,
            slurm=_slurm_from_cluster(spec),
        )

    def collect(self, run_dir: str) -> TrainResult:
        rd = Path(run_dir)
        final = str(rd / "final_model")
        ok = is_hf_model_dir(final)
        meta = {}
        meta_path = rd / "dataos_meta.json"
        if meta_path.is_file():
            try:
                meta = read_json(meta_path)
            except (OSError, ValueError):
                meta = {}
        rows = self._count_rows(rd)
        result = TrainResult(
            spec_id="",
            status=Status.COMPLETED if ok else Status.FAILED,
            model_path=final if ok else None,
            base_model=meta.get("base_model"),
            metrics=parse_metrics(run_dir),
            rows=rows,
            artifacts={
                "final_model": final,
                "data": str(rd / "data" / "rl"),
                "ckpt": str(rd / "verl" / "ckpt"),
                "logs": str(rd / "logs"),
            },
            report_fragment={
                "stages": {"train": {"status": "completed" if ok else "failed"}},
                "model": {
                    "path": final,
                    "base_model": meta.get("base_model"),
                    "finetuning_type": meta.get("finetuning_type", "full"),
                    "mode": meta.get("mode", "grpo"),
                },
                "dataset": {"rows": rows},
            },
        )
        if not ok:
            result.error = "final_model missing/incomplete (verl checkpoint merge may have failed)"
            result.log_tail = _log_tail(rd)
        return result

    @staticmethod
    def _count_rows(rd: Path) -> int | None:
        p = rd / "data" / "rl" / "_count.json"
        if p.is_file():
            try:
                return int(read_json(p).get("rows"))
            except (OSError, ValueError, TypeError):
                return None
        return None


def _slurm_from_cluster(spec: TrainSpec) -> SlurmExtras:
    c = spec.cluster or {}
    return SlurmExtras(
        partition=c.get("partition"),
        qos=c.get("qos"),
        gres=c.get("gres"),
        nodelist=c.get("nodelist"),
        cpus_per_task=c.get("cpus_per_task"),
        account=c.get("account"),
    )


def _local_attention_heads(base_model: str) -> int | None:
    config_path = Path(base_model) / "config.json"
    if not config_path.is_file():
        return None
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    value = raw.get("num_attention_heads")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _log_tail(rd: Path) -> str:
    candidates = sorted(rd.glob("slurm-*.out"))
    if (rd / "logs").is_dir():
        candidates += sorted((rd / "logs").glob("*.out"))
    for f in candidates:
        text = tail_text(f, 50)
        if text:
            return text
    return ""
