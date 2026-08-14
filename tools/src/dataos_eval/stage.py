"""The evaluation stage: what an evaluation IS, and how this deployment runs one.

The spec here is written by the tool, not by an agent, and that is the point rather than an
omission. PostTrainBench's protocol is frozen: the bench list, the chat template, the judge, the
limit semantics are all declared in ``configs/tasks/<task>/eval/protocol.yaml``, and an agent
that could state any of them could change its own score without changing its model. So an
``EvalSpec`` is assembled from the protocol plus the four things the caller legitimately names
— which bench, which model, where to put the outputs, how many items — and the rest is resolved
here, from config and from the model's own ``config.json``.

It is still a :class:`~dataos_launch.spec.Spec`. Authorship differs between the three stages;
the noun does not.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import yaml
from dataos_launch import Job, Resources
from dataos_launch.spec import Spec
from dataos_launch.stage import Stage


class EvalSpec(Spec):
    """One bench, one model, one output dir.

    ``chat_template`` and ``needs_judge`` are resolved rather than supplied: both come from the
    protocol, and the template is keyed on the MODEL's ``model_type``. A wrong chat template
    caps every score with nothing in any log, so it is not a knob anyone gets to reach.
    """

    stage = "eval"

    task: str
    model_path: str
    model_name: str
    output_dir: str
    limit: int = -1
    chat_template: str
    needs_judge: bool = False


def resolve_eval_template(repo: Path, task_config: str, bench: str, model_path: str,
                          protocol: dict) -> Path:
    """The protocol chat template for this model, from the MODEL's own config.json.

    Never from ``--model-name``/``--model-path`` substrings: that label is filled by the
    experimenter (task.yaml's ``ctx.label``), so guessing from it let an agent change the eval
    protocol by choosing a name. Keyed on ``model_type`` because a continuation node's model
    path is the parent's ``final_model`` dir and carries no family name, while a saved model
    always keeps its ``model_type``. Raises rather than guessing.
    """
    cfg = Path(model_path) / "config.json"
    try:
        model_type = json.loads(cfg.read_text(encoding="utf-8")).get("model_type")
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read {cfg} to resolve the chat template: {exc}") from exc
    family = (protocol.get("model_families") or {}).get(str(model_type))
    if not family:
        raise SystemExit(
            f"no chat template declared for model_type {model_type!r}; add it to "
            f"configs/tasks/{task_config}/eval/protocol.yaml model_families "
            "(together with the matching train_template)"
        )
    name = family.get(f"eval_template_{bench}") or family.get("eval_template")
    path = repo / "configs" / "tasks" / task_config / "eval" / "lmeval" / "templates" / name
    if not path.is_file():
        raise SystemExit(f"declared chat template does not exist: {path}")
    return path


class EvalStage(Stage):
    """Runs the frozen PostTrainBench evaluator, wherever this environment says to run things."""

    name = "eval"

    def __init__(self, env: dict, *, repo: Path, task_config: str,
                 task_cfg: dict, protocol: dict) -> None:
        super().__init__(env)
        self.repo = repo
        self.task_config = task_config
        self.task_cfg = task_cfg
        self.protocol = protocol

    @classmethod
    def from_repo(cls, repo: Path, environment: str, task_config: str = "posttrainbench",
                  **_: object) -> EvalStage:
        """The deployment plus the TASK contract — the only stage that needs both."""
        task_dir = repo / "configs" / "tasks" / task_config
        read = lambda p: yaml.safe_load(p.read_text(encoding="utf-8"))  # noqa: E731
        return cls(
            cls.load_env(repo, environment),
            repo=repo,
            task_config=task_config,
            task_cfg=read(task_dir / "task.yaml"),
            protocol=read(task_dir / "eval" / "protocol.yaml"),
        )

    @property
    def benches(self) -> dict:
        return self.protocol.get("benches") or {}

    def spec_for(self, *, task: str, model_path: str, model_name: str, output_dir: str,
                 run_root: str, limit: int, resources: Resources) -> EvalSpec:
        """Assemble the spec: what the caller named, plus what the protocol decides."""
        bench = self.benches.get(task)
        if bench is None:
            raise SystemExit(
                f"bench {task!r} is not declared in configs/tasks/{self.task_config}/eval/"
                "protocol.yaml benches"
            )
        template = resolve_eval_template(
            self.repo, self.task_config, task, model_path, self.protocol
        )
        return EvalSpec(
            name=f"eval_{task}",
            run_dir=run_root,
            resources=resources,
            task=task,
            model_path=model_path,
            model_name=model_name,
            output_dir=output_dir,
            limit=limit,
            chat_template=str(template),
            needs_judge=bool(bench.get("judge")),
        )

    def validate(self, spec: EvalSpec) -> list[str]:
        errors = []
        if spec.task not in self.benches:
            errors.append(
                f"bench {spec.task!r} is not declared in configs/tasks/{self.task_config}/"
                "eval/protocol.yaml benches"
            )
        if not Path(spec.model_path, "config.json").is_file():
            errors.append(f"{spec.model_path} is not a model directory (no config.json)")
        if not Path(spec.chat_template).is_file():
            errors.append(f"chat template does not exist: {spec.chat_template}")
        return errors

    def build(self, spec: EvalSpec) -> Job:
        """Render this evaluation as a launcher-agnostic Job.

        Every hard-won environment lesson still lives here (short TMPDIR, libcuda link, cache
        redirects, offline HF) — but as ``pre_cmds`` and ``env`` rather than text baked into a
        bash template, so the SAME evaluation can be run by any launcher.

        Two things are deliberately explicit:

        * ``PYTORCH_CUDA_ALLOC_CONF`` is set to the empty string. ``build_preamble`` injects
          ``expandable_segments:True`` for TRAINING jobs, where fragmentation costs OOMs — but
          on an eval job it changes how vLLM's reservation is accounted (audit 12: a nominal 26
          GiB reported as 24320 MiB), and evaluation must not silently acquire a training
          default.
        * ``TMPDIR`` uses ``$SLURM_JOB_ID`` when Slurm sets it and the pid otherwise: the path
          must stay short (the AF_UNIX socket limit is 108 chars) and unique per job, on both a
          shared cluster node and a workstation.
        """
        run_root = shlex.quote(spec.run_dir)
        eval_py = f"{self.env['tools']['eval_env']['path']}/bin/python"
        bridge = self.repo / "configs" / "tasks" / self.task_config / "eval" / "scripts"
        benchmarks = self.repo / "configs" / "tasks" / self.task_config / "benchmarks"

        pre_cmds = [
            "export TMPDIR=/tmp/job_${SLURM_JOB_ID:-$$}",
            f'mkdir -p "$TMPDIR" "$TMPDIR/culib" {run_root}/cache/torch {run_root}/cache/triton '
            f"{run_root}/cache/inductor",
            "trap 'rm -rf \"$TMPDIR\"' EXIT",
            # triton/inductor compile needs libcuda.so; some nodes ship only libcuda.so.1
            "if [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then",
            '  ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$TMPDIR/culib/libcuda.so"',
            "elif [ -e /usr/lib64/libcuda.so.1 ]; then",
            '  ln -sf /usr/lib64/libcuda.so.1 "$TMPDIR/culib/libcuda.so"',
            "fi",
            'export LIBRARY_PATH="$TMPDIR/culib:${LIBRARY_PATH:-}"',
            'export TRITON_LIBCUDA_PATH="$TMPDIR/culib"',
        ]
        job_env = {
            "TORCH_HOME": f"{spec.run_dir}/cache/torch",
            "TRITON_CACHE_DIR": f"{spec.run_dir}/cache/triton",
            "TORCHINDUCTOR_CACHE_DIR": f"{spec.run_dir}/cache/inductor",
            "POSTTRAINBENCH_BENCHMARKS_DIR": str(benchmarks),
            "HF_DATASETS_CACHE": f"{benchmarks}/hf_datasets",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "",  # see docstring: a training default, not an eval one
            # Server benches start a vLLM server and poll ITS OWN /health on 127.0.0.1. With an
            # HTTP proxy in the environment and no NO_PROXY, that request goes to the proxy,
            # which cannot reach a port on this machine — so the server comes up healthy and the
            # poller still times out after 10 minutes, blaming vLLM. No proxy is set on the
            # cluster, which is why this only ever appeared on a workstation.
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
        return Job(
            backend="lmeval",
            job_name=spec.name,
            command=[
                eval_py, str(bridge / "run_lmeval_eval.py"),
                "--task", spec.task,
                "--model-path", spec.model_path,
                "--model-name", spec.model_name,
                "--chat-template", spec.chat_template,
                "--output-dir", spec.output_dir,
                "--limit", str(spec.limit),
            ],
            env=job_env,
            run_dir=spec.run_dir,
            resources=spec.resources,
            pre_cmds=pre_cmds,
        )

    def judge_command(self, task: str, output_dir: str, summary_path: Path) -> str:
        """The judge run, on the login node — the only place with internet."""
        judge = (self.task_cfg.get("scoring") or {}).get("judge") or {}
        gateway = self.env.get("gateway") or {}
        bridge = self.repo / "configs" / "tasks" / self.task_config / "eval" / "scripts"
        benchmarks = self.repo / "configs" / "tasks" / self.task_config / "benchmarks"
        env_vars = (
            f"OPENAI_API_KEY=$(cat {shlex.quote(gateway['key_file'])}) "
            f"OPENAI_BASE_URL={shlex.quote(gateway['base_url'])} "
            f"POSTTRAINBENCH_JUDGE_MODEL={shlex.quote(str(judge.get('model')))} "
            f"POSTTRAINBENCH_BENCHMARKS_DIR={shlex.quote(str(benchmarks))}"
        )
        return (
            f"{env_vars} {self.env['tools']['eval_env']['path']}/bin/python "
            f"{bridge}/run_lmeval_judge.py --bench {shlex.quote(task)} "
            f"--predictions {shlex.quote(output_dir)}/predictions.jsonl "
            f"--output {shlex.quote(str(summary_path))} --workers {judge.get('workers', 32)}"
        )
