"""LLaMA-Factory backend — formalizes the DataOS PLAYBOOK train step.

build() renders the LF config + a ``run_train.sh`` (canonical->alpaca convert, train, and
for LoRA the export/merge) as deterministic assets; the launcher runs them. collect() turns
the on-disk outputs into a canonical TrainResult whose ``report_fragment`` slots into the
DataOS report.json train/model/dataset sections.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from dataos_launch.io import read_json
from dataos_launch.types import Job, LaunchPlan, SlurmExtras, Status

from ...spec import DPO, KTO, ORPO, PPO, TrainSpec
from ...types import Capabilities, TrainResult
from ..base import is_hf_model_dir, tail_text
from .config import (
    SUPPORTED_METHODS,
    LlamaFactoryMethod,
    method_data_subdir,
    method_dataset_name,
    render_train_config,
    to_yaml,
)
from .ext.templates import DATAOS_TEMPLATES, model_family, resolve_template

_HOOKS = ["step", "reweight", "select"]

# PPO's TRL-pinned dependency set conflicts with the standard LF env (verified on
# yzc-3090-1; see envs/llamafactory-ppo-cu121-py311.constraints.txt), so it resolves
# its own env domain instead of the backend-level one.
ENV_DOMAINS = {"ppo": "llamafactory:ppo"}


class LlamaFactoryBackend:
    name = "llamafactory"

    def capabilities(self) -> Capabilities:
        return Capabilities(
            name=self.name,
            methods=list(SUPPORTED_METHODS),
            finetuning_types=["lora", "full", "freeze"],
            launch_kinds=["direct", "torchrun", "deepspeed"],
            online_hook_points=_HOOKS,
            produces_format="hf",
            needs_separate_env=False,
            env_domains=dict(ENV_DOMAINS),
            notes="SFT/DPO/PPO/KTO/ORPO via llamafactory-cli; lora/qlora "
            "(quantization_bit) merged to final_model via export; full/freeze write "
            "final_model directly. PPO requires a reward_model and its own env domain "
            "'llamafactory:ppo' (TRL pin conflicts with the standard LF env).",
        )

    def validate(self, spec: TrainSpec) -> list[str]:
        errors: list[str] = []
        m = spec.method
        if getattr(m, "kind", None) not in SUPPORTED_METHODS:
            errors.append(f"llamafactory backend supports method.kind in {SUPPORTED_METHODS}")
            return errors
        if not spec.base_model:
            errors.append("base_model is required")
        if spec.dataset is None:
            errors.append("dataset is required (path to canonical data)")
        if m.finetuning_type not in ("lora", "full", "freeze"):
            errors.append("finetuning_type must be 'lora', 'full', or 'freeze'")
        if m.template is not None and not str(m.template).strip():
            errors.append("template must be a non-empty name, or omitted to derive it from "
                          "the model's config.json model_type")
        if "model_name_or_path" in (m.extra or {}):
            errors.append("method.extra must not set model_name_or_path; code injects base_model")
        if isinstance(m, PPO):
            if not m.reward_model:
                errors.append("PPO requires method.reward_model (path or reward API URL)")
            elif m.reward_model_type == "api" and not m.reward_model.startswith("http"):
                errors.append(
                    "PPO reward_model_type='api' requires reward_model to be an HTTP(S) URL"
                )
            if m.reward_model_type == "lora" and m.finetuning_type != "lora":
                errors.append("PPO reward_model_type='lora' requires finetuning_type='lora'")
            ppo_domain = ENV_DOMAINS["ppo"]
            # A family-scoped domain counts as "an env is configured" — otherwise this check
            # silently stops firing for environments that scope by model family.
            configured = any(
                key == self.name or key.startswith(f"{self.name}@") for key in spec.env.backends
            )
            if configured and not any(k.startswith(ppo_domain) for k in spec.env.backends):
                errors.append(
                    f"PPO runs in its own env domain {ppo_domain!r} (its TRL pin conflicts "
                    "with the standard llamafactory env); set "
                    f"env.backends[{ppo_domain!r}] — point it at the same env only if that "
                    "env satisfies the PPO constraints"
                )
        for op in spec.online_ops:
            if op.hook is not None and op.hook not in _HOOKS:
                errors.append(f"unsupported online hook {op.hook!r}")
            self._check_online_op_resolvable(op.op, errors)
        self._warn_cutoff_vs_data(spec)
        return errors

    def _family_scoped(self, spec: TrainSpec) -> bool:
        """Whether the environment declares family-scoped env domains for this backend."""
        return any(key.startswith(f"{self.name}@") for key in spec.env.backends)

    def _env_for(self, spec: TrainSpec, method_kind: str, family: str | None) -> str | None:
        """The env this job runs in — and a hard error when the config covers no such family.

        An env whose transformers predates the architecture fails at model load, so the
        alternative to erroring here is 100+ dead jobs plus agents hand-rolling scripts that
        borrow another env to train in (2026-08-11 smol3 batch). Falling back to the unscoped
        domain would reproduce exactly that, so once the config opts into family scoping an
        undeclared family stops the build instead.
        """
        if self._family_scoped(spec) and f"{self.name}@{family}" not in spec.env.backends:
            declared = sorted(k for k in spec.env.backends if k.startswith(f"{self.name}@"))
            raise ValueError(
                f"no env declares model family {family!r}: env.backends has {declared} but "
                f"nothing for {self.name}@{family}. Add {family!r} to the `model_families` "
                "list of the env that can train it (the environment's env.yaml), or point a "
                "new env at it — an env whose transformers does not know this architecture "
                "fails at model load."
            )
        return spec.env.env_for(self.name, method_kind, family)

    @staticmethod
    def _warn_cutoff_vs_data(spec: TrainSpec) -> None:
        """Heuristic warning when many samples are longer than cutoff_len.

        Truncation mid-answer teaches the model to never stop (verified on
        Qwen3-1.7B-Base + LoRA on 3090: >40% truncated -> degenerate infinite
        generation). Chars/3 is a crude token estimate; stay a warning, not an error.
        """
        import json as _json
        import warnings

        m = spec.method
        cutoff = getattr(m, "cutoff_len", None)
        if not cutoff or spec.dataset is None:
            return
        path = Path(spec.dataset.path)
        if path.is_dir():
            path = path / "train.json"
        if not path.is_file() or path.suffix not in {".jsonl", ".json"}:
            return
        try:
            if path.suffix == ".jsonl":
                rows = [_json.loads(line) for _, line in zip(range(200), path.open(), strict=False)]
            else:
                rows = _json.loads(path.read_text())[:200]
        except (OSError, ValueError):
            return
        lengths = []
        for rec in rows:
            if isinstance(rec, dict) and "messages" in rec:
                text = "".join(str(msg.get("content", "")) for msg in rec["messages"])
            elif isinstance(rec, dict):
                text = str(rec.get("instruction", "")) + str(rec.get("input", "")) + str(
                    rec.get("output", "")
                )
            else:
                continue
            lengths.append(len(text))
        if not lengths:
            return
        over = sum(1 for n in lengths if n > cutoff * 3) / len(lengths)
        if over > 0.25:
            warnings.warn(
                f"{over:.0%} of sampled records exceed ~cutoff_len ({cutoff}) by a "
                "chars/3 token estimate — mid-answer truncation teaches non-termination. "
                "Raise cutoff_len or length-filter the data (see docs/training_pitfalls.md)",
                stacklevel=2,
            )

    @staticmethod
    def _check_online_op_resolvable(name: str, errors: list[str]) -> None:
        """Best-effort resolvability check for an online op. Unknown names are errors;
        an unconfirmable entry-point scan (provider not installed here) is a warning —
        never silent, and the probe stays torch-free."""
        import warnings

        from ...online import known_online_op_names, probe_online_op

        known = probe_online_op(name)
        if known is False:
            errors.append(
                f"unknown online op {name!r}; known: {known_online_op_names()} "
                "(external providers ship ops via the 'dataos_train.online_ops' entry-point "
                "group and must be installed in the validating env)"
            )
        elif known is None:
            warnings.warn(
                f"could not confirm online op {name!r}: the entry-point metadata scan "
                "failed, so resolvability is unverified (the job will fail at startup if "
                "the op is not registered there)",
                stacklevel=2,
            )

    def build(self, spec: TrainSpec) -> Job:
        if spec.dataset is None:
            raise ValueError("dataset is required")
        m: LlamaFactoryMethod = spec.method  # type: ignore[assignment]  # validated in validate()
        run_dir = spec.run_dir
        data_dir = f"{run_dir}/data/{method_data_subdir(m)}"
        dataset_name = method_dataset_name(m)
        adapter_dir = f"{run_dir}/adapter"
        final_dir = f"{run_dir}/final_model"
        output_dir = adapter_dir if m.finetuning_type == "lora" else final_dir
        # The model's own model_type drives BOTH halves of "how this model runs": which chat
        # template LF renders, and — when the environment declares family-scoped env domains —
        # which env it trains in. Resolve it ONCE, here, and carry the template on the method
        # copy so the rendered config, the assets meta, the export command and the entry
        # routing can never disagree.
        family = model_family(spec.base_model)
        derived = resolve_template(spec.base_model, family)
        # The template is DERIVED, never accepted. It is a function of the model, so a caller that
        # names a different one is asking for the failure this refuses: training under one family's
        # turn framing while the frozen scorer sends another's. That is not hypothetical — the
        # 2026-08-11 gsm8k batch produced its top score that way (audit 13.2), because the
        # online-ops entry point could not register `smollm3` and switching template was the only
        # way past the crash. Both halves are fixed: the entry registers, and this refuses.
        if m.template is not None and m.template != derived:
            raise ValueError(
                f"method.template={m.template!r} contradicts the model: {spec.base_model} is "
                f"model_type {family!r}, whose declared template is {derived!r}. The chat "
                "template is not a knob — it is derived from the model so training and the frozen "
                "scorer cannot diverge. Remove the field."
            )
        if m.template != derived:
            m = m.model_copy(update={"template": derived})

        cfg = render_train_config(spec, m, data_dir, output_dir, dataset_name)
        # LF 0.9.x only loads YAML correctly (JSON branch is buggy).
        assets = {
            "train_config.yaml": to_yaml(cfg),
            "dataos_meta.json": json.dumps(
                {
                    "base_model": spec.base_model,
                    "finetuning_type": m.finetuning_type,
                    "template": m.template,
                    "mode": m.kind,
                    "method": m.kind,
                },
                indent=2,
            ),
        }

        env: dict[str, str] = {}
        if spec.resources.gpus > 1:
            env["FORCE_TORCHRUN"] = "1"  # LF: use all visible GPUs via torchrun
        env.update(spec.env.extra)

        pythonpath = list(spec.env.pythonpath)
        if spec.env.deploy_root and not pythonpath:
            pythonpath = [f"{spec.env.deploy_root}/src", spec.env.deploy_root]

        # Self-sufficient script: the launcher preamble already activates the env, but the
        # script is also run on its own (by hand, or from a helper job). Without activation
        # the convert step falls through to /usr/bin/python and dies on
        # "No module named 'dataos_train'" — seen in 4b2 v5, gsm8k round-0002.
        train_env = self._env_for(spec, m.kind, family)
        steps = ["#!/bin/bash", "set -euo pipefail"]
        if spec.env.conda_sh and train_env:
            steps.append(
                f"source {shlex.quote(spec.env.conda_sh)} && "
                f"conda activate {shlex.quote(train_env)}"
            )
        elif train_env and train_env.startswith("/"):
            steps.append(f"source {shlex.quote(train_env)}/bin/activate")
        steps += [
            "python -m dataos_train.data.convert "
            f"--in {shlex.quote(spec.dataset.path)} --informat {shlex.quote(spec.dataset.format)} "
            f"--out {shlex.quote(data_dir)} --to {self._convert_target(m)} "
            f"--name {shlex.quote(dataset_name)}",
            f"{self._train_entry(spec, m.template)} {shlex.quote(run_dir)}/train_config.yaml",
        ]
        if m.finetuning_type == "lora":
            steps.append(
                f"{self._export_entry(m.template)} "
                f"--model_name_or_path {shlex.quote(spec.base_model)} "
                f"--adapter_name_or_path {shlex.quote(adapter_dir)} "
                f"--template {shlex.quote(m.template)} --finetuning_type lora "
                f"--export_dir {shlex.quote(final_dir)} --export_size 5"
            )
        assets["run_train.sh"] = "\n".join(steps) + "\n"

        return Job(
            backend=self.name,
            job_name=spec.name or m.kind,
            command=["bash", "run_train.sh"],
            env=env,
            run_dir=run_dir,
            resources=spec.resources,
            launch=LaunchPlan(kind="direct"),  # llamafactory-cli manages its own distribution
            conda_sh=spec.env.conda_sh,
            conda_env=train_env,
            pythonpath=pythonpath,
            assets=assets,
            slurm=_slurm_from_cluster(spec),
        )

    @staticmethod
    def _convert_target(method: LlamaFactoryMethod) -> str:
        if isinstance(method, (DPO, ORPO)):
            return "preference"
        if isinstance(method, KTO):
            return "kto"
        if isinstance(method, PPO):
            return "ppo"
        return "alpaca"

    def _train_entry(self, spec: TrainSpec, template: str) -> str:
        # Online operators require the patched entrypoint (Milestone 3).
        if spec.online_ops:
            ops = json.dumps([op.model_dump() for op in spec.online_ops])
            return (
                "python -m dataos_train.backends.llamafactory.ext.entry "
                f"--online-ops {shlex.quote(ops)} --"
            )
        # Templates LF does not ship (see ext/templates.py) need our registering wrapper; every
        # other template keeps LF's own console script, so this path carries only what needs it.
        if template in DATAOS_TEMPLATES:
            return "python -m dataos_train.backends.llamafactory.ext.lf_entry train"
        return "llamafactory-cli train"

    @staticmethod
    def _export_entry(template: str) -> str:
        if template in DATAOS_TEMPLATES:
            return "python -m dataos_train.backends.llamafactory.ext.lf_entry export"
        return "llamafactory-cli export"

    def collect(self, run_dir: str) -> TrainResult:
        rd = Path(run_dir)
        final = str(rd / "final_model")
        meta = {}
        meta_path = rd / "dataos_meta.json"
        if meta_path.is_file():
            try:
                meta = read_json(meta_path)
            except (OSError, ValueError):
                meta = {}
        ft = meta.get("finetuning_type", "lora")
        base = meta.get("base_model")
        template = meta.get("template")
        method = meta.get("method", meta.get("mode", "sft"))
        data_subdir = method if method in SUPPORTED_METHODS else "sft"
        ok = is_hf_model_dir(final)

        rows = self._count_rows(rd, data_subdir)
        artifacts = {
            "final_model": final,
            "data": str(rd / "data" / data_subdir),
            "logs": str(rd / "logs"),
        }
        if ft == "lora":
            artifacts["adapter"] = str(rd / "adapter")

        result = TrainResult(
            spec_id="",
            status=Status.COMPLETED if ok else Status.FAILED,
            model_path=final if ok else None,
            base_model=base,
            metrics=self._read_metrics(rd),
            rows=rows,
            artifacts=artifacts,
            report_fragment={
                "stages": {"train": {"status": "completed" if ok else "failed"}},
                "model": {
                    "path": final,
                    "base_model": base,
                    "finetuning_type": ft,
                    "template": template,
                    "mode": method,
                    "method": method,
                },
                "dataset": {"rows": rows},
            },
        )
        if not ok:
            result.error = "final_model is not a complete HF model dir"
            result.log_tail = self._log_tail(rd)
        return result

    @staticmethod
    def _read_metrics(rd: Path) -> dict:
        for sub in ("adapter", "final_model"):
            f = rd / sub / "all_results.json"
            if f.is_file():
                try:
                    data = read_json(f)
                except (OSError, ValueError):
                    continue
                if isinstance(data, dict):
                    return {k: v for k, v in data.items() if isinstance(v, (int, float))}
        for sub in ("adapter", "final_model"):
            f = rd / sub / "trainer_state.json"
            if f.is_file():
                try:
                    data = read_json(f)
                except (OSError, ValueError):
                    continue
                hist = data.get("log_history", []) if isinstance(data, dict) else []
                if hist:
                    return {k: v for k, v in hist[-1].items() if isinstance(v, (int, float))}
        return {}

    @staticmethod
    def _count_rows(rd: Path, data_subdir: str = "sft") -> int | None:
        p = rd / "data" / data_subdir / "train.json"
        if p.is_file():
            try:
                return len(read_json(p))
            except (OSError, ValueError):
                return None
        return None

    @staticmethod
    def _log_tail(rd: Path) -> str:
        candidates = sorted(rd.glob("slurm-*.out"))
        if (rd / "logs").is_dir():
            candidates += sorted((rd / "logs").glob("*.out"))
        for f in candidates:
            text = tail_text(f, 50)
            if text:
                return text
        return ""


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
