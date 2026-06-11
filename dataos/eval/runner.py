"""Drive the lm-evaluation-harness bridge from the eval stage.

``run_eval_stage`` is the single entry point. It builds an
:class:`EvalConfig` from the open ``config`` dict and routes either to the
original DataOSLM adapter bridge or to lm-eval's native model backends.
The :class:`LmEvalRunOutput` is then packaged into one :class:`Eval`
whose meta carries the per-task results, the headline score, a protocol
snapshot, and a budget-trimmed copy of the raw lm-eval payload. Legacy
``provider="stub"`` configs still emit a local stub eval for smoke tests and
dry runs.

The model argument is any :class:`dataos.experiment.Model` (structurally a
``ModelLike``): the runner only reads ``.path`` / ``.meta`` / ``.artifact_id``
/ ``.producer``.
"""

from __future__ import annotations

import dataclasses
import json
import re
import warnings
from typing import TYPE_CHECKING, Any

from dataos.eval.config import EvalConfig
from dataos._artifact import Eval, Model
from dataos._ids import _make_id
from dataos._redact import redact

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dataos.workspace import Workspace

_RAW_TRIM_MAX_KEYS = 50


def run_eval_stage(
    model: Model,
    config: dict[str, Any] | None,
    *,
    workspace: "Workspace",
) -> Eval:
    """Run an lm-eval task suite and emit a single :class:`Eval`."""
    eval_config = dict(config or {})
    if "suite" in eval_config:  # key presence, not truthiness — {"suite": []} must hit the validator
        return _suite_eval_artifact(model, eval_config, workspace=workspace)
    if eval_config.get("provider") == "stub":
        evaluation = _stub_eval_artifact(workspace, model, eval_config)
        workspace.log(
            "eval_model",
            _build_stub_log_message(model, evaluation, eval_config),
        )
        return evaluation
    if eval_config.get("provider") == "external":
        evaluation = _external_eval_artifact(workspace, model, eval_config)
        workspace.log("eval_model", _build_external_log_message(model, evaluation))
        return evaluation
    eval_config.pop("provider", None)
    eval_config.pop("provider_config", None)
    # eval config maps to EvalConfig's fields (a broad but fixed set). Tolerate unknown keys
    # instead of crashing with a raw TypeError — drop them with a warning.
    known = {f.name for f in dataclasses.fields(EvalConfig)}
    unknown = sorted(set(eval_config) - known)
    if unknown:
        warnings.warn(f"eval: ignoring config keys not in EvalConfig: {unknown}", stacklevel=2)
        eval_config = {k: v for k, v in eval_config.items() if k in known}
    config_obj = EvalConfig(**eval_config)
    mode = config_obj.normalized_mode()
    if mode == "lm_eval_native":
        output = _run_lm_eval_native(model, config_obj)
    else:
        adapter = _make_adapter(config_obj.adapter, config_obj.adapter_config)
        output = _run_lm_eval(model, adapter, config_obj)
    evaluation = _eval_output_to_artifact(output, workspace, model, config_obj)
    workspace.log("eval_model", _build_log_message(model, evaluation, output, config_obj))
    return evaluation


def _entry_slug(name: Any, index: int, used: set[str]) -> str:
    """Directory-safe, unique suite-entry name (no path escapes, no clash with results.json)."""
    slug = re.sub(r"[^0-9A-Za-z._-]+", "-", str(name)).strip(".-") or f"entry{index}"
    if slug == "results.json":
        slug = f"entry{index}"
    candidate, counter = slug, 1
    while candidate in used:
        counter += 1
        candidate = f"{slug}#{counter}"
    used.add(candidate)
    return candidate


def _merge_unique(target: dict[str, Any], key: str, prefix: str, value: Any) -> None:
    """Insert without ever overwriting: collide -> '<prefix>::<key>', then '...#2', '...#3'."""
    candidate = key
    if candidate in target:
        candidate = f"{prefix}::{key}"
        counter = 1
        while candidate in target:
            counter += 1
            candidate = f"{prefix}::{key}#{counter}"
    target[candidate] = value


class _SuiteEntryWorkspace:
    """Route one suite entry's artifacts into ``eval/<entry>/`` so entries don't clobber
    each other (the flat workspace layout keeps a single dir per artifact kind)."""

    def __init__(self, base: "Workspace", entry_dir: Any) -> None:
        self._base = base
        self._entry_dir = entry_dir

    def artifact_dir(self, kind: str, _id: str) -> Any:
        del kind, _id
        self._entry_dir.mkdir(parents=True, exist_ok=True)
        return self._entry_dir

    def log(self, name: str, message: str) -> Any:
        return self._base.log(name, message)


def _suite_eval_artifact(
    model: Model,
    eval_config: dict[str, Any],
    *,
    workspace: "Workspace",
) -> Eval:
    """Combine several eval runs (lm-eval / external / stub entries) into ONE :class:`Eval`.

    This is how a single experiment gets scored against a whole scenario (e.g. PostTrainBench
    = one lm-eval entry covering 4 tasks + 3 external harness entries). Each entry runs through
    the normal single-provider path (keeping its seeds / receipts / task isolation); results
    merge into one artifact: union ``metrics``, union ``per_task``, ``macro_average`` across
    every bench, and entry-level failure isolation (a broken harness lands in ``skipped_tasks``
    instead of voiding the others).
    """
    entries = eval_config.get("suite")
    if not isinstance(entries, list) or not entries:
        raise ValueError("eval suite must be a non-empty list of eval-config dicts")
    for index, entry in enumerate(entries):  # validate the WHOLE shape before running anything
        if not isinstance(entry, dict):
            raise ValueError(f"suite entry #{index} must be a dict eval config")
        if entry.get("suite"):
            raise ValueError("nested eval suites are not supported")
    eval_id = _make_id("eval")
    eval_dir = workspace.artifact_dir("eval", eval_id)
    merged_metrics: dict[str, float] = {}
    per_task: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    receipts: list[dict[str, Any]] = []
    seeds: dict[str, Any] | None = None
    used_names: set[str] = set()
    for index, entry in enumerate(entries):
        entry_cfg = dict(entry)
        name = _entry_slug(
            entry_cfg.pop("name", None)
            or entry_cfg.get("provider")
            or entry_cfg.get("mode")
            or f"entry{index}",
            index,
            used_names,
        )
        sub_workspace = _SuiteEntryWorkspace(workspace, eval_dir / name)
        try:
            sub = run_eval_stage(model, entry_cfg, workspace=sub_workspace)
        except Exception as exc:  # noqa: BLE001 - isolate entries like lm-eval tasks
            skipped[name] = redact(str(exc))  # error text may quote harness stderr — scrub it
            continue
        for key, value in (sub.metrics or {}).items():
            _merge_unique(merged_metrics, key, name, value)
        sub_per_task = sub.meta.get("per_task")
        if isinstance(sub_per_task, dict) and sub_per_task:
            for task, value in sub_per_task.items():
                _merge_unique(per_task, task, name, value)
        else:
            _merge_unique(per_task, name, name, sub.score)
        if seeds is None and sub.meta.get("seeds"):
            seeds = dict(sub.meta["seeds"])
        for task, reason in (sub.meta.get("skipped_tasks") or {}).items():
            skipped[f"{name}::{task}"] = redact(str(reason))
        receipts.append(
            {
                "name": name,
                "backend": sub.meta.get("backend"),
                "mode": sub.meta.get("mode"),
                "tasks": sub.meta.get("tasks"),
                "command": sub.meta.get("command"),
                "harness": sub.meta.get("harness"),
                "seeds": sub.meta.get("seeds"),
                "score": sub.score,
                "metrics": dict(sub.metrics or {}),
                "artifact_id": sub.artifact_id,
                "path": str(sub.path),
            }
        )
    if not receipts:
        raise RuntimeError(f"all suite entries failed: {skipped}")
    present = [value for value in per_task.values() if isinstance(value, (int, float))]
    macro = sum(present) / len(present) if present else None
    summary: dict[str, Any] = {
        "backend": "suite",
        "suite": receipts,
        "n_entries": len(entries),
        "score": macro,
        "headline_score": macro,
        "macro_average": macro,
        "per_task": per_task,
        "skipped_tasks": skipped,
        "seeds": seeds,
        "source_model_id": model.artifact_id,
        "base_model": model.meta.get("base_model"),
    }
    (eval_dir / "results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    workspace.log(
        "eval_model",
        f"model={model.artifact_id}\neval={eval_id}\nbackend=suite\n"
        f"entries={len(receipts)}/{len(entries)}\n"
        f"score={macro if macro is not None else 'n/a'}\n",
    )
    return Eval(
        eval_id=eval_id,
        path=eval_dir,
        metrics=merged_metrics,
        score=float(macro) if macro is not None else None,
        meta=summary,
        producer="eval_model",
    )


def _stub_eval_artifact(
    workspace: "Workspace",
    model: Model,
    eval_config: dict[str, Any],
) -> Eval:
    provider_config = dict(eval_config.get("provider_config", {}))
    score = float(provider_config.get("score", eval_config.get("score", 0.0)))
    eval_id = _make_id("eval")
    eval_dir = workspace.artifact_dir("eval", eval_id)
    summary: dict[str, Any] = {
        "backend": "stub",
        "provider": "stub",
        "score": score,
        "headline_score": score,
        "source_model_id": model.artifact_id,
        "base_model": model.meta.get("base_model"),
        "provider_config": provider_config,
    }
    (eval_dir / "results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return Eval(
        eval_id=eval_id,
        path=eval_dir,
        metrics={},
        score=score,
        meta=summary,
        producer="eval_model",
    )


def _external_eval_artifact(
    workspace: "Workspace",
    model: Model,
    eval_config: dict[str, Any],
) -> Eval:
    """Run an external benchmark harness (Arena-Hard / BFCL / HealthBench style) and ingest
    its score receipt.

    IoC: DataOS only runs the command deterministically and records the result — the
    judge/harness itself lives outside. The command must write JSON
    ``{"score": <float>, "metrics": {...}}`` to the ``{results_file}`` placeholder path.
    Secrets (judge API keys, HF tokens) go through the process env, never the config.
    """
    import os
    import subprocess

    command = eval_config.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError("external eval needs config['command'] (a shell command string)")
    eval_id = _make_id("eval")
    eval_dir = workspace.artifact_dir("eval", eval_id)
    results_path = eval_dir / str(eval_config.get("results_file") or "external_results.json")
    rendered = command
    for key, value in {
        "model_path": str(model.path),
        "base_model": str(model.meta.get("base_model") or ""),
        "output_dir": str(eval_dir),
        "results_file": str(results_path),
    }.items():
        rendered = rendered.replace("{" + key + "}", value)

    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in (eval_config.get("env") or {}).items()})
    timeout = eval_config.get("timeout")
    proc = subprocess.run(  # noqa: S602 - command is authored by the runtime via the spec
        rendered,
        shell=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=float(timeout) if timeout else None,
    )
    # harness output may echo env/tokens — scrub before anything reaches disk or error text
    (eval_dir / "external_stdout.log").write_text(redact(proc.stdout[-20000:]), encoding="utf-8")
    (eval_dir / "external_stderr.log").write_text(redact(proc.stderr[-20000:]), encoding="utf-8")
    if proc.returncode != 0:
        tail = redact((proc.stderr or proc.stdout or "")[-800:])
        raise RuntimeError(f"external eval command failed (exit {proc.returncode}): {tail}")
    if not results_path.is_file():
        raise RuntimeError(
            f"external eval produced no results file at {results_path} — the command must write "
            'JSON {"score": <float>, "metrics": {...}} there (use the {results_file} / '
            "{output_dir} placeholders)"
        )
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    metrics = {
        str(key): float(value)
        for key, value in (payload.get("metrics") or {}).items()
        if isinstance(value, (int, float))
    }
    score = payload.get("score")
    if score is None and metrics:
        headline = eval_config.get("headline_metric")
        score = metrics[headline] if headline in metrics else next(iter(metrics.values()))
    summary: dict[str, Any] = {
        "backend": "external",
        "command": redact(rendered),  # the receipt for the manifest; token-looking parts scrubbed
        # frozen-harness identity ({repo, commit, judge_model, ...}) — fill it when freezing
        # the harness so the manifest can prove which scorer produced the number
        "harness": redact(eval_config.get("harness")),
        "results_file": str(results_path),
        "score": score,
        "headline_score": score,
        "metrics": metrics,
        "extra": redact({k: v for k, v in payload.items() if k not in {"score", "metrics"}}),
        "source_model_id": model.artifact_id,
        "base_model": model.meta.get("base_model"),
    }
    (eval_dir / "results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return Eval(
        eval_id=eval_id,
        path=eval_dir,
        metrics=metrics,
        score=float(score) if score is not None else None,
        meta=summary,
        producer="eval_model",
    )


def _build_external_log_message(model: Model, evaluation: Eval) -> str:
    lines = [
        f"model={model.artifact_id}",
        f"eval={evaluation.artifact_id}",
        "backend=external",
        f"score={evaluation.score if evaluation.score is not None else 'n/a'}",
    ]
    return "\n".join(lines) + "\n"


def _make_adapter(name: str, adapter_config: dict[str, Any]) -> Any:
    del adapter_config  # adapters read their knobs off AdapterCallContext at call time
    if name == "transformers":
        from dataos.eval.adapters.transformers import TransformersAdapter

        return TransformersAdapter()
    if name == "openai_compatible":
        from dataos.eval.adapters.openai_compatible import OpenAICompatibleAdapter

        return OpenAICompatibleAdapter()
    raise ValueError(f"Unknown eval adapter {name!r}; expected 'transformers' or 'openai_compatible'")


def _run_lm_eval(model: Model, adapter: Any, config: EvalConfig):
    from dataos.eval.backend import run_lm_eval

    # Forward the seed hierarchy to lm-eval so the legacy-adapter path is reproducible too
    # (native mode already does this; this closes the gap).
    seeds = {
        "random_seed": config.random_seed,
        "numpy_random_seed": config.numpy_random_seed,
        "torch_random_seed": config.torch_random_seed,
        "fewshot_random_seed": config.fewshot_random_seed,
    }
    return run_lm_eval(
        model,
        adapter,
        tasks=list(config.tasks),
        limit=config.limit,
        num_fewshot=config.num_fewshot,
        batch_size=config.batch_size,
        max_new_tokens=config.max_new_tokens,
        apply_chat_template=config.apply_chat_template,
        confirm_run_unsafe_code=config.confirm_run_unsafe_code,
        adapter_config=dict(config.adapter_config),
        extra_simple_evaluate_kwargs=seeds,
    )


def _run_lm_eval_native(model: Model, config: EvalConfig):
    from dataos.eval.backend import run_lm_eval_native

    return run_lm_eval_native(model, config)


def _eval_seeds(config: EvalConfig) -> dict[str, Any]:
    """The effective seed hierarchy lm-eval ran with (EvalConfig defaults if user didn't set)."""
    return {
        "seed": config.seed,
        "random_seed": config.random_seed,
        "numpy_random_seed": config.numpy_random_seed,
        "torch_random_seed": config.torch_random_seed,
        "fewshot_random_seed": config.fewshot_random_seed,
    }


def _eval_output_to_artifact(
    output: Any,
    workspace: "Workspace",
    model: Model,
    config: EvalConfig,
) -> Eval:
    eval_id = _make_id("eval")
    eval_dir = workspace.artifact_dir("eval", eval_id)
    metrics = _flatten_metrics(output.results)
    headline = _pick_headline(metrics, config.tasks, config.headline_metric)
    per_task, macro_average = _macro_average(metrics, list(config.tasks), config.headline_metric)
    protocol = _protocol_snapshot(output, model, config)
    summary: dict[str, Any] = {
        "backend": "lm_eval",
        "mode": config.normalized_mode(),
        "tasks": list(config.tasks),
        "limit": config.limit,
        "num_fewshot": config.num_fewshot,
        "batch_size": config.batch_size,
        "model": protocol["resolved"].get("lm_eval_model"),
        "adapter": config.adapter if config.normalized_mode() == "legacy_adapter" else None,
        "results": dict(output.results),
        "score": headline,
        "headline_score": headline,
        "macro_average": macro_average,  # mean of per-bench primary metrics — the multi-task aggregate
        "per_task": per_task,            # {task: its primary metric value}
        "n_tasks": len(config.tasks),
        "skipped_tasks": dict(protocol["resolved"].get("skipped_tasks") or {}),
        "source_model_id": model.artifact_id,
        "base_model": model.meta.get("base_model"),
        "seeds": _eval_seeds(config),  # effective seeds (incl. EvalConfig defaults) for repro
    }
    (eval_dir / "results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (eval_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    raw = _trim_raw(output.raw)
    (eval_dir / "lm_eval_results.json").write_text(
        json.dumps(raw, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    if output.samples:
        for task_name, samples in output.samples.items():
            safe = task_name.replace("/", "_")
            with (eval_dir / f"samples-{safe}.jsonl").open("w", encoding="utf-8") as handle:
                for record in samples:
                    handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return Eval(
        eval_id=eval_id,
        path=eval_dir,
        metrics=metrics,
        score=headline,
        meta=summary,
        producer="eval_model",
    )


def _flatten_metrics(results: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for task_name, metrics in (results or {}).items():
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and "stderr" not in key and key != "alias":
                flat[f"{task_name}::{key}"] = float(value)
    return flat


def _pick_headline(
    metrics: dict[str, float], tasks: list[str], headline_metric: str | None = None
) -> float | None:
    # A task can emit several metrics (e.g. gsm8k -> strict-match AND flexible-extract). If the
    # caller pinned one via config["headline_metric"], honor it; otherwise fall back to the first.
    if headline_metric:
        for task in tasks:
            key = f"{task}::{headline_metric}"
            if key in metrics:
                return metrics[key]
    for task in tasks:
        for key, value in metrics.items():
            if key.startswith(f"{task}::"):
                return value
    return None


def _macro_average(
    metrics: dict[str, float], tasks: list[str], headline_metric: str | None
) -> tuple[dict[str, float | None], float | None]:
    """Each task's primary metric + their mean — the multi-bench aggregate. The built-in
    ``score`` is only the FIRST task's metric, so multi-task evals need this to rank by all benches."""
    per_task: dict[str, float | None] = {}
    for task in tasks:
        value: float | None = None
        if headline_metric and f"{task}::{headline_metric}" in metrics:
            value = metrics[f"{task}::{headline_metric}"]
        else:
            for key, val in metrics.items():
                if key.startswith(f"{task}::"):
                    value = val
                    break
        per_task[task] = value
    present = [v for v in per_task.values() if isinstance(v, (int, float))]
    macro = sum(present) / len(present) if present else None
    return per_task, macro


def _trim_raw(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"_note": "raw payload was not a dict", "_type": type(raw).__name__}
    keep = {
        "results",
        "configs",
        "config",
        "versions",
        "n-shot",
        "task_hashes",
        "model_source",
        "model_configs",
        "higher_is_better",
        "git_hash",
        "date",
    }
    trimmed = {key: value for key, value in raw.items() if key in keep}
    if len(trimmed) < min(len(raw), _RAW_TRIM_MAX_KEYS):
        for key in list(raw.keys())[: _RAW_TRIM_MAX_KEYS - len(trimmed)]:
            if key not in trimmed and key != "samples":
                trimmed[key] = raw[key]
    return trimmed


def _protocol_snapshot(output: Any, model: Model, config: EvalConfig) -> dict[str, Any]:
    return {
        "eval_config": config.to_record(),
        "source_model": {
            "artifact_id": model.artifact_id,
            "path": str(model.path),
            "base_model": model.meta.get("base_model"),
            "finetuning_type": model.meta.get("finetuning_type"),
            "producer": model.producer,
        },
        "resolved": dict(getattr(output, "protocol", {}) or {}),
    }


def _build_log_message(
    model: Model,
    evaluation: Eval,
    output: Any,
    config: EvalConfig,
) -> str:
    headline = evaluation.meta.get("headline_score")
    lines = [
        f"model={model.artifact_id}",
        f"eval={evaluation.artifact_id}",
        "backend=lm_eval",
        f"mode={config.normalized_mode()}",
        f"tasks={','.join(config.tasks)}",
        f"score={headline if headline is not None else 'n/a'}",
    ]
    if config.normalized_mode() == "lm_eval_native":
        lines.append(f"lm_eval_model={evaluation.meta.get('model') or config.model or 'hf'}")
    else:
        lines.append(f"adapter={config.adapter}")
    if config.limit is not None:
        lines.append(f"limit={config.limit}")
    return "\n".join(lines) + "\n"


def _build_stub_log_message(
    model: Model,
    evaluation: Eval,
    eval_config: dict[str, Any],
) -> str:
    score = evaluation.meta.get("score")
    lines = [
        f"model={model.artifact_id}",
        f"eval={evaluation.artifact_id}",
        "backend=stub",
        f"score={score if score is not None else 'n/a'}",
    ]
    if eval_config.get("provider_config"):
        lines.append(f"provider_config={json.dumps(eval_config['provider_config'], sort_keys=True)}")
    return "\n".join(lines) + "\n"
