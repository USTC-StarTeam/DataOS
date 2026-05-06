from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

from dataos.core.artifact import DatasetArtifact, EvalArtifact, ModelArtifact, make_artifact_id
from dataos.core.sandbox import Sandbox
from dataos.core.task import TaskTemplate
from dataos.operators import build_default_registry
from dataos.train import run_train_stage


def _ensure_workspace(workspace: Path, overwrite: bool) -> None:
    if overwrite and workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)


def _write_parquet(dataset: Dataset, target: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    target.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(dataset))
    pq.write_table(table, target)


def _load_alpaca_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("alpaca-eval json must be a non-empty JSON array")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"alpaca-eval row {index} is not an object")
        instruction = str(row.get("instruction", "")).strip()
        output = str(row.get("output", "")).strip()
        if not instruction or not output:
            continue
        normalized.append(dict(row))
    if len(normalized) < 12:
        raise ValueError("alpaca-eval json did not contain enough usable rows")
    return normalized


def _matches_category(name: str, text: str) -> bool:
    lowered = text.lower()
    patterns = {
        "translation": ("translate", "translation"),
        "math": ("solve", "calculate", "equation", "math"),
        "code": ("python", "code", "function", "algorithm", "sql", "program"),
        "classification": ("classify", "label", "sentiment", "category"),
        "extraction": ("extract", "identify", "find all", "from the passage"),
        "writing": ("write", "draft", "compose", "summarize", "essay"),
    }
    if name == "math" and any(ch in text for ch in "+-*/"):
        return True
    return any(pattern in lowered for pattern in patterns[name])


def _pick_seed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    seen_instructions: set[str] = set()
    wanted = ["translation", "math", "code", "classification", "extraction", "writing"]
    for category in wanted:
        for row in rows:
            instruction = str(row["instruction"]).strip()
            if instruction in seen_instructions:
                continue
            if _matches_category(category, instruction):
                chosen.append(row)
                seen_instructions.add(instruction)
                break
    for row in rows:
        instruction = str(row["instruction"]).strip()
        if instruction in seen_instructions:
            continue
        chosen.append(row)
        seen_instructions.add(instruction)
        if len(chosen) >= 12:
            break
    if len(chosen) < 12:
        raise ValueError("Unable to build a diverse seed set from alpaca-eval")
    return chosen[:12]


def _make_raw_train_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    train_rows: list[dict[str, Any]] = []
    for index, row in enumerate(seed_rows[:8]):
        prompt = str(row["instruction"]).strip()
        answer = str(row["output"]).strip()
        context = f"alpaca_eval dataset={row.get('dataset', 'unknown')} generator={row.get('generator', 'unknown')}"
        if index == 0:
            context += " contact alice@example.com and review https://example.com/style-guide before release."
        elif index == 1:
            context += " reviewer phone +1 555 123 4567."
        elif index == 2:
            context += " " + "Keep the explanation detailed. " * 30
        train_rows.append(
            {
                "prompt": prompt,
                "context": context,
                "answer": answer,
                "source_dataset": row.get("dataset", "alpaca_eval"),
                "source_generator": row.get("generator", "unknown"),
            }
        )

    duplicate = dict(train_rows[0])
    duplicate["context"] = "Duplicate row used to verify dedup on instruction/response keys."
    train_rows.insert(1, duplicate)
    return train_rows


def _make_raw_validation_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in seed_rows[8:12]:
        rows.append(
            {
                "prompt": str(row["instruction"]).strip(),
                "context": f"alpaca_eval validation dataset={row.get('dataset', 'unknown')}",
                "answer": str(row["output"]).strip(),
                "source_dataset": row.get("dataset", "alpaca_eval"),
                "source_generator": row.get("generator", "unknown"),
            }
        )
    return rows


def _save_raw_dataset(workspace: Path, alpaca_eval_json: Path) -> tuple[Path, Path, dict[str, int]]:
    rows = _load_alpaca_rows(alpaca_eval_json)
    seeds = _pick_seed_rows(rows)
    train_rows = _make_raw_train_rows(seeds)
    validation_rows = _make_raw_validation_rows(seeds)
    dataset = DatasetDict(
        {
            "train": Dataset.from_list(train_rows),
            "validation": Dataset.from_list(validation_rows),
        }
    )
    hf_local_path = workspace / "raw_hf_local"
    if hf_local_path.exists():
        shutil.rmtree(hf_local_path)
    dataset.save_to_disk(str(hf_local_path))

    parquet_path = workspace / "raw_train.parquet"
    _write_parquet(dataset["train"], parquet_path)
    return hf_local_path, parquet_path, {
        "alpaca_eval_total_rows": len(rows),
        "seed_rows_used": len(seeds),
        "raw_train_rows": dataset["train"].num_rows,
        "raw_validation_rows": dataset["validation"].num_rows,
    }


def _artifact_from_path(*, artifact_id: str, name: str, path: Path, producer: str) -> DatasetArtifact:
    artifact = DatasetArtifact(
        artifact_id=artifact_id,
        name=name,
        path=path,
        parent_ids=[],
        producer=producer,
    )
    artifact.refresh_metadata_from_disk()
    artifact.save_meta()
    return artifact


def _require_live_judge_env() -> dict[str, str]:
    base_url = os.environ.get("DASHSCOPE_BASE_URL") or os.environ.get("STARTEAM_BASE_URL")
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("STARTEAM_API_KEY")
    model = os.environ.get("DASHSCOPE_MODEL") or os.environ.get("STARTEAM_MODEL")
    missing = [
        name
        for name, value in {
            "DASHSCOPE_BASE_URL/STARTEAM_BASE_URL": base_url,
            "DASHSCOPE_API_KEY/STARTEAM_API_KEY": api_key,
            "DASHSCOPE_MODEL/STARTEAM_MODEL": model,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing live-judge environment variables: {', '.join(missing)}")
    return {"base_url": str(base_url), "api_key": str(api_key), "model": str(model)}


def _stub_task(task_id: str) -> TaskTemplate:
    return TaskTemplate.from_dict(
        {
            "task_id": task_id,
            "name": "Operator Real Checks Stub Runtime",
            "goal": "Exercise real operators and then produce stub runtime artifacts.",
            "input_description": "Processed alpaca-eval-derived DatasetArtifact.",
            "output_description": "Stub model and stub eval artifacts for integration validation.",
            "constraints": {"dataset_format": "hf_datasets", "execution_mode": "fixed_runtime"},
            "budget": {"max_experiments": 1},
            "max_rounds": 1,
            "pipeline_hint": ["dedup", "sample"],
            "train_config": {"mode": "stub"},
            "eval_config": {"mode": "stub", "tasks": ["alpaca_eval_stub"]},
        }
    )


def _run_stub_eval(
    *,
    sandbox: Sandbox,
    dataset_artifact: DatasetArtifact,
    model_artifact: ModelArtifact,
    task_name: str,
) -> EvalArtifact:
    dataset_obj = dataset_artifact.load()
    if isinstance(dataset_obj, DatasetDict):
        split_sizes = {split: split_dataset.num_rows for split, split_dataset in dataset_obj.items()}
    else:
        split_sizes = {"train": dataset_obj.num_rows}
    artifact_id = make_artifact_id("eval")
    eval_dir = sandbox.artifact_dir("eval", artifact_id)
    payload = {
        "backend": "stub",
        "tasks": [task_name],
        "source_model_id": model_artifact.artifact_id,
        "source_dataset_id": dataset_artifact.artifact_id,
        "split_sizes": split_sizes,
        "score": None,
        "headline_score": None,
        "note": "Operator integration check only; training and evaluation are intentionally stubbed.",
    }
    (eval_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    (eval_dir / "lm_eval_results.json").write_text(
        json.dumps({"backend": "stub", "tasks": [task_name]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    artifact = EvalArtifact(
        artifact_id=artifact_id,
        name=f"eval-{model_artifact.artifact_id}",
        path=eval_dir,
        parent_ids=[model_artifact.artifact_id],
        producer="stub_eval",
        meta=payload,
    )
    artifact.save_meta()
    sandbox.log(
        "stub_eval",
        (
            f"dataset={dataset_artifact.artifact_id}\n"
            f"model={model_artifact.artifact_id}\n"
            f"eval={artifact.artifact_id}\n"
            f"task={task_name}\n"
        ),
    )
    return artifact


def run_checks(
    *,
    workspace: Path,
    alpaca_eval_json: Path,
    base_model: Path,
    overwrite: bool,
    quality_filter_mode: str,
    less_mode: str,
    less_repo_path: Path | None,
) -> dict[str, Any]:
    _ensure_workspace(workspace, overwrite)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sandbox = Sandbox(workspace / "sandbox")
    sandbox.prepare()
    registry = build_default_registry()

    hf_local_path, parquet_path, source_stats = _save_raw_dataset(workspace, alpaca_eval_json)
    raw_artifact = _artifact_from_path(
        artifact_id="dataset-raw-local",
        name="raw-local",
        path=hf_local_path,
        producer="run_all_operator_real_checks",
    )

    op_import = registry.get("import_dataset")
    imported_hf = op_import.run(
        [],
        sandbox,
        source="hf_local",
        path=str(hf_local_path),
        rename={"prompt": "instruction", "context": "input", "answer": "response"},
        keep_columns=["instruction", "input", "response", "source_dataset", "source_generator"],
        required_columns=["instruction", "response"],
        name="imported-hf-local",
    )
    imported_hf.save_meta()

    imported_parquet = op_import.run(
        [],
        sandbox,
        source="parquet",
        path=str(parquet_path),
        rename={"prompt": "instruction", "context": "input", "answer": "response"},
        keep_columns=["instruction", "input", "response", "source_dataset", "source_generator"],
        required_columns=["instruction", "response"],
        train_test_split={"test_size": 0.25, "seed": 7},
        max_train_samples=6,
        max_test_samples=2,
        name="imported-parquet",
    )
    imported_parquet.save_meta()

    schema_guard = registry.get("schema_guard").run(
        [raw_artifact],
        sandbox,
        mode="repair",
        keep_extra_columns=False,
        drop_empty=True,
    )
    pii_redact = registry.get("pii_redact").run([schema_guard], sandbox)
    dedup = registry.get("dedup").run([pii_redact], sandbox, keys=["instruction", "response"])
    length_filter = registry.get("length_filter").run(
        [dedup],
        sandbox,
        unit="words",
        min_length=5,
        max_length=140,
        per_column_min={"response": 3},
    )
    lang_filter = registry.get("lang_filter").run(
        [length_filter],
        sandbox,
        allowed_languages=["en"],
        annotate_column="dataos_lang",
    )

    quality_kwargs: dict[str, Any] = {
        "mode": quality_filter_mode,
        "min_score": 0.55,
        "attach_score_column": "dataos_quality_score",
        "attach_reason_column": "dataos_quality_reason",
    }
    if quality_filter_mode == "llm_judge":
        quality_kwargs.update(_require_live_judge_env())
    quality_filter = registry.get("quality_filter").run([lang_filter], sandbox, **quality_kwargs)
    category_tag = registry.get("category_tag").run([quality_filter], sandbox)
    category_balance = registry.get("category_balance").run(
        [category_tag],
        sandbox,
        split="train",
        strategy="cap",
        target_per_category=1,
    )
    curriculum_sort = registry.get("curriculum_sort").run(
        [category_balance],
        sandbox,
        split="train",
        order="easy_to_hard",
    )
    synth = registry.get("instruction_synthesis_template").run(
        [curriculum_sort],
        sandbox,
        split="train",
        templates=["brief_answer", "answer_only"],
        max_new_rows=4,
        preserve_original=True,
    )
    sample = registry.get("sample").run([synth], sandbox, max_train=6, max_per_split={"validation": 3})
    format_convert = registry.get("format_convert").run([sample], sandbox, target="messages")
    mixture = registry.get("mixture").run(
        [sample, quality_filter],
        sandbox,
        source_artifact_ids=[sample.artifact_id, quality_filter.artifact_id],
        split="train",
        max_rows=6,
        weights={sample.artifact_id: 2, quality_filter.artifact_id: 1},
    )
    less = registry.get("less").run(
        [mixture, sample],
        sandbox,
        target_artifact_id=sample.artifact_id,
        base_model=str(base_model),
        mode=less_mode,
        top_k=2,
        warmup_epochs=1,
        projection_dim=64,
        lora_r=8,
        lora_alpha=16,
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        less_repo_path=str(less_repo_path) if less_repo_path else None,
        force_recompute=True,
    )

    task = _stub_task("alpaca-eval-operator-suite")
    model_artifact = run_train_stage([less], sandbox=sandbox, task=task)
    model_artifact.save_meta()
    eval_artifact = _run_stub_eval(
        sandbox=sandbox,
        dataset_artifact=less,
        model_artifact=model_artifact,
        task_name="alpaca_eval_stub",
    )

    imported_hf_obj = imported_hf.load()
    imported_parquet_obj = imported_parquet.load()
    less_obj = less.load()
    quality_obj = quality_filter.load()
    format_obj = format_convert.load()
    mixture_obj = mixture.load()

    summary = {
        "workspace": str(workspace),
        "alpaca_eval_json": str(alpaca_eval_json),
        "base_model": str(base_model),
        "quality_filter_mode": quality_filter_mode,
        "less_mode": less_mode,
        "less_repo_path": str(less_repo_path) if less_repo_path else None,
        "source_stats": source_stats,
        "operators_exercised": [
            "import_dataset[hf_local]",
            "import_dataset[parquet]",
            "schema_guard",
            "pii_redact",
            "dedup",
            "length_filter",
            "lang_filter",
            f"quality_filter[{quality_filter_mode}]",
            "category_tag",
            "category_balance",
            "curriculum_sort",
            "instruction_synthesis_template",
            "sample",
            "format_convert",
            "mixture",
            f"less[{less_mode}]",
        ],
        "checks": {
            "import_hf_local_splits": list(imported_hf_obj.keys()) if isinstance(imported_hf_obj, DatasetDict) else ["train"],
            "import_hf_local_train_rows": imported_hf_obj["train"].num_rows if isinstance(imported_hf_obj, DatasetDict) else imported_hf_obj.num_rows,
            "import_parquet_splits": list(imported_parquet_obj.keys()) if isinstance(imported_parquet_obj, DatasetDict) else ["train"],
            "schema_guard_kept_counts": schema_guard.meta.get("schema_guard_kept_counts"),
            "pii_redact_counts": pii_redact.meta.get("pii_redact_counts"),
            "dedup_removed_counts": dedup.meta.get("dedup_removed_counts"),
            "length_filter_removed_counts": length_filter.meta.get("length_filter_removed_counts"),
            "lang_filter_detected_counts": lang_filter.meta.get("lang_filter_detected_counts"),
            "quality_filter_kept_counts": quality_filter.meta.get("quality_filter_kept_counts"),
            "quality_filter_removed_counts": quality_filter.meta.get("quality_filter_removed_counts"),
            "category_counts": category_tag.meta.get("category_tag_counts"),
            "category_balance_output_counts": category_balance.meta.get("category_balance_output_counts"),
            "curriculum_score_range": [
                curriculum_sort.meta.get("curriculum_sort_min_score"),
                curriculum_sort.meta.get("curriculum_sort_max_score"),
            ],
            "synth_added_rows": synth.meta.get("instruction_synthesis_added_rows"),
            "sample_kept_counts": sample.meta.get("sample_kept_counts"),
            "format_convert_target": format_convert.meta.get("format_target"),
            "format_convert_columns": format_obj["train"].column_names if isinstance(format_obj, DatasetDict) else format_obj.column_names,
            "mixture_rows": mixture_obj["train"].num_rows if isinstance(mixture_obj, DatasetDict) else mixture_obj.num_rows,
            "quality_columns": quality_obj["train"].column_names if isinstance(quality_obj, DatasetDict) else quality_obj.column_names,
            "less_selected_count": less.meta.get("less_selected_count"),
            "less_train_rows": less_obj["train"].num_rows if isinstance(less_obj, DatasetDict) else less_obj.num_rows,
        },
        "runtime": {
            "train_artifact": model_artifact.summary(),
            "eval_artifact": eval_artifact.summary(),
        },
        "artifacts": {
            "schema_guard": str(schema_guard.path),
            "pii_redact": str(pii_redact.path),
            "dedup": str(dedup.path),
            "length_filter": str(length_filter.path),
            "lang_filter": str(lang_filter.path),
            "quality_filter": str(quality_filter.path),
            "category_tag": str(category_tag.path),
            "category_balance": str(category_balance.path),
            "curriculum_sort": str(curriculum_sort.path),
            "instruction_synthesis_template": str(synth.path),
            "sample": str(sample.path),
            "format_convert": str(format_convert.path),
            "mixture": str(mixture.path),
            "less": str(less.path),
            "model": str(model_artifact.path),
            "eval": str(eval_artifact.path),
        },
    }
    summary_path = workspace / "operator_real_checks_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real alpaca-eval-backed checks for all built-in DataOS operators.")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--alpaca-eval-json", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quality-filter-mode", choices=["heuristic", "llm_judge"], default="llm_judge")
    parser.add_argument("--less-mode", choices=["stub", "less"], default="stub")
    parser.add_argument("--less-repo-path", type=Path, default=None)
    args = parser.parse_args()

    summary = run_checks(
        workspace=args.workspace.expanduser().resolve(),
        alpaca_eval_json=args.alpaca_eval_json.expanduser().resolve(),
        base_model=args.base_model.expanduser().resolve(),
        overwrite=args.overwrite,
        quality_filter_mode=args.quality_filter_mode,
        less_mode=args.less_mode,
        less_repo_path=args.less_repo_path.expanduser().resolve() if args.less_repo_path else None,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
