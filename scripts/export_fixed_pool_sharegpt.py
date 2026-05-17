from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk

PROVENANCE_COLUMNS = [
    "source_id",
    "source_repo",
    "source_config",
    "source_split",
    "source_row_index",
    "raw_id",
    "domain_tags",
    "benchmark_tags",
    "license",
    "contamination_notes",
]
CORE_COLUMNS = [
    "conversations",
    "system",
    *PROVENANCE_COLUMNS,
    "conversion_native_format",
    "conversion_turn_count",
    "conversion_lossless",
    "conversion_notes",
]
DEFAULT_NOTES = (
    "Derived from normalized instruction/input/output/system fields; native multi-turn "
    "structure is not recovered."
)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = read_json(manifest_path)
    input_root = Path(args.input_root or manifest["processed_root"]).expanduser()
    output_root = Path(args.output_root or f"{input_root}_sharegpt").expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    selected_source_ids = set(args.source_id or [])
    sources = [
        source
        for source in manifest["sources"]
        if not selected_source_ids or source["source_id"] in selected_source_ids
    ]
    if selected_source_ids:
        missing = selected_source_ids - {source["source_id"] for source in sources}
        if missing:
            raise ValueError(f"Requested source_id not found in manifest: {sorted(missing)}")

    summaries = []
    summaries_dir = output_root / "_summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        summary = convert_source(
            source=source,
            input_root=input_root,
            output_root=output_root,
            num_proc=args.num_proc,
            batch_size=args.batch_size,
            assistant_role=args.assistant_role,
            overwrite=args.overwrite,
            strict=args.strict,
            limit_rows=args.limit_rows,
        )
        summaries.append(summary)
        (summaries_dir / f"{source['source_id']}.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"{summary['source_id']}: rows={summary['rows_written']}/"
            f"{summary['rows_seen']} skipped={summary['rows_skipped']}"
        )

    output_manifest = build_output_manifest(
        source_manifest=manifest,
        output_root=output_root,
        summaries=summaries,
        assistant_role=args.assistant_role,
    )
    (output_root / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_root / "registry.json").write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote ShareGPT pool manifest: {output_root / 'manifest.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a normalized fixed training pool as per-source ShareGPT datasets."
    )
    parser.add_argument(
        "--manifest",
        default="dataos/templates/fixed_pools/posttrainbench_fixed_pool_v1_full.json",
        help="Path to the fixed-pool manifest JSON.",
    )
    parser.add_argument("--input-root", help="Override manifest processed_root.")
    parser.add_argument(
        "--output-root",
        help="Output directory. Defaults to <input-root>_sharegpt.",
    )
    parser.add_argument(
        "--source-id",
        action="append",
        help="Convert only this source_id. Can be repeated.",
    )
    parser.add_argument(
        "--assistant-role",
        choices=["gpt", "assistant"],
        default="gpt",
        help="Assistant role tag to write into conversations.",
    )
    parser.add_argument("--num-proc", type=int, default=1, help="datasets.map num_proc.")
    parser.add_argument("--batch-size", type=int, default=1000, help="datasets.map batch_size.")
    parser.add_argument("--limit-rows", type=int, help="Smoke-test limit per split/source.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite completed output sources.")
    parser.add_argument("--strict", action="store_true", help="Fail instead of skipping empty rows.")
    return parser.parse_args()


def convert_source(
    *,
    source: dict[str, Any],
    input_root: Path,
    output_root: Path,
    num_proc: int,
    batch_size: int,
    assistant_role: str,
    overwrite: bool,
    strict: bool,
    limit_rows: int | None,
) -> dict[str, Any]:
    source_id = source["source_id"]
    input_path = resolve_source_path(source, input_root)
    output_path = output_root / source_id
    complete_marker = output_path / "_DATAOS_SOURCE_COMPLETE"
    if complete_marker.exists() and not overwrite:
        summary_path = output_path / "source_summary.json"
        if summary_path.exists():
            return read_json(summary_path)

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} already exists but is not marked complete")
        shutil.rmtree(output_path)

    dataset_obj = load_from_disk(str(input_path))
    dataset_dict, was_dict = ensure_dataset_dict(dataset_obj)
    converted_splits: dict[str, Dataset] = {}
    split_summaries: dict[str, dict[str, int]] = {}
    total_seen = 0
    total_written = 0
    for split, dataset in dataset_dict.items():
        if limit_rows is not None:
            dataset = dataset.select(range(min(limit_rows, dataset.num_rows)))
        converted = convert_dataset(
            dataset=dataset,
            source=source,
            num_proc=num_proc,
            batch_size=batch_size,
            assistant_role=assistant_role,
            strict=strict,
            split=split,
        )
        converted_splits[split] = converted
        seen = dataset.num_rows
        written = converted.num_rows
        total_seen += seen
        total_written += written
        split_summaries[split] = {
            "rows_seen": seen,
            "rows_written": written,
            "rows_skipped": seen - written,
        }

    output_obj: Dataset | DatasetDict
    output_obj = DatasetDict(converted_splits) if was_dict else next(iter(converted_splits.values()))
    output_obj.save_to_disk(str(output_path))

    summary = {
        "source_id": source_id,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows_seen": total_seen,
        "rows_written": total_written,
        "rows_skipped": total_seen - total_written,
        "splits": split_summaries,
        "assistant_role": assistant_role,
        "conversion_native_format": "alpaca_like",
        "conversion_lossless": True,
        "conversion_notes": DEFAULT_NOTES,
        "converted_at": datetime.now(UTC).isoformat(),
    }
    (output_path / "source_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    complete_marker.write_text("ok\n", encoding="utf-8")
    return summary


def convert_dataset(
    *,
    dataset: Dataset,
    source: dict[str, Any],
    num_proc: int,
    batch_size: int,
    assistant_role: str,
    strict: bool,
    split: str,
) -> Dataset:
    missing = {"instruction", "output"} - set(dataset.column_names)
    if missing:
        raise ValueError(f"{source['source_id']} split {split} missing required columns: {sorted(missing)}")

    def convert_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        row_count = len(batch["instruction"])
        output: dict[str, list[Any]] = {column: [] for column in CORE_COLUMNS}
        for row_index in range(row_count):
            instruction = normalize_text(batch.get("instruction", [""] * row_count)[row_index])
            extra_input = normalize_text(batch.get("input", [""] * row_count)[row_index])
            answer = normalize_text(batch.get("output", [""] * row_count)[row_index])
            prompt = build_prompt(instruction, extra_input)
            if not prompt or not answer:
                if strict:
                    raise ValueError(
                        f"{source['source_id']} split {split} has an empty prompt/output row"
                    )
                continue

            output["conversations"].append(
                [
                    {"from": "human", "value": prompt},
                    {"from": assistant_role, "value": answer},
                ]
            )
            output["system"].append(normalize_text(batch.get("system", [""] * row_count)[row_index]))
            for column in PROVENANCE_COLUMNS:
                values = batch.get(column)
                if values is None:
                    output[column].append(source.get(column) or "")
                else:
                    output[column].append(values[row_index])
            output["conversion_native_format"].append("alpaca_like")
            output["conversion_turn_count"].append(1)
            output["conversion_lossless"].append(True)
            output["conversion_notes"].append(DEFAULT_NOTES)
        return output

    return dataset.map(
        convert_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=max(1, num_proc),
        remove_columns=dataset.column_names,
        desc=f"sharegpt:{source['source_id']}:{split}",
    )


def build_output_manifest(
    *,
    source_manifest: dict[str, Any],
    output_root: Path,
    summaries: list[dict[str, Any]],
    assistant_role: str,
) -> dict[str, Any]:
    summary_by_source = {summary["source_id"]: summary for summary in summaries}
    sources = []
    for source in source_manifest["sources"]:
        summary = summary_by_source.get(source["source_id"])
        if summary is None:
            continue
        updated = dict(source)
        updated["path"] = summary["output_path"]
        updated["num_rows"] = summary["rows_written"]
        updated["raw_rows_seen"] = summary["rows_seen"]
        updated["skipped_empty_instruction_or_output"] = summary["rows_skipped"]
        updated["sharegpt_view_of"] = source.get("path")
        sources.append(updated)

    manifest = dict(source_manifest)
    manifest["pool_id"] = f"{source_manifest['pool_id']}_sharegpt"
    manifest["name"] = f"{source_manifest['name']} (ShareGPT View)"
    manifest["description"] = (
        f"ShareGPT conversation view of {source_manifest['pool_id']}. "
        f"Each source remains independent and preserves provenance columns."
    )
    manifest["notes"] = [
        note
        for note in source_manifest.get("notes", [])
        if not str(note).startswith("Rows are normalized to instruction/input/output/system")
    ]
    manifest["notes"].append(
        "Rows are stored as single-turn ShareGPT conversations plus system/provenance columns; "
        "the source view remains per-dataset rather than capped or merged."
    )
    manifest["processed_root"] = str(output_root)
    manifest["sharegpt_view_of"] = source_manifest["pool_id"]
    manifest["processed_on"] = datetime.now(UTC).date().isoformat()
    manifest["num_sources"] = len(sources)
    manifest["format"] = {
        "columns": CORE_COLUMNS,
        "split": source_manifest.get("format", {}).get("split", "train"),
        "storage": "one_huggingface_datasetdict_save_to_disk_per_source",
        "conversation_tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": assistant_role,
        },
    }
    manifest["conversion"] = {
        "native_format": "alpaca_like",
        "lossless_from_processed_pool": True,
        "notes": DEFAULT_NOTES,
    }
    manifest["sources"] = sources
    return manifest


def ensure_dataset_dict(dataset_obj: Dataset | DatasetDict) -> tuple[DatasetDict, bool]:
    if isinstance(dataset_obj, DatasetDict):
        return dataset_obj, True
    return DatasetDict({"train": dataset_obj}), False


def resolve_source_path(source: dict[str, Any], input_root: Path) -> Path:
    source_path = Path(source["path"])
    if source_path.exists():
        return source_path
    candidate = input_root / source["source_id"]
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find source path for {source['source_id']}: {source_path}")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_prompt(instruction: str, extra_input: str) -> str:
    if instruction and extra_input:
        return f"{instruction}\n\n{extra_input}".strip()
    return instruction or extra_input


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
