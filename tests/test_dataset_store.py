import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dataos.core import DatasetVersion, LineageDependency, LineageRecord
from dataos.storage import DatasetStore


class DatasetStoreTests(unittest.TestCase):
    def test_dataset_store_saves_and_loads_versioned_jsonl_dataset(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store = DatasetStore(Path(temp_dir) / "datasets")
            dataset = DatasetVersion(
                dataset_id="alpaca-clean",
                version="v2",
                source={"kind": "derived"},
                schema={"fields": [{"name": "messages"}]},
                statistics={"rows": 2},
                tags=["cleaned", "sft"],
                lineage=[
                    LineageRecord(
                        operator_id="rewrite_dialog",
                        run_id="run-1",
                        step_id="step-1",
                        upstreams=[
                            LineageDependency(
                                dataset_id="alpaca-raw",
                                version="v1",
                                storage_uri="artifacts/datasets/alpaca-raw/v1/data.jsonl",
                                role="input",
                            )
                        ],
                    )
                ],
                created_by_run="run-1",
            )
            rows = [
                {"id": "sample-1", "messages": [{"role": "user", "content": "hi"}]},
                {"id": "sample-2", "messages": [{"role": "assistant", "content": "hello"}]},
            ]

            stored = store.save(dataset, rows)
            restored_dataset, restored_rows = store.load("alpaca-clean", "v2")

            self.assertEqual(restored_rows, rows)
            self.assertEqual(restored_dataset, stored)
            self.assertEqual(stored.storage_uri, store.dataset_dir("alpaca-clean", "v2").as_posix())
            self.assertEqual(stored.source["format"], "jsonl")

    def test_dataset_store_writes_metadata_and_lineage_sidecars(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir) / "datasets"
            store = DatasetStore(root_dir)
            dataset = DatasetVersion(
                dataset_id="mixture",
                version="2026-03-13",
                source={"kind": "mixed", "uri": "memory://mixture"},
                statistics={"rows": 1},
                lineage=[
                    {
                        "operator_id": "mix_datasets",
                        "event_type": "mix",
                        "upstreams": [
                            {
                                "dataset_id": "dataset-a",
                                "version": "v5",
                                "role": "input",
                            }
                        ],
                    }
                ],
            )

            store.save_dataset(dataset, [{"prompt": "hello"}])

            dataset_dir = root_dir / "mixture" / "2026-03-13"
            metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
            lineage = json.loads((dataset_dir / "lineage.json").read_text(encoding="utf-8"))

            self.assertEqual(metadata["dataset_id"], "mixture")
            self.assertEqual(metadata["version"], "2026-03-13")
            self.assertEqual(metadata["storage_format"], "jsonl")
            self.assertEqual(metadata["data_filename"], "data.jsonl")
            self.assertNotIn("lineage", metadata)
            self.assertEqual(lineage, dataset.lineage_dict())

    def test_dataset_store_normalizes_source_format_to_actual_storage_format(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store = DatasetStore(Path(temp_dir) / "datasets")
            dataset = DatasetVersion(
                dataset_id="alpaca-clean",
                version="v3",
                source={"kind": "derived", "format": "PARQUET"},
            )

            stored = store.save(dataset, [{"id": "sample-1"}], format="jsonl")
            restored_dataset, restored_rows = store.load("alpaca-clean", "v3")

            self.assertEqual(stored.source["format"], "jsonl")
            self.assertEqual(restored_dataset.source["format"], "jsonl")
            self.assertEqual(restored_rows, [{"id": "sample-1"}])

    def test_dataset_store_exposes_parquet_interfaces_for_future_backends(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store = DatasetStore(Path(temp_dir) / "datasets")

            with self.assertRaises(NotImplementedError):
                store.save_parquet(Path(temp_dir) / "data.parquet", [])

            with self.assertRaises(NotImplementedError):
                store.load_parquet(Path(temp_dir) / "data.parquet")
