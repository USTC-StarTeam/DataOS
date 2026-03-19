from datetime import datetime
import unittest

from dataos.core import DatasetVersion, LineageDependency, LineageRecord


class DatasetVersionTests(unittest.TestCase):
    def test_dataset_version_tracks_source_and_upstreams(self) -> None:
        dataset = DatasetVersion(
            dataset_id="alpaca-clean",
            version="v2",
            source={
                "kind": "derived",
                "uri": "artifacts/datasets/alpaca-clean/v2/data.jsonl",
                "format": "jsonl",
            },
            schema={"fields": [{"name": "instruction"}, {"name": "output"}]},
            statistics={"rows": 1280},
            tags=["cleaned", "sft", "cleaned"],
            lineage=[
                {
                    "operator_id": "filter_by_rule",
                    "run_id": "run-42",
                    "step_id": "step-filter",
                    "config": {"min_score": 0.8},
                    "artifacts": {"report": "artifacts/runs/run-42/steps/step-filter.json"},
                    "upstreams": [
                        {
                            "dataset_id": "alpaca-raw",
                            "version": "v1",
                            "storage_uri": "artifacts/datasets/alpaca-raw/v1/data.jsonl",
                            "role": "input",
                        }
                    ],
                }
            ],
            storage_uri="artifacts/datasets/alpaca-clean/v2",
            created_at="2026-03-13T12:00:00Z",
            created_by_run="run-42",
        )

        self.assertEqual(dataset.ref, "alpaca-clean:v2")
        self.assertEqual(dataset.source["kind"], "derived")
        self.assertEqual(dataset.tags, ["cleaned", "sft"])
        self.assertEqual(len(dataset.lineage), 1)
        self.assertIsInstance(dataset.lineage[0], LineageRecord)
        self.assertEqual(dataset.lineage[0].operator_id, "filter_by_rule")
        self.assertEqual(
            dataset.upstream_dependencies,
            [
                LineageDependency(
                    dataset_id="alpaca-raw",
                    version="v1",
                    storage_uri="artifacts/datasets/alpaca-raw/v1/data.jsonl",
                    role="input",
                )
            ],
        )

    def test_dataset_version_round_trips_to_storage_friendly_dicts(self) -> None:
        dataset = DatasetVersion(
            dataset_id="mixture",
            version="2026-03-13",
            source={"kind": "mixed", "uri": "memory://mixture"},
            schema={"fields": [{"name": "prompt"}]},
            statistics={"rows": 2048, "avg_tokens": 312.4},
            tags=["mixture"],
            lineage=[
                LineageRecord(
                    operator_id="mix_datasets",
                    run_id="run-99",
                    step_id="step-mix",
                    event_type="mix",
                    config={"weights": {"a": 0.6, "b": 0.4}},
                    upstreams=[
                        LineageDependency(dataset_id="dataset-a", version="v5"),
                        LineageDependency(dataset_id="dataset-b", version="v3"),
                    ],
                )
            ],
            storage_uri="artifacts/datasets/mixture/2026-03-13",
            created_at="2026-03-13T09:30:00Z",
            created_by_run="run-99",
        )

        payload = dataset.to_dict()
        restored = DatasetVersion.from_dict(payload)

        self.assertEqual(restored, dataset)
        self.assertEqual(
            restored.metadata_dict(),
            {
                "dataset_id": "mixture",
                "version": "2026-03-13",
                "source": {"kind": "mixed", "uri": "memory://mixture"},
                "schema": {"fields": [{"name": "prompt"}]},
                "statistics": {"rows": 2048, "avg_tokens": 312.4},
                "tags": ["mixture"],
                "storage_uri": "artifacts/datasets/mixture/2026-03-13",
                "created_at": "2026-03-13T09:30:00Z",
                "created_by_run": "run-99",
            },
        )
        self.assertEqual(restored.lineage_dict(), payload["lineage"])

    def test_dataset_version_defaults_support_pre_storage_instances(self) -> None:
        dataset = DatasetVersion(dataset_id="seed", version="v1", source={"kind": "external"})

        self.assertEqual(dataset.schema, {})
        self.assertEqual(dataset.statistics, {})
        self.assertEqual(dataset.tags, [])
        self.assertEqual(dataset.lineage, [])
        self.assertIsNone(dataset.storage_uri)
        self.assertIsNone(dataset.created_by_run)
        self.assertTrue(datetime.fromisoformat(dataset.created_at.replace("Z", "+00:00")))

    def test_lineage_dependency_requires_dataset_version_identity(self) -> None:
        with self.assertRaises(ValueError):
            LineageDependency(dataset_id="", version="v1")
