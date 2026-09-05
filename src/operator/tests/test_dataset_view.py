from __future__ import annotations

import json

from datasets import Dataset as HFDataset
from datasets import Features, List, Value, load_from_disk
from datasets import Image as HFImage
from PIL import Image

from dataos_operator import ConversationFeatureEvaluator, DatasetView, WeightedReservoirSelector
from dataos_operator.integrations.curationbench import (
    load_manifest_indices,
    save_submission,
    with_reference_indices,
)


def _source() -> HFDataset:
    return HFDataset.from_list(
        [
            {
                "id": f"r{index}",
                "subset": "a" if index < 3 else "b",
                "texts": [{"user": f"q{index}", "assistant": f"a{index}"}],
            }
            for index in range(6)
        ]
    )


def test_dataset_view_keeps_indices_and_batches_only_requested_columns() -> None:
    view = DatasetView(_source()).with_columns({"metadata.score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
    selected = view.select_positions([4, 1])

    assert list(selected.source_indices) == [4, 1]
    batch = next(selected.iter_batches(["id", "metadata.score"]))
    assert batch["id"] == ("r4", "r1")
    assert batch["metadata.score"] == (0.5, 0.2)
    assert selected[0]["metadata"]["score"] == 0.5


def test_curation_submission_preserves_source_rows_and_writes_manifest(tmp_path) -> None:
    view = with_reference_indices(DatasetView(_source()), [1, 4])
    selected = view.select_positions([4, 1])
    output = tmp_path / "submission"
    manifest = tmp_path / "selection.jsonl"

    save_submission(selected, output, manifest_path=manifest)
    restored = load_from_disk(str(output))

    assert restored["id"] == ["r4", "r1"]
    assert "metadata" not in restored.column_names
    assert load_manifest_indices(manifest) == [4, 1]
    assert [json.loads(line)["order"] for line in manifest.read_text().splitlines()] == [0, 1]


def test_indexed_pipeline_preserves_scalar_hf_image_feature(tmp_path) -> None:
    features = Features(
        {
            "id": Value("string"),
            "subset": Value("string"),
            "images": HFImage(),
            "texts": List({"user": Value("string"), "assistant": Value("string")}),
        }
    )
    source = HFDataset.from_dict(
        {
            "id": ["image-row"],
            "subset": ["coco"],
            "images": [Image.new("RGB", (4, 3), color=(1, 2, 3))],
            "texts": [[{"user": "What is shown?", "assistant": "A test image."}]],
        },
        features=features,
    )
    featured = ConversationFeatureEvaluator().run(DatasetView(source))
    selected = WeightedReservoirSelector(sample_size=1, seed=1).run(featured)
    output = tmp_path / "scalar-image-submission"

    save_submission(selected, output)
    restored = load_from_disk(str(output))

    assert restored.features == source.features
    assert restored[0]["images"].size == (4, 3)
    assert restored[0]["texts"] == source[0]["texts"]
