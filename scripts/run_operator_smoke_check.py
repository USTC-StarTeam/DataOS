from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from datasets import Dataset, DatasetDict

from dataos.core.artifact import DatasetArtifact, make_artifact_id
from dataos.core.sandbox import Sandbox
from dataos.operators import build_default_registry


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="dataos-operator-smoke-") as tmpdir:
        root = Path(tmpdir)
        sandbox = Sandbox(root / "sandbox")
        sandbox.prepare()

        dataset_dir = root / "seed-dataset"
        DatasetDict(
            {
                "train": Dataset.from_list(
                    [
                        {
                            "instruction": "Rewrite the sentence in a friendlier way.",
                            "input": "Please send the report today.",
                            "output": "Could you please send the report today?",
                        },
                        {
                            "instruction": "Rewrite the sentence in a friendlier way.",
                            "input": "Please send the report today.",
                            "output": "Could you please send the report today?",
                        },
                        {
                            "instruction": "Answer the question.",
                            "input": "What is the capital of France?",
                            "output": "Paris is the capital of France.",
                        },
                    ]
                )
            }
        ).save_to_disk(str(dataset_dir))

        source_artifact_id = make_artifact_id("dataset")
        source_path = sandbox.inputs_dir / source_artifact_id
        shutil.copytree(dataset_dir, source_path)
        source = DatasetArtifact(
            artifact_id=source_artifact_id,
            name="seed",
            path=source_path,
            producer="smoke_check",
            meta={"dataset_format": "alpaca"},
        )
        source.refresh_metadata_from_disk()

        registry = build_default_registry()
        mapped = registry.get("map_to_alpaca").run([source], sandbox=sandbox)
        deduped = registry.get("filter_exact_dedup").run(
            [mapped],
            sandbox=sandbox,
            keys=["instruction", "input", "output"],
        )
        scored = registry.get("evaluate_text_quality").run([deduped], sandbox=sandbox)
        selected = registry.get("select_top_by_score").run(
            [scored],
            sandbox=sandbox,
            score_column="quality_score",
            max_rows=2,
        )

        payload = {
            "source": source.summary(),
            "mapped": mapped.summary(),
            "deduped": deduped.summary(),
            "scored": scored.summary(),
            "selected": selected.summary(),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
