from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, cast

import pytest

from dataos_operator import Dataset
from dataos_operator.core import (
    ClustererOperator,
    DataOperator,
    EvaluatorOperator,
    GeneratorOperator,
    MapperOperator,
    MixerOperator,
    OperatorKind,
    OperatorSpec,
    Record,
    RecordOperator,
    SelectorOperator,
    get_field,
    is_dataset,
    require_field,
    set_field,
)


def test_operator_spec_accepts_known_kind() -> None:
    spec = OperatorSpec(
        name="placeholder-evaluator",
        kind="evaluator",
        description="Only used to validate the framework contract.",
    )

    assert spec.kind == "evaluator"


def test_data_operator_run_always_returns_dataset() -> None:
    class AddSeenFlag(RecordOperator):
        """为每条 record 增加标记字段。"""

        operator_name: ClassVar[str] = "add_seen_flag"
        operator_kind: ClassVar[OperatorKind] = "mapper"

        def transform_record(self, record: Record) -> Record:
            new_record = dict(record)
            new_record["seen"] = True
            return new_record

    result = AddSeenFlag().run({"id": "sample-1"})

    assert isinstance(result, Dataset)
    assert result[0] == {"id": "sample-1", "seen": True}


def test_data_operator_supports_dataset_chaining() -> None:
    class AddSeenFlag(RecordOperator):
        """为每条 record 增加标记字段。"""

        operator_name: ClassVar[str] = "add_seen_flag"
        operator_kind: ClassVar[OperatorKind] = "mapper"

        def transform_record(self, record: Record) -> Record:
            new_record = dict(record)
            new_record["seen"] = True
            return new_record

    class KeepSeenOnly(SelectorOperator):
        """只保留 seen 的 records。"""

        operator_name: ClassVar[str] = "keep_seen_only"

        def select(self, data: Sequence[Sequence[Record]]) -> Sequence[Record]:
            return [record for dataset in data for record in dataset if record.get("seen")]

    dataset = Dataset.from_records([{"id": "sample-1"}], metadata={"source": "unit"})
    first_pass = AddSeenFlag().run(dataset)
    second_pass = KeepSeenOnly().run(first_pass)

    assert second_pass.metadata["source"] == "unit"
    assert second_pass[0]["seen"] is True
    assert second_pass.metadata["operator"] == "keep_seen_only"
    assert second_pass.metadata["history"] == [
        {
            "operator": "add_seen_flag",
            "kind": "mapper",
            "input_records": 1,
            "output_records": 1,
            "input_items": 1,
            "output_items": 1,
        },
        {
            "operator": "keep_seen_only",
            "kind": "selector",
            "input_records": 1,
            "output_records": 1,
            "input_items": 1,
            "output_items": 1,
        },
    ]


def test_data_operator_is_abstract() -> None:
    abstract_operator = cast(Any, DataOperator)
    with pytest.raises(TypeError):
        abstract_operator()


def test_field_path_helpers_read_require_and_write_nested_fields() -> None:
    record = {
        "text": "hello",
        "metadata": {"eval": {"score": 0.8}},
        "messages": [{"content": "first"}],
    }

    assert get_field(record, "metadata.eval.score") == 0.8
    assert get_field(record, "messages.0.content") == "first"
    assert get_field(record, "missing.path", default="fallback") == "fallback"
    assert require_field(record, "text") == "hello"
    with pytest.raises(KeyError):
        require_field(record, "metadata.eval.reason")

    updated = set_field(record, "metadata.eval.reason", "ok")
    assert updated["metadata"]["eval"]["reason"] == "ok"
    assert "reason" not in record["metadata"]["eval"]


def test_operator_preserves_config_and_resource_metadata() -> None:
    class ConfiguredScore(EvaluatorOperator):
        """为配置指定的输入字段打分。"""

        operator_name: ClassVar[str] = "configured_score"

        def __init__(
            self,
            model: str,
            prompt_path: str,
            output_key: str,
            **kwargs: Any,
        ) -> None:
            super().__init__(
                model=model,
                prompt_path=prompt_path,
                output_key=output_key,
                **kwargs,
            )
            self.model = model
            self.prompt_path = prompt_path
            self.output_key = output_key

        def evaluate_record(self, record: Record) -> Record:
            text = self.require_field(record, self.text_key)
            return self.set_field(record, self.output_key, f"{self.model}:{len(str(text))}")

    operator = ConfiguredScore(
        model="example-model",
        prompt_path="configs/prompts/eval.md",
        text_key="payload.text",
        output_key="metadata.eval.score",
    )
    result = operator.run({"payload": {"text": "hello"}})

    assert result[0] == {
        "payload": {"text": "hello"},
        "metadata": {"eval": {"score": "example-model:5"}},
    }
    assert operator.operator_config["output_key"] == "metadata.eval.score"
    assert dict(operator.resource_config) == {
        "model": "example-model",
        "prompt_path": "configs/prompts/eval.md",
    }
    assert result.metadata["resources"] == {
        "model": "example-model",
        "prompt_path": "configs/prompts/eval.md",
    }


def test_evaluator_operator_lifts_record_logic_to_dataset() -> None:
    class LengthScore(EvaluatorOperator):
        """按文本长度为 records 打分。"""

        def evaluate_record(self, record: Record) -> Record:
            new_record = dict(record)
            new_record["score"] = len(str(record.get("text", "")))
            return new_record

    operator = LengthScore()

    assert operator.run([{"text": "abc"}, {"text": "hello"}]).to_records() == [
        {"text": "abc", "score": 3},
        {"text": "hello", "score": 5},
    ]


def test_evaluator_operator_can_override_dataset_logic() -> None:
    class DatasetSizeScore(EvaluatorOperator):
        """使用 dataset 级上下文为 records 打分。"""

        def evaluate_record(self, record: Record) -> Record:
            return record

        def evaluate_dataset(self, data: Sequence[Record]) -> Sequence[Record]:
            size = len(data)
            return [dict(record, dataset_size=size) for record in data]

    operator = DatasetSizeScore()

    assert operator.run([{"id": "a"}, {"id": "b"}]).to_records() == [
        {"id": "a", "dataset_size": 2},
        {"id": "b", "dataset_size": 2},
    ]


def test_selector_operator_accepts_single_dataset() -> None:
    class FirstOnly(SelectorOperator):
        """只保留第一条 record。"""

        def select(self, data: Sequence[Sequence[Record]]) -> Sequence[Record]:
            return list(data[0][:1])

    operator = FirstOnly()

    assert operator.run([{"id": "a"}, {"id": "b"}]).to_records() == [{"id": "a"}]


def test_selector_operator_accepts_multiple_datasets() -> None:
    class TakeFirstFromEach(SelectorOperator):
        """从每个 dataset 取第一条 record。"""

        def select(self, data: Sequence[Sequence[Record]]) -> Sequence[Record]:
            return [dataset[0] for dataset in data if len(dataset) > 0]

    operator = TakeFirstFromEach()

    assert operator.run([[{"id": "a1"}, {"id": "a2"}], [{"id": "b1"}]]).to_records() == [
        {"id": "a1"},
        {"id": "b1"},
    ]


def test_generator_operator_accepts_single_dataset() -> None:
    class EchoGenerator(GeneratorOperator):
        """把输入 dataset 原样作为生成结果返回。"""

        def generate(self, data: Sequence[Sequence[Record]]) -> Sequence[Record]:
            return [record for dataset in data for record in dataset]

    operator = EchoGenerator()

    assert operator.run([{"id": "seed-1"}]).to_records() == [{"id": "seed-1"}]


def test_clusterer_operator_accepts_multiple_datasets() -> None:
    class DatasetIndexClusterer(ClustererOperator):
        """给 record 打上所属 dataset 序号。"""

        def cluster(self, data: Sequence[Sequence[Record]]) -> Sequence[Record]:
            return [
                dict(record, dataset_index=dataset_index)
                for dataset_index, dataset in enumerate(data)
                for record in dataset
            ]

    operator = DatasetIndexClusterer()

    assert operator.run([[{"id": "a"}], [{"id": "b"}]]).to_records() == [
        {"id": "a", "dataset_index": 0},
        {"id": "b", "dataset_index": 1},
    ]


def test_mixer_operator_accepts_single_dataset() -> None:
    class AppendAll(MixerOperator):
        """追加所有 datasets 中的 records。"""

        def mix(self, data: Sequence[Sequence[Record]]) -> Sequence[Record]:
            return [record for dataset in data for record in dataset]

    operator = AppendAll()

    assert operator.run([{"id": "a"}]).to_records() == [{"id": "a"}]


def test_mixer_operator_accepts_multiple_datasets() -> None:
    class AppendAll(MixerOperator):
        """追加所有 datasets 中的 records。"""

        def mix(self, data: Sequence[Sequence[Record]]) -> Sequence[Record]:
            return [record for dataset in data for record in dataset]

    operator = AppendAll()

    assert operator.run([[{"id": "a"}], [{"id": "b"}]]).to_records() == [
        {"id": "a"},
        {"id": "b"},
    ]


def test_operator_accepts_a_huggingface_dataset() -> None:
    """A loaded source IS a ``datasets.Dataset``, the toolbox's most natural input.

    It sizes, indexes to a Mapping and iterates records, but it is not REGISTERED as
    ``collections.abc.Sequence`` — so an isinstance-only check rejected it and every operator
    raised "expected a record or dataset". Cost a build_data round in 4b2 v5 (aime2025 r0004).
    """
    hf = pytest.importorskip("datasets")

    class Tag(MapperOperator):
        """给每条 record 加标记。"""

        def map_record(self, record: Record) -> Record:
            return {**record, "tagged": True}

    dataset = hf.Dataset.from_list([{"id": "a"}, {"id": "b"}])
    assert not isinstance(dataset, Sequence)  # exactly why the old check failed
    assert Tag().run(dataset).to_records() == [
        {"id": "a", "tagged": True},
        {"id": "b", "tagged": True},
    ]


def test_a_non_record_indexable_is_not_a_dataset() -> None:
    """The duck-typing must not swallow every len()+[]-able object (e.g. an array of numbers)."""

    class Numbers:
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> int:
            return index

    assert not is_dataset(Numbers())
    assert not is_dataset("ab")
    assert is_dataset([])
