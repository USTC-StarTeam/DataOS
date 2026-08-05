"""面向 data-centric 的算子核心契约。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence, Sized
from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Final,
    Generic,
    Literal,
    TypeAlias,
    TypeGuard,
    TypeVar,
    cast,
)

if TYPE_CHECKING:
    from dataos_operator.dataset import Dataset as RuntimeDataset
    from dataos_operator.runtime.types import (
        BackwardSignal,
        BatchView,
        ForwardSignal,
        OperatorState,
        PolicyAction,
        RuntimeContext,
    )

OperatorKind: TypeAlias = Literal[
    "evaluator",
    "generator",
    "mapper",
    "mixer",
    "clusterer",
    "reweighter",
    "refiner",
    "selector",
]

OPERATOR_KINDS: Final[tuple[OperatorKind, ...]] = (
    "evaluator",
    "generator",
    "mapper",
    "mixer",
    "clusterer",
    "reweighter",
    "refiner",
    "selector",
)

Record: TypeAlias = Mapping[str, Any]
Dataset: TypeAlias = Sequence[Record]
RecordOrDataset: TypeAlias = Record | Dataset
DatasetInput: TypeAlias = Dataset | Sequence[Dataset]
DOperatorOutput = TypeVar("DOperatorOutput")
InputT = TypeVar("InputT")

DEFAULT_FIELD_KEYS: Final[Mapping[str, str]] = {
    "id_key": "id",
    "text_key": "text",
    "query_key": "query",
    "answer_key": "answer",
    "metadata_key": "metadata",
    "score_key": "score",
    "weight_key": "weight",
}

DEFAULT_RESOURCE_CONFIG_KEYS: Final[frozenset[str]] = frozenset(
    {
        "model",
        "model_name",
        "model_path",
        "prompt_path",
        "reference_path",
        "taxonomy_path",
        "resource_path",
        "resources",
    }
)


@dataclass(frozen=True)
class OperatorSpec:
    """描述算子的元信息。"""

    name: str
    kind: OperatorKind
    description: str
    operator_class: str | None = None


def is_record(value: object) -> TypeGuard[Record]:
    return isinstance(value, Mapping)


def is_dataset(value: object) -> TypeGuard[Dataset]:
    if isinstance(value, str | bytes | bytearray) or is_record(value):
        return False
    if not isinstance(value, Sequence):
        return False
    return all(is_record(item) for item in value)


def is_multi_dataset(value: object) -> TypeGuard[Sequence[Dataset]]:
    if isinstance(value, str | bytes | bytearray) or is_record(value):
        return False
    if not isinstance(value, Sequence) or len(value) == 0:
        return False
    return all(is_dataset(item) for item in value)




def _normalize_dataset_input(value: object, *, operator_name: str) -> list[Dataset]:
    if is_dataset(value):
        return [value]
    if is_multi_dataset(value):
        return list(value)
    raise TypeError(
        f"{operator_name} expected a dataset or sequence of datasets, got {type(value)!r}"
    )


def get_field(record: Record, key: str, default: Any = None) -> Any:
    """使用点分字段路径从 record 中读取值。"""

    try:
        return _resolve_field_path(record, key)
    except (KeyError, IndexError, TypeError, ValueError):
        return default


def require_field(record: Record, key: str) -> Any:
    """使用点分字段路径从 record 中读取必需字段。"""

    try:
        return _resolve_field_path(record, key)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise KeyError(f"Required field is missing: {key}") from exc


def set_field(record: Record, key: str, value: Any) -> Record:
    """返回一个写入点分字段路径的新 record。"""

    parts = _split_field_path(key)
    return cast(Record, _set_field_path(record, parts, value))


def _record_count(value: object) -> int | None:
    if is_record(value):
        return 1
    if is_dataset(value):
        return len(value)
    if is_multi_dataset(value):
        return sum(len(dataset) for dataset in value)
    return None


def _item_count(value: object) -> int | None:
    if is_record(value):
        return 1
    if not isinstance(value, Sized):
        return None
    return len(value)


def _runtime_dataset_class() -> type[RuntimeDataset]:
    from dataos_operator.dataset import Dataset as RuntimeDatasetClass

    return RuntimeDatasetClass


def _dataset_metadata(value: object) -> dict[str, Any]:
    runtime_dataset_class = _runtime_dataset_class()
    if isinstance(value, runtime_dataset_class):
        return dict(value.metadata)
    return {}


def _runtime_dataset_records(value: object) -> list[Record]:
    runtime_dataset_class = _runtime_dataset_class()
    if isinstance(value, runtime_dataset_class):
        return value.to_records()
    if is_record(value):
        return [dict(value)]
    if is_dataset(value):
        return [dict(record) for record in value]
    raise TypeError(f"Operator output must be a record or dataset, got {type(value)!r}")


def _history_entries(metadata: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    history = metadata.get("history")
    if not isinstance(history, Sequence) or isinstance(history, str | bytes | bytearray):
        return []
    return [entry for entry in history if isinstance(entry, Mapping)]


def _split_field_path(key: str) -> list[str]:
    if not key or not isinstance(key, str):
        raise ValueError(f"Field key must be a non-empty string: {key!r}")
    return key.split(".")


def _resolve_field_path(value: Any, key: str) -> Any:
    current = value
    for part in _split_field_path(key):
        if isinstance(current, Mapping):
            current = current[part]
        elif _is_sequence_container(current):
            current = current[_parse_list_index(part)]
        else:
            raise TypeError(f"Cannot access {part!r} on {type(current)!r}")
    return current


def _set_field_path(current: Any, parts: Sequence[str], value: Any) -> Any:
    part = parts[0]
    is_leaf = len(parts) == 1

    if isinstance(current, Mapping):
        updated_mapping = dict(current)
        if is_leaf:
            updated_mapping[part] = value
        else:
            child = updated_mapping.get(part, _new_container_for(parts[1]))
            updated_mapping[part] = _set_field_path(child, parts[1:], value)
        return updated_mapping

    if _is_sequence_container(current):
        index = _parse_list_index(part)
        updated_list = list(current)
        if index >= len(updated_list):
            raise IndexError(f"List index {index} is out of range for field path")
        if is_leaf:
            updated_list[index] = value
        else:
            updated_list[index] = _set_field_path(updated_list[index], parts[1:], value)
        return updated_list

    raise TypeError(f"Cannot set {part!r} on {type(current)!r}")


def _new_container_for(next_part: str) -> dict[str, Any] | list[Any]:
    if _is_list_index(next_part):
        return []
    return {}


def _is_sequence_container(value: object) -> TypeGuard[Sequence[Any]]:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)


def _is_list_index(value: str) -> bool:
    return value.isdigit()


def _parse_list_index(value: str) -> int:
    if not _is_list_index(value):
        raise ValueError(f"Expected list index in field path, got {value!r}")
    return int(value)


def _extract_resource_config(
    config: Mapping[str, Any],
    extra_resource_keys: Sequence[str],
) -> dict[str, Any]:
    resource_keys = set(DEFAULT_RESOURCE_CONFIG_KEYS).union(extra_resource_keys)
    resources: dict[str, Any] = {}
    for key, value in config.items():
        if key in resource_keys or key.endswith(("_path", "_paths", "_uri", "_url")):
            resources[key] = value
    return resources


class DataOperator(ABC, Generic[InputT, DOperatorOutput]):
    """所有具体算子的对象接口基类。"""

    operator_name: ClassVar[str] = ""
    operator_kind: ClassVar[OperatorKind]
    resource_config_keys: ClassVar[tuple[str, ...]] = ()
    id_key: str
    text_key: str
    query_key: str
    answer_key: str
    metadata_key: str
    score_key: str
    weight_key: str
    _operator_config: dict[str, Any]
    _resource_config: dict[str, Any]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._operator_config: dict[str, Any] = dict(kwargs)
        self._resource_config: dict[str, Any] = _extract_resource_config(
            self._operator_config,
            self.resource_config_keys,
        )
        self._apply_common_field_keys(self._operator_config)

    @property
    def operator_config(self) -> Mapping[str, Any]:
        return MappingProxyType(self._operator_config)

    @property
    def resource_config(self) -> Mapping[str, Any]:
        return MappingProxyType(self._resource_config)

    def set_operator_config(self, config: Mapping[str, Any]) -> None:
        """保存实例化该算子时使用的完整配置。"""

        self._operator_config = dict(config)
        self._resource_config = _extract_resource_config(
            self._operator_config,
            self.resource_config_keys,
        )
        self._apply_common_field_keys(self._operator_config)

    @property
    def spec(self) -> OperatorSpec:
        return OperatorSpec(
            name=self.name,
            kind=self.operator_kind,
            description=(self.__class__.__doc__ or "").strip(),
            operator_class=f"{self.__class__.__module__}.{self.__class__.__qualname__}",
        )

    @property
    def name(self) -> str:
        return self.operator_name or self.__class__.__module__.rsplit(".", 1)[-1]

    def run(self, data: InputT) -> RuntimeDataset:
        """运行算子，并统一返回可继续链式处理的 Dataset。"""

        runtime_dataset_class = _runtime_dataset_class()
        output = self.transform(data)
        base_metadata = _dataset_metadata(data)
        output_metadata = _dataset_metadata(output)
        metadata: dict[str, Any] = {
            **base_metadata,
            **{key: value for key, value in output_metadata.items() if key != "history"},
        }
        run_metadata: dict[str, Any] = {
            "operator": self.name,
            "kind": self.operator_kind,
        }
        if self._resource_config:
            run_metadata["resources"] = dict(self._resource_config)
        input_records = _record_count(data)
        output_records = _record_count(output)
        if input_records is not None:
            run_metadata["input_records"] = input_records
        if output_records is not None:
            run_metadata["output_records"] = output_records
        input_items = _item_count(data)
        output_items = _item_count(output)
        if input_items is not None:
            run_metadata["input_items"] = input_items
        if output_items is not None:
            run_metadata["output_items"] = output_items

        history = _history_entries(base_metadata)
        history.append(run_metadata)
        metadata.update(run_metadata)
        metadata["history"] = history

        records = _runtime_dataset_records(output)
        metadata["records"] = len(records)
        return runtime_dataset_class.from_records(records, metadata=metadata)

    def get_field(self, record: Record, key: str, default: Any = None) -> Any:
        return get_field(record, key, default)

    def require_field(self, record: Record, key: str) -> Any:
        return require_field(record, key)

    def set_field(self, record: Record, key: str, value: Any) -> Record:
        return set_field(record, key, value)

    @abstractmethod
    def transform(self, data: InputT) -> DOperatorOutput:
        """把算子输入转换为输出数据。"""

    def _apply_common_field_keys(self, config: Mapping[str, Any]) -> None:
        for attr, default in DEFAULT_FIELD_KEYS.items():
            setattr(self, attr, config.get(attr, default))

    def prepare(
        self,
        *,
        datasets: Mapping[str, RuntimeDataset] | None,
        state: OperatorState,
        context: RuntimeContext,
    ) -> tuple[OperatorState, PolicyAction | None]:
        """Optional runtime preparation hook for offline/online execution."""

        del datasets, context
        return state, None

    def before_batch(
        self,
        *,
        batch: BatchView,
        state: OperatorState,
        context: RuntimeContext,
        **kwargs: Any,
    ) -> PolicyAction | None:
        """Optional hook before a trainer consumes the next batch."""

        del batch, state, context, kwargs
        return None

    def after_forward(
        self,
        *,
        batch: BatchView,
        signal: ForwardSignal,
        state: OperatorState,
        context: RuntimeContext,
        **kwargs: Any,
    ) -> None:
        """Optional hook after forward propagation or offline scoring."""

        del batch, signal, state, context, kwargs

    def before_backward(
        self,
        *,
        batch: BatchView,
        signal: ForwardSignal,
        state: OperatorState,
        context: RuntimeContext,
        **kwargs: Any,
    ) -> PolicyAction | None:
        """Optional hook before backward propagation."""

        del batch, signal, state, context, kwargs
        return None

    def after_backward(
        self,
        *,
        batch: BatchView,
        signal: BackwardSignal,
        state: OperatorState,
        context: RuntimeContext,
        **kwargs: Any,
    ) -> None:
        """Optional hook after backward propagation."""

        del batch, signal, state, context, kwargs

    def on_update_boundary(
        self,
        *,
        state: OperatorState,
        context: RuntimeContext,
        **kwargs: Any,
    ) -> PolicyAction | None:
        """Optional hook at update boundaries such as step or epoch refresh."""

        del state, context, kwargs
        return None

    def finalize(
        self,
        *,
        state: OperatorState,
        context: RuntimeContext,
        **kwargs: Any,
    ) -> PolicyAction | None:
        """Optional hook when a runtime session is ending."""

        del state, context, kwargs
        return None


class RecordOperator(DataOperator[RecordOrDataset, RecordOrDataset]):
    """最细操作粒度为单条 record 的算子。"""

    def transform(self, data: RecordOrDataset) -> RecordOrDataset:
        if is_record(data):
            return self.transform_record(data)
        if not is_dataset(data):
            raise TypeError(f"RecordOperator expected a record or dataset, got {type(data)!r}")
        return [self.transform_record(record) for record in data]

    @abstractmethod
    def transform_record(self, record: Record) -> Record:
        """转换单条 record。"""


class DatasetOperator(DataOperator[InputT, Dataset], Generic[InputT]):
    """最细操作粒度为 dataset 级整体的算子。"""

    def transform(self, data: InputT) -> Dataset:
        return self.transform_dataset(data)

    @abstractmethod
    def transform_dataset(self, data: InputT) -> Dataset:
        """转换 dataset 级输入。"""


class EvaluatorOperator(RecordOperator):
    """以 record 为优先粒度的 evaluator，并支持 dataset 级执行。"""

    operator_kind: ClassVar[OperatorKind] = "evaluator"

    def transform(self, data: RecordOrDataset) -> RecordOrDataset:
        if is_record(data):
            return self.evaluate_record(data)
        if not is_dataset(data):
            raise TypeError(f"EvaluatorOperator expected a record or dataset, got {type(data)!r}")
        return self.evaluate_dataset(data)

    def transform_record(self, record: Record) -> Record:
        return self.evaluate_record(record)

    def evaluate_dataset(self, data: Dataset) -> Dataset:
        return [self.evaluate_record(record) for record in data]

    @abstractmethod
    def evaluate_record(self, record: Record) -> Record:
        """评价单条 record。"""


class MapperOperator(RecordOperator):
    """以 record 为优先粒度的 mapper，并支持 dataset 级执行。"""

    operator_kind: ClassVar[OperatorKind] = "mapper"

    def transform(self, data: RecordOrDataset) -> RecordOrDataset:
        if is_record(data):
            return self.map_record(data)
        if not is_dataset(data):
            raise TypeError(f"MapperOperator expected a record or dataset, got {type(data)!r}")
        return self.map_dataset(data)

    def transform_record(self, record: Record) -> Record:
        return self.map_record(record)

    def map_dataset(self, data: Dataset) -> Dataset:
        return [self.map_record(record) for record in data]

    @abstractmethod
    def map_record(self, record: Record) -> Record:
        """映射单条 record。"""


class RefinerOperator(RecordOperator):
    """以 record 为优先粒度的 refiner，并支持 dataset 级执行。"""

    operator_kind: ClassVar[OperatorKind] = "refiner"

    def transform(self, data: RecordOrDataset) -> RecordOrDataset:
        if is_record(data):
            return self.refine_record(data)
        if not is_dataset(data):
            raise TypeError(f"RefinerOperator expected a record or dataset, got {type(data)!r}")
        return self.refine_dataset(data)

    def transform_record(self, record: Record) -> Record:
        return self.refine_record(record)

    def refine_dataset(self, data: Dataset) -> Dataset:
        return [self.refine_record(record) for record in data]

    @abstractmethod
    def refine_record(self, record: Record) -> Record:
        """改写单条 record。"""


class ReweighterOperator(RecordOperator):
    """以 record 为优先粒度的 reweighter，并支持 dataset 级执行。"""

    operator_kind: ClassVar[OperatorKind] = "reweighter"

    def transform(self, data: RecordOrDataset) -> RecordOrDataset:
        if is_record(data):
            return self.reweight_record(data)
        if not is_dataset(data):
            raise TypeError(f"ReweighterOperator expected a record or dataset, got {type(data)!r}")
        return self.reweight_dataset(data)

    def transform_record(self, record: Record) -> Record:
        return self.reweight_record(record)

    def reweight_dataset(self, data: Dataset) -> Dataset:
        return [self.reweight_record(record) for record in data]

    @abstractmethod
    def reweight_record(self, record: Record) -> Record:
        """为单条 record 重新加权。"""


class SelectorOperator(DatasetOperator[DatasetInput]):
    """面向一个或多个 dataset 的 selector。"""

    operator_kind: ClassVar[OperatorKind] = "selector"

    def transform_dataset(self, data: DatasetInput) -> Dataset:
        return self.select(_normalize_dataset_input(data, operator_name="SelectorOperator"))

    @abstractmethod
    def select(self, data: Sequence[Dataset]) -> Dataset:
        """从一个或多个 dataset 中选择 records。"""


class GeneratorOperator(DatasetOperator[DatasetInput]):
    """面向一个或多个 dataset 的 generator。"""

    operator_kind: ClassVar[OperatorKind] = "generator"

    def transform_dataset(self, data: DatasetInput) -> Dataset:
        return self.generate(_normalize_dataset_input(data, operator_name="GeneratorOperator"))

    @abstractmethod
    def generate(self, data: Sequence[Dataset]) -> Dataset:
        """从一个或多个可选种子 dataset 生成 records。"""


class ClustererOperator(DatasetOperator[DatasetInput]):
    """面向一个或多个 dataset 的 clusterer。"""

    operator_kind: ClassVar[OperatorKind] = "clusterer"

    def transform_dataset(self, data: DatasetInput) -> Dataset:
        return self.cluster(_normalize_dataset_input(data, operator_name="ClustererOperator"))

    @abstractmethod
    def cluster(self, data: Sequence[Dataset]) -> Dataset:
        """对一个或多个 dataset 中的 records 聚类。"""


class MixerOperator(DatasetOperator[DatasetInput]):
    """面向一个或多个 dataset 的 mixer。"""

    operator_kind: ClassVar[OperatorKind] = "mixer"

    def transform_dataset(self, data: DatasetInput) -> Dataset:
        return self.mix(_normalize_dataset_input(data, operator_name="MixerOperator"))

    @abstractmethod
    def mix(self, data: Sequence[Dataset]) -> Dataset:
        """把一个或多个 dataset 混合成一个 dataset。"""
