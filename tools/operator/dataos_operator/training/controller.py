"""Trainer-facing policy controller for online DataOS operators.

This module intentionally has no dependency on LLaMA-Factory internals. Framework
adapters should feed it record ids, domains, tensors, and per-sample losses, then
apply the returned sampler decisions or loss weights in their own trainer hooks.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Sampler

from dataos_operator import Dataset
from dataos_operator.operators.mixer import DoremiMixer, OdmMixer
from dataos_operator.operators.reweighter import LossReweighter
from dataos_operator.operators.selector import (
    DeltaLossSelector,
    LessSelector,
    LossSelector,
    NearSelector,
    NiceSelector,
    TsdsSelector,
)
from dataos_operator.runtime import BatchView, ForwardSignal, PolicyRuntime, RuntimeContext
from dataos_operator.runtime.state import save_state_snapshot
from dataos_operator.runtime.text_backend import TextModelBackend

SELECTOR_REGISTRY = {
    "loss_selector": LossSelector,
    "delta_loss_selector": DeltaLossSelector,
    "less_selector": LessSelector,
    "nice_selector": NiceSelector,
    "near_selector": NearSelector,
    "tsds_selector": TsdsSelector,
}

MIXER_REGISTRY = {
    "doremi_mixer": DoremiMixer,
    "odm_mixer": OdmMixer,
}

REWEIGHTER_REGISTRY = {
    "loss_reweighter": LossReweighter,
}


def load_dataos_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        return json.loads(text)

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - yaml is available on the server path
        raise RuntimeError("PyYAML is required for non-JSON DataOS configs.") from exc

    payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError("DataOS config must decode to a mapping.")
    return payload


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    dataset = Dataset.from_local(path)
    return dataset.to_records()


def _load_datasets(paths: list[str] | None) -> list[Dataset]:
    if not paths:
        return []
    return [Dataset.from_local(path) for path in paths]


def _maybe_get_record_id(record: dict[str, Any], fallback: int) -> str:
    value = record.get("dataos_id", record.get("id"))
    if value is None:
        return f"row-{fallback}"
    return str(value)


class AttachedTextBackend(TextModelBackend):
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        *,
        max_length: int = 256,
        max_new_tokens: int = 48,
        parameter_filters: tuple[str, ...] = (),
    ) -> None:
        super().__init__(
            getattr(getattr(model, "config", None), "name_or_path", "attached-model"),
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            parameter_filters=parameter_filters,
        )
        self._model = model
        self._tokenizer = tokenizer

    def _ensure_model_loaded(self) -> None:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("AttachedTextBackend requires an attached model and tokenizer.")


class DataOSPolicySampler(Sampler[int]):
    def __init__(self, dataset: Any, controller: DataOSController, *, seed: int) -> None:
        self.dataset = dataset
        self.controller = controller
        self.seed = seed
        id_column = "dataos_id" if "dataos_id" in dataset.column_names else "_dataos_id"
        domain_column = (
            "dataos_domain" if "dataos_domain" in dataset.column_names else "_dataos_domain"
        )
        self._ids = [str(value) for value in dataset[id_column]]
        self._domains = [
            str(value) if value is not None else "unknown" for value in dataset[domain_column]
        ]
        self._domain_to_indices: dict[str, list[int]] = defaultdict(list)
        for index, domain in enumerate(self._domains):
            self._domain_to_indices[domain].append(index)

    def __len__(self) -> int:
        return len(self._ids)

    def __iter__(self):
        rng = random.Random(self.seed + self.controller.sampler_epoch)
        active_ids = self.controller.active_ids
        if active_ids:
            active_indices = [
                index for index, record_id in enumerate(self._ids) if record_id in active_ids
            ]
        else:
            active_indices = list(range(len(self._ids)))
        if not active_indices:
            active_indices = list(range(len(self._ids)))

        domain_weights = self.controller.current_domain_weights
        if domain_weights:
            active_by_domain: dict[str, list[int]] = defaultdict(list)
            for index in active_indices:
                active_by_domain[self._domains[index]].append(index)
            domains = sorted(active_by_domain)
            weights = [max(float(domain_weights.get(domain, 0.0)), 0.0) for domain in domains]
            if sum(weights) <= 0:
                weights = [1.0 for _ in domains]
            for _ in range(len(self)):
                domain = rng.choices(domains, weights=weights, k=1)[0]
                yield rng.choice(active_by_domain[domain])
            return

        shuffled = active_indices[:]
        rng.shuffle(shuffled)
        if len(shuffled) >= len(self):
            yield from shuffled[: len(self)]
            return
        for _ in range(len(self)):
            yield rng.choice(shuffled)


@dataclass
class DataOSController:
    config_path: Path
    config: dict[str, Any]
    runtime: PolicyRuntime
    raw_train_records: list[dict[str, Any]]
    output_dir: Path
    eval_datasets: list[Dataset]
    query_datasets: list[Dataset]
    active_ids: set[str] | None = None
    current_domain_weights: dict[str, float] | None = None
    sampler_epoch: int = 0
    backend: AttachedTextBackend | None = None

    def __post_init__(self) -> None:
        self.id_to_record = {
            _maybe_get_record_id(record, fallback=index): record
            for index, record in enumerate(self.raw_train_records)
        }
        self.id_to_index = {
            _maybe_get_record_id(record, fallback=index): index
            for index, record in enumerate(self.raw_train_records)
        }
        self.all_ids = set(self.id_to_record)
        self.current_domain_weights = self._initial_domain_weights()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(
        cls,
        *,
        config_path: str | Path,
        output_dir: str | Path,
    ) -> DataOSController:
        resolved_config_path = Path(config_path).expanduser().resolve()
        config = load_dataos_config(resolved_config_path)
        operators = cls._build_operators(config)
        runtime = PolicyRuntime(
            operators,
            context=RuntimeContext(
                mode=config.get("mode", "online"),
                update_interval=int(config.get("update_interval", 1)),
                trainer_backend="llamafactory",
                artifact_dir=str(Path(output_dir).resolve() / "dataos"),
                metadata={"config_path": str(resolved_config_path)},
            ),
        )
        train_paths = config.get("train_dataset_paths") or config.get("candidate_dataset_paths")
        if not train_paths:
            raise ValueError(
                "DataOS config must include `train_dataset_paths` or `candidate_dataset_paths`."
            )
        raw_train_records: list[dict[str, Any]] = []
        for path in train_paths:
            raw_train_records.extend(_load_records(path))
        eval_datasets = _load_datasets(config.get("eval_dataset_paths"))
        query_datasets = _load_datasets(config.get("query_dataset_paths"))
        return cls(
            config_path=resolved_config_path,
            config=config,
            runtime=runtime,
            raw_train_records=raw_train_records,
            output_dir=Path(output_dir).resolve() / "dataos",
            eval_datasets=eval_datasets,
            query_datasets=query_datasets,
        )

    @staticmethod
    def _build_operators(config: dict[str, Any]) -> list[Any]:
        operators: list[Any] = []
        eval_datasets = _load_datasets(config.get("eval_dataset_paths"))
        query_datasets = _load_datasets(config.get("query_dataset_paths"))
        for item in config.get("operators", []):
            operator_name = str(item.get("operator", "")).strip().lower()
            kwargs = dict(item.get("kwargs", {}))
            if operator_name in {"less_selector", "nice_selector"}:
                if not eval_datasets:
                    raise ValueError(
                        f"{operator_name} requires `eval_dataset_paths` in the DataOS config."
                    )
                kwargs.setdefault("eval_datasets", eval_datasets)
            if operator_name in {"near_selector", "tsds_selector"}:
                if not query_datasets:
                    raise ValueError(
                        f"{operator_name} requires `query_dataset_paths` in the DataOS config."
                    )
                kwargs.setdefault("query_datasets", query_datasets)
            registry = SELECTOR_REGISTRY | MIXER_REGISTRY | REWEIGHTER_REGISTRY
            operator_cls = registry.get(operator_name)
            if operator_cls is None:
                raise ValueError(f"Unknown DataOS operator: {operator_name}")
            operators.append(operator_cls(**kwargs))
        if not operators:
            raise ValueError("DataOS config must declare at least one operator.")
        return operators

    def attach(
        self, *, trainer: Any, tokenizer: Any, train_dataset: Any, total_updates: int
    ) -> None:
        backend_config = self.config.get("backend", {})
        parameter_filters = tuple(backend_config.get("parameter_filters", ()))
        self.backend = AttachedTextBackend(
            trainer.model,
            tokenizer,
            max_length=int(backend_config.get("max_length", 256)),
            max_new_tokens=int(backend_config.get("max_new_tokens", 48)),
            parameter_filters=parameter_filters,
        )
        self.runtime.context.metadata["backend"] = self.backend
        self.runtime.context.metadata["total_updates"] = int(total_updates)
        self.runtime.context.update_interval = int(self.config.get("update_interval", 1))
        self.runtime.prepare(
            datasets={
                "train": Dataset.from_records(self.raw_train_records),
                "eval": Dataset.from_records(
                    [record for dataset in self.eval_datasets for record in dataset.to_records()]
                ),
            }
        )
        self._run_initial_offline_selection()
        self.refresh_policy(step=0, force=True)
        self.sampler = DataOSPolicySampler(
            train_dataset, self, seed=int(getattr(trainer.args, "seed", 42))
        )

    def make_batch(
        self,
        *,
        record_ids: list[str],
        domains: list[str],
        tensors: dict[str, Any] | None = None,
    ) -> BatchView:
        indices = [self.id_to_index.get(record_id, -1) for record_id in record_ids]
        records = [
            dict(self.id_to_record.get(record_id, {"dataos_id": record_id}))
            for record_id in record_ids
        ]
        return BatchView(records=records, indices=indices, domain_ids=domains, tensors=tensors)

    def record_forward(
        self,
        *,
        batch: BatchView,
        per_sample_losses: list[float],
        reference_losses: list[float | None],
    ) -> None:
        self.runtime.after_forward(
            batch,
            ForwardSignal(per_sample_losses=per_sample_losses),
            reference_losses=reference_losses,
        )

    def batch_weights(
        self,
        *,
        batch: BatchView,
        per_sample_losses: list[float],
    ) -> list[float]:
        actions = self.runtime.before_backward(
            batch, ForwardSignal(per_sample_losses=per_sample_losses)
        )
        weights = torch.ones(len(batch.indices), dtype=torch.float32)

        if self.active_ids:
            active_mask = torch.tensor(
                [record.get("dataos_id") in self.active_ids for record in batch.records],
                dtype=torch.float32,
            )
            weights = weights * active_mask

        if self.current_domain_weights and batch.domain_ids is not None:
            domain_weights = torch.tensor(
                [
                    float(self.current_domain_weights.get(str(domain_id), 0.0))
                    for domain_id in batch.domain_ids
                ],
                dtype=torch.float32,
            )
            weights = weights * domain_weights

        for action in actions:
            if action.sample_weights is not None:
                weights = weights * torch.tensor(action.sample_weights, dtype=torch.float32)

        if torch.sum(weights) <= 0:
            weights = torch.ones_like(weights)
        weights = weights / torch.sum(weights)
        return weights.tolist()

    def refresh_policy(self, *, step: int, force: bool = False) -> None:
        self.runtime.context.step = int(step)
        interval = max(int(self.runtime.context.update_interval), 1)
        if not force and step > 0 and step % interval != 0:
            return
        actions = self.runtime.on_update_boundary(
            candidates=self.raw_train_records,
            backend=self.backend,
            total_updates=self.runtime.context.metadata.get("total_updates", 1),
        )
        self._apply_actions(actions)

    def finalize(self) -> None:
        actions = self.runtime.finalize(backend=self.backend)
        self._apply_actions(actions)
        save_state_snapshot(self.output_dir / "final_state.pt", self.runtime.states)
        summary = {
            "config_path": str(self.config_path),
            "active_ids": sorted(self.active_ids) if self.active_ids is not None else None,
            "domain_weights": self.current_domain_weights,
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def save_snapshot(self, *, step: int) -> None:
        save_state_snapshot(self.output_dir / f"state_step_{step}.pt", self.runtime.states)

    def _apply_actions(self, actions: list[Any]) -> None:
        selector_sets: list[set[str]] = []
        for action in actions:
            if action.selected_indices is not None:
                selected_ids = {
                    _maybe_get_record_id(self.raw_train_records[index], fallback=index)
                    for index in action.selected_indices
                    if 0 <= int(index) < len(self.raw_train_records)
                }
                selector_sets.append(selected_ids)
            if action.domain_weights is not None:
                self.current_domain_weights = {
                    str(key): float(value) for key, value in action.domain_weights.items()
                }
        if selector_sets:
            intersection = selector_sets[0]
            for selector_set in selector_sets[1:]:
                intersection = intersection.intersection(selector_set)
            self.active_ids = intersection

    def _run_initial_offline_selection(self) -> None:
        train_dataset = Dataset.from_records(self.raw_train_records)
        selector_sets: list[set[str]] = []
        for operator in self.runtime.operators:
            if isinstance(operator, (NearSelector, TsdsSelector)):
                selected_dataset = operator.run(train_dataset)
                selector_sets.append(
                    {
                        _maybe_get_record_id(dict(record), fallback=index)
                        for index, record in enumerate(selected_dataset)
                    }
                )
        if selector_sets:
            active_ids = selector_sets[0]
            for selector_set in selector_sets[1:]:
                active_ids = active_ids.intersection(selector_set)
            self.active_ids = active_ids

    def _initial_domain_weights(self) -> dict[str, float] | None:
        domains: dict[str, int] = defaultdict(int)
        for record in self.raw_train_records:
            domain = record.get("dataos_domain")
            if domain is None:
                metadata = record.get("metadata", {})
                if isinstance(metadata, dict):
                    domain = metadata.get("source")
            if domain is not None:
                domains[str(domain)] += 1
        if not domains:
            return None
        weight = 1.0 / len(domains)
        return {domain: weight for domain in sorted(domains)}
