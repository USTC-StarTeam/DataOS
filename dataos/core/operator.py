from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from dataos.core.artifact import Artifact
from dataos.core.sandbox import Sandbox


@dataclass(slots=True)
class OperatorParameter:
    name: str
    type: str
    description: str
    required: bool = False
    default: Any = None
    choices: list[Any] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "required": self.required,
        }
        if self.default is not None:
            record["default"] = self.default
        if self.choices:
            record["choices"] = list(self.choices)
        return record


class Operator(ABC):
    """Abstract operator executed by the fixed runtime."""

    name = ""
    category = ""
    description = ""
    chinese_description = ""
    parameter_specs: list[OperatorParameter] = []

    @abstractmethod
    def run(self, inputs: list[Artifact], sandbox: Sandbox, **kwargs) -> Artifact | list[Artifact]:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "chinese_description": self.chinese_description,
            "parameters": [spec.to_record() for spec in self.parameter_specs],
            "default_params": self.default_params(),
        }

    def default_params(self) -> dict[str, Any]:
        defaults: dict[str, Any] = {}
        for spec in self.parameter_specs:
            if spec.default is not None:
                defaults[spec.name] = spec.default
        return defaults

    def normalize_params(self, params: dict[str, Any] | None) -> dict[str, Any]:
        params = dict(params or {})
        normalized: dict[str, Any] = {}
        allowed = {spec.name for spec in self.parameter_specs}
        unknown = sorted(set(params) - allowed)
        if unknown:
            joined = ", ".join(unknown)
            raise ValueError(f"{self.name}: unsupported params: {joined}")

        for spec in self.parameter_specs:
            if spec.name in params:
                value = params[spec.name]
            elif spec.default is not None:
                value = spec.default
            elif spec.required:
                raise ValueError(f"{self.name}: missing required param '{spec.name}'")
            else:
                continue
            normalized[spec.name] = self._coerce_value(spec, value)
        return normalized

    def _coerce_value(self, spec: OperatorParameter, value: Any) -> Any:
        if spec.type == "str":
            coerced = str(value)
        elif spec.type == "int":
            coerced = int(value)
        elif spec.type == "float":
            coerced = float(value)
        elif spec.type == "bool":
            if isinstance(value, bool):
                coerced = value
            elif isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "1", "yes"}:
                    coerced = True
                elif lowered in {"false", "0", "no"}:
                    coerced = False
                else:
                    raise ValueError(f"{self.name}: invalid boolean for '{spec.name}': {value!r}")
            else:
                coerced = bool(value)
        elif spec.type == "list[str]":
            if not isinstance(value, list):
                raise ValueError(f"{self.name}: param '{spec.name}' must be a list of strings")
            coerced = [str(item) for item in value]
        else:
            raise ValueError(f"{self.name}: unsupported param type '{spec.type}' in operator spec")

        if spec.choices and coerced not in spec.choices:
            joined = ", ".join(repr(item) for item in spec.choices)
            raise ValueError(f"{self.name}: param '{spec.name}' must be one of {joined}")
        return coerced
