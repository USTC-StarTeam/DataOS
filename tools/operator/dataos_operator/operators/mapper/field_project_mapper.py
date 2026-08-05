"""保留指定字段并丢弃其他字段。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from dataos_operator.core import MapperOperator, Record, get_field, set_field

_MISSING = object()


class FieldProjectMapper(MapperOperator):
    """Project records onto a field allow-list, keeping only wanted fields (e.g. to trim intermediate fields before export).

    Implementation: read each path in ``keep_fields`` (dotted paths supported) and rebuild a
    new record; with ``strict=True`` a missing field raises, otherwise it is skipped.
    Dependency: pure-rule operator, no external dependencies.
    """

    operator_name = "field_project_mapper"

    def __init__(
        self,
        keep_fields: Sequence[str],
        strict: bool = True,
        **kwargs: Any,
    ) -> None:
        if not keep_fields:
            raise ValueError("keep_fields must contain at least one field path")
        super().__init__(keep_fields=tuple(keep_fields), strict=strict, **kwargs)
        self.keep_fields = tuple(keep_fields)
        self.strict = strict

    def map_record(self, record: Record) -> Record:
        projected: Record = {}
        for field in self.keep_fields:
            value = get_field(record, field, _MISSING)
            if value is _MISSING:
                if self.strict:
                    raise KeyError(f"Required field is missing: {field}")
                continue
            projected = set_field(projected, field, value)
        return projected
