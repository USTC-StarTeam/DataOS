"""算子命名空间。

具体算子应放在匹配类别的包下。

``__all__`` names the CATEGORIES; the operators themselves resolve by bare name through
``__getattr__`` (``from dataos_operator.operators import CodeCompletionFormatter``), because that
is the second of the three namespaces experimenters reach for and it used to be the second one to
fail. See :mod:`dataos_operator._lookup`.
"""

from typing import Any as _Any

from dataos_operator._lookup import dir_with_operators as _dir_with_operators
from dataos_operator._lookup import resolve as _resolve

__all__ = [
    "clusterer",
    "evaluator",
    "generator",
    "mapper",
    "mixer",
    "refiner",
    "reweighter",
    "selector",
]


def __getattr__(name: str) -> _Any:
    return _resolve(__name__, name)


def __dir__() -> list[str]:
    return _dir_with_operators(globals())
