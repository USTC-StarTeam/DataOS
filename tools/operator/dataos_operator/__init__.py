"""包入口：DataOS Operator。

Operator classes resolve here by bare name — ``from dataos_operator import ConcatMixer`` — even
though they are not imported eagerly. See :mod:`dataos_operator._lookup` for why that is a
``__getattr__`` rather than 52 more entries in ``__all__``.
"""

from importlib import metadata as _metadata
from typing import Any as _Any

from dataos_operator._lookup import dir_with_operators as _dir_with_operators
from dataos_operator._lookup import resolve as _resolve
from dataos_operator.dataset import Dataset
from dataos_operator.usage import catalog, guide, operator_docs, reference, usage

#: Eagerly imported names only. Operators are reachable by name without being listed here.
__all__ = ["Dataset", "catalog", "usage", "guide", "reference", "operator_docs", "__version__"]


def __getattr__(name: str) -> _Any:
    return _resolve(__name__, name)


def __dir__() -> list[str]:
    return _dir_with_operators(globals())

# Read from the installed distribution instead of a second hand-maintained literal: this
# constant had drifted to 0.5.0 while pyproject said 0.8.x, and `guide()` reports it to the
# agent — so the version an experimenter saw was three minors stale. One source of truth now.
__version__ = _metadata.version("dataos-operator")
