"""包入口：DataOS Operator。"""

from importlib import metadata as _metadata

from dataos_operator.dataset import Dataset
from dataos_operator.usage import catalog, guide, operator_docs, reference, usage

__all__ = ["Dataset", "catalog", "usage", "guide", "reference", "operator_docs", "__version__"]

# Read from the installed distribution instead of a second hand-maintained literal: this
# constant had drifted to 0.5.0 while pyproject said 0.8.x, and `guide()` reports it to the
# agent — so the version an experimenter saw was three minors stale. One source of truth now.
__version__ = _metadata.version("dataos-operator")
