"""包入口：DataOS Operator。"""

from dataos_operator.dataset import Dataset
from dataos_operator.usage import catalog, guide, operator_docs, reference, usage

__all__ = ["Dataset", "catalog", "usage", "guide", "reference", "operator_docs", "__version__"]

__version__ = "0.5.0"
