"""LLaMA-Factory integration entrypoints.

The objects are loaded lazily so importing this namespace does not require the
optional training stack unless the caller actually asks for trainer integration.
"""

from __future__ import annotations

import importlib

__all__ = [
    "AttachedTextBackend",
    "DataOSCallback",
    "DataOSController",
    "DataOSLanguageModelingCollator",
    "DataOSPolicySampler",
    "DataOSTrainingController",
    "causal_per_sample_losses",
    "load_dataos_config",
]

_LAZY_EXPORTS = {
    "AttachedTextBackend": ("dataos_operator.training", "AttachedTextBackend"),
    "DataOSCallback": ("dataos_operator.integrations.llamafactory.callback", "DataOSCallback"),
    "DataOSController": ("dataos_operator.training", "DataOSController"),
    "DataOSLanguageModelingCollator": (
        "dataos_operator.integrations.llamafactory.collator",
        "DataOSLanguageModelingCollator",
    ),
    "DataOSPolicySampler": ("dataos_operator.training", "DataOSPolicySampler"),
    "DataOSTrainingController": ("dataos_operator.training", "DataOSTrainingController"),
    "causal_per_sample_losses": ("dataos_operator.training", "causal_per_sample_losses"),
    "load_dataos_config": ("dataos_operator.training", "load_dataos_config"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
