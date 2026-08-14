"""Resolving an operator class by its bare name, on the package root or on ``operators``.

Every category package already exports its operators (selector 27, mapper 10, evaluator 10,
mixer 4, reweighter 1) and the full dotted path works. What did not work was the shortest guess::

    from dataos_operator import ConcatMixer                              # ImportError
    from dataos_operator.operators import CodeCompletionFormatter        # AttributeError
    from dataos_operator.operators.mixer.concat_mixer import ConcatMixer # the only one that worked

and the shortest guess is the one the experimenter contract teaches by example, because
``from dataos_operator import Dataset`` is the only import it shows. Three of the four runs of the
2026-08-13 batch spent rounds rediscovering that, and one gave up a different way: it introspected
the module and reported ``operator class unavailable for code_completion_formatter;
exports=['Dataset', '__all__', '__builtins__', ...]`` — a listing with no operators in it.

Two properties matter more than the convenience:

* **Lazy.** A name already in a module's dict never reaches ``__getattr__``, so importing the
  package costs exactly what it did before. That is not a micro-optimisation: writing all 52 names
  into ``__all__`` would pull the evaluators — and the torch some of them import — into every
  ``import dataos_operator``, on a login node where the cluster discipline forbids exactly that.
* **One list.** The index is built from :func:`list_operator_specs`, the same scan ``usage()`` and
  ``catalog()`` read, so operators registered through ``DATAOS_OPERATOR_PLUGINS`` resolve here too
  — which matters because writing a run-local custom operator is routine.
"""

from __future__ import annotations

import difflib
import importlib
from typing import Any

_index: dict[str, str] | None = None


def operator_index() -> dict[str, str]:
    """``{ClassName: dotted.path.to.ClassName}`` for every built-in and plugin operator."""
    global _index
    if _index is None:
        from dataos_operator.core.catalog import list_operator_specs

        _index = {
            spec.operator_class.rpartition(".")[2]: spec.operator_class
            for spec in list_operator_specs()
            if spec.operator_class
        }
    return _index


def resolve(module_name: str, name: str) -> Any:
    """The body of a module-level ``__getattr__``. Raises AttributeError like one."""
    # Only CapWords are looked up. Anything else — `__path__`, `__wrapped__`, the attributes
    # inspection tools and pytest probe modules for — must stay a cheap miss rather than trigger
    # the scan that imports all 52 operator modules.
    if not name[:1].isupper():
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
    index = operator_index()
    dotted = index.get(name)
    if dotted is None:
        near = difflib.get_close_matches(name, index, n=3)
        hint = f"; closest names: {', '.join(near)}" if near else ""
        raise AttributeError(
            f"module {module_name!r} has no operator {name!r}{hint}. "
            f"dir({module_name}) lists every operator, and dataos_operator.catalog() describes them"
        )
    module, _, qualname = dotted.rpartition(".")
    return getattr(importlib.import_module(module), qualname)


def dir_with_operators(module_globals: dict[str, Any]) -> list[str]:
    """``__dir__`` that shows the operators.

    The failure this is for was not an import error: an experimenter listed the module's exports,
    saw ``['Dataset', '__all__', '__builtins__', ...]``, concluded the operator was unavailable and
    stopped. Introspection has to be able to answer the question the operators exist to answer.
    """
    return sorted({*module_globals, *operator_index()})
