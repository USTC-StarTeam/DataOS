"""Resolving an operator by bare name, from either of the two namespaces people reach for.

The package has always exported every operator from its CATEGORY package, and the full dotted
path has always worked. The three of four runs of the 2026-08-13 batch that lost rounds to
``ImportError: cannot import name 'ConcatMixer' from 'dataos_operator'`` were not failing to find
the operators — they were guessing the shortest of three plausible namespaces, which is the one
the experimenter contract teaches by example (``from dataos_operator import Dataset`` is the only
import it shows).
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_the_root_package_resolves_an_operator_by_name():
    from dataos_operator import ConcatMixer
    from dataos_operator.operators.mixer.concat_mixer import ConcatMixer as ViaDottedPath

    assert ConcatMixer is ViaDottedPath


@pytest.mark.parametrize(
    "name",
    [
        "AnswerLetterFilter",       # selector
        "CodeCompletionFormatter",  # mapper
        "IFDEvaluator",             # evaluator
        "ConcatMixer",              # mixer
        "LossReweighter",           # reweighter
    ],
)
def test_every_category_is_reachable_from_both_namespaces(name):
    """One operator per category, through both of the namespaces that used to fail."""
    import dataos_operator
    from dataos_operator import operators

    assert getattr(dataos_operator, name).__name__ == name
    assert getattr(operators, name).__name__ == name


def test_dir_lists_the_operators():
    """The failure this is for was introspection, not import.

    An experimenter listed the module's exports, got ``['Dataset', '__all__', '__builtins__',
    ...]``, concluded ``operator class unavailable for code_completion_formatter`` and stopped.
    """
    import dataos_operator

    listed = dir(dataos_operator)
    assert "CodeCompletionFormatter" in listed
    assert sum(1 for n in listed if n[:1].isupper()) > 40


def test_a_wrong_guess_names_the_right_one():
    import dataos_operator

    with pytest.raises(AttributeError, match="ConcatMixer"):
        dataos_operator.ConcatMixxer


def test_importing_the_package_stays_lazy():
    """A fresh interpreter, because the index is process-wide once built.

    Writing all 52 names into ``__all__`` would be the obvious fix and would import the evaluators
    on every ``import dataos_operator`` — on a login node, where the cluster discipline forbids
    importing exactly this package's heavy dependencies. The dunder probes are the other half:
    inspection tools and test collectors ask modules for ``__path__`` / ``__wrapped__``, and a
    lookup that scanned on those would pay the cost anyway.
    """
    code = (
        "import sys, dataos_operator\n"
        "from dataos_operator import _lookup\n"
        "assert _lookup._index is None, 'index built at import time'\n"
        "for probe in ('__path__', '__wrapped__', '_pytest', '__all__'):\n"
        "    getattr(dataos_operator, probe, None)\n"
        "assert _lookup._index is None, 'a dunder probe triggered the scan'\n"
        "assert not [m for m in ('torch', 'transformers') if m in sys.modules], 'heavy import'\n"
        "dataos_operator.ConcatMixer\n"
        "assert _lookup._index, 'index not built on a real lookup'\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_plugin_operators_resolve_from_the_root_too(tmp_path, monkeypatch):
    """The index is the catalog scan, so DATAOS_OPERATOR_PLUGINS is covered for free.

    Not incidental: writing a run-local custom operator is routine, and a second hand-maintained
    export list would have covered the built-ins and silently missed exactly those.
    """
    (tmp_path / "my_op.py").write_text(
        "from collections.abc import Sequence\n"
        "from typing import Any\n"
        "\n"
        "from dataos_operator.core import Record, SelectorOperator\n"
        "\n"
        "\n"
        "class MyRunLocalSelector(SelectorOperator):\n"
        '    """Keep everything.\n\n    Effect: no-op.\n    """\n'
        "\n"
        "    def select(self, records: Sequence[Record], **kwargs: Any) -> list[Record]:\n"
        "        return list(records)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DATAOS_OPERATOR_PLUGINS", str(tmp_path))
    from dataos_operator import _lookup

    monkeypatch.setattr(_lookup, "_index", None)  # the index is cached per process
    import dataos_operator

    assert dataos_operator.MyRunLocalSelector.__name__ == "MyRunLocalSelector"
