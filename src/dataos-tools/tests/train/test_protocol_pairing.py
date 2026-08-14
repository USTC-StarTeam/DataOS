"""dataos-train's model_type -> LF template map must match the task config's declaration.

The pairing is declared once, in ``configs/tasks/<task>/eval/protocol.yaml``
(``model_families[*]``: ``eval_template`` + ``train_template`` on the same line). This package
cannot read that file at import time — it runs in a venv that has no access to the task tree —
so it mirrors the train half, and this test is what stops the two from drifting.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[4]
PROTOCOL = REPO / "configs" / "tasks" / "posttrainbench" / "eval" / "protocol.yaml"


def _protocol() -> dict:
    return yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))


def test_dataos_train_agrees_with_the_declaration():
    """dataos-train keeps its own model_type -> LF template map (it runs in a venv with no
    access to this file at import time); this test is what keeps the two from drifting."""
    from dataos_train.backends.llamafactory.ext.templates import MODEL_TYPE_TEMPLATES

    declared = {
        family: spec["train_template"] for family, spec in _protocol()["model_families"].items()
    }
    assert MODEL_TYPE_TEMPLATES == declared, (
        "configs/tasks/posttrainbench/eval/protocol.yaml model_families[*].train_template and "
        "dataos_train ... ext/templates.MODEL_TYPE_TEMPLATES disagree — bump them together"
    )


