"""Make ``src/`` importable so the suite runs without an editable install."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# A REAL model dir, because build() now derives the chat template from the model's own
# config.json instead of accepting one from the spec (audit 13.2: an accepted template let a run
# train under another family's turn framing while the frozen scorer sent this family's). Tests
# that used a bare "/m" path therefore need something readable; this is that, session-wide.
def _make_model_dir(model_type: str = "qwen3") -> str:
    import json
    import tempfile

    d = Path(tempfile.mkdtemp(prefix=f"ptb-model-{model_type}-"))
    (d / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    return str(d)


#: Importable as ``from conftest import MODEL_DIR`` or via the fixture below.
MODEL_DIR = _make_model_dir()


import pytest  # noqa: E402


@pytest.fixture(scope="session")
def model_dir() -> str:
    return MODEL_DIR
