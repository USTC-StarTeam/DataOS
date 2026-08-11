"""The eval protocol is CONFIG, and both tools read the same declaration.

Two defects motivated this (2026-08-11 smol3 batch):

* the chat template was guessed from ``f"{model_name} {model_path}"`` substrings, and
  ``--model-name`` is filled by the experimenter — so an agent could change the eval protocol
  by choosing a label. gpqamain's baseline died for want of a template while humaneval's agent
  invented a "protocol mirror" and got one; the two groups ended up scored under DIFFERENT
  templates and nobody noticed.
* ``dataos-eval`` held a hardcoded ``JUDGE_BENCHES`` table, i.e. the tool owned bench knowledge.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
TASK = REPO / "configs" / "tasks" / "posttrainbench"
PROTOCOL = TASK / "eval" / "protocol.yaml"
BENCHES = ("gsm8k", "bfcl", "aime2025", "healthbench", "arenahardwriting", "humaneval", "gpqamain")


def _protocol() -> dict:
    return yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))


def test_every_bench_is_declared():
    declared = _protocol()["benches"]
    assert set(declared) == set(BENCHES)
    for name, cfg in declared.items():
        assert isinstance(cfg.get("judge"), bool), name
        assert cfg.get("runner") in {"direct", "server"}, name
        # a runner setting must be explicit, never a tool-side default
        key = "gpu_memory_utilization" if cfg["runner"] == "direct" else "num_concurrent"
        assert key in cfg, f"{name} must declare {key}"
        assert isinstance(cfg.get("timeout"), int), name


def test_declared_eval_templates_exist():
    families = _protocol()["model_families"]
    templates = TASK / "eval" / "lmeval" / "templates"
    for family, spec in families.items():
        for key, name in spec.items():
            if key.startswith("eval_template"):
                assert (templates / name).is_file(), f"{family}.{key} -> missing {name}"


def test_train_and_eval_templates_are_declared_together():
    """The pairing is the point: one line declares both halves, so 'the training template must
    render the same turn framing as the eval jinja' has a single home."""
    for family, spec in _protocol()["model_families"].items():
        assert spec.get("eval_template"), family
        assert spec.get("train_template"), family


def test_the_eval_tool_owns_no_bench_knowledge():
    """The acceptance criterion for the refactor: the TOOL is a pure executor. Bench names
    belong to the task config, not to dataos-eval."""
    src = (REPO / "tools" / "eval" / "src").rglob("*.py")
    offenders = [
        (p.relative_to(REPO), n, line.strip())
        for p in src
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        for b in BENCHES
        if b in line
    ]
    assert not offenders, offenders
