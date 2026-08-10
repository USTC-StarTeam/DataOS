"""Byte-exact golden snapshots of the three roles' context prompts.

The prompts are tuned over many rounds and are fragile, so every structural refactor
(moving files, splitting context/turn, de-duplicating blocks across roles) must leave the
COMPOSED text byte-identical. This test is that proof; the signature-based assertions in
``test_prompt_compose.py`` move with the refactor and cannot serve as one.

A deliberate content change is expected to fail this test. Review the diff it prints, and
only then refresh the fixture:

    uv run python -c "
    import pathlib
    from dataos.agent.backend import system_prompt
    from dataos.agent.config import AutoRunConfig
    cfg = AutoRunConfig(project_id='t', workspace='/tmp/x')
    out = pathlib.Path('tests/agent/fixtures/prompts')
    for r in ('leader', 'experimenter', 'investigator'):
        (out / f'{r}.txt').write_text(system_prompt(r, cfg), encoding='utf-8')
    "
"""

import difflib
from pathlib import Path

import pytest

from dataos.agent.backend import system_prompt
from dataos.agent.config import AutoRunConfig

_FIXTURES = Path(__file__).parent / "fixtures" / "prompts"
# Fixed config: context blocks render run facts (budget / GPUs / bench), so the snapshot needs a
# deterministic one. Defaults give 10h / 1 GPU; target_bench is set so the snapshot exercises the
# real fact rather than its "the target benchmark" fallback.
_CFG = AutoRunConfig(project_id="t", workspace="/tmp/x", target_bench="gsm8k")


@pytest.mark.parametrize("role", ["leader", "experimenter", "investigator"])
def test_context_prompt_matches_golden(role):
    expected = (_FIXTURES / f"{role}.txt").read_text(encoding="utf-8")
    actual = system_prompt(role, _CFG)
    if actual != expected:
        diff = "\n".join(
            difflib.unified_diff(
                expected.splitlines(), actual.splitlines(),
                fromfile=f"golden/{role}", tofile=f"rendered/{role}", lineterm="",
            )
        )
        pytest.fail(f"{role} context prompt changed:\n{diff}")
