"""Prompt composition: role system prompts = skeleton + kb blocks, single-sourced.

Guards the kb move: every block renders standalone, each role's composed system prompt
contains its blocks' signature lines, the investigator carries NO training prose, and
signature discipline lines appear exactly once across system+task (no drifted duplicates).
"""

import pytest

from dataos import prompt
from dataos.agent.backend import _BLOCKS, prompt_facts, system_prompt
from dataos.agent.config import AutoRunConfig

_CFG = AutoRunConfig(project_id="t", workspace="/tmp/x")

# Derived from _BLOCKS, never hand-maintained: membership is already asserted by
# test_prompt_layers and content byte-exactly by test_prompt_golden, so this file only checks
# that each block still renders and still carries its signature line (i.e. was not gutted).
KB_BLOCKS = [b for blocks in _BLOCKS.values() for b in blocks]

#: A line each block must keep. Blocks without an entry are only checked for rendering.
SIGNATURES = {
    "leader/context/operator_selection": "PICK BY `effect`",
    "leader/context/search_discipline": "SEARCH DISCIPLINE",
    "experimenter/context/operator_use": "LOOK BEFORE YOU WRITE",
    "experimenter/context/operator_custom": "HAND-WRITE IT",
    "experimenter/context/training": "DETERMINISTIC via dataos-train",
    "experimenter/context/cluster_discipline": "LOGIN-NODE DISCIPLINE",
    "experimenter/context/eval_contract": "FORMAT PRE-CHECK",
    "experimenter/context/report_contract": "REPORT CONTRACT",
    "experimenter/context/failure_diagnosis": "DIAGNOSE, DON'T BLINDLY RETRY",
    "investigator/context/pool_survey": "EXPLORE the",
}


@pytest.mark.parametrize("block", KB_BLOCKS)
def test_kb_block_renders_standalone(block):
    text = prompt.render(block, **prompt_facts(_CFG))
    sig = SIGNATURES.get(block)
    assert text.strip(), f"{block} rendered empty"
    if sig:
        assert sig in text, f"{block} lost its signature line {sig!r}"


def test_block_sets_cover_all_roles():
    assert set(_BLOCKS) == {"leader", "experimenter", "investigator"}


@pytest.mark.parametrize("role", ["leader", "experimenter", "investigator"])
def test_role_system_prompt_contains_its_blocks(role):
    text = system_prompt(role, _CFG)
    for block in _BLOCKS[role]:
        sig = SIGNATURES.get(block)
        if sig:
            assert sig in text, f"{role} missing {block}"


def test_leader_prompt_is_bare_of_prior_tips():
    text = system_prompt("leader", _CFG)
    assert "TIPS" not in text                   # bare leader: role + tools + facts only
    assert "10h budget" in text                 # budget_hours rendered into the skeleton
    assert "You are the LEADER" in text


def test_investigator_has_no_training_prose():
    text = system_prompt("investigator", _CFG)
    assert "dataos-train" not in text
    assert "TRAIN SPEC" not in text
    assert "You are the INVESTIGATOR" in text
    assert "do NOT train" in text


def test_experimenter_discipline_appears_once_across_system_and_task(tmp_path):
    """The pre-refactor prose duplicated these lines in system AND task; now single-sourced."""
    from pathlib import Path

    import dataos.agent.subagent_executor as se

    system = system_prompt("experimenter", _CFG)
    cfg = AutoRunConfig(
        project_id="t", workspace=str(tmp_path), target_bench="gsm8k", ptb_profile="smoke",
        h100_base_model="/m", runtime_config={},
    )
    spec = {"name": "e0", "data": {}, "train": {}, "init_model": "/m"}
    try:
        task = se._task_prompt(
            cfg=cfg, study_id="round-0000", spec=spec, exp_dir=Path(tmp_path) / "e0",
            eval_limit=4,
        )
    except RuntimeError:
        pytest.skip("DataOS-Runtime checkout not available for the task composer")
    combined = system + "\n\n" + task
    for sig in ["LOGIN-NODE DISCIPLINE", "FORMAT PRE-CHECK", "REPORT CONTRACT",
                "SUBMIT, THEN WAIT PATIENTLY", "LOOK BEFORE YOU WRITE"]:
        assert combined.count(sig) == 1, f"{sig!r} appears {combined.count(sig)} times"


def test_unknown_role_rejected():
    with pytest.raises(ValueError, match="unknown role"):
        system_prompt("nobody", _CFG)
