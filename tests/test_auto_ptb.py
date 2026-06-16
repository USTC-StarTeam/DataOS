"""agent-version-ptb: eval is fixed to the PostTrainBench suite (no GPU/LLM/network)."""

from dataos_auto.config import AutoRunConfig
from dataos_auto.eval_freeze import freeze_eval
from dataos_auto.ptb import ptb_suite

_EXPECTED = [
    "ptb_gsm8k", "ptb_bfcl", "ptb_humaneval", "ptb_aime2025",
    "ptb_healthbench", "ptb_arenahardwriting", "ptb_gpqamain",
]


def test_ptb_suite_has_7_external_benches():
    entries = ptb_suite(profile="formal")["suite"]
    assert [e["name"] for e in entries] == _EXPECTED
    for e in entries:
        assert e["provider"] == "external"
        assert "run_ptb.sh" in e["command"]
        assert "{model_path}" in e["command"] and "{results_file}" in e["command"]
        assert e["harness"]["repo"] == "aisa-group/PostTrainBench"
        assert isinstance(e["timeout"], int)


def test_profile_changes_limits():
    formal = {e["name"]: e["command"] for e in ptb_suite(profile="formal")["suite"]}
    smoke = {e["name"]: e["command"] for e in ptb_suite(profile="smoke")["suite"]}
    assert formal["ptb_gsm8k"].endswith(" 200")
    assert smoke["ptb_gsm8k"].endswith(" 20")
    assert formal != smoke


def test_harness_path_override():
    s = ptb_suite(profile="formal", harness="/x/run_ptb.sh")
    assert "/x/run_ptb.sh" in s["suite"][0]["command"]


def test_unknown_profile_rejected():
    import pytest

    with pytest.raises(ValueError):
        ptb_suite(profile="bogus")


def test_freeze_eval_is_fixed_to_ptb_and_ignores_eval_cfg():
    cfg = AutoRunConfig(project_id="p", eval_cfg={"tasks": ["gsm8k"]}, ptb_profile="formal")
    fz = freeze_eval(cfg=cfg)
    assert "suite" in fz and len(fz["suite"]) == 7  # eval_cfg ignored; PTB fixed


def test_freeze_eval_stub_still_works():
    cfg = AutoRunConfig(project_id="p", stub=True)
    assert freeze_eval(cfg=cfg) == {"provider": "stub", "provider_config": {"score": 0.5}}
