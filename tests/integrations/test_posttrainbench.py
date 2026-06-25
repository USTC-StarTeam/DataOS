"""agent-version-ptb: eval is fixed to the PostTrainBench suite (no GPU/LLM/network)."""

import pytest

from dataos.agent.config import AutoRunConfig
from dataos.runtime.eval_freeze import freeze_eval
from dataos.integrations.posttrainbench.adapter import PTB_BENCHES, ptb_eval, ptb_suite

_EXPECTED = [
    "ptb_gsm8k", "ptb_bfcl", "ptb_humaneval", "ptb_aime2025",
    "ptb_healthbench", "ptb_arenahardwriting", "ptb_gpqamain",
]


def test_ptb_suite_has_7_lmeval_benches():
    entries = ptb_suite(profile="formal")["suite"]
    assert [e["name"] for e in entries] == _EXPECTED
    for e in entries:
        assert e["provider"] == "lm_eval"
        assert e["bench"] in PTB_BENCHES
        assert isinstance(e["limit"], int)
        assert isinstance(e["timeout"], int)


def test_profile_changes_limits():
    formal = {e["bench"]: e["limit"] for e in ptb_suite(profile="formal")["suite"]}
    smoke = {e["bench"]: e["limit"] for e in ptb_suite(profile="smoke")["suite"]}
    assert formal["gsm8k"] == 200
    assert smoke["gsm8k"] == 20
    assert formal != smoke


def test_unknown_profile_rejected():
    with pytest.raises(ValueError):
        ptb_suite(profile="bogus")


def test_freeze_eval_is_fixed_to_ptb_and_ignores_eval_cfg():
    cfg = AutoRunConfig(project_id="p", eval_cfg={"tasks": ["gsm8k"]}, ptb_profile="formal")
    fz = freeze_eval(cfg=cfg)
    assert "suite" in fz and len(fz["suite"]) == 7  # eval_cfg ignored; PTB fixed


def test_freeze_eval_stub_still_works():
    cfg = AutoRunConfig(project_id="p", stub=True)
    assert freeze_eval(cfg=cfg) == {"provider": "stub", "provider_config": {"score": 0.5}}


def test_ptb_eval_single_bench():
    e = ptb_eval("gsm8k", profile="formal")
    assert "suite" not in e
    assert e["provider"] == "lm_eval" and e["bench"] == "gsm8k"
    assert e["limit"] == 200
    assert ptb_eval("gsm8k", profile="smoke")["limit"] == 20


def test_ptb_eval_unknown_bench():
    with pytest.raises(ValueError):
        ptb_eval("not_a_bench")


def test_ptb_benches_list():
    assert PTB_BENCHES == ["gsm8k", "bfcl", "humaneval", "aime2025",
                           "healthbench", "arenahardwriting", "gpqamain"]


def test_freeze_eval_target_bench_is_single():
    cfg = AutoRunConfig(project_id="p", target_bench="gsm8k")
    fz = freeze_eval(cfg=cfg)
    assert "suite" not in fz and fz["provider"] == "lm_eval" and fz["bench"] == "gsm8k"


def test_freeze_eval_no_target_is_suite():
    cfg = AutoRunConfig(project_id="p")
    fz = freeze_eval(cfg=cfg)
    assert "suite" in fz and len(fz["suite"]) == 7
