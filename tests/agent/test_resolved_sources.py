"""``resolved_sources`` — the input the contract promised and the task never carried.

``operator_use.yaml`` has told experimenters "the task gives you `resolved_sources` (exact on-disk
paths)" since it was written, and grepping the rendered prompt found that string exactly once: in
that sentence. So every run guessed the path convention itself (four different spellings across the
2026-08-13 batch) and then discovered the shape by crashing — every one of the pool's 106 sources
is a ``DatasetDict``, and 5 of them have no ``train`` split, so neither the shape nor the split name
is a rule anyone could have written down. recon had already profiled all of it.
"""

from __future__ import annotations

import json

from dataos.agent import roles
from dataos.agent.config import AutoRunConfig

_POOL = "/pool/v2"


def _cfg(tmp_path, profiles: list[dict]) -> AutoRunConfig:
    cfg = AutoRunConfig(project_id="p", workspace=str(tmp_path), target_bench="gsm8k")
    project = tmp_path / "p"
    project.mkdir(parents=True, exist_ok=True)
    (project / "recon.json").write_text(
        json.dumps({"source_profiles": profiles}), encoding="utf-8"
    )
    cfg.runtime_config = {"paths": {"fixed_pool": _POOL}}
    return cfg


def test_a_source_resolves_to_its_path_splits_and_shape(tmp_path):
    cfg = _cfg(tmp_path, [
        {"source": "metamathqa", "splits": {"train": 395000}, "shape": "qa", "n_rows": 395000},
    ])
    assert roles._resolved_sources(cfg, ["metamathqa"]) == [
        {
            "id": "metamathqa",
            "path": f"{_POOL}/metamathqa",
            "splits": {"train": 395000},
            "shape": "qa",
            "n_rows": 395000,
        }
    ]


def test_the_split_names_survive_when_there_is_no_train(tmp_path):
    """The case a documented rule cannot cover.

    gpqamain's run crashed on exactly this source: ``Invalid key: 0. Please first select a split.
    For example: `my_dataset_dictionary['test'][0]`.`` — and ``test`` is not the one it wanted.
    """
    cfg = _cfg(tmp_path, [
        {"source": "mmlu_aux",
         "splits": {"test": 14042, "validation": 1531, "dev": 285, "auxiliary_train": 99842},
         "shape": "mcq", "n_rows": 14042},
        {"source": "agentinstruct",
         "splits": {"os": 195, "db": 538, "alfworld": 336, "webshop": 351,
                    "kg": 324, "mind2web": 122},
         "shape": "chat", "n_rows": 195},
    ])
    resolved = {r["id"]: r for r in roles._resolved_sources(cfg, ["mmlu_aux", "agentinstruct"])}
    assert "auxiliary_train" in resolved["mmlu_aux"]["splits"]
    assert "train" not in resolved["agentinstruct"]["splits"]
    assert resolved["agentinstruct"]["splits"]["db"] == 538


def test_an_unprofiled_source_still_gets_its_path(tmp_path):
    """A source recon never saw is a plan bug caught elsewhere; do not also lose the path."""
    cfg = _cfg(tmp_path, [])
    (only,) = roles._resolved_sources(cfg, ["never_profiled"])
    assert only["path"] == f"{_POOL}/never_profiled"
    assert only["splits"] is None


def test_no_recon_and_no_pool_degrade_without_raising(tmp_path):
    """The baseline node plans no sources at all, and smoke configs carry no pool."""
    cfg = AutoRunConfig(project_id="p", workspace=str(tmp_path), target_bench="gsm8k")
    assert roles._resolved_sources(cfg, []) == []
    (only,) = roles._resolved_sources(cfg, ["x"])
    assert only["path"] is None


def test_the_task_actually_carries_it(tmp_path, monkeypatch):
    """The half that was missing: resolution is useless if it never reaches the prompt."""
    cfg = _cfg(tmp_path, [
        {"source": "mmlu_aux", "splits": {"auxiliary_train": 99842}, "shape": "mcq",
         "n_rows": 99842},
    ])
    from dataos.agent import runtime_compose, stages

    monkeypatch.setattr(roles.runtime, "_run_root", lambda *a, **k: "/runs/x")
    monkeypatch.setattr(runtime_compose, "render_sections", lambda *a, **k: ("", "", {"task": {}}))
    monkeypatch.setattr(stages, "build_stages", lambda *a, **k: [])
    monkeypatch.setattr(stages, "render_stage_blocks", lambda *a, **k: "")
    prompt = roles._task_prompt(
        cfg=cfg,
        study_id="round-0001",
        spec={"name": "e1", "data": {"sources": ["mmlu_aux"], "recipe": []}, "train": {}},
        exp_dir=tmp_path / "exp",
        eval_limit=-1,
    )
    assert "RESOLVED SOURCES" in prompt
    assert f"{_POOL}/mmlu_aux" in prompt
    assert "auxiliary_train" in prompt
