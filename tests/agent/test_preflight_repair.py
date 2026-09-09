from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dataos.agent import preflight_repair as repair
from dataos.agent.schemas import ExperimentPlan
from dataos.agent.state_tree import TreeState


def _report(capacity=578):
    return {
        "id": "experiment", "status": "failed",
        "stages": {
            "data": {
                "failure_kind": "quota_capacity", "rows": 0,
                "submission_materialized": False, "official_submit_called": False,
                "error": "quota capacity unavailable", "handle_path": "/run/data_handle.json",
                "capacity_diagnostic": {
                    "capacities": {"mcq": capacity, "other": 20000},
                    "requested_quotas": {"mcq": 1500, "other": 8500},
                    "shortfalls": {"mcq": 1500 - capacity},
                },
            },
            "train": {"status": "not_started"}, "eval": {"status": "not_started"},
        },
    }


def _setup(tmp_path, monkeypatch):
    tree = TreeState.load(tmp_path, "/base")
    plan = ExperimentPlan(parent_node_id=tree.root_id, name="candidate", hypothesis="coverage",
                          data={"recipe": [{"op": "selector", "params": {"quotas": {"mcq": 1500}}}]})
    node = tree.get(tree.add_planned_node(parent=tree.root_id, round_idx=2, plan=plan.model_dump()))
    node.update(iteration=3, suite_target="task_iter3")
    tree.save()
    path = tmp_path / "candidate" / "report.json"
    path.parent.mkdir()
    path.write_text(json.dumps(_report()))
    monkeypatch.setattr(repair.iterations, "preflight_check", lambda *args: {"eligible": True})
    return tree, node, path


def _attempt(tree, node, path, **kwargs):
    def propose(context):
        assert len(tree.get(node["id"])["preflight_repairs"]) == context["revision"]
        return ExperimentPlan(
            parent_node_id=node["parent"], name=node["experiment_name"], hypothesis="feasible coverage",
            data={"recipe": [{"op": "selector", "params": {"strata_weights": {"mcq": 2}}}]},
        )
    return repair.attempt(cfg=SimpleNamespace(), node=node, tree=tree, report_path=path,
                          policy={"max_revisions": 2}, propose=kwargs.get("propose", propose),
                          validate=lambda plan: [])


@pytest.mark.parametrize("change", [
    lambda r: r.update(status="completed"),
    lambda r: r["stages"]["data"].update(submission_materialized=True),
    lambda r: r["stages"]["data"].update(official_submit_called=True),
    lambda r: r["stages"]["data"].update(rows=10000),
    lambda r: r["stages"]["train"].update(status="failed", error="CUDA OOM"),
    lambda r: r["stages"]["train"].update(handle="/run/train.handle.json"),
    lambda r: r["stages"]["eval"].update(score=0.5),
    lambda r: r["stages"]["data"].update(failure_kind="image_decode", capacity_diagnostic={}),
])
def test_only_unsubmitted_capacity_failures_are_repairable(change):
    report = _report()
    change(report)
    assert repair.diagnosis(report) is None


def test_legacy_capacity_layouts_are_recognized_without_error_string_matching():
    report = _report()
    data = report["stages"]["data"]
    data.pop("failure_kind")
    data.pop("capacity_diagnostic")
    data["selection_statistics"] = {
        "post_deduplication_post_image_cap_capacities": {"mcq": 578},
        "quota_shortfalls": {"mcq": 922}, "requested_quotas": {"mcq": 1500},
    }
    assert repair.diagnosis(report)["evidence"]["shortfalls"] == {"mcq": 922}
    data.pop("selection_statistics")
    data["preflight"] = {
        "infeasible_required_cells_at_raw_pool_level": {"source|one": {"capacity": 0, "quota": 200}},
        "final_eligible_source_by_qa_bin": {"source|many": 10000},
    }
    assert repair.diagnosis(report) is not None
    data.pop("preflight")
    data.pop("submission_materialized")
    data["official_submission_created"] = False
    report["stages"]["train"]["status"] = "not_run"
    data["failure_evidence"] = {"exhausted_cell": {
        "key": "ocr|short", "deficit": 14, "qualified_after_fixed_rules": 2,
        "n0004_final_quota": 16,
    }}
    assert repair.diagnosis(report)["evidence"]["capacities"] == {"ocr|short": 2}


def test_revision_is_archived_and_activated_on_the_same_node(tmp_path, monkeypatch):
    tree, node, path = _setup(tmp_path, monkeypatch)
    old_plan = copy.deepcopy(node["plan"])
    old_report = json.loads(path.read_text())
    assert _attempt(tree, node, path)
    assert node["plan"] == old_plan and json.loads(path.read_text()) == old_report
    request = json.loads(Path(node["preflight_repairs"][0]["request_path"]).read_text())
    assert request["report"] == old_report and request["previous_plan"] == old_plan
    # Reload between proposal and activation, as on a process crash.
    tree = TreeState.load(tmp_path, "/base")
    node = tree.get(node["id"])
    repair.activate_pending(cfg=None, node=node, tree=tree, report_path=path)
    assert node["iteration"] == 3 and node["suite_target"] == "task_iter3"
    assert node["plan_revision"] == 1 and node["plan"] != old_plan
    assert json.loads(path.read_text())["status"] == "data"
    assert len(tree.nodes) == 2
    repair.activate_pending(cfg=None, node=node, tree=tree, report_path=path)
    assert len(node["preflight_repairs"]) == 1


def test_unchanged_capacity_error_stops_after_revised_selection(tmp_path, monkeypatch):
    tree, node, path = _setup(tmp_path, monkeypatch)
    assert _attempt(tree, node, path)
    repair.activate_pending(cfg=None, node=node, tree=tree, report_path=path)
    path.write_text(json.dumps(_report()))
    assert not _attempt(tree, node, path)
    assert node["preflight_repair_stop"] == "preflight:unchanged_capacity_failure"
    assert len(node["preflight_repairs"]) == 1


def test_revision_limit_survives_resume_and_interrupted_proposals(tmp_path, monkeypatch):
    tree, node, path = _setup(tmp_path, monkeypatch)

    def interrupt(context):
        raise RuntimeError("transport failed")

    for _ in range(2):
        assert not _attempt(tree, node, path, propose=interrupt)
        tree = TreeState.load(tmp_path, "/base")
        node = tree.get(node["id"])
    assert not _attempt(tree, node, path)
    assert node["preflight_repair_stop"] == "preflight:revision_limit"
    assert len(node["preflight_repairs"]) == 2


@pytest.mark.parametrize("field,value", [("name", "new-node"), ("parent_node_id", "other"),
                                       ("train", {"learning_rate": 0.1}), ("eval_limit", 1)])
def test_repair_cannot_change_identity_or_frozen_stages(tmp_path, monkeypatch, field, value):
    tree, node, path = _setup(tmp_path, monkeypatch)

    def propose(context):
        plan = copy.deepcopy(node["plan"])
        plan["data"] = {"new": "selector"}
        plan[field] = value
        return ExperimentPlan(**plan)

    assert not _attempt(tree, node, path, propose=propose)
    assert json.loads(path.read_text())["status"] == "failed"


def test_prose_or_seed_only_revision_is_rejected(tmp_path, monkeypatch):
    tree, node, path = _setup(tmp_path, monkeypatch)

    def propose(context):
        plan = copy.deepcopy(node["plan"])
        plan["data"].update(seed=1234, evidence="new explanation")
        return ExperimentPlan(**plan)

    assert not _attempt(tree, node, path, propose=propose)
    assert "did not change DATA" in node["preflight_repairs"][0]["error"]


def test_remote_proof_blocks_repair_and_is_rechecked_before_activation(tmp_path, monkeypatch):
    tree, node, path = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(repair.iterations, "preflight_check", lambda *args: {"eligible": False})
    assert not _attempt(tree, node, path)
    assert not node["preflight_repairs"]
    monkeypatch.setattr(repair.iterations, "preflight_check", lambda *args: {"eligible": True})
    assert _attempt(tree, node, path)
    monkeypatch.setattr(repair.iterations, "preflight_check", lambda *args: {"eligible": False})
    with pytest.raises(RuntimeError, match="unsafe"):
        repair.activate_pending(cfg=None, node=node, tree=tree, report_path=path)
    assert json.loads(path.read_text())["status"] == "failed"


def test_activation_recovers_a_crash_after_report_reopening(tmp_path, monkeypatch):
    tree, node, path = _setup(tmp_path, monkeypatch)
    assert _attempt(tree, node, path)
    repair.activate_pending(cfg=None, node=node, tree=tree, report_path=path)
    current = json.loads(path.read_text())
    current["stages"]["data"]["progress"] = "preserve this"
    path.write_text(json.dumps(current))
    node["preflight_repairs"][-1]["state"] = "ready"
    repair.activate_pending(cfg=None, node=node, tree=tree, report_path=path)
    assert json.loads(path.read_text())["stages"]["data"]["progress"] == "preserve this"
