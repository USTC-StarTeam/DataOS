"""Bounded, durable DATA-plan corrections before an official submission or training."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from dataos.core.io import atomic_write_json

from . import iterations


log = logging.getLogger(__name__)


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _selection_digest(data: dict) -> str:
    ignored = {"seed", "random_seed", "evidence", "expected_data_change", "description",
               "rationale", "primary_intervention", "bounded_sample_check", "reference_node_id",
               "pretraining_comparison", "required_pretraining_checks", "controlled_factors"}

    def strip(value):
        if isinstance(value, dict):
            return {key: strip(item) for key, item in value.items() if key not in ignored}
        if isinstance(value, list):
            return [strip(item) for item in value]
        return value

    return _digest(strip(data))


def diagnosis(report: dict) -> dict | None:
    if report.get("status") != "failed":
        return None
    stages = report.get("stages") or {}
    data = stages.get("data") or {}
    materialized = data.get("submission_materialized", data.get("official_submission_created"))
    if materialized is not False or data.get("rows") != 0:
        return None
    if data.get("official_submit_called") is True:
        return None
    for name in ("train", "eval"):
        stage = stages.get(name) or {}
        if stage.get("status", stage.get("state")) not in (None, "not_started", "not_run", "skipped"):
            return None
        if any(value for key, value in stage.items() if key not in {"status", "state", "reason"}):
            return None
    evidence = data.get("capacity_diagnostic")
    if not (data.get("failure_kind") == "quota_capacity" and isinstance(evidence, dict)
            and evidence.get("capacities") and evidence.get("shortfalls")):
        # Historical reports contain these same structured measurements under the
        # preflight layouts already emitted by the deployed selectors. Never match error text.
        stats = data.get("selection_statistics") or {}
        preflight = data.get("preflight") or {}
        failure = data.get("failure_evidence") or {}
        cell = failure.get("exhausted_cell") or {}
        if stats.get("quota_shortfalls") and stats.get("post_deduplication_post_image_cap_capacities"):
            evidence = {
                "capacities": stats["post_deduplication_post_image_cap_capacities"],
                "requested_quotas": stats.get("requested_quotas"),
                "shortfalls": stats["quota_shortfalls"],
            }
        elif preflight.get("infeasible_required_cells_at_raw_pool_level"):
            evidence = {
                "capacities": preflight.get("final_eligible_source_by_qa_bin"),
                "shortfalls": preflight["infeasible_required_cells_at_raw_pool_level"],
            }
        elif (cell.get("key") and isinstance(cell.get("deficit"), int) and cell["deficit"] > 0
              and isinstance(cell.get("qualified_after_fixed_rules"), int)
              and isinstance(cell.get("n0004_final_quota"), int)):
            evidence = {
                "capacities": {cell["key"]: cell["qualified_after_fixed_rules"]},
                "requested_quotas": {cell["key"]: cell["n0004_final_quota"]},
                "shortfalls": {cell["key"]: cell["deficit"]},
                "source_qualified": failure.get("source_qualified"),
                "qualified_rows": failure.get("qualified_rows"),
            }
        else:
            return None
    handle = data.get("handle_path") or data.get("handle") or data.get("data_handle")
    if not isinstance(handle, str) or not handle:
        return None
    return {"error": data.get("error"), "evidence": evidence, "data_handle": handle}


def attempt(*, cfg, node: dict, tree, report_path: Path, policy: dict, propose, validate) -> bool:
    """Reserve one revision before asking the leader; a ready revision is applied next loop."""
    if not policy or node.get("round", 0) <= 0:
        return False  # the official random baseline cannot be replanned
    report = json.loads(report_path.read_text())
    diagnostic = diagnosis(report)
    if diagnostic is None:
        return False
    history = node.setdefault("preflight_repairs", [])
    maximum = min(2, int(policy.get("max_revisions", 0)))
    signature = _digest(diagnostic["evidence"])
    if len(history) >= maximum:
        node["preflight_repair_stop"] = "preflight:revision_limit"
        tree.save()
        return False
    if any(item["signature"] == signature and item["state"] == "applied" for item in history):
        node["preflight_repair_stop"] = "preflight:unchanged_capacity_failure"
        tree.save()
        return False
    proof = iterations.preflight_check(cfg, node, diagnostic["data_handle"])
    if not proof.get("eligible"):
        node["preflight_repair_stop"] = f"preflight:{proof.get('reason', 'unsafe_target')}"
        tree.save()
        return False

    number = len(history) + 1
    directory = report_path.parent / "preflight_revisions" / f"{number:02d}"
    request = {
        "node_id": node["id"], "iteration": node["iteration"], "target_key": node["suite_target"],
        "revision": number, "max_revisions": maximum, "remaining_revisions": maximum - number,
        "previous_plan": node["plan"], "diagnostic": diagnostic, "remote_proof": proof,
        "report": report,
    }
    atomic_write_json(directory / "request.json", request)
    record = {"number": number, "signature": signature, "state": "requested",
              "request_path": str(directory / "request.json")}
    history.append(record)
    node.pop("preflight_repair_stop", None)
    tree.save()  # crashes and repeated --resume cannot reset the correction budget
    try:
        plan = propose(request)
        errors = validate(plan)
        old = node["plan"]
        if plan.name != old["name"] or plan.parent_node_id != old["parent_node_id"]:
            errors.append("preflight repair must preserve node name and parent")
        if plan.train or plan.eval_limit != old.get("eval_limit"):
            errors.append("preflight repair cannot change train/eval")
        if _selection_digest(plan.data) == _selection_digest(old.get("data") or {}):
            errors.append("preflight repair did not change DATA")
        if errors:
            raise ValueError("; ".join(errors))
        atomic_write_json(directory / "plan.json", plan.model_dump())
        record["state"] = "ready"
        record["plan_path"] = str(directory / "plan.json")
        tree.save()
        return True
    except Exception as exc:
        record["state"] = "rejected"
        record["error"] = f"{type(exc).__name__}: {exc}"
        node["preflight_repair_stop"] = "preflight:invalid_revision"
        tree.save()
        log.error("preflight revision rejected for %s: %s", node["id"], exc)
        return False


def activate_pending(*, cfg, node: dict, tree, report_path: Path) -> None:
    """Idempotently reconcile an accepted revision after a crash, before any execution."""
    history = node.get("preflight_repairs") or []
    if not history or history[-1]["state"] != "ready":
        return
    record = history[-1]
    request = json.loads(Path(record["request_path"]).read_text())
    current = json.loads(report_path.read_text())
    proof = iterations.preflight_check(cfg, node, request["diagnostic"]["data_handle"])
    if not proof.get("eligible"):
        raise RuntimeError(f"preflight repair became unsafe before activation: {proof}")
    plan = json.loads(Path(record["plan_path"]).read_text())
    node["plan"] = plan
    node["hypothesis"] = plan["hypothesis"]
    node["estimated_hours"] = plan.get("estimated_hours")
    node["plan_revision"] = record["number"]
    if (current.get("stages", {}).get("data") or {}).get("preflight_revision") != record["number"]:
        if _digest(current) != _digest(request["report"]):
            raise RuntimeError("preflight report changed after proposal; refusing to overwrite")
        reopened = {
            **request["report"], "status": "data", "error": None,
            "stages": {"data": {
                "preflight_revision": record["number"],
                "diagnostic_path": record["request_path"],
                "previous_error": request["diagnostic"]["error"],
            }},
        }
        atomic_write_json(report_path, reopened)
    record["state"] = "applied"
    node["status"] = "planned"
    tree.save()
