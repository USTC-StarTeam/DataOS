"""Persistent experiment tree for the agentic main loop.

Root is the starting base model. Every child node is one experiment that continues
training from its parent's output model. Claude Code chooses the parent and plan;
Python owns node ids, parent-model injection, status updates, and persistence.

``tree.json`` lives at the project root as a file, not a directory, because
``Project.studies`` treats project subdirectories as studies.
"""

from __future__ import annotations

import json
from pathlib import Path

from dataos.core.io import atomic_write_json


class TreeState:
    def __init__(self, path: Path, data: dict) -> None:
        self.path = path
        self.root_id: str = data["root"]
        self.nodes: list[dict] = data["nodes"]

    @classmethod
    def load(cls, project_dir, base_model: str | None) -> "TreeState":
        path = Path(project_dir) / "tree.json"
        if path.is_file():
            return cls(path, json.loads(path.read_text(encoding="utf-8")))
        root_model = base_model or "base"
        root = {
            "id": "n0000",
            "parent": None,
            "depth": 0,
            "round": -1,
            "status": "completed",
            "experiment_name": "root",
            "init_model_path": None,
            "model_path": root_model,
            "score": None,
            "plan": None,
            "report_path": None,
            "review_path": None,
            "artifacts": {},
            "scripts": {},
        }
        return cls(path, {"root": "n0000", "nodes": [root]})

    def save(self) -> None:
        atomic_write_json(self.path, {"root": self.root_id, "nodes": self.nodes})

    def get(self, node_id: str) -> dict:
        for node in self.nodes:
            if node["id"] == node_id:
                return node
        raise KeyError(f"unknown tree node {node_id!r}")

    def has_node(self, node_id: str) -> bool:
        return any(node["id"] == node_id for node in self.nodes)

    def output_model(self, node_id: str) -> str | None:
        return self.get(node_id).get("model_path")

    def add_planned_node(self, *, parent: str, round_idx: int, plan: dict) -> str:
        parent_node = self.get(parent)
        init_model = parent_node.get("model_path")
        nid = f"n{len(self.nodes):04d}"
        self.nodes.append(
            {
                "id": nid,
                "parent": parent,
                "depth": int(parent_node.get("depth") or 0) + 1,
                "round": round_idx,
                "status": "planned",
                "experiment_name": plan["name"],
                "hypothesis": plan.get("hypothesis", ""),
                "estimated_hours": plan.get("estimated_hours"),
                "init_model_path": init_model,
                "model_path": None,
                "score": None,
                "plan": dict(plan),
                "report_path": None,
                "review_path": None,
                "artifacts": {},
                "scripts": {},
            }
        )
        return nid

    def mark_running(self, node_id: str) -> None:
        self.get(node_id)["status"] = "running"

    def attach_report(
        self,
        node_id: str,
        *,
        report_path: str,
        status: str | None,
        score,
        model_path,
        artifacts: dict | None = None,
        scripts: dict | None = None,
    ) -> None:
        node = self.get(node_id)
        node["status"] = status or "unknown"
        node["score"] = score
        if model_path:
            node["model_path"] = model_path
        node["report_path"] = report_path
        node["artifacts"] = artifacts or {}
        node["scripts"] = scripts or {}

    def attach_review(self, node_id: str, *, review_path: str, review: dict) -> None:
        node = self.get(node_id)
        node["review_path"] = review_path
        node["review"] = dict(review)

    def nodes_for_round(self, round_idx: int) -> list[str]:
        return [node["id"] for node in self.nodes if node.get("round") == round_idx]

    def experiment_names(self) -> set[str]:
        return {
            str(node.get("experiment_name"))
            for node in self.nodes
            if node.get("experiment_name") and node["id"] != self.root_id
        }

    def growable_nodes(self) -> list[dict]:
        return [
            node
            for node in self.nodes
            if node.get("model_path") and node.get("status") in {"completed", "unknown"}
        ]

    def _scored(self) -> list[dict]:
        return sorted(
            (
                node
                for node in self.nodes
                if isinstance(node.get("score"), (int, float)) and node.get("model_path")
            ),
            key=lambda node: node["score"],
            reverse=True,
        )

    def best_node(self) -> dict | None:
        scored = self._scored()
        return scored[0] if scored else None

    def stats(self) -> dict:
        return {
            "n_nodes": len(self.nodes),
            "n_growable": len(self.growable_nodes()),
            "max_depth": max((int(node.get("depth") or 0) for node in self.nodes), default=0),
        }

    def to_context(self, *, max_nodes: int = 40) -> dict:
        """Compact tree view for the main-agent context bundle."""

        def summarize(node: dict) -> dict:
            review = node.get("review") or {}
            return {
                "node_id": node["id"],
                "parent_node_id": node.get("parent"),
                "depth": node.get("depth"),
                "round": node.get("round"),
                "status": node.get("status"),
                "experiment_name": node.get("experiment_name"),
                "has_model": bool(node.get("model_path")),
                "score": node.get("score"),
                "hypothesis": node.get("hypothesis"),
                "review": {
                    "lesson": review.get("lesson"),
                    "recommendation": review.get("recommendation_for_future_growth"),
                }
                if review
                else None,
            }

        nodes = self.nodes[-max_nodes:]
        return {
            "root_node_id": self.root_id,
            "stats": self.stats(),
            "growable_node_ids": [node["id"] for node in self.growable_nodes()],
            "best_node_id": (self.best_node() or {}).get("id"),
            "nodes": [summarize(node) for node in nodes],
        }
