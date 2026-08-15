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
        eval_limit: int | None = None,
        scripts: dict | None = None,
    ) -> None:
        node = self.get(node_id)
        # A cache of the report's own cursor — report.json is the source of truth. `unknown` used
        # to stand in when the report could not be read at all; a node whose report says nothing is
        # a node that finished with nothing, and `failed` says that without inventing a sixth word.
        node["status"] = status or "failed"
        node["score"] = score
        node["eval_limit"] = eval_limit  # how many eval items produced that score (-1 = full)
        if model_path:
            node["model_path"] = model_path
        node["report_path"] = report_path
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
            if node.get("model_path") and node.get("status") == "completed"
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

    @staticmethod
    def _score_at(node: dict, full_size: int | None = None) -> str | None:
        """``score`` annotated with the eval size that produced it — ``"74.00@200/1319"``,
        ``"84.69@full"``, or ``"84.69@?"`` when the report didn't record one.

        The leader picks each experiment's eval size, so raw floats in the tree are silently
        cross-caliber: a 25-item probe and the full benchmark both arrive as bare numbers, and
        "did this beat its parent?" becomes a comparison between two different measurements.
        Attaching the caliber to the number itself makes that impossible to miss, and leaves the
        judgement where it belongs — with the leader."""
        score = node.get("score")
        if not isinstance(score, (int, float)):
            return None
        limit = node.get("eval_limit")
        if limit == -1:
            at = "full"
        elif limit is None:
            at = "?"
        elif full_size:
            at = f"{limit}/{full_size}"  # a subset, and what fraction of the benchmark it is
        else:
            at = str(limit)
        return f"{float(score):.2f}@{at}"

    #: How many of the most recent nodes carry their ``hypothesis`` in the tree view. Measured on
    #: a real 31-node run, prose was 92% of the view's 40K while the structure a parent choice
    #: actually needs (ids, score, status, depth) was 8%. Older nodes keep the structure only;
    #: the whole tree is on disk in tree.json, which the leader can read for an old node's
    #: reasoning. Per-round reviews are NOT repeated here — the bundle's ``recent_reviews``
    #: carries them whole (interpretation + lesson + recommendation + caution).
    PROSE_NODES = 8

    def to_context(self, *, max_nodes: int = 40, full_size: int | None = None) -> dict:
        """Tiered tree view for the leader context bundle.

        ``full_size`` (the benchmark's real item count) turns a subset score into
        ``"74.00@200/1319"`` instead of a bare ``@200``. The newest :attr:`PROSE_NODES` nodes are
        rendered in full; earlier ones keep only the fields a parent choice needs, because this
        view is re-sent every round and grows with the tree."""

        def summarize(node: dict, *, prose: bool) -> dict:
            out = {
                "node_id": node["id"],
                "parent_node_id": node.get("parent"),
                "depth": node.get("depth"),
                "round": node.get("round"),
                "status": node.get("status"),
                "experiment_name": node.get("experiment_name"),
                "has_model": bool(node.get("model_path")),
                # "<score>@<eval items>" — never a bare float, so a subset score can't be read
                # as if it were comparable to a full-benchmark one
                "score": self._score_at(node, full_size),
            }
            if prose:
                out["hypothesis"] = node.get("hypothesis")
            return out

        nodes = self.nodes[-max_nodes:]
        prose_from = len(nodes) - self.PROSE_NODES
        return {
            "root_node_id": self.root_id,
            "stats": self.stats(),
            "growable_node_ids": [node["id"] for node in self.growable_nodes()],
            "best_node_id": (self.best_node() or {}).get("id"),
            "nodes": [summarize(n, prose=i >= prose_from) for i, n in enumerate(nodes)],
            "note": (
                f"hypothesis is shown for the {self.PROSE_NODES} most recent nodes; for older "
                f"ones, and for any node's full plan, read tree.json in the project dir. "
                f"Per-round reviews are in recent_reviews."
            ),
        }
