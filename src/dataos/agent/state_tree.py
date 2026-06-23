"""The training-state TREE — externalized so the skeleton (not the agent's head) owns it.

Root = the base model. Each node is one "base + recipe" SFT result with a score; each edge
is one training that modifies the parent's recipe. This is B-only: every node trains FROM
the fixed base, so the tree records the RECIPE lineage (which mixture/hparams were derived
from which), not weight continuation. The skeleton owns node selection (2 exploit + 1
explore) and persistence; the final model is the highest-scoring node over the whole tree.

``tree.json`` lives at the PROJECT ROOT as a file (not a subdir — ``Project.studies`` counts
every subdir as a study, the same reason ``final_model.json`` sits there).
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
        root = {
            "id": "n0000", "parent": None, "depth": 0, "round": -1,
            "role": "root", "recipe": None, "model_path": base_model, "score": None,
        }
        return cls(path, {"root": "n0000", "nodes": [root]})

    def save(self) -> None:
        atomic_write_json(self.path, {"root": self.root_id, "nodes": self.nodes})

    def get(self, node_id: str) -> dict:
        return next(n for n in self.nodes if n["id"] == node_id)

    def add_node(self, *, parent: str, role: str, round_idx: int, recipe: dict) -> str:
        nid = f"n{len(self.nodes):04d}"
        self.nodes.append({
            "id": nid, "parent": parent, "depth": self.get(parent)["depth"] + 1,
            "round": round_idx, "role": role, "recipe": recipe,
            "model_path": None, "score": None, "artifacts": {}, "scripts": {},
        })
        return nid

    def attach(self, node_id: str, score, model_path, artifacts: dict | None = None,
               scripts: dict | None = None) -> None:
        """Record an experiment's result on its node for full provenance: score, the merged-model
        path, the artifact layout (run_root / data / adapter / model / eval / logs / report), and
        the SCRIPTS actually used (the data-build code + the exact training command)."""
        n = self.get(node_id)
        n["score"], n["model_path"] = score, model_path
        n["artifacts"] = artifacts or {}
        n["scripts"] = scripts or {}

    def nodes_for_round(self, round_idx: int) -> list[str]:
        return [n["id"] for n in self.nodes if n["round"] == round_idx]

    def _scored(self) -> list[dict]:
        return sorted(
            (n for n in self.nodes if isinstance(n["score"], (int, float))),
            key=lambda n: n["score"], reverse=True,
        )

    def select_targets(self, round_idx: int) -> list[tuple[str, str]]:
        """The one selection mechanism: 3 (node_id, role) targets per round —
        2 EXPLOIT (small refinements of the top-2 scored nodes) + 1 EXPLORE (a fresh
        recipe from the root base). Early rounds with <2 scored nodes degrade to explore."""
        scored = self._scored()
        root = self.root_id
        if len(scored) >= 2:
            return [(scored[0]["id"], "exploit"), (scored[1]["id"], "exploit"), (root, "explore")]
        if len(scored) == 1:
            return [(scored[0]["id"], "exploit"), (root, "explore"), (root, "explore")]
        return [(root, "explore"), (root, "explore"), (root, "explore")]

    def best_node(self) -> dict | None:
        scored = self._scored()
        return scored[0] if scored else None

    def stats(self) -> dict:
        return {
            "n_nodes": len(self.nodes),
            "max_depth": max((n["depth"] for n in self.nodes), default=0),
        }
