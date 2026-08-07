"""dataos-eval — the deterministic GPU-eval lifecycle (submit / status / collect).

Agents never write sbatch for evaluations: this CLI renders the job script with every
environment fact baked in (single node, node pool, gres, caches, libcuda, offline HF),
submits it, tracks it, and collects the frozen score. Judge benches additionally score
on the login node at collect time.
"""

from __future__ import annotations

__version__ = "0.1.0"
