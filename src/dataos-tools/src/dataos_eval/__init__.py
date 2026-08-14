"""dataos-eval — the deterministic GPU-eval lifecycle (submit / status / collect).

Agents never write sbatch for evaluations: this CLI renders the job script with every
environment fact baked in (single node, node pool, gres, caches, libcuda, offline HF),
submits it, tracks it, and collects the frozen score. Judge benches additionally score
on the login node at collect time.
"""

from __future__ import annotations

from importlib import metadata as _metadata

# Read from the installed distribution, not a second hand-maintained literal: this had
# drifted to 0.1.0 while pyproject said 0.1.4.
# One distribution now (dataos-tools): the launcher and the three instruments always ship
# together, so they share a version. Read from the installed dist rather than a literal —
# a second hand-maintained copy is how this constant once sat three minors stale.
__version__ = _metadata.version("dataos-tools")
