"""The one pydantic base every type in this package derives from.

There were three, one per distribution that used to exist here — ``dataos_launch.types``,
``dataos_train.spec`` and ``dataos_train.modeljob.spec`` each declared a private ``_Model``. They
had drifted: only the first disabled ``protected_namespaces``, so a field named ``model_path`` or
``model_name`` warns in two of the three files and not in the third, for no reason anyone chose.

``extra="forbid"`` is the load-bearing half. These types are the CLI's wire format — JSON in,
JSON out, written to disk and read back rounds later — and a silently-dropped key in a spec is a
setting the caller believes it set. Forbidding extras turns that into an error at the boundary,
which is the only place it can still be attributed to whoever wrote it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())
