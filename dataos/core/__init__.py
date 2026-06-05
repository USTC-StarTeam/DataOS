"""Core DataOS abstractions.

The staged SDK keeps only :class:`TaskTemplate` here — the old artifact /
sandbox / store / operator / registry / agent stack has been removed. The
three runtime types live in :mod:`dataos.experiment`.
"""

from __future__ import annotations

from dataos.core.task import TaskTemplate

__all__ = ["TaskTemplate"]
