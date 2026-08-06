"""Single source of truth for "now".

``datetime.utcnow()`` is deprecated from Python 3.12 (this project's floor) and
returns a *naive* datetime, which silently compares wrong against anything
timezone-aware and carries no marker of the zone it nominally represents.
Everything in the package uses :func:`utcnow` instead.
"""

from __future__ import annotations

from datetime import UTC, datetime

__all__ = ["utcnow"]


def utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)
