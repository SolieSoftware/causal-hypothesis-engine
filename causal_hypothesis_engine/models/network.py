from __future__ import annotations

import uuid
from datetime import datetime

from .._time import utcnow
from enum import Enum

from pydantic import BaseModel, Field


class AdapterType(str, Enum):
    None_ = "none"
    # Domain-agnostic: any CSV/Parquet with a declared outcome column.
    Tabular = "Tabular"
    Insurance = "Insurance"
    Financial = "Financial"
    # NOTE: Clinical has no implementation. `causal-engine new --adapter
    # Clinical` is rejected up front rather than silently creating a network
    # that can never be backtested.
    Clinical = "Clinical"


class HypothesisNetwork(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    domain: str = ""
    adapter: AdapterType = AdapterType.None_
    created_at: datetime = Field(default_factory=utcnow)
    # Versions and sessions are loaded from DB on demand; only IDs stored here.
    version_ids: list[str] = Field(default_factory=list)
    session_ids: list[str] = Field(default_factory=list)
