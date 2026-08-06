from __future__ import annotations

import uuid
from datetime import datetime

from .._time import utcnow
from enum import Enum

from pydantic import BaseModel, Field


class AdapterType(str, Enum):
    None_ = "none"
    Insurance = "Insurance"
    Financial = "Financial"
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
