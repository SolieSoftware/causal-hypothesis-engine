from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SessionMode(str, Enum):
    DAGCreation = "DAGCreation"
    Modification = "Modification"


class ModificationMode(str, Enum):
    Theory = "Theory"
    Backtest = "Backtest"
    Hybrid = "Hybrid"


class SessionStatus(str, Enum):
    Active = "Active"
    Confirmed = "Confirmed"
    Abandoned = "Abandoned"


class Session(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    network_id: str
    mode: SessionMode
    modification_mode: ModificationMode | None = None
    checkpoint_path: str = ""
    conversation_history: list[dict] = Field(default_factory=list)  # list of message dicts
    status: SessionStatus = SessionStatus.Active
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    exchange_count: int = 0  # incremented each exchange; triggers checkpoint every 10
