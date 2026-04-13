from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class BacktestResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    adapter_type: str  # e.g. "insurance"
    baseline_score: float
    dag_score: float
    lift: float  # dag_score - baseline_score
    node_contributions: dict[str, float] = Field(default_factory=dict)  # node_id -> lift contribution
    notes: str = ""
