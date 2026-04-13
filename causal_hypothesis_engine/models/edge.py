from __future__ import annotations

from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class EdgeDirection(str, Enum):
    Forward = "Forward"   # source -> target
    Backward = "Backward"  # target -> source


class Edge(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    id: str = Field(default_factory=lambda: str(uuid4()))
    source_node_id: str
    target_node_id: str
    label: str = ""
    description: str = ""
