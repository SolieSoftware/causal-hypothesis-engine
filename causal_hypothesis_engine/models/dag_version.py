from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from .backtest_result import BacktestResult
from .edge import Edge
from .node import Node


class DAGVersionStatus(str, Enum):
    Draft = "Draft"
    Tested = "Tested"
    Archived = "Archived"


class DAGVersion(BaseModel):
    version_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    network_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    parent_version_id: str | None = None
    modification_rationale: str = ""
    backtest_result: BacktestResult | None = None
    status: DAGVersionStatus = DAGVersionStatus.Draft

    @model_validator(mode="after")
    def tested_requires_backtest_result(self) -> "DAGVersion":
        if self.status == DAGVersionStatus.Tested and self.backtest_result is None:
            raise ValueError(
                "A DAGVersion with status Tested must have a backtest_result attached."
            )
        return self

    @property
    def is_immutable(self) -> bool:
        """True when the version must not be modified in place."""
        return self.status in (DAGVersionStatus.Tested, DAGVersionStatus.Archived)

    def node_by_id(self, node_id: str) -> Node | None:
        """Return the node with the given id, or None if not found."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
