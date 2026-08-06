from __future__ import annotations

import uuid
from datetime import datetime

from .._time import utcnow
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
    created_at: datetime = Field(default_factory=utcnow)
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

    @model_validator(mode="after")
    def must_be_acyclic(self) -> "DAGVersion":
        """Enforce the defining invariant: the graph is *directed acyclic*.

        Without this, a cycle could be constructed, saved, diffed, exported and
        backtested with nothing objecting — the acyclicity requirement existed
        only as prose inside LLM system prompts.
        """
        from ..graph import find_cycle

        cycle = find_cycle(self.nodes, self.edges)
        if cycle is not None:
            by_id = {node.id: node.label for node in self.nodes}
            path = [by_id.get(node_id, node_id[:8]) for node_id in cycle]
            raise ValueError(
                "A causal hypothesis network must be acyclic, but this graph "
                "contains a cycle: " + " -> ".join([*path, path[0]])
            )
        return self

    def structural_warnings(self) -> list[str]:
        """Advisory (non-blocking) structural issues: duplicate labels, orphans."""
        from ..graph import validate_structure

        _errors, warnings = validate_structure(self.nodes, self.edges)
        return warnings

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
