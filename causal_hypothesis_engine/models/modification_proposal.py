"""ModificationProposal — structured output schema for ModificationAgent.

Every change the agent proposes must be expressed as a ModificationProposal.
No freeform prose is accepted as a change record.

Change types
------------
add_node      — add a new node to the DAG
remove_node   — remove an existing node (and its connected edges)
update_node   — change label, description, type, or measurability of a node
add_edge      — add a directed edge between two existing nodes
remove_edge   — remove an existing edge
update_edge   — change label or description of an existing edge
"""

from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Change payloads — one per change type
# ---------------------------------------------------------------------------


class AddNodeChange(BaseModel):
    change_type: Literal["add_node"] = "add_node"
    label: str
    node_type: str          # NodeType value string
    measurability_state: str = "Hypothetical"
    description: str = ""


class RemoveNodeChange(BaseModel):
    change_type: Literal["remove_node"] = "remove_node"
    node_id: str            # must match an existing node.id


class UpdateNodeChange(BaseModel):
    change_type: Literal["update_node"] = "update_node"
    node_id: str
    label: str | None = None
    description: str | None = None
    node_type: str | None = None
    measurability_state: str | None = None


class AddEdgeChange(BaseModel):
    change_type: Literal["add_edge"] = "add_edge"
    source_node_id: str
    target_node_id: str
    label: str = ""
    description: str = ""


class RemoveEdgeChange(BaseModel):
    change_type: Literal["remove_edge"] = "remove_edge"
    edge_id: str            # must match an existing edge.id


class UpdateEdgeChange(BaseModel):
    change_type: Literal["update_edge"] = "update_edge"
    edge_id: str
    label: str | None = None
    description: str | None = None


# Discriminated union — Pydantic v2 picks the right model from change_type.
ProposedChange = Union[
    AddNodeChange,
    RemoveNodeChange,
    UpdateNodeChange,
    AddEdgeChange,
    RemoveEdgeChange,
    UpdateEdgeChange,
]


# ---------------------------------------------------------------------------
# ModificationProposal — top-level structured output
# ---------------------------------------------------------------------------


class ModificationProposal(BaseModel):
    """A single atomic proposal from the ModificationAgent.

    The agent must always emit this schema — never freeform prose — when
    proposing a structural change to the DAG.

    Attributes:
        source:         Where the proposal originates.
                        "theory"  — pure causal reasoning
                        "data"    — driven by backtest results
                        "both"    — theory and data agree (HYBRID mode)
        confidence:     0.0–1.0, agent's subjective confidence in the proposal.
        rationale:      One-paragraph justification.  Must explain *why*, not
                        just *what*.
        proposed_change: The structural mutation to apply if accepted.
        theory_data_agreement: Only populated in HYBRID mode.  True if theory
                        and data signals agree, False if they conflict.
    """

    source: Literal["theory", "data", "both"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    proposed_change: ProposedChange = Field(discriminator="change_type")
    theory_data_agreement: bool | None = None   # HYBRID mode only
