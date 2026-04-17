"""DAGVersion diff tool.

Computes a structural diff between two DAGVersions and renders it with Rich.

Diff covers:
  - Nodes: added, removed, modified (label / description / node_type /
    measurability_state)
  - Edges: added, removed, modified (label / description)
  - Version metadata: status, backtest result presence, modification_rationale,
    parent linkage

The comparison is identity-based: nodes and edges are matched by their `id`
fields.  If the same logical variable was re-created with a new id it will
appear as a removal + addition rather than a modification — this is intentional,
as the id is the stable identifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models.dag_version import DAGVersion
from .models.edge import Edge
from .models.node import Node


# ---------------------------------------------------------------------------
# Diff data classes
# ---------------------------------------------------------------------------


@dataclass
class NodeChange:
    """A field-level change for a single node that exists in both versions."""

    node_id: str
    label_v1: str
    label_v2: str
    field: str          # which field changed
    old_value: str
    new_value: str


@dataclass
class EdgeChange:
    """A field-level change for a single edge that exists in both versions."""

    edge_id: str
    label_v1: str       # edge label in v1 (for display)
    label_v2: str       # edge label in v2 (for display)
    field: str
    old_value: str
    new_value: str


@dataclass
class DAGDiff:
    """Full structural diff between two DAGVersions."""

    version_id_1: str
    version_id_2: str

    # Nodes
    nodes_added: list[Node] = field(default_factory=list)       # in v2, not v1
    nodes_removed: list[Node] = field(default_factory=list)     # in v1, not v2
    nodes_modified: list[NodeChange] = field(default_factory=list)

    # Edges
    edges_added: list[Edge] = field(default_factory=list)
    edges_removed: list[Edge] = field(default_factory=list)
    edges_modified: list[EdgeChange] = field(default_factory=list)

    # Metadata changes
    metadata_changes: list[tuple[str, str, str]] = field(default_factory=list)
    # Each entry: (field_name, old_value, new_value)

    @property
    def is_identical(self) -> bool:
        return (
            not self.nodes_added
            and not self.nodes_removed
            and not self.nodes_modified
            and not self.edges_added
            and not self.edges_removed
            and not self.edges_modified
            and not self.metadata_changes
        )

    @property
    def summary_line(self) -> str:
        parts = []
        if self.nodes_added:
            parts.append(f"+{len(self.nodes_added)} node(s)")
        if self.nodes_removed:
            parts.append(f"-{len(self.nodes_removed)} node(s)")
        if self.nodes_modified:
            parts.append(f"~{len(self.nodes_modified)} node change(s)")
        if self.edges_added:
            parts.append(f"+{len(self.edges_added)} edge(s)")
        if self.edges_removed:
            parts.append(f"-{len(self.edges_removed)} edge(s)")
        if self.edges_modified:
            parts.append(f"~{len(self.edges_modified)} edge change(s)")
        if self.metadata_changes:
            parts.append(f"{len(self.metadata_changes)} metadata change(s)")
        return "  ".join(parts) if parts else "identical"


# ---------------------------------------------------------------------------
# Compute diff
# ---------------------------------------------------------------------------

# Fields compared on nodes and edges.
_NODE_FIELDS = ("label", "description", "node_type", "measurability_state")
_EDGE_FIELDS = ("label", "description")


def compute_diff(v1: DAGVersion, v2: DAGVersion) -> DAGDiff:
    """Return a DAGDiff describing what changed from *v1* to *v2*."""
    diff = DAGDiff(version_id_1=v1.version_id, version_id_2=v2.version_id)

    # -- Nodes ---------------------------------------------------------------
    v1_nodes = {n.id: n for n in v1.nodes}
    v2_nodes = {n.id: n for n in v2.nodes}

    for nid, node in v2_nodes.items():
        if nid not in v1_nodes:
            diff.nodes_added.append(node)

    for nid, node in v1_nodes.items():
        if nid not in v2_nodes:
            diff.nodes_removed.append(node)

    for nid in v1_nodes:
        if nid not in v2_nodes:
            continue
        n1, n2 = v1_nodes[nid], v2_nodes[nid]
        for f in _NODE_FIELDS:
            old = _node_field_str(n1, f)
            new = _node_field_str(n2, f)
            if old != new:
                diff.nodes_modified.append(
                    NodeChange(
                        node_id=nid,
                        label_v1=n1.label,
                        label_v2=n2.label,
                        field=f,
                        old_value=old,
                        new_value=new,
                    )
                )

    # -- Edges ---------------------------------------------------------------
    v1_edges = {e.id: e for e in v1.edges}
    v2_edges = {e.id: e for e in v2.edges}

    for eid, edge in v2_edges.items():
        if eid not in v1_edges:
            diff.edges_added.append(edge)

    for eid, edge in v1_edges.items():
        if eid not in v2_edges:
            diff.edges_removed.append(edge)

    for eid in v1_edges:
        if eid not in v2_edges:
            continue
        e1, e2 = v1_edges[eid], v2_edges[eid]
        for f in _EDGE_FIELDS:
            old = getattr(e1, f, "") or ""
            new = getattr(e2, f, "") or ""
            if old != new:
                diff.edges_modified.append(
                    EdgeChange(
                        edge_id=eid,
                        label_v1=e1.label,
                        label_v2=e2.label or new,
                        field=f,
                        old_value=old,
                        new_value=new,
                    )
                )

    # -- Metadata ------------------------------------------------------------
    _compare_meta(diff, "status", v1.status.value, v2.status.value)
    _compare_meta(
        diff,
        "backtest_result",
        "present" if v1.backtest_result else "none",
        "present" if v2.backtest_result else "none",
    )
    _compare_meta(
        diff,
        "modification_rationale",
        v1.modification_rationale or "",
        v2.modification_rationale or "",
    )
    _compare_meta(
        diff,
        "parent_version_id",
        v1.parent_version_id or "none",
        v2.parent_version_id or "none",
    )

    return diff


def _node_field_str(node: Node, field_name: str) -> str:
    val = getattr(node, field_name)
    if hasattr(val, "value"):
        return val.value
    return str(val) if val is not None else ""


def _compare_meta(
    diff: DAGDiff, field_name: str, old: str, new: str
) -> None:
    if old != new:
        diff.metadata_changes.append((field_name, old, new))


# ---------------------------------------------------------------------------
# Rich renderer
# ---------------------------------------------------------------------------


def render_diff(diff: DAGDiff, console: Any, v1: DAGVersion, v2: DAGVersion) -> None:
    """Print a Rich-formatted diff report to *console*."""
    from rich.panel import Panel
    from rich.table import Table

    console.print(
        Panel(
            f"[dim]v1:[/dim] [cyan]{diff.version_id_1}[/cyan]  "
            f"({v1.status.value}, {len(v1.nodes)} nodes, {len(v1.edges)} edges)\n"
            f"[dim]v2:[/dim] [cyan]{diff.version_id_2}[/cyan]  "
            f"({v2.status.value}, {len(v2.nodes)} nodes, {len(v2.edges)} edges)\n\n"
            f"[bold]{diff.summary_line}[/bold]",
            title="DAGVersion Diff",
        )
    )

    if diff.is_identical:
        console.print("[green]Versions are structurally identical.[/green]")
        return

    # Build a node label map from both versions combined.
    node_labels: dict[str, str] = {n.id: n.label for n in v1.nodes}
    node_labels.update({n.id: n.label for n in v2.nodes})

    # -- Nodes added
    if diff.nodes_added:
        t = Table(title="[green]Nodes Added[/green]", show_lines=False, box=None)
        t.add_column("ID", style="dim")
        t.add_column("Label", style="green")
        t.add_column("Type")
        t.add_column("Measurability")
        for n in diff.nodes_added:
            t.add_row(
                n.id[:8], n.label,
                n.node_type.value, n.measurability_state.value,
            )
        console.print(t)

    # -- Nodes removed
    if diff.nodes_removed:
        t = Table(title="[red]Nodes Removed[/red]", show_lines=False, box=None)
        t.add_column("ID", style="dim")
        t.add_column("Label", style="red")
        t.add_column("Type")
        t.add_column("Measurability")
        for n in diff.nodes_removed:
            t.add_row(
                n.id[:8], n.label,
                n.node_type.value, n.measurability_state.value,
            )
        console.print(t)

    # -- Nodes modified
    if diff.nodes_modified:
        t = Table(title="[yellow]Nodes Modified[/yellow]", show_lines=False, box=None)
        t.add_column("ID", style="dim")
        t.add_column("Label")
        t.add_column("Field", style="yellow")
        t.add_column("v1", style="red")
        t.add_column("v2", style="green")
        for ch in diff.nodes_modified:
            label = ch.label_v2 if ch.label_v1 != ch.label_v2 else ch.label_v1
            t.add_row(
                ch.node_id[:8], label, ch.field,
                ch.old_value, ch.new_value,
            )
        console.print(t)

    # -- Edges added
    if diff.edges_added:
        t = Table(title="[green]Edges Added[/green]", show_lines=False, box=None)
        t.add_column("ID", style="dim")
        t.add_column("From", style="green")
        t.add_column("To", style="green")
        t.add_column("Label")
        for e in diff.edges_added:
            t.add_row(
                e.id[:8],
                node_labels.get(e.source_node_id, e.source_node_id[:8]),
                node_labels.get(e.target_node_id, e.target_node_id[:8]),
                e.label,
            )
        console.print(t)

    # -- Edges removed
    if diff.edges_removed:
        t = Table(title="[red]Edges Removed[/red]", show_lines=False, box=None)
        t.add_column("ID", style="dim")
        t.add_column("From", style="red")
        t.add_column("To", style="red")
        t.add_column("Label")
        for e in diff.edges_removed:
            t.add_row(
                e.id[:8],
                node_labels.get(e.source_node_id, e.source_node_id[:8]),
                node_labels.get(e.target_node_id, e.target_node_id[:8]),
                e.label,
            )
        console.print(t)

    # -- Edges modified
    if diff.edges_modified:
        t = Table(title="[yellow]Edges Modified[/yellow]", show_lines=False, box=None)
        t.add_column("ID", style="dim")
        t.add_column("Field", style="yellow")
        t.add_column("v1", style="red")
        t.add_column("v2", style="green")
        for ch in diff.edges_modified:
            t.add_row(ch.edge_id[:8], ch.field, ch.old_value, ch.new_value)
        console.print(t)

    # -- Metadata
    if diff.metadata_changes:
        t = Table(title="[yellow]Metadata Changes[/yellow]", show_lines=False, box=None)
        t.add_column("Field", style="yellow")
        t.add_column("v1", style="red")
        t.add_column("v2", style="green")
        for field_name, old, new in diff.metadata_changes:
            # Truncate long rationales for display.
            old_disp = old[:80] + "…" if len(old) > 80 else old
            new_disp = new[:80] + "…" if len(new) > 80 else new
            t.add_row(field_name, old_disp, new_disp)
        console.print(t)
