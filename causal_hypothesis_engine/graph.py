"""Graph invariants and causal-structure queries.

This module is the single place that knows a causal hypothesis network is a
*directed acyclic graph*, not merely a bag of nodes and edges. Prior to this
module the acyclicity requirement existed only as English prose inside LLM
system prompts, which meant a cycle could be constructed, saved, diffed,
exported and backtested without any component objecting.

Two groups of functions live here:

Structural validation
  ``find_cycle``, ``find_dangling_edges``, ``find_duplicate_labels`` and the
  aggregate ``validate_structure``. These operate on plain ``(nodes, edges)``
  sequences rather than a ``DAGVersion`` so that the in-progress draft states
  used by the agents can be checked *before* an edge is committed, letting the
  model self-correct mid-conversation instead of failing at save time.

Causal-structure queries
  ``d_separated`` and ``testable_implications`` expose the part of causal
  reasoning that depends only on graph topology. These are what make the
  node types (Exposure/Confounder/Mediator/Collider) and the edge directions
  computationally meaningful rather than decorative.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import combinations
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .models.edge import Edge
    from .models.node import Node


class _HasId(Protocol):
    id: str


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------


def build_adjacency(
    nodes: Sequence["Node"],
    edges: Sequence["Edge"],
) -> dict[str, list[str]]:
    """Return a ``node_id -> [child_node_id, ...]`` map.

    Edges whose endpoints are not present in *nodes* are excluded; use
    :func:`find_dangling_edges` to detect those separately.
    """
    known = {node.id for node in nodes}
    adjacency: dict[str, list[str]] = {node.id: [] for node in nodes}
    for edge in edges:
        if edge.source_node_id in known and edge.target_node_id in known:
            adjacency[edge.source_node_id].append(edge.target_node_id)
    return adjacency


def find_cycle(
    nodes: Sequence["Node"],
    edges: Sequence["Edge"],
) -> list[str] | None:
    """Return one cycle as a list of node ids, or ``None`` if the graph is acyclic.

    Uses an iterative colouring DFS (white/grey/black) so that deep graphs
    cannot blow the Python recursion limit. The returned list is the cycle in
    traversal order with the repeated node appearing only once, e.g.
    ``[a, b, c]`` for ``a -> b -> c -> a``.
    """
    adjacency = build_adjacency(nodes, edges)

    WHITE, GREY, BLACK = 0, 1, 2
    colour: dict[str, int] = {node_id: WHITE for node_id in adjacency}
    parent: dict[str, str | None] = {node_id: None for node_id in adjacency}

    for root in adjacency:
        if colour[root] != WHITE:
            continue
        # Stack holds (node, iterator over its children).
        stack: list[tuple[str, Iterable[str]]] = [(root, iter(adjacency[root]))]
        colour[root] = GREY
        while stack:
            current, children = stack[-1]
            advanced = False
            for child in children:
                if colour[child] == GREY:
                    # Found a back edge: walk parents from `current` back to
                    # `child` to reconstruct the cycle.
                    cycle = [current]
                    walker = current
                    while walker != child and parent[walker] is not None:
                        walker = parent[walker]  # type: ignore[assignment]
                        cycle.append(walker)
                    cycle.reverse()
                    return cycle
                if colour[child] == WHITE:
                    colour[child] = GREY
                    parent[child] = current
                    stack.append((child, iter(adjacency[child])))
                    advanced = True
                    break
            if not advanced:
                colour[current] = BLACK
                stack.pop()
    return None


def would_create_cycle(
    nodes: Sequence["Node"],
    edges: Sequence["Edge"],
    source_node_id: str,
    target_node_id: str,
) -> bool:
    """True if adding ``source -> target`` would introduce a cycle.

    Checked by asking whether *source* is already reachable from *target*.
    A self-loop is always a cycle.
    """
    if source_node_id == target_node_id:
        return True
    adjacency = build_adjacency(nodes, edges)
    if target_node_id not in adjacency:
        return False
    seen: set[str] = set()
    stack = [target_node_id]
    while stack:
        current = stack.pop()
        if current == source_node_id:
            return True
        if current in seen:
            continue
        seen.add(current)
        stack.extend(adjacency.get(current, ()))
    return False


def find_dangling_edges(
    nodes: Sequence["Node"],
    edges: Sequence["Edge"],
) -> list["Edge"]:
    """Return edges referencing a node id that is not in *nodes*."""
    known = {node.id for node in nodes}
    return [
        edge
        for edge in edges
        if edge.source_node_id not in known or edge.target_node_id not in known
    ]


def find_duplicate_labels(nodes: Sequence["Node"]) -> list[str]:
    """Return labels shared by more than one node, case-insensitively.

    Duplicate labels are not a structural error but they break manifest
    label-matching and make a diff unreadable, so they are reported as a
    warning by :func:`validate_structure`.
    """
    seen: dict[str, int] = {}
    for node in nodes:
        key = node.label.strip().lower()
        seen[key] = seen.get(key, 0) + 1
    return sorted(label for label, count in seen.items() if count > 1)


def validate_structure(
    nodes: Sequence["Node"],
    edges: Sequence["Edge"],
) -> tuple[list[str], list[str]]:
    """Validate graph structure.

    Returns ``(errors, warnings)``. Errors make the object an invalid DAG and
    should block a save; warnings are advisory and should be surfaced but not
    enforced.
    """
    errors: list[str] = []
    warnings: list[str] = []

    dangling = find_dangling_edges(nodes, edges)
    if dangling:
        known = {node.id for node in nodes}
        for edge in dangling:
            missing = [
                ref
                for ref in (edge.source_node_id, edge.target_node_id)
                if ref not in known
            ]
            errors.append(
                f"Edge {edge.id[:8]} references unknown node id(s): "
                f"{', '.join(m[:8] for m in missing)}"
            )

    # Only meaningful once dangling edges are excluded, which build_adjacency does.
    cycle = find_cycle(nodes, edges)
    if cycle is not None:
        labels = _labels_for(nodes, cycle)
        errors.append(
            "Graph contains a cycle: " + " -> ".join([*labels, labels[0]])
            + ". A causal hypothesis network must be acyclic."
        )

    duplicates = find_duplicate_labels(nodes)
    if duplicates:
        warnings.append(
            "Duplicate node labels (case-insensitive): "
            + ", ".join(repr(d) for d in duplicates)
            + ". Manifest label matching and diffs will be ambiguous."
        )

    known_ids = {node.id for node in nodes}
    connected: set[str] = set()
    for edge in edges:
        if edge.source_node_id in known_ids and edge.target_node_id in known_ids:
            connected.add(edge.source_node_id)
            connected.add(edge.target_node_id)
    orphans = [node.label for node in nodes if node.id not in connected]
    if orphans and len(nodes) > 1:
        warnings.append(
            "Node(s) with no edges: " + ", ".join(repr(o) for o in orphans) + "."
        )

    return errors, warnings


def _labels_for(nodes: Sequence["Node"], node_ids: Sequence[str]) -> list[str]:
    """Map node ids to labels, falling back to a short id when not found."""
    by_id = {node.id: node.label for node in nodes}
    return [by_id.get(node_id, node_id[:8]) for node_id in node_ids]


# ---------------------------------------------------------------------------
# Causal-structure queries
# ---------------------------------------------------------------------------


def _undirected_with_direction(
    nodes: Sequence["Node"],
    edges: Sequence["Edge"],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Return ``(parents, children)`` maps keyed by node id."""
    known = {node.id for node in nodes}
    parents: dict[str, set[str]] = {node_id: set() for node_id in known}
    children: dict[str, set[str]] = {node_id: set() for node_id in known}
    for edge in edges:
        if edge.source_node_id in known and edge.target_node_id in known:
            children[edge.source_node_id].add(edge.target_node_id)
            parents[edge.target_node_id].add(edge.source_node_id)
    return parents, children


def _ancestors(
    node_ids: Iterable[str],
    parents: dict[str, set[str]],
) -> set[str]:
    """Return *node_ids* plus all of their ancestors."""
    result: set[str] = set()
    stack = list(node_ids)
    while stack:
        current = stack.pop()
        if current in result:
            continue
        result.add(current)
        stack.extend(parents.get(current, ()))
    return result


def d_separated(
    nodes: Sequence["Node"],
    edges: Sequence["Edge"],
    x: str,
    y: str,
    given: Iterable[str] = (),
) -> bool:
    """True if *x* and *y* are d-separated by the conditioning set *given*.

    Implemented with the Bayes-ball / reachability formulation: starting from
    *x*, walk the graph tracking whether each node was entered along an arrow
    pointing *into* it or *out of* it, and apply the collider rules. A path is
    open through a chain or fork only when the middle node is not conditioned
    on, and open through a collider only when the collider *or one of its
    descendants* is conditioned on.

    All arguments are node ids.
    """
    conditioned = set(given)
    parents, children = _undirected_with_direction(nodes, edges)
    if x not in parents or y not in parents:
        return True  # Unknown nodes are trivially separated.

    # A collider is "activated" if it or any descendant is conditioned on.
    activated = _ancestors(conditioned, parents)

    # State: (node, arriving_from_parent). arriving_from_parent=True means we
    # entered `node` along an edge pointing into it (parent -> node).
    visited: set[tuple[str, bool]] = set()
    # Seed: leaving x in both directions.
    stack: list[tuple[str, bool]] = [(x, True), (x, False)]

    while stack:
        node, from_parent = stack.pop()
        if (node, from_parent) in visited:
            continue
        visited.add((node, from_parent))

        if node == y:
            return False  # Found an open path.

        is_conditioned = node in conditioned

        if not from_parent:
            # Entered along node -> child, i.e. we are travelling against an
            # arrow out of `node`; `node` is a chain/fork middle.
            if not is_conditioned:
                for parent in parents[node]:
                    stack.append((parent, False))
                for child in children[node]:
                    stack.append((child, True))
        else:
            # Entered along parent -> node.
            if not is_conditioned:
                # Chain: continue to children.
                for child in children[node]:
                    stack.append((child, True))
            if node in activated:
                # Collider that is conditioned on (or has a conditioned
                # descendant): the path opens back up through its parents.
                for parent in parents[node]:
                    stack.append((parent, False))

    return True


def testable_implications(
    nodes: Sequence["Node"],
    edges: Sequence["Edge"],
    max_conditioning_set: int = 3,
) -> list[tuple[str, str, tuple[str, ...]]]:
    """Enumerate conditional independencies implied by the graph.

    Returns a list of ``(x_id, y_id, conditioning_ids)`` triples meaning
    "*x* is independent of *y* given *conditioning*". These are the graph's
    falsifiable claims: each one can be checked against data, which is what
    makes the DAG's *structure* testable rather than decorative.

    For each non-adjacent pair the smallest separating set found is reported,
    so the caller gets one implication per pair rather than a combinatorial
    explosion. *max_conditioning_set* bounds the search.
    """
    parents, children = _undirected_with_direction(nodes, edges)
    adjacent: set[frozenset[str]] = set()
    known = {node.id for node in nodes}
    for edge in edges:
        if edge.source_node_id in known and edge.target_node_id in known:
            adjacent.add(frozenset((edge.source_node_id, edge.target_node_id)))

    implications: list[tuple[str, str, tuple[str, ...]]] = []
    node_ids = [node.id for node in nodes]

    for x, y in combinations(node_ids, 2):
        if frozenset((x, y)) in adjacent:
            continue  # Adjacent nodes are never d-separated.
        others = [n for n in node_ids if n not in (x, y)]
        found: tuple[str, ...] | None = None
        limit = min(max_conditioning_set, len(others))
        for size in range(0, limit + 1):
            for subset in combinations(others, size):
                if d_separated(nodes, edges, x, y, subset):
                    found = subset
                    break
            if found is not None:
                break
        if found is not None:
            implications.append((x, y, found))

    return implications
