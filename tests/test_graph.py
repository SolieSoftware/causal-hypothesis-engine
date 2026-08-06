"""Tests for graph invariants and causal-structure queries.

These cover the invariant that was previously enforced only by a sentence in
an LLM system prompt: a causal hypothesis network must be acyclic.
"""

from __future__ import annotations

import pytest

from causal_hypothesis_engine.graph import (
    d_separated,
    find_cycle,
    find_dangling_edges,
    find_duplicate_labels,
    validate_structure,
    would_create_cycle,
)
# Aliased: pytest would otherwise collect `testable_implications` as a test.
from causal_hypothesis_engine.graph import (
    testable_implications as implications_of,
)
from causal_hypothesis_engine.models import DAGVersion, Edge, Node
from causal_hypothesis_engine.models.node import NodeType


def _node(label: str, node_type: NodeType = NodeType.Mediator) -> Node:
    return Node(label=label, node_type=node_type)


def _edge(source: Node, target: Node) -> Edge:
    return Edge(source_node_id=source.id, target_node_id=target.id)


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------


class TestFindCycle:
    def test_acyclic_chain_has_no_cycle(self) -> None:
        a, b, c = _node("A"), _node("B"), _node("C")
        edges = [_edge(a, b), _edge(b, c)]
        assert find_cycle([a, b, c], edges) is None

    def test_diamond_is_acyclic(self) -> None:
        """A -> B -> D, A -> C -> D. Converging paths are not a cycle."""
        a, b, c, d = _node("A"), _node("B"), _node("C"), _node("D")
        edges = [_edge(a, b), _edge(a, c), _edge(b, d), _edge(c, d)]
        assert find_cycle([a, b, c, d], edges) is None

    def test_three_node_cycle_detected(self) -> None:
        a, b, c = _node("A"), _node("B"), _node("C")
        edges = [_edge(a, b), _edge(b, c), _edge(c, a)]
        cycle = find_cycle([a, b, c], edges)
        assert cycle is not None
        assert set(cycle) == {a.id, b.id, c.id}

    def test_self_loop_detected(self) -> None:
        a = _node("A")
        assert find_cycle([a], [_edge(a, a)]) == [a.id]

    def test_two_node_cycle_detected(self) -> None:
        a, b = _node("A"), _node("B")
        cycle = find_cycle([a, b], [_edge(a, b), _edge(b, a)])
        assert cycle is not None
        assert set(cycle) == {a.id, b.id}

    def test_cycle_in_one_component_of_disconnected_graph(self) -> None:
        a, b = _node("A"), _node("B")
        x, y = _node("X"), _node("Y")
        edges = [_edge(a, b), _edge(x, y), _edge(y, x)]
        cycle = find_cycle([a, b, x, y], edges)
        assert cycle is not None
        assert set(cycle) == {x.id, y.id}

    def test_deep_chain_does_not_hit_recursion_limit(self) -> None:
        """Iterative DFS: a 5,000-node chain must not raise RecursionError."""
        nodes = [_node(f"N{i}") for i in range(5_000)]
        edges = [_edge(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
        assert find_cycle(nodes, edges) is None


class TestWouldCreateCycle:
    def test_forward_edge_is_safe(self) -> None:
        a, b, c = _node("A"), _node("B"), _node("C")
        edges = [_edge(a, b), _edge(b, c)]
        assert not would_create_cycle([a, b, c], edges, a.id, c.id)

    def test_back_edge_would_cycle(self) -> None:
        a, b, c = _node("A"), _node("B"), _node("C")
        edges = [_edge(a, b), _edge(b, c)]
        assert would_create_cycle([a, b, c], edges, c.id, a.id)

    def test_self_loop_would_cycle(self) -> None:
        a = _node("A")
        assert would_create_cycle([a], [], a.id, a.id)


# ---------------------------------------------------------------------------
# The model-level invariant
# ---------------------------------------------------------------------------


class TestDAGVersionAcyclicity:
    def test_cyclic_version_is_rejected(self) -> None:
        """Regression: A -> B -> C -> A used to construct and persist cleanly."""
        a, b, c = _node("A"), _node("B"), _node("C")
        with pytest.raises(ValueError, match="acyclic"):
            DAGVersion(
                network_id="net-1",
                nodes=[a, b, c],
                edges=[_edge(a, b), _edge(b, c), _edge(c, a)],
            )

    def test_cycle_error_names_the_offending_nodes(self) -> None:
        a, b = _node("Rainfall"), _node("Flood Risk")
        with pytest.raises(ValueError) as excinfo:
            DAGVersion(
                network_id="net-1",
                nodes=[a, b],
                edges=[_edge(a, b), _edge(b, a)],
            )
        message = str(excinfo.value)
        assert "Rainfall" in message and "Flood Risk" in message

    def test_self_loop_is_rejected(self) -> None:
        a = _node("A")
        with pytest.raises(ValueError, match="acyclic"):
            DAGVersion(network_id="net-1", nodes=[a], edges=[_edge(a, a)])

    def test_acyclic_version_is_accepted(self) -> None:
        a, b, c = _node("A"), _node("B"), _node("C")
        version = DAGVersion(
            network_id="net-1",
            nodes=[a, b, c],
            edges=[_edge(a, b), _edge(b, c)],
        )
        assert len(version.edges) == 2

    def test_orphan_node_is_a_warning_not_an_error(self) -> None:
        a, b, lonely = _node("A"), _node("B"), _node("Lonely")
        version = DAGVersion(
            network_id="net-1", nodes=[a, b, lonely], edges=[_edge(a, b)]
        )
        assert any("Lonely" in w for w in version.structural_warnings())


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------


class TestValidateStructure:
    def test_dangling_edge_is_an_error(self) -> None:
        a = _node("A")
        ghost = _node("Ghost")
        errors, _warnings = validate_structure([a], [_edge(a, ghost)])
        assert any("unknown node id" in e for e in errors)

    def test_find_dangling_edges(self) -> None:
        a, ghost = _node("A"), _node("Ghost")
        edge = _edge(a, ghost)
        assert find_dangling_edges([a], [edge]) == [edge]

    def test_duplicate_labels_detected_case_insensitively(self) -> None:
        assert find_duplicate_labels([_node("Rainfall"), _node("rainfall")]) == [
            "rainfall"
        ]

    def test_clean_graph_has_no_errors(self) -> None:
        a, b = _node("A"), _node("B")
        errors, warnings = validate_structure([a, b], [_edge(a, b)])
        assert errors == []
        assert warnings == []


# ---------------------------------------------------------------------------
# d-separation — the first place edge direction actually matters
# ---------------------------------------------------------------------------


class TestDSeparation:
    def test_chain_is_blocked_by_conditioning_on_the_mediator(self) -> None:
        """A -> B -> C: A and C are dependent, but independent given B."""
        a, b, c = _node("A"), _node("B"), _node("C")
        nodes, edges = [a, b, c], [_edge(a, b), _edge(b, c)]
        assert not d_separated(nodes, edges, a.id, c.id)
        assert d_separated(nodes, edges, a.id, c.id, [b.id])

    def test_fork_is_blocked_by_conditioning_on_the_confounder(self) -> None:
        """B <- A -> C: B and C are dependent, but independent given A."""
        a, b, c = _node("A"), _node("B"), _node("C")
        nodes, edges = [a, b, c], [_edge(a, b), _edge(a, c)]
        assert not d_separated(nodes, edges, b.id, c.id)
        assert d_separated(nodes, edges, b.id, c.id, [a.id])

    def test_collider_is_opened_by_conditioning_on_it(self) -> None:
        """A -> C <- B: A and B are independent until you condition on C."""
        a, b, c = _node("A"), _node("B"), _node("C")
        nodes, edges = [a, b, c], [_edge(a, c), _edge(b, c)]
        assert d_separated(nodes, edges, a.id, b.id)
        assert not d_separated(nodes, edges, a.id, b.id, [c.id])

    def test_collider_opened_by_conditioning_on_its_descendant(self) -> None:
        """A -> C <- B, C -> D. Conditioning on D also opens the collider."""
        a, b, c, d = _node("A"), _node("B"), _node("C"), _node("D")
        nodes = [a, b, c, d]
        edges = [_edge(a, c), _edge(b, c), _edge(c, d)]
        assert d_separated(nodes, edges, a.id, b.id)
        assert not d_separated(nodes, edges, a.id, b.id, [d.id])

    def test_adjacent_nodes_are_never_separated(self) -> None:
        a, b = _node("A"), _node("B")
        assert not d_separated([a, b], [_edge(a, b)], a.id, b.id)


class TestTestableImplications:
    def test_chain_implies_one_independence(self) -> None:
        a, b, c = _node("A"), _node("B"), _node("C")
        implications = implications_of([a, b, c], [_edge(a, b), _edge(b, c)])
        assert len(implications) == 1
        x, y, given = implications[0]
        assert {x, y} == {a.id, c.id}
        assert given == (b.id,)

    def test_complete_graph_implies_nothing(self) -> None:
        """Every pair adjacent means no testable claim — an unfalsifiable DAG."""
        a, b, c = _node("A"), _node("B"), _node("C")
        edges = [_edge(a, b), _edge(a, c), _edge(b, c)]
        assert implications_of([a, b, c], edges) == []

    def test_collider_implies_unconditional_independence(self) -> None:
        a, b, c = _node("A"), _node("B"), _node("C")
        implications = implications_of([a, b, c], [_edge(a, c), _edge(b, c)])
        assert len(implications) == 1
        x, y, given = implications[0]
        assert {x, y} == {a.id, b.id}
        assert given == ()
