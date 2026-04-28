"""Tests for export.py — Mermaid, DOT, JSON, and 3D HTML output."""

from __future__ import annotations

import json

import pytest

from causal_hypothesis_engine.export import to_dot, to_html_3d, to_json, to_mermaid
from causal_hypothesis_engine.models.dag_version import DAGVersion
from causal_hypothesis_engine.models.edge import Edge
from causal_hypothesis_engine.models.node import MeasurabilityState, Node, NodeType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_version() -> DAGVersion:
    fed = Node(
        label="Fed Funds Rate",
        description="Overnight rate set by the Fed",
        node_type=NodeType.Exposure,
        measurability_state=MeasurabilityState.Proxied,
    )
    treasury = Node(
        label="10Y Treasury",
        description="10-year treasury yield",
        node_type=NodeType.Mediator,
        measurability_state=MeasurabilityState.Identified,
    )
    sp500 = Node(
        label="S&P 500",
        description="Equity market index",
        node_type=NodeType.Outcome,
        measurability_state=MeasurabilityState.Validated,
    )
    e1 = Edge(
        source_node_id=fed.id,
        target_node_id=treasury.id,
        label="raises",
    )
    e2 = Edge(
        source_node_id=treasury.id,
        target_node_id=sp500.id,
        label="pressures",
    )
    return DAGVersion(
        network_id="test-net",
        nodes=[fed, treasury, sp500],
        edges=[e1, e2],
        modification_rationale="test",
    )


def _make_empty_version() -> DAGVersion:
    return DAGVersion(
        network_id="test-net",
        nodes=[],
        edges=[],
        modification_rationale="empty",
    )


# ---------------------------------------------------------------------------
# Mermaid
# ---------------------------------------------------------------------------


def test_mermaid_contains_flowchart_header() -> None:
    v = _make_version()
    result = to_mermaid(v)
    assert result.startswith("flowchart TD")


def test_mermaid_contains_all_node_labels() -> None:
    v = _make_version()
    result = to_mermaid(v)
    assert "Fed Funds Rate" in result
    assert "10Y Treasury" in result
    assert "S&P 500" in result


def test_mermaid_contains_node_types() -> None:
    v = _make_version()
    result = to_mermaid(v)
    assert "Exposure" in result
    assert "Mediator" in result
    assert "Outcome" in result


def test_mermaid_contains_edge_labels() -> None:
    v = _make_version()
    result = to_mermaid(v)
    assert "raises" in result
    assert "pressures" in result


def test_mermaid_contains_style_classes() -> None:
    v = _make_version()
    result = to_mermaid(v)
    assert "classDef exposure" in result
    assert "classDef outcome" in result


def test_mermaid_empty_version() -> None:
    v = _make_empty_version()
    result = to_mermaid(v)
    assert "flowchart TD" in result


def test_mermaid_edge_without_label() -> None:
    n1 = Node(label="A", node_type=NodeType.Exposure, measurability_state=MeasurabilityState.Hypothetical)
    n2 = Node(label="B", node_type=NodeType.Outcome, measurability_state=MeasurabilityState.Hypothetical)
    e = Edge(source_node_id=n1.id, target_node_id=n2.id)  # no label
    v = DAGVersion(network_id="x", nodes=[n1, n2], edges=[e], modification_rationale="t")
    result = to_mermaid(v)
    assert "-->" in result
    assert "n0 --> n1" in result


# ---------------------------------------------------------------------------
# DOT
# ---------------------------------------------------------------------------


def test_dot_starts_with_digraph() -> None:
    v = _make_version()
    result = to_dot(v)
    assert result.startswith("digraph causal_dag {")
    assert result.strip().endswith("}")


def test_dot_contains_node_ids() -> None:
    v = _make_version()
    result = to_dot(v)
    for node in v.nodes:
        assert node.id in result


def test_dot_contains_edge_arrows() -> None:
    v = _make_version()
    result = to_dot(v)
    assert "->" in result


def test_dot_contains_fill_colours() -> None:
    v = _make_version()
    result = to_dot(v)
    assert "fillcolor" in result
    assert "#4f86c6" in result  # Exposure colour


def test_dot_empty_version() -> None:
    v = _make_empty_version()
    result = to_dot(v)
    assert "digraph" in result


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def test_json_is_valid_json() -> None:
    v = _make_version()
    result = to_json(v)
    parsed = json.loads(result)
    assert isinstance(parsed, dict)


def test_json_contains_nodes_and_edges() -> None:
    v = _make_version()
    parsed = json.loads(to_json(v))
    assert "nodes" in parsed
    assert "edges" in parsed
    assert len(parsed["nodes"]) == 3
    assert len(parsed["edges"]) == 2


def test_json_round_trips_version_id() -> None:
    v = _make_version()
    parsed = json.loads(to_json(v))
    assert parsed["version_id"] == v.version_id


def test_json_empty_version() -> None:
    v = _make_empty_version()
    parsed = json.loads(to_json(v))
    assert parsed["nodes"] == []
    assert parsed["edges"] == []


# ---------------------------------------------------------------------------
# 3D HTML
# ---------------------------------------------------------------------------


def test_html_is_valid_html_structure() -> None:
    v = _make_version()
    result = to_html_3d(v)
    assert "<!DOCTYPE html>" in result
    assert "<html" in result
    assert "</html>" in result


def test_html_contains_graph_data() -> None:
    v = _make_version()
    result = to_html_3d(v)
    assert "Fed Funds Rate" in result
    assert "10Y Treasury" in result
    assert "S&P 500" in result


def test_html_embeds_3d_force_graph_cdn() -> None:
    v = _make_version()
    result = to_html_3d(v)
    assert "3d-force-graph" in result


def test_html_contains_node_colours() -> None:
    v = _make_version()
    result = to_html_3d(v)
    assert "#4f86c6" in result  # Exposure
    assert "#e05c5c" in result  # Outcome
    assert "#5cb85c" in result  # Mediator


def test_html_node_count_in_header() -> None:
    v = _make_version()
    result = to_html_3d(v)
    assert "3 nodes" in result
    assert "2 edges" in result


def test_html_empty_version() -> None:
    v = _make_empty_version()
    result = to_html_3d(v)
    assert "0 nodes" in result
    assert "<!DOCTYPE html>" in result
