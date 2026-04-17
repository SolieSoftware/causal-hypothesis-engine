"""Tests for the DAGVersion diff tool (comparison.py).

No Anthropic API calls, no DB required for the core diff logic.
DB-dependent tests verify the CLI helper path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from causal_hypothesis_engine.comparison import (
    DAGDiff,
    EdgeChange,
    NodeChange,
    compute_diff,
    render_diff,
)
from causal_hypothesis_engine.models.backtest_result import BacktestResult
from causal_hypothesis_engine.models.dag_version import DAGVersion, DAGVersionStatus
from causal_hypothesis_engine.models.edge import Edge
from causal_hypothesis_engine.models.network import AdapterType, HypothesisNetwork
from causal_hypothesis_engine.models.node import MeasurabilityState, Node, NodeType
from causal_hypothesis_engine.persistence.db import Database


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node(label: str, node_type: NodeType = NodeType.Exposure,
          state: MeasurabilityState = MeasurabilityState.Hypothetical,
          description: str = "") -> Node:
    return Node(label=label, node_type=node_type,
                measurability_state=state, description=description)


def _edge(src: Node, tgt: Node, label: str = "causes") -> Edge:
    return Edge(source_node_id=src.id, target_node_id=tgt.id, label=label)


def _version(network_id: str = "net-1",
             nodes: list[Node] | None = None,
             edges: list[Edge] | None = None,
             status: DAGVersionStatus = DAGVersionStatus.Draft,
             rationale: str = "",
             parent_id: str | None = None) -> DAGVersion:
    return DAGVersion(
        network_id=network_id,
        nodes=nodes or [],
        edges=edges or [],
        status=status,
        modification_rationale=rationale,
        parent_version_id=parent_id,
    )


# ---------------------------------------------------------------------------
# Null console (suppresses Rich output)
# ---------------------------------------------------------------------------


class _NullConsole:
    def print(self, *a, **kw) -> None:
        pass


# ---------------------------------------------------------------------------
# Identical versions
# ---------------------------------------------------------------------------


class TestIdenticalVersions:
    def test_same_version_is_identical(self) -> None:
        rain = _node("Rain")
        flood = _node("Flood", NodeType.Outcome)
        e = _edge(rain, flood)
        v = _version(nodes=[rain, flood], edges=[e])
        diff = compute_diff(v, v)
        assert diff.is_identical

    def test_empty_versions_are_identical(self) -> None:
        v1 = _version()
        v2 = _version()
        diff = compute_diff(v1, v2)
        assert diff.is_identical

    def test_summary_line_identical(self) -> None:
        v = _version()
        diff = compute_diff(v, v)
        assert diff.summary_line == "identical"


# ---------------------------------------------------------------------------
# Node diffs
# ---------------------------------------------------------------------------


class TestNodeDiffs:
    def test_node_added(self) -> None:
        rain = _node("Rain")
        flood = _node("Flood", NodeType.Outcome)
        v1 = _version(nodes=[rain])
        v2 = _version(nodes=[rain, flood])
        diff = compute_diff(v1, v2)
        assert len(diff.nodes_added) == 1
        assert diff.nodes_added[0].id == flood.id
        assert not diff.nodes_removed
        assert not diff.nodes_modified

    def test_node_removed(self) -> None:
        rain = _node("Rain")
        flood = _node("Flood", NodeType.Outcome)
        v1 = _version(nodes=[rain, flood])
        v2 = _version(nodes=[rain])
        diff = compute_diff(v1, v2)
        assert len(diff.nodes_removed) == 1
        assert diff.nodes_removed[0].id == flood.id
        assert not diff.nodes_added

    def test_node_label_modified(self) -> None:
        n = _node("Rain")
        v1 = _version(nodes=[n])
        n2 = n.model_copy(update={"label": "Heavy Rain"})
        v2 = _version(nodes=[n2])
        diff = compute_diff(v1, v2)
        assert len(diff.nodes_modified) == 1
        ch = diff.nodes_modified[0]
        assert ch.field == "label"
        assert ch.old_value == "Rain"
        assert ch.new_value == "Heavy Rain"

    def test_node_type_modified(self) -> None:
        n = _node("Rainfall", NodeType.Exposure)
        v1 = _version(nodes=[n])
        n2 = n.model_copy(update={"node_type": NodeType.Confounder})
        v2 = _version(nodes=[n2])
        diff = compute_diff(v1, v2)
        fields = [ch.field for ch in diff.nodes_modified]
        assert "node_type" in fields

    def test_node_measurability_modified(self) -> None:
        n = _node("Rainfall", state=MeasurabilityState.Hypothetical)
        v1 = _version(nodes=[n])
        n2 = n.model_copy(update={"measurability_state": MeasurabilityState.Proxied})
        v2 = _version(nodes=[n2])
        diff = compute_diff(v1, v2)
        fields = [ch.field for ch in diff.nodes_modified]
        assert "measurability_state" in fields

    def test_node_description_modified(self) -> None:
        n = _node("Rain", description="original")
        v1 = _version(nodes=[n])
        n2 = n.model_copy(update={"description": "updated description"})
        v2 = _version(nodes=[n2])
        diff = compute_diff(v1, v2)
        fields = [ch.field for ch in diff.nodes_modified]
        assert "description" in fields

    def test_multiple_field_changes_on_same_node(self) -> None:
        n = _node("Rain", NodeType.Exposure, MeasurabilityState.Hypothetical)
        v1 = _version(nodes=[n])
        n2 = n.model_copy(update={
            "label": "Rainfall",
            "measurability_state": MeasurabilityState.Identified,
        })
        v2 = _version(nodes=[n2])
        diff = compute_diff(v1, v2)
        assert len(diff.nodes_modified) == 2
        fields = {ch.field for ch in diff.nodes_modified}
        assert fields == {"label", "measurability_state"}

    def test_unchanged_node_not_in_modified(self) -> None:
        rain = _node("Rain")
        flood = _node("Flood", NodeType.Outcome)
        v1 = _version(nodes=[rain, flood])
        flood2 = flood.model_copy(update={"label": "Major Flood"})
        v2 = _version(nodes=[rain, flood2])
        diff = compute_diff(v1, v2)
        # Only flood should appear as modified, not rain.
        modified_ids = {ch.node_id for ch in diff.nodes_modified}
        assert flood.id in modified_ids
        assert rain.id not in modified_ids


# ---------------------------------------------------------------------------
# Edge diffs
# ---------------------------------------------------------------------------


class TestEdgeDiffs:
    def test_edge_added(self) -> None:
        rain = _node("Rain")
        flood = _node("Flood", NodeType.Outcome)
        e = _edge(rain, flood)
        v1 = _version(nodes=[rain, flood])
        v2 = _version(nodes=[rain, flood], edges=[e])
        diff = compute_diff(v1, v2)
        assert len(diff.edges_added) == 1
        assert diff.edges_added[0].id == e.id

    def test_edge_removed(self) -> None:
        rain = _node("Rain")
        flood = _node("Flood", NodeType.Outcome)
        e = _edge(rain, flood)
        v1 = _version(nodes=[rain, flood], edges=[e])
        v2 = _version(nodes=[rain, flood])
        diff = compute_diff(v1, v2)
        assert len(diff.edges_removed) == 1
        assert diff.edges_removed[0].id == e.id

    def test_edge_label_modified(self) -> None:
        rain = _node("Rain")
        flood = _node("Flood", NodeType.Outcome)
        e = _edge(rain, flood, label="causes")
        v1 = _version(nodes=[rain, flood], edges=[e])
        e2 = e.model_copy(update={"label": "directly causes"})
        v2 = _version(nodes=[rain, flood], edges=[e2])
        diff = compute_diff(v1, v2)
        assert len(diff.edges_modified) == 1
        assert diff.edges_modified[0].field == "label"
        assert diff.edges_modified[0].old_value == "causes"
        assert diff.edges_modified[0].new_value == "directly causes"

    def test_edge_description_modified(self) -> None:
        rain = _node("Rain")
        flood = _node("Flood", NodeType.Outcome)
        e = Edge(source_node_id=rain.id, target_node_id=flood.id,
                 label="causes", description="old desc")
        v1 = _version(nodes=[rain, flood], edges=[e])
        e2 = e.model_copy(update={"description": "new desc"})
        v2 = _version(nodes=[rain, flood], edges=[e2])
        diff = compute_diff(v1, v2)
        fields = [ch.field for ch in diff.edges_modified]
        assert "description" in fields

    def test_unchanged_edge_not_in_modified(self) -> None:
        rain = _node("Rain")
        flood = _node("Flood", NodeType.Outcome)
        drought = _node("Drought", NodeType.Confounder)
        e1 = _edge(rain, flood, label="causes")
        e2 = _edge(drought, flood, label="worsens")
        v1 = _version(nodes=[rain, flood, drought], edges=[e1, e2])
        e2_mod = e2.model_copy(update={"label": "amplifies"})
        v2 = _version(nodes=[rain, flood, drought], edges=[e1, e2_mod])
        diff = compute_diff(v1, v2)
        modified_ids = {ch.edge_id for ch in diff.edges_modified}
        assert e2.id in modified_ids
        assert e1.id not in modified_ids


# ---------------------------------------------------------------------------
# Metadata diffs
# ---------------------------------------------------------------------------


class TestMetadataDiffs:
    def test_status_change(self) -> None:
        v1 = _version(status=DAGVersionStatus.Draft)
        br = BacktestResult(
            version_id="v1", adapter_type="Insurance",
            baseline_score=0.5, dag_score=0.6, lift=0.1,
        )
        # DAGVersion validator requires backtest_result when status=Tested.
        v2 = DAGVersion(
            network_id="net-1", nodes=[], edges=[],
            status=DAGVersionStatus.Tested,
            backtest_result=br,
        )
        diff = compute_diff(v1, v2)
        meta_fields = {f for f, _, _ in diff.metadata_changes}
        assert "status" in meta_fields

    def test_rationale_change(self) -> None:
        v1 = _version(rationale="")
        v2 = _version(rationale="Added confounder based on domain knowledge.")
        diff = compute_diff(v1, v2)
        meta_fields = {f for f, _, _ in diff.metadata_changes}
        assert "modification_rationale" in meta_fields

    def test_parent_version_change(self) -> None:
        v1 = _version(parent_id=None)
        v2 = _version(parent_id="abc-123")
        diff = compute_diff(v1, v2)
        meta_fields = {f for f, _, _ in diff.metadata_changes}
        assert "parent_version_id" in meta_fields

    def test_backtest_result_added(self) -> None:
        v1 = _version()
        br = BacktestResult(
            version_id="v1", adapter_type="Insurance",
            baseline_score=0.5, dag_score=0.6, lift=0.1,
        )
        v2 = _version()
        v2 = v2.model_copy(update={"backtest_result": br})
        diff = compute_diff(v1, v2)
        meta_fields = {f for f, _, _ in diff.metadata_changes}
        assert "backtest_result" in meta_fields

    def test_identical_metadata_no_changes(self) -> None:
        v1 = _version(rationale="same", parent_id="p-1")
        v2 = _version(rationale="same", parent_id="p-1")
        diff = compute_diff(v1, v2)
        # Status and backtest are both Draft/none — no metadata changes.
        assert not diff.metadata_changes


# ---------------------------------------------------------------------------
# Summary line
# ---------------------------------------------------------------------------


class TestSummaryLine:
    def test_summary_includes_counts(self) -> None:
        rain = _node("Rain")
        flood = _node("Flood", NodeType.Outcome)
        v1 = _version(nodes=[rain])
        v2 = _version(nodes=[rain, flood])
        diff = compute_diff(v1, v2)
        assert "+1 node(s)" in diff.summary_line

    def test_summary_multiple_change_types(self) -> None:
        rain = _node("Rain")
        flood = _node("Flood", NodeType.Outcome)
        drought = _node("Drought", NodeType.Confounder)
        e = _edge(rain, flood)
        v1 = _version(nodes=[rain, flood], edges=[e])
        v2 = _version(nodes=[rain, drought])  # flood removed, drought added, edge removed
        diff = compute_diff(v1, v2)
        summary = diff.summary_line
        assert "+1 node(s)" in summary
        assert "-1 node(s)" in summary
        assert "-1 edge(s)" in summary


# ---------------------------------------------------------------------------
# DAGDiff.is_identical
# ---------------------------------------------------------------------------


class TestIsIdentical:
    def test_false_when_node_added(self) -> None:
        v1 = _version(nodes=[_node("A")])
        v2 = _version(nodes=[_node("A"), _node("B")])
        diff = compute_diff(v1, v2)
        assert not diff.is_identical

    def test_true_when_all_empty(self) -> None:
        diff = DAGDiff(version_id_1="v1", version_id_2="v2")
        assert diff.is_identical


# ---------------------------------------------------------------------------
# render_diff — smoke test (no assertions on content, just no exceptions)
# ---------------------------------------------------------------------------


class TestRenderDiff:
    def test_render_identical_no_crash(self) -> None:
        rain = _node("Rain")
        v = _version(nodes=[rain])
        diff = compute_diff(v, v)
        render_diff(diff, _NullConsole(), v, v)

    def test_render_full_diff_no_crash(self) -> None:
        rain = _node("Rain")
        flood = _node("Flood", NodeType.Outcome)
        drought = _node("Drought", NodeType.Confounder)
        e = _edge(rain, flood)
        v1 = _version(nodes=[rain, flood], edges=[e])
        rain2 = rain.model_copy(update={"label": "Heavy Rain"})
        v2 = _version(nodes=[rain2, drought])
        diff = compute_diff(v1, v2)
        render_diff(diff, _NullConsole(), v1, v2)

    def test_render_with_metadata_changes_no_crash(self) -> None:
        v1 = _version(rationale="old rationale")
        v2 = _version(rationale="new rationale")
        diff = compute_diff(v1, v2)
        render_diff(diff, _NullConsole(), v1, v2)


# ---------------------------------------------------------------------------
# Integration: compute_diff on versions loaded from DB
# ---------------------------------------------------------------------------


class TestCompareDiffFromDB:
    def test_round_trip_via_db(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "test.db")
        net = HypothesisNetwork(name="N", domain="d", adapter=AdapterType.None_)
        db.create_network(net)

        rain = _node("Rain")
        flood = _node("Flood", NodeType.Outcome)
        e = _edge(rain, flood)
        v1 = _version(net.id, nodes=[rain, flood], edges=[e])
        db.save_version(v1)

        # v2: rain → Heavy Rain, drought added, edge removed
        rain2 = rain.model_copy(update={"label": "Heavy Rain"})
        drought = _node("Drought", NodeType.Confounder)
        v2 = _version(net.id, nodes=[rain2, flood, drought], parent_id=v1.version_id)
        db.save_version(v2)

        loaded_v1 = db.get_version(v1.version_id)
        loaded_v2 = db.get_version(v2.version_id)
        diff = compute_diff(loaded_v1, loaded_v2)

        assert len(diff.nodes_added) == 1
        assert diff.nodes_added[0].label == "Drought"
        assert len(diff.nodes_modified) == 1
        assert diff.nodes_modified[0].field == "label"
        assert len(diff.edges_removed) == 1
