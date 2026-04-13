"""Tests for Node and Edge models."""
from __future__ import annotations

import pytest

from causal_hypothesis_engine.models.edge import Edge, EdgeDirection
from causal_hypothesis_engine.models.node import (
    BaseNodeMetadata,
    InsuranceNodeMetadata,
    MeasurabilityState,
    Node,
    NodeType,
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestNodeTypeEnum:
    def test_all_values_present(self) -> None:
        values = {e.value for e in NodeType}
        assert values == {"Exposure", "Outcome", "Confounder", "Mediator", "Collider"}

    def test_members_accessible_by_name(self) -> None:
        assert NodeType.Exposure == NodeType("Exposure")
        assert NodeType.Collider == NodeType("Collider")


class TestMeasurabilityStateEnum:
    def test_all_values_present(self) -> None:
        values = {e.value for e in MeasurabilityState}
        assert values == {"Hypothetical", "Identified", "Proxied", "Validated"}

    def test_members_accessible_by_name(self) -> None:
        assert MeasurabilityState.Hypothetical == MeasurabilityState("Hypothetical")
        assert MeasurabilityState.Validated == MeasurabilityState("Validated")


class TestEdgeDirectionEnum:
    def test_all_values_present(self) -> None:
        values = {e.value for e in EdgeDirection}
        assert values == {"Forward", "Backward"}


# ---------------------------------------------------------------------------
# Node creation and defaults
# ---------------------------------------------------------------------------


class TestNodeCreation:
    def test_minimal_creation(self) -> None:
        node = Node(label="Weather", node_type=NodeType.Exposure)
        assert node.label == "Weather"
        assert node.node_type is NodeType.Exposure
        assert node.description == ""
        assert node.measurability_state is MeasurabilityState.Hypothetical
        assert node.edge_ids == []
        assert node.adapter_metadata is None

    def test_id_auto_generated(self) -> None:
        n1 = Node(label="A", node_type=NodeType.Outcome)
        n2 = Node(label="B", node_type=NodeType.Outcome)
        assert n1.id != n2.id
        assert len(n1.id) == 36  # UUID4 string

    def test_explicit_id(self) -> None:
        node = Node(id="custom-id", label="X", node_type=NodeType.Confounder)
        assert node.id == "custom-id"

    def test_enum_values_preserved_as_enum(self) -> None:
        """use_enum_values=False means enum members are not coerced to raw values."""
        node = Node(label="Y", node_type=NodeType.Mediator)
        assert isinstance(node.node_type, NodeType)
        assert isinstance(node.measurability_state, MeasurabilityState)

    def test_all_node_types_accepted(self) -> None:
        for nt in NodeType:
            node = Node(label="n", node_type=nt)
            assert node.node_type is nt

    def test_all_measurability_states_accepted(self) -> None:
        for ms in MeasurabilityState:
            node = Node(
                label="n",
                node_type=NodeType.Exposure,
                measurability_state=ms,
            )
            assert node.measurability_state is ms


# ---------------------------------------------------------------------------
# Node serialisation roundtrip
# ---------------------------------------------------------------------------


class TestNodeSerialisation:
    def test_model_dump_contains_expected_keys(self) -> None:
        node = Node(label="Rain", node_type=NodeType.Exposure)
        data = node.model_dump()
        assert set(data.keys()) == {
            "id",
            "label",
            "description",
            "node_type",
            "measurability_state",
            "edge_ids",
            "adapter_metadata",
        }

    def test_model_validate_roundtrip(self) -> None:
        node = Node(
            label="Flood",
            description="A flood event",
            node_type=NodeType.Outcome,
            measurability_state=MeasurabilityState.Proxied,
            edge_ids=["edge-1", "edge-2"],
        )
        data = node.model_dump()
        restored = Node.model_validate(data)
        assert restored.id == node.id
        assert restored.label == node.label
        assert restored.description == node.description
        assert restored.node_type is NodeType.Outcome
        assert restored.measurability_state is MeasurabilityState.Proxied
        assert restored.edge_ids == ["edge-1", "edge-2"]

    def test_model_dump_json_roundtrip(self) -> None:
        node = Node(label="Loss", node_type=NodeType.Collider)
        json_str = node.model_dump_json()
        restored = Node.model_validate_json(json_str)
        assert restored.id == node.id
        assert restored.node_type is NodeType.Collider


# ---------------------------------------------------------------------------
# Edge creation
# ---------------------------------------------------------------------------


class TestEdgeCreation:
    def test_minimal_creation(self) -> None:
        edge = Edge(source_node_id="n1", target_node_id="n2")
        assert edge.source_node_id == "n1"
        assert edge.target_node_id == "n2"
        assert edge.label == ""
        assert edge.description == ""

    def test_id_auto_generated(self) -> None:
        e1 = Edge(source_node_id="a", target_node_id="b")
        e2 = Edge(source_node_id="a", target_node_id="b")
        assert e1.id != e2.id

    def test_full_creation(self) -> None:
        edge = Edge(
            id="e-999",
            source_node_id="src",
            target_node_id="tgt",
            label="causes",
            description="Direct causal link",
        )
        assert edge.id == "e-999"
        assert edge.label == "causes"
        assert edge.description == "Direct causal link"

    def test_serialisation_roundtrip(self) -> None:
        edge = Edge(source_node_id="a", target_node_id="b", label="link")
        restored = Edge.model_validate(edge.model_dump())
        assert restored.id == edge.id
        assert restored.source_node_id == "a"
        assert restored.target_node_id == "b"
        assert restored.label == "link"


# ---------------------------------------------------------------------------
# Node with InsuranceNodeMetadata adapter_metadata
# ---------------------------------------------------------------------------


class TestInsuranceNodeMetadata:
    def _make_insurance_meta(self) -> InsuranceNodeMetadata:
        meta: InsuranceNodeMetadata = {
            "claim_type": "flood",
            "proxy_variables": ["rainfall_mm", "distance_to_river"],
            "data_source": "bigquery://claims.flood_events",
            "domain_metadata": {
                "gdacs_score": 3.2,
                "geospatial_resolution": "1km",
            },
        }
        return meta

    def test_node_accepts_insurance_metadata(self) -> None:
        meta = self._make_insurance_meta()
        node = Node(
            label="Flood Claim",
            node_type=NodeType.Outcome,
            measurability_state=MeasurabilityState.Proxied,
            adapter_metadata=meta,
        )
        assert node.adapter_metadata is not None
        assert node.adapter_metadata["claim_type"] == "flood"  # type: ignore[typeddict-item]

    def test_node_metadata_serialises_correctly(self) -> None:
        meta = self._make_insurance_meta()
        node = Node(
            label="Flood Claim",
            node_type=NodeType.Outcome,
            adapter_metadata=meta,
        )
        data = node.model_dump()
        assert data["adapter_metadata"]["claim_type"] == "flood"
        assert data["adapter_metadata"]["proxy_variables"] == [
            "rainfall_mm",
            "distance_to_river",
        ]
        assert data["adapter_metadata"]["domain_metadata"]["gdacs_score"] == 3.2

    def test_node_metadata_roundtrip(self) -> None:
        meta = self._make_insurance_meta()
        node = Node(
            label="Flood Claim",
            node_type=NodeType.Outcome,
            adapter_metadata=meta,
        )
        restored = Node.model_validate(node.model_dump())
        # After roundtrip the metadata comes back as a plain dict (TypedDict is
        # structurally compatible with dict at runtime).
        assert restored.adapter_metadata is not None
        assert restored.adapter_metadata["data_source"] == "bigquery://claims.flood_events"  # type: ignore[typeddict-item]

    def test_base_metadata_is_subset_compatible(self) -> None:
        """A plain BaseNodeMetadata (empty) is also a valid adapter_metadata."""
        meta: BaseNodeMetadata = {}
        node = Node(label="Hypothetical", node_type=NodeType.Exposure, adapter_metadata=meta)
        assert node.adapter_metadata == {}

    def test_partial_insurance_metadata(self) -> None:
        """InsuranceNodeMetadata is total=False so partial dicts are fine."""
        meta: InsuranceNodeMetadata = {"claim_type": "fire"}
        node = Node(
            label="Fire Claim",
            node_type=NodeType.Outcome,
            adapter_metadata=meta,
        )
        assert node.adapter_metadata["claim_type"] == "fire"  # type: ignore[typeddict-item]
        assert "proxy_variables" not in node.adapter_metadata
