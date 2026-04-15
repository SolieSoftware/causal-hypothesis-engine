"""Tests for ModificationProposal schema and ModificationAgent core logic.

No Anthropic API calls are made in these tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from causal_hypothesis_engine.agents.modification_agent import (
    ModificationAgent,
    MutableDAG,
    _extract_proposal_json,
    _trim_history,
)
from causal_hypothesis_engine.models.dag_version import DAGVersion, DAGVersionStatus
from causal_hypothesis_engine.models.modification_proposal import (
    AddEdgeChange,
    AddNodeChange,
    ModificationProposal,
    RemoveEdgeChange,
    RemoveNodeChange,
    UpdateEdgeChange,
    UpdateNodeChange,
)
from causal_hypothesis_engine.models.network import AdapterType, HypothesisNetwork
from causal_hypothesis_engine.models.node import MeasurabilityState, Node, NodeType
from causal_hypothesis_engine.models.edge import Edge
from causal_hypothesis_engine.models.session import ModificationMode, SessionMode
from causal_hypothesis_engine.persistence.db import Database


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db(tmp_path: Path) -> Database:
    return Database(tmp_path / "test.db")


@pytest.fixture
def sample_network() -> HypothesisNetwork:
    return HypothesisNetwork(name="Net", domain="test", adapter=AdapterType.None_)


@pytest.fixture
def rain_node() -> Node:
    return Node(label="Rain", node_type=NodeType.Exposure,
                measurability_state=MeasurabilityState.Hypothetical)


@pytest.fixture
def flood_node() -> Node:
    return Node(label="Flood", node_type=NodeType.Outcome,
                measurability_state=MeasurabilityState.Hypothetical)


@pytest.fixture
def rain_flood_edge(rain_node: Node, flood_node: Node) -> Edge:
    return Edge(source_node_id=rain_node.id, target_node_id=flood_node.id, label="causes")


@pytest.fixture
def base_version(sample_network: HypothesisNetwork, rain_node: Node,
                 flood_node: Node, rain_flood_edge: Edge) -> DAGVersion:
    return DAGVersion(
        network_id=sample_network.id,
        nodes=[rain_node, flood_node],
        edges=[rain_flood_edge],
    )


# ---------------------------------------------------------------------------
# ModificationProposal schema
# ---------------------------------------------------------------------------


class TestModificationProposalSchema:
    def test_add_node_roundtrip(self) -> None:
        proposal = ModificationProposal(
            source="theory",
            confidence=0.8,
            rationale="Drought is a mediating variable.",
            proposed_change=AddNodeChange(
                label="Drought",
                node_type="Confounder",
                measurability_state="Hypothetical",
                description="Lack of rainfall",
            ),
        )
        data = proposal.model_dump()
        restored = ModificationProposal.model_validate(data)
        assert restored.proposed_change.change_type == "add_node"
        assert restored.proposed_change.label == "Drought"  # type: ignore[union-attr]

    def test_remove_node_roundtrip(self) -> None:
        proposal = ModificationProposal(
            source="theory",
            confidence=0.6,
            rationale="Node is redundant.",
            proposed_change=RemoveNodeChange(node_id="abc-123"),
        )
        data = proposal.model_dump()
        restored = ModificationProposal.model_validate(data)
        assert restored.proposed_change.change_type == "remove_node"

    def test_add_edge_roundtrip(self) -> None:
        proposal = ModificationProposal(
            source="theory",
            confidence=0.9,
            rationale="Direct causal path.",
            proposed_change=AddEdgeChange(
                source_node_id="n1",
                target_node_id="n2",
                label="causes",
            ),
        )
        data = proposal.model_dump()
        restored = ModificationProposal.model_validate(data)
        assert restored.proposed_change.change_type == "add_edge"

    def test_confidence_bounds(self) -> None:
        with pytest.raises(Exception):
            ModificationProposal(
                source="theory",
                confidence=1.5,  # out of range
                rationale="x",
                proposed_change=RemoveNodeChange(node_id="x"),
            )
        with pytest.raises(Exception):
            ModificationProposal(
                source="theory",
                confidence=-0.1,
                rationale="x",
                proposed_change=RemoveNodeChange(node_id="x"),
            )

    def test_theory_data_agreement_optional(self) -> None:
        p = ModificationProposal(
            source="both",
            confidence=0.85,
            rationale="both agree",
            proposed_change=AddNodeChange(label="X", node_type="Exposure"),
            theory_data_agreement=True,
        )
        assert p.theory_data_agreement is True

    def test_json_serialisation_discriminated_union(self) -> None:
        """Pydantic discriminated union must survive a JSON roundtrip."""
        for change in [
            AddNodeChange(label="X", node_type="Exposure"),
            RemoveNodeChange(node_id="id-1"),
            UpdateNodeChange(node_id="id-1", label="Y"),
            AddEdgeChange(source_node_id="a", target_node_id="b"),
            RemoveEdgeChange(edge_id="e-1"),
            UpdateEdgeChange(edge_id="e-1", label="new"),
        ]:
            p = ModificationProposal(
                source="theory", confidence=0.5, rationale="r", proposed_change=change
            )
            restored = ModificationProposal.model_validate_json(p.model_dump_json())
            assert restored.proposed_change.change_type == change.change_type


# ---------------------------------------------------------------------------
# MutableDAG — apply changes
# ---------------------------------------------------------------------------


class TestMutableDAGApply:
    def test_add_node(self, base_version: DAGVersion) -> None:
        dag = MutableDAG(base_version)
        proposal = ModificationProposal(
            source="theory", confidence=0.8, rationale="r",
            proposed_change=AddNodeChange(label="Drought", node_type="Confounder"),
        )
        ok, msg = dag.apply(proposal)
        assert ok
        assert len(dag.nodes) == 3
        assert any(n.label == "Drought" for n in dag.nodes)

    def test_remove_node_cleans_edges(
        self, base_version: DAGVersion, rain_node: Node, rain_flood_edge: Edge
    ) -> None:
        dag = MutableDAG(base_version)
        assert len(dag.edges) == 1
        proposal = ModificationProposal(
            source="theory", confidence=0.7, rationale="r",
            proposed_change=RemoveNodeChange(node_id=rain_node.id),
        )
        ok, _ = dag.apply(proposal)
        assert ok
        assert len(dag.nodes) == 1
        assert len(dag.edges) == 0

    def test_remove_node_not_found(self, base_version: DAGVersion) -> None:
        dag = MutableDAG(base_version)
        proposal = ModificationProposal(
            source="theory", confidence=0.5, rationale="r",
            proposed_change=RemoveNodeChange(node_id="nonexistent"),
        )
        ok, msg = dag.apply(proposal)
        assert not ok
        assert "not found" in msg

    def test_update_node(
        self, base_version: DAGVersion, rain_node: Node
    ) -> None:
        dag = MutableDAG(base_version)
        proposal = ModificationProposal(
            source="theory", confidence=0.9, rationale="r",
            proposed_change=UpdateNodeChange(
                node_id=rain_node.id,
                label="Heavy Rain",
                measurability_state="Proxied",
            ),
        )
        ok, _ = dag.apply(proposal)
        assert ok
        updated = dag.get_node(rain_node.id)
        assert updated.label == "Heavy Rain"
        assert updated.measurability_state == MeasurabilityState.Proxied

    def test_add_edge(
        self, base_version: DAGVersion, rain_node: Node, flood_node: Node
    ) -> None:
        dag = MutableDAG(base_version)
        # Add a third node first.
        dag.nodes.append(Node(label="Dam Failure", node_type=NodeType.Mediator))
        dam = dag.nodes[-1]
        proposal = ModificationProposal(
            source="theory", confidence=0.75, rationale="r",
            proposed_change=AddEdgeChange(
                source_node_id=flood_node.id,
                target_node_id=dam.id,
                label="triggers",
            ),
        )
        ok, msg = dag.apply(proposal)
        assert ok
        assert len(dag.edges) == 2

    def test_add_edge_missing_source(self, base_version: DAGVersion, flood_node: Node) -> None:
        dag = MutableDAG(base_version)
        proposal = ModificationProposal(
            source="theory", confidence=0.5, rationale="r",
            proposed_change=AddEdgeChange(
                source_node_id="nonexistent",
                target_node_id=flood_node.id,
            ),
        )
        ok, msg = dag.apply(proposal)
        assert not ok
        assert "not found" in msg

    def test_remove_edge(
        self, base_version: DAGVersion, rain_flood_edge: Edge
    ) -> None:
        dag = MutableDAG(base_version)
        proposal = ModificationProposal(
            source="theory", confidence=0.6, rationale="r",
            proposed_change=RemoveEdgeChange(edge_id=rain_flood_edge.id),
        )
        ok, _ = dag.apply(proposal)
        assert ok
        assert len(dag.edges) == 0

    def test_remove_edge_not_found(self, base_version: DAGVersion) -> None:
        dag = MutableDAG(base_version)
        proposal = ModificationProposal(
            source="theory", confidence=0.5, rationale="r",
            proposed_change=RemoveEdgeChange(edge_id="nonexistent"),
        )
        ok, msg = dag.apply(proposal)
        assert not ok

    def test_update_edge(
        self, base_version: DAGVersion, rain_flood_edge: Edge
    ) -> None:
        dag = MutableDAG(base_version)
        proposal = ModificationProposal(
            source="theory", confidence=0.7, rationale="r",
            proposed_change=UpdateEdgeChange(
                edge_id=rain_flood_edge.id, label="directly causes"
            ),
        )
        ok, _ = dag.apply(proposal)
        assert ok
        edge = dag.get_edge(rain_flood_edge.id)
        assert edge.label == "directly causes"

    def test_source_version_not_mutated(self, base_version: DAGVersion) -> None:
        """MutableDAG must deepcopy — changes must not bleed into the source."""
        dag = MutableDAG(base_version)
        proposal = ModificationProposal(
            source="theory", confidence=0.8, rationale="r",
            proposed_change=AddNodeChange(label="Extra", node_type="Collider"),
        )
        dag.apply(proposal)
        assert len(base_version.nodes) == 2  # unchanged

    def test_to_dag_version_parent_linkage(self, base_version: DAGVersion) -> None:
        dag = MutableDAG(base_version)
        new_version = dag.to_dag_version("net-1", base_version.version_id)
        assert new_version.parent_version_id == base_version.version_id
        assert new_version.status == DAGVersionStatus.Draft


# ---------------------------------------------------------------------------
# _extract_proposal_json
# ---------------------------------------------------------------------------


class TestExtractProposalJson:
    def _make_reply(self, proposal: ModificationProposal) -> str:
        data = proposal.model_dump()
        return (
            "Here is my proposal:\n\n"
            f"```json\n{json.dumps(data, indent=2)}\n```\n\n"
            "This change adds a new node."
        )

    def test_valid_proposal_extracted(self) -> None:
        proposal = ModificationProposal(
            source="theory", confidence=0.8, rationale="r",
            proposed_change=AddNodeChange(label="X", node_type="Exposure"),
        )
        reply = self._make_reply(proposal)
        extracted = _extract_proposal_json(reply)
        assert extracted is not None
        assert extracted.proposed_change.change_type == "add_node"

    def test_no_json_block_returns_none(self) -> None:
        assert _extract_proposal_json("Just a plain text reply with no code block.") is None

    def test_invalid_json_returns_none(self) -> None:
        reply = "```json\n{not valid json}\n```"
        assert _extract_proposal_json(reply) is None

    def test_valid_json_wrong_schema_returns_none(self) -> None:
        reply = '```json\n{"foo": "bar"}\n```'
        assert _extract_proposal_json(reply) is None


# ---------------------------------------------------------------------------
# ModificationAgent construction
# ---------------------------------------------------------------------------


class TestModificationAgentInit:
    def test_theory_mode_stays_theory(
        self, tmp_db: Database, sample_network: HypothesisNetwork,
        base_version: DAGVersion, tmp_path: Path
    ) -> None:
        tmp_db.create_network(sample_network)
        tmp_db.save_version(base_version)
        agent = ModificationAgent(
            db=tmp_db, network=sample_network, source_version=base_version,
            modification_mode=ModificationMode.Theory,
            checkpoint_dir=tmp_path,
        )
        assert agent.modification_mode == ModificationMode.Theory
        assert not agent._fell_back

    def test_backtest_mode_falls_back_without_result(
        self, tmp_db: Database, sample_network: HypothesisNetwork,
        base_version: DAGVersion, tmp_path: Path
    ) -> None:
        tmp_db.create_network(sample_network)
        tmp_db.save_version(base_version)
        # base_version has no backtest_result
        agent = ModificationAgent(
            db=tmp_db, network=sample_network, source_version=base_version,
            modification_mode=ModificationMode.Backtest,
            checkpoint_dir=tmp_path,
        )
        assert agent.modification_mode == ModificationMode.Theory
        assert agent._fell_back

    def test_hybrid_mode_falls_back_without_result(
        self, tmp_db: Database, sample_network: HypothesisNetwork,
        base_version: DAGVersion, tmp_path: Path
    ) -> None:
        tmp_db.create_network(sample_network)
        tmp_db.save_version(base_version)
        agent = ModificationAgent(
            db=tmp_db, network=sample_network, source_version=base_version,
            modification_mode=ModificationMode.Hybrid,
            checkpoint_dir=tmp_path,
        )
        assert agent.modification_mode == ModificationMode.Theory
        assert agent._fell_back

    def test_creates_session_in_db(
        self, tmp_db: Database, sample_network: HypothesisNetwork,
        base_version: DAGVersion, tmp_path: Path
    ) -> None:
        tmp_db.create_network(sample_network)
        tmp_db.save_version(base_version)
        agent = ModificationAgent(
            db=tmp_db, network=sample_network, source_version=base_version,
            checkpoint_dir=tmp_path,
        )
        session = tmp_db.get_session(agent.session.id)
        assert session is not None
        assert session.mode == SessionMode.Modification

    def test_mutable_dag_starts_as_copy(
        self, tmp_db: Database, sample_network: HypothesisNetwork,
        base_version: DAGVersion, tmp_path: Path
    ) -> None:
        tmp_db.create_network(sample_network)
        tmp_db.save_version(base_version)
        agent = ModificationAgent(
            db=tmp_db, network=sample_network, source_version=base_version,
            checkpoint_dir=tmp_path,
        )
        assert len(agent.dag.nodes) == len(base_version.nodes)
        assert len(agent.dag.edges) == len(base_version.edges)


# ---------------------------------------------------------------------------
# _trim_history for ModificationAgent
# ---------------------------------------------------------------------------


class TestModificationAgentTrimHistory:
    def test_short_history_unchanged(self, base_version: DAGVersion) -> None:
        dag = MutableDAG(base_version)
        history = [{"role": "user", "content": f"m{i}"} for i in range(10)]
        result = _trim_history(history, dag, window=20)
        assert result == history

    def test_long_history_trimmed_with_summary(self, base_version: DAGVersion) -> None:
        dag = MutableDAG(base_version)
        history = [{"role": "user", "content": f"m{i}"} for i in range(30)]
        result = _trim_history(history, dag, window=20)
        assert len(result) == 21
        assert "CONTEXT SUMMARY" in result[0]["content"]
        assert "Rain" in result[0]["content"]
