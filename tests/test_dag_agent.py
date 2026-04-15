"""Tests for DAGAgent and DraftState.

These tests never call the Anthropic API — they exercise DraftState,
history trimming, checkpoint integration, and the DB wiring.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from causal_hypothesis_engine.agents.dag_agent import (
    DAGAgent,
    DraftState,
    _trim_history,
    _checkpoint_path,
)
from causal_hypothesis_engine.models.dag_version import DAGVersion, DAGVersionStatus
from causal_hypothesis_engine.models.edge import Edge
from causal_hypothesis_engine.models.network import AdapterType, HypothesisNetwork
from causal_hypothesis_engine.models.node import MeasurabilityState, Node, NodeType
from causal_hypothesis_engine.models.session import (
    Session,
    SessionMode,
    SessionStatus,
)
from causal_hypothesis_engine.persistence.checkpoint import load_checkpoint
from causal_hypothesis_engine.persistence.db import Database


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db(tmp_path: Path) -> Database:
    return Database(tmp_path / "test.db")


@pytest.fixture
def sample_network() -> HypothesisNetwork:
    return HypothesisNetwork(name="TestNet", domain="climate", adapter=AdapterType.None_)


# ---------------------------------------------------------------------------
# DraftState
# ---------------------------------------------------------------------------


class TestDraftState:
    def test_add_and_get_node(self) -> None:
        draft = DraftState()
        node = Node(label="Rainfall", node_type=NodeType.Exposure)
        draft.add_node(node)
        assert draft.get_node(node.id) is node

    def test_find_node_by_label(self) -> None:
        draft = DraftState()
        node = Node(label="Flood", node_type=NodeType.Outcome)
        draft.add_node(node)
        assert draft.find_node_by_label("flood") is node
        assert draft.find_node_by_label("FLOOD") is node
        assert draft.find_node_by_label("unknown") is None

    def test_remove_node_also_removes_edges(self) -> None:
        draft = DraftState()
        src = Node(label="A", node_type=NodeType.Exposure)
        tgt = Node(label="B", node_type=NodeType.Outcome)
        edge = Edge(source_node_id=src.id, target_node_id=tgt.id, label="causes")
        draft.add_node(src)
        draft.add_node(tgt)
        draft.add_edge(edge)
        assert len(draft.edges) == 1
        draft.remove_node(src.id)
        assert len(draft.nodes) == 1
        assert len(draft.edges) == 0

    def test_remove_nonexistent_node_returns_false(self) -> None:
        draft = DraftState()
        assert draft.remove_node("nonexistent") is False

    def test_add_edge(self) -> None:
        draft = DraftState()
        src = Node(label="Rain", node_type=NodeType.Exposure)
        tgt = Node(label="Flood", node_type=NodeType.Outcome)
        draft.add_node(src)
        draft.add_node(tgt)
        edge = Edge(source_node_id=src.id, target_node_id=tgt.id, label="causes")
        draft.add_edge(edge)
        assert len(draft.edges) == 1

    def test_summary_empty(self) -> None:
        draft = DraftState()
        assert draft.summary() == "No nodes yet."

    def test_summary_with_nodes_and_edges(self) -> None:
        draft = DraftState()
        src = Node(label="Rain", node_type=NodeType.Exposure)
        tgt = Node(label="Flood", node_type=NodeType.Outcome)
        draft.add_node(src)
        draft.add_node(tgt)
        edge = Edge(source_node_id=src.id, target_node_id=tgt.id, label="causes")
        draft.add_edge(edge)
        summary = draft.summary()
        assert "Rain" in summary
        assert "Flood" in summary
        assert "causes" in summary
        assert "→" in summary

    def test_to_dag_version(self) -> None:
        draft = DraftState()
        draft.add_node(Node(label="X", node_type=NodeType.Exposure))
        version = draft.to_dag_version("net-1")
        assert version.network_id == "net-1"
        assert len(version.nodes) == 1
        assert version.status == DAGVersionStatus.Draft

    def test_from_dag_version(self) -> None:
        nodes = [Node(label="X", node_type=NodeType.Exposure)]
        version = DAGVersion(network_id="net-1", nodes=nodes)
        draft = DraftState.from_dag_version(version)
        assert len(draft.nodes) == 1
        assert draft.nodes[0].label == "X"

    def test_mutation_does_not_affect_original_version(self) -> None:
        """to_dag_version must deepcopy nodes so later mutations don't corrupt."""
        draft = DraftState()
        node = Node(label="X", node_type=NodeType.Exposure)
        draft.add_node(node)
        version = draft.to_dag_version("net-1")
        # Mutate the draft.
        draft.add_node(Node(label="Y", node_type=NodeType.Outcome))
        assert len(version.nodes) == 1  # snapshot is unchanged


# ---------------------------------------------------------------------------
# Rolling history trim
# ---------------------------------------------------------------------------


class TestTrimHistory:
    def _make_history(self, n: int) -> list[dict[str, str]]:
        history = []
        for i in range(n):
            history.append({"role": "user", "content": f"msg {i}"})
            history.append({"role": "assistant", "content": f"reply {i}"})
        return history

    def test_short_history_unchanged(self) -> None:
        history = self._make_history(5)
        draft = DraftState()
        result = _trim_history(history, draft, window=20)
        assert result == history

    def test_long_history_trimmed(self) -> None:
        history = self._make_history(15)  # 30 messages
        draft = DraftState()
        result = _trim_history(history, draft, window=20)
        # Should be summary + last 20 messages
        assert len(result) == 21
        assert result[0]["role"] == "user"
        assert "CONTEXT SUMMARY" in result[0]["content"]

    def test_trim_injects_dag_summary(self) -> None:
        history = self._make_history(15)
        draft = DraftState()
        draft.add_node(Node(label="Rain", node_type=NodeType.Exposure))
        result = _trim_history(history, draft, window=20)
        assert "Rain" in result[0]["content"]

    def test_exact_window_boundary(self) -> None:
        history = self._make_history(10)  # exactly 20 messages
        draft = DraftState()
        result = _trim_history(history, draft, window=20)
        assert result == history


# ---------------------------------------------------------------------------
# Checkpoint path helper
# ---------------------------------------------------------------------------


class TestCheckpointPath:
    def test_generates_path_under_dir(self, tmp_path: Path) -> None:
        path = _checkpoint_path("sess-abc", tmp_path)
        assert path.parent == tmp_path / ".checkpoints"
        assert "sess-abc" in path.name

    def test_uses_default_dir(self) -> None:
        path = _checkpoint_path("sess-xyz")
        assert "sess-xyz" in path.name


# ---------------------------------------------------------------------------
# DAGAgent construction and DB wiring
# ---------------------------------------------------------------------------


class TestDAGAgentInit:
    def test_creates_session_in_db(
        self, tmp_db: Database, sample_network: HypothesisNetwork, tmp_path: Path
    ) -> None:
        tmp_db.create_network(sample_network)
        agent = DAGAgent(
            db=tmp_db,
            network=sample_network,
            checkpoint_dir=tmp_path,
        )
        session = tmp_db.get_session(agent.session.id)
        assert session is not None
        assert session.network_id == sample_network.id
        assert session.mode == SessionMode.DAGCreation

    def test_registers_session_in_network(
        self, tmp_db: Database, sample_network: HypothesisNetwork, tmp_path: Path
    ) -> None:
        tmp_db.create_network(sample_network)
        agent = DAGAgent(
            db=tmp_db,
            network=sample_network,
            checkpoint_dir=tmp_path,
        )
        network = tmp_db.get_network(sample_network.id)
        assert agent.session.id in network.session_ids

    def test_checkpoint_path_set(
        self, tmp_db: Database, sample_network: HypothesisNetwork, tmp_path: Path
    ) -> None:
        tmp_db.create_network(sample_network)
        agent = DAGAgent(
            db=tmp_db,
            network=sample_network,
            checkpoint_dir=tmp_path,
        )
        assert agent.session.checkpoint_path != ""
        assert agent.session.id in agent.session.checkpoint_path

    def test_resumes_existing_session(
        self, tmp_db: Database, sample_network: HypothesisNetwork, tmp_path: Path
    ) -> None:
        tmp_db.create_network(sample_network)
        # First: create agent and checkpoint some state.
        agent = DAGAgent(
            db=tmp_db,
            network=sample_network,
            checkpoint_dir=tmp_path,
        )
        agent.add_node("Rain", NodeType.Exposure, "Rainfall")
        agent._do_checkpoint()

        # Reload: pass the existing session back.
        session = tmp_db.get_session(agent.session.id)
        _, draft_version = load_checkpoint(session.checkpoint_path)
        draft = DraftState.from_dag_version(draft_version)

        agent2 = DAGAgent(
            db=tmp_db,
            network=sample_network,
            session=session,
            draft=draft,
            checkpoint_dir=tmp_path,
        )
        assert len(agent2.draft.nodes) == 1
        assert agent2.draft.nodes[0].label == "Rain"


# ---------------------------------------------------------------------------
# DAGAgent: add_node / add_edge helpers
# ---------------------------------------------------------------------------


class TestDAGAgentMutations:
    @pytest.fixture
    def agent(
        self, tmp_db: Database, sample_network: HypothesisNetwork, tmp_path: Path
    ) -> DAGAgent:
        tmp_db.create_network(sample_network)
        return DAGAgent(db=tmp_db, network=sample_network, checkpoint_dir=tmp_path)

    def test_add_node(self, agent: DAGAgent) -> None:
        node = agent.add_node("Drought", NodeType.Exposure)
        assert node.label == "Drought"
        assert len(agent.draft.nodes) == 1

    def test_add_edge_by_label(self, agent: DAGAgent) -> None:
        agent.add_node("Rain", NodeType.Exposure)
        agent.add_node("Flood", NodeType.Outcome)
        edge = agent.add_edge("Rain", "Flood", label="causes")
        assert edge is not None
        assert len(agent.draft.edges) == 1

    def test_add_edge_missing_node_returns_none(self, agent: DAGAgent) -> None:
        agent.add_node("Rain", NodeType.Exposure)
        edge = agent.add_edge("Rain", "Nonexistent", label="causes")
        assert edge is None
        assert len(agent.draft.edges) == 0


# ---------------------------------------------------------------------------
# DAGAgent: _do_checkpoint + DB update interaction
# ---------------------------------------------------------------------------


class TestDAGAgentCheckpoint:
    def test_checkpoint_file_written(
        self, tmp_db: Database, sample_network: HypothesisNetwork, tmp_path: Path
    ) -> None:
        tmp_db.create_network(sample_network)
        agent = DAGAgent(db=tmp_db, network=sample_network, checkpoint_dir=tmp_path)
        agent.add_node("Rain", NodeType.Exposure)
        agent._do_checkpoint()
        assert Path(agent.session.checkpoint_path).exists()

    def test_checkpoint_roundtrip_preserves_nodes(
        self, tmp_db: Database, sample_network: HypothesisNetwork, tmp_path: Path
    ) -> None:
        tmp_db.create_network(sample_network)
        agent = DAGAgent(db=tmp_db, network=sample_network, checkpoint_dir=tmp_path)
        agent.add_node("Rain", NodeType.Exposure)
        agent.add_node("Flood", NodeType.Outcome)
        agent._do_checkpoint()

        _, loaded_version = load_checkpoint(agent.session.checkpoint_path)
        assert len(loaded_version.nodes) == 2
        labels = {n.label for n in loaded_version.nodes}
        assert labels == {"Rain", "Flood"}
