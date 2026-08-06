"""Tests for the SQLite persistence layer and atomic checkpoint writer."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from causal_hypothesis_engine.models.backtest_result import BacktestResult
from causal_hypothesis_engine.models.dag_version import DAGVersion, DAGVersionStatus
from causal_hypothesis_engine.models.edge import Edge, EdgeDirection
from causal_hypothesis_engine.models.network import AdapterType, HypothesisNetwork
from causal_hypothesis_engine.models.node import MeasurabilityState, Node, NodeType
from causal_hypothesis_engine.models.session import (
    ModificationMode,
    Session,
    SessionMode,
    SessionStatus,
)
from causal_hypothesis_engine.persistence.checkpoint import load_checkpoint, write_checkpoint
from causal_hypothesis_engine.persistence.db import Database


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db(tmp_path: Path) -> Database:
    return Database(tmp_path / "test.db")


@pytest.fixture
def sample_network() -> HypothesisNetwork:
    return HypothesisNetwork(name="TestNet", domain="test", adapter=AdapterType.None_)


@pytest.fixture
def sample_node() -> Node:
    return Node(
        label="Rainfall",
        description="Amount of rainfall",
        node_type=NodeType.Exposure,
        measurability_state=MeasurabilityState.Hypothetical,
    )


@pytest.fixture
def sample_edge(sample_node: Node) -> Edge:
    target = Node(
        label="Flood",
        description="Flood occurrence",
        node_type=NodeType.Outcome,
        measurability_state=MeasurabilityState.Hypothetical,
    )
    return Edge(
        source_node_id=sample_node.id,
        target_node_id=target.id,
        label="causes",
        description="Rainfall causes flooding",
    )


@pytest.fixture
def sample_version(sample_network: HypothesisNetwork, sample_node: Node, sample_edge: Edge) -> DAGVersion:
    return DAGVersion(
        network_id=sample_network.id,
        nodes=[sample_node],
        edges=[sample_edge],
    )


@pytest.fixture
def sample_session(sample_network: HypothesisNetwork) -> Session:
    return Session(
        network_id=sample_network.id,
        mode=SessionMode.DAGCreation,
    )


# ---------------------------------------------------------------------------
# Database: schema
# ---------------------------------------------------------------------------


class TestDatabaseSchema:
    def test_creates_db_file(self, tmp_path: Path) -> None:
        db_path = tmp_path / "sub" / "test.db"
        Database(db_path)
        assert db_path.exists()

    def test_schema_version_matches_current(self, tmp_db: Database) -> None:
        from causal_hypothesis_engine.persistence.db import CURRENT_SCHEMA_VERSION

        with tmp_db._connect() as conn:
            row = conn.execute("SELECT version FROM schema_version").fetchone()
        assert row["version"] == CURRENT_SCHEMA_VERSION

    def test_idempotent_reinit(self, tmp_path: Path) -> None:
        """Opening the same DB twice must not raise or reset schema_version."""
        from causal_hypothesis_engine.persistence.db import CURRENT_SCHEMA_VERSION

        db_path = tmp_path / "test.db"
        Database(db_path)
        db2 = Database(db_path)
        with db2._connect() as conn:
            row = conn.execute("SELECT version FROM schema_version").fetchone()
        assert row["version"] == CURRENT_SCHEMA_VERSION

    def test_wal_mode_enabled(self, tmp_db: Database) -> None:
        with tmp_db._connect() as conn:
            row = conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0] == "wal"


# ---------------------------------------------------------------------------
# Database: Networks
# ---------------------------------------------------------------------------


class TestNetworks:
    def test_create_and_get(self, tmp_db: Database, sample_network: HypothesisNetwork) -> None:
        tmp_db.create_network(sample_network)
        fetched = tmp_db.get_network(sample_network.id)
        assert fetched is not None
        assert fetched.id == sample_network.id
        assert fetched.name == "TestNet"
        assert fetched.adapter == AdapterType.None_

    def test_get_missing_returns_none(self, tmp_db: Database) -> None:
        assert tmp_db.get_network("nonexistent-id") is None

    def test_list_networks_empty(self, tmp_db: Database) -> None:
        assert tmp_db.list_networks() == []

    def test_list_networks_summary(self, tmp_db: Database, sample_network: HypothesisNetwork) -> None:
        tmp_db.create_network(sample_network)
        summaries = tmp_db.list_networks()
        assert len(summaries) == 1
        s = summaries[0]
        assert s["id"] == sample_network.id
        assert s["version_count"] == 0
        assert s["session_count"] == 0

    def test_update_network_ids(self, tmp_db: Database, sample_network: HypothesisNetwork) -> None:
        tmp_db.create_network(sample_network)
        tmp_db.update_network_ids(sample_network.id, ["v1", "v2"], ["s1"])
        fetched = tmp_db.get_network(sample_network.id)
        assert fetched.version_ids == ["v1", "v2"]
        assert fetched.session_ids == ["s1"]

    def test_delete_network(self, tmp_db: Database, sample_network: HypothesisNetwork) -> None:
        tmp_db.create_network(sample_network)
        tmp_db.delete_network(sample_network.id)
        assert tmp_db.get_network(sample_network.id) is None

    def test_list_networks_count_reflects_ids(
        self, tmp_db: Database, sample_network: HypothesisNetwork
    ) -> None:
        tmp_db.create_network(sample_network)
        tmp_db.update_network_ids(sample_network.id, ["v1", "v2", "v3"], ["s1", "s2"])
        summaries = tmp_db.list_networks()
        assert summaries[0]["version_count"] == 3
        assert summaries[0]["session_count"] == 2


# ---------------------------------------------------------------------------
# Database: DAGVersions
# ---------------------------------------------------------------------------


class TestDAGVersions:
    def test_save_and_get(
        self,
        tmp_db: Database,
        sample_network: HypothesisNetwork,
        sample_version: DAGVersion,
    ) -> None:
        tmp_db.create_network(sample_network)
        tmp_db.save_version(sample_version)
        fetched = tmp_db.get_version(sample_version.version_id)
        assert fetched is not None
        assert fetched.version_id == sample_version.version_id
        assert len(fetched.nodes) == 1
        assert len(fetched.edges) == 1

    def test_get_missing_returns_none(self, tmp_db: Database) -> None:
        assert tmp_db.get_version("nonexistent-id") is None

    def test_nodes_roundtrip(
        self,
        tmp_db: Database,
        sample_network: HypothesisNetwork,
        sample_version: DAGVersion,
        sample_node: Node,
    ) -> None:
        tmp_db.create_network(sample_network)
        tmp_db.save_version(sample_version)
        fetched = tmp_db.get_version(sample_version.version_id)
        assert fetched.nodes[0].label == sample_node.label
        assert fetched.nodes[0].node_type == NodeType.Exposure

    def test_get_versions_for_network(
        self,
        tmp_db: Database,
        sample_network: HypothesisNetwork,
    ) -> None:
        tmp_db.create_network(sample_network)
        v1 = DAGVersion(network_id=sample_network.id)
        v2 = DAGVersion(network_id=sample_network.id, parent_version_id=v1.version_id)
        tmp_db.save_version(v1)
        tmp_db.save_version(v2)
        versions = tmp_db.get_versions_for_network(sample_network.id)
        assert len(versions) == 2
        ids = {v.version_id for v in versions}
        assert v1.version_id in ids
        assert v2.version_id in ids

    def test_update_version_status(
        self,
        tmp_db: Database,
        sample_network: HypothesisNetwork,
        sample_version: DAGVersion,
    ) -> None:
        tmp_db.create_network(sample_network)
        tmp_db.save_version(sample_version)
        br = BacktestResult(
            version_id=sample_version.version_id,
            adapter_type=AdapterType.Insurance,
            baseline_score=0.5,
            dag_score=0.65,
            lift=0.15,
        )
        tmp_db.update_version_status(sample_version.version_id, DAGVersionStatus.Tested, br)
        fetched = tmp_db.get_version(sample_version.version_id)
        assert fetched.status == DAGVersionStatus.Tested
        assert fetched.backtest_result is not None
        assert fetched.backtest_result.lift == pytest.approx(0.15)

    def test_save_version_upsert(
        self,
        tmp_db: Database,
        sample_network: HypothesisNetwork,
        sample_version: DAGVersion,
    ) -> None:
        tmp_db.create_network(sample_network)
        tmp_db.save_version(sample_version)
        # Modify and re-save (INSERT OR REPLACE).
        updated = sample_version.model_copy(update={"modification_rationale": "refined"})
        tmp_db.save_version(updated)
        fetched = tmp_db.get_version(sample_version.version_id)
        assert fetched.modification_rationale == "refined"


# ---------------------------------------------------------------------------
# Database: Sessions
# ---------------------------------------------------------------------------


class TestSessions:
    def test_save_and_get(
        self,
        tmp_db: Database,
        sample_network: HypothesisNetwork,
        sample_session: Session,
    ) -> None:
        tmp_db.create_network(sample_network)
        tmp_db.save_session(sample_session)
        fetched = tmp_db.get_session(sample_session.id)
        assert fetched is not None
        assert fetched.id == sample_session.id
        assert fetched.mode == SessionMode.DAGCreation

    def test_get_missing_returns_none(self, tmp_db: Database) -> None:
        assert tmp_db.get_session("nonexistent-id") is None

    def test_update_session(
        self,
        tmp_db: Database,
        sample_network: HypothesisNetwork,
        sample_session: Session,
    ) -> None:
        tmp_db.create_network(sample_network)
        tmp_db.save_session(sample_session)
        updated = sample_session.model_copy(
            update={
                "status": SessionStatus.Confirmed,
                "exchange_count": 5,
                "conversation_history": [{"role": "user", "content": "hello"}],
            }
        )
        tmp_db.update_session(updated)
        fetched = tmp_db.get_session(sample_session.id)
        assert fetched.status == SessionStatus.Confirmed
        assert fetched.exchange_count == 5
        assert fetched.conversation_history[0]["content"] == "hello"

    def test_modification_mode_roundtrip(
        self,
        tmp_db: Database,
        sample_network: HypothesisNetwork,
    ) -> None:
        tmp_db.create_network(sample_network)
        session = Session(
            network_id=sample_network.id,
            mode=SessionMode.Modification,
            modification_mode=ModificationMode.Hybrid,
        )
        tmp_db.save_session(session)
        fetched = tmp_db.get_session(session.id)
        assert fetched.modification_mode == ModificationMode.Hybrid

    def test_list_sessions_for_network(
        self,
        tmp_db: Database,
        sample_network: HypothesisNetwork,
    ) -> None:
        tmp_db.create_network(sample_network)
        s1 = Session(network_id=sample_network.id, mode=SessionMode.DAGCreation)
        s2 = Session(network_id=sample_network.id, mode=SessionMode.Modification)
        tmp_db.save_session(s1)
        tmp_db.save_session(s2)
        sessions = tmp_db.list_sessions_for_network(sample_network.id)
        assert len(sessions) == 2


# ---------------------------------------------------------------------------
# Checkpoint: write + load
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_write_and_load(
        self, tmp_path: Path, sample_session: Session, sample_version: DAGVersion
    ) -> None:
        cp = tmp_path / "session.checkpoint"
        write_checkpoint(sample_session, sample_version, cp)
        assert cp.exists()
        loaded_session, loaded_version = load_checkpoint(cp)
        assert loaded_session.id == sample_session.id
        assert loaded_version.version_id == sample_version.version_id

    def test_no_temp_file_left_after_write(
        self, tmp_path: Path, sample_session: Session, sample_version: DAGVersion
    ) -> None:
        cp = tmp_path / "session.checkpoint"
        write_checkpoint(sample_session, sample_version, cp)
        assert not (tmp_path / "session.checkpoint.tmp").exists()

    def test_creates_parent_directory(
        self, tmp_path: Path, sample_session: Session, sample_version: DAGVersion
    ) -> None:
        cp = tmp_path / "deep" / "nested" / "session.checkpoint"
        write_checkpoint(sample_session, sample_version, cp)
        assert cp.exists()

    def test_overwrite_is_atomic(
        self, tmp_path: Path, sample_session: Session, sample_version: DAGVersion
    ) -> None:
        cp = tmp_path / "session.checkpoint"
        write_checkpoint(sample_session, sample_version, cp)
        # Write again with updated exchange_count.
        updated = sample_session.model_copy(update={"exchange_count": 10})
        write_checkpoint(updated, sample_version, cp)
        loaded_session, _ = load_checkpoint(cp)
        assert loaded_session.exchange_count == 10

    def test_load_missing_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "nonexistent.checkpoint")

    def test_load_corrupt_raises_value_error(self, tmp_path: Path) -> None:
        cp = tmp_path / "corrupt.checkpoint"
        cp.write_text("this is not valid json", encoding="utf-8")
        with pytest.raises(ValueError):
            load_checkpoint(cp)

    def test_conversation_history_roundtrip(
        self, tmp_path: Path, sample_version: DAGVersion
    ) -> None:
        session = sample_session = Session(
            network_id=sample_version.network_id,
            mode=SessionMode.DAGCreation,
            conversation_history=[
                {"role": "user", "content": "add a rainfall node"},
                {"role": "assistant", "content": "Added Rainfall node."},
            ],
        )
        cp = tmp_path / "session.checkpoint"
        write_checkpoint(session, sample_version, cp)
        loaded_session, _ = load_checkpoint(cp)
        assert len(loaded_session.conversation_history) == 2
        assert loaded_session.conversation_history[1]["content"] == "Added Rainfall node."
