"""Tests for BacktestAgent — automated backtesting pipeline.

No Anthropic API calls.  Uses synthetic CSV data and the InsuranceClaimsAdapter.
"""

from __future__ import annotations

import csv
import io
import random
from pathlib import Path

import pandas as pd
import pytest

from causal_hypothesis_engine.agents.backtest_agent import (
    BacktestAgent,
    BacktestPreflightError,
    check_can_backtest,
    get_adapter,
)
from causal_hypothesis_engine.adapters.insurance import InsuranceClaimsAdapter
from causal_hypothesis_engine.models.backtest_result import BacktestResult
from causal_hypothesis_engine.models.dag_version import DAGVersion, DAGVersionStatus
from causal_hypothesis_engine.models.network import AdapterType, HypothesisNetwork
from causal_hypothesis_engine.models.node import MeasurabilityState, Node, NodeType
from causal_hypothesis_engine.models.edge import Edge
from causal_hypothesis_engine.persistence.db import Database


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_claims_csv(tmp_path: Path, n: int = 150, seed: int = 99) -> Path:
    rng = random.Random(seed)
    claim_types = ["fire", "flood", "burst_pipe", "storm", "other"]
    rows = []
    for i in range(n):
        ct = rng.choice(claim_types)
        rainfall = rng.uniform(0, 200)
        wind = rng.uniform(0, 120)
        amount = rng.uniform(500, 50_000)
        is_large = int(amount > 20_000 or rainfall > 150)
        rows.append({
            "claim_id": f"C{i:04d}",
            "claim_type": ct,
            "claim_amount": amount,
            "is_large_claim": is_large,
            "rainfall_mm": rainfall,
            "max_wind_speed": wind,
            "flood_zone": int(rainfall > 100),
            "fire_risk_score": rng.uniform(0, 1),
        })
    df = pd.DataFrame(rows)
    path = tmp_path / "claims.csv"
    df.to_csv(path, index=False)
    return path


def _make_network(adapter: AdapterType = AdapterType.Insurance) -> HypothesisNetwork:
    return HypothesisNetwork(name="TestNet", domain="insurance", adapter=adapter)


def _make_version_with_proxied(network_id: str) -> DAGVersion:
    rain = Node(
        label="Rainfall",
        node_type=NodeType.Exposure,
        measurability_state=MeasurabilityState.Proxied,
        adapter_metadata={"proxy_variables": ["rainfall_mm", "flood_zone"], "claim_type": "flood"},
    )
    wind = Node(
        label="Wind",
        node_type=NodeType.Exposure,
        measurability_state=MeasurabilityState.Proxied,
        adapter_metadata={"proxy_variables": ["max_wind_speed"], "claim_type": "storm"},
    )
    outcome = Node(label="Large Claim", node_type=NodeType.Outcome,
                   measurability_state=MeasurabilityState.Hypothetical)
    return DAGVersion(network_id=network_id, nodes=[rain, wind, outcome])


def _make_version_no_proxied(network_id: str) -> DAGVersion:
    node = Node(label="X", node_type=NodeType.Exposure,
                measurability_state=MeasurabilityState.Hypothetical)
    return DAGVersion(network_id=network_id, nodes=[node])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db(tmp_path: Path) -> Database:
    return Database(tmp_path / "test.db")


@pytest.fixture
def network() -> HypothesisNetwork:
    return _make_network()


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    return _make_claims_csv(tmp_path)


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------


class TestGetAdapter:
    def test_returns_insurance_adapter(self) -> None:
        adapter = get_adapter(AdapterType.Insurance)
        assert isinstance(adapter, InsuranceClaimsAdapter)

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="No adapter registered"):
            get_adapter(AdapterType.None_)


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------


class TestCheckCanBacktest:
    def test_passes_valid_setup(self, network: HypothesisNetwork) -> None:
        version = _make_version_with_proxied(network.id)
        check_can_backtest(network, version)  # should not raise

    def test_fails_no_adapter(self) -> None:
        network = _make_network(AdapterType.None_)
        version = _make_version_with_proxied(network.id)
        with pytest.raises(BacktestPreflightError, match="no adapter"):
            check_can_backtest(network, version)

    def test_fails_unregistered_adapter(self) -> None:
        network = _make_network(AdapterType.Financial)
        version = _make_version_with_proxied(network.id)
        with pytest.raises(BacktestPreflightError, match="not yet implemented"):
            check_can_backtest(network, version)

    def test_fails_no_proxied_nodes(self, network: HypothesisNetwork) -> None:
        version = _make_version_no_proxied(network.id)
        with pytest.raises(BacktestPreflightError, match="No Proxied nodes"):
            check_can_backtest(network, version)

    def test_fails_already_tested(self, network: HypothesisNetwork) -> None:
        from causal_hypothesis_engine.models.backtest_result import BacktestResult
        br = BacktestResult(
            version_id="v1", adapter_type="Insurance",
            baseline_score=0.5, dag_score=0.6, lift=0.1,
        )
        version = _make_version_with_proxied(network.id)
        tested = version.model_copy(update={
            "status": DAGVersionStatus.Tested,
            "backtest_result": br,
        })
        with pytest.raises(BacktestPreflightError, match="immutable"):
            check_can_backtest(network, tested)

    def test_fails_archived(self, network: HypothesisNetwork) -> None:
        version = _make_version_with_proxied(network.id)
        archived = version.model_copy(update={"status": DAGVersionStatus.Archived})
        with pytest.raises(BacktestPreflightError, match="immutable"):
            check_can_backtest(network, archived)


# ---------------------------------------------------------------------------
# BacktestAgent construction
# ---------------------------------------------------------------------------


class TestBacktestAgentInit:
    def test_creates_with_valid_inputs(
        self, tmp_db: Database, network: HypothesisNetwork, csv_path: Path
    ) -> None:
        version = _make_version_with_proxied(network.id)
        agent = BacktestAgent(db=tmp_db, network=network, version=version, data_path=csv_path)
        assert isinstance(agent.adapter, InsuranceClaimsAdapter)
        assert agent.data_path == csv_path


# ---------------------------------------------------------------------------
# BacktestAgent.run — full pipeline
# ---------------------------------------------------------------------------


class TestBacktestAgentRun:
    def test_run_returns_backtest_result(
        self, tmp_db: Database, network: HypothesisNetwork, csv_path: Path
    ) -> None:
        tmp_db.create_network(network)
        version = _make_version_with_proxied(network.id)
        tmp_db.save_version(version)

        agent = BacktestAgent(db=tmp_db, network=network, version=version, data_path=csv_path)
        result = agent.run(_NullConsole())

        assert isinstance(result, BacktestResult)
        assert result.version_id == version.version_id
        assert 0.0 <= result.baseline_score <= 1.0
        assert 0.0 <= result.dag_score <= 1.0
        assert isinstance(result.lift, float)

    def test_version_status_updated_to_tested(
        self, tmp_db: Database, network: HypothesisNetwork, csv_path: Path
    ) -> None:
        tmp_db.create_network(network)
        version = _make_version_with_proxied(network.id)
        tmp_db.save_version(version)

        agent = BacktestAgent(db=tmp_db, network=network, version=version, data_path=csv_path)
        agent.run(_NullConsole())

        stored = tmp_db.get_version(version.version_id)
        assert stored.status == DAGVersionStatus.Tested
        assert stored.backtest_result is not None

    def test_backtest_result_persisted(
        self, tmp_db: Database, network: HypothesisNetwork, csv_path: Path
    ) -> None:
        tmp_db.create_network(network)
        version = _make_version_with_proxied(network.id)
        tmp_db.save_version(version)

        agent = BacktestAgent(db=tmp_db, network=network, version=version, data_path=csv_path)
        result = agent.run(_NullConsole())

        stored = tmp_db.get_version(version.version_id)
        assert stored.backtest_result.id == result.id
        assert stored.backtest_result.baseline_score == result.baseline_score

    def test_node_contributions_keyed_by_proxied_ids(
        self, tmp_db: Database, network: HypothesisNetwork, csv_path: Path
    ) -> None:
        tmp_db.create_network(network)
        version = _make_version_with_proxied(network.id)
        tmp_db.save_version(version)

        agent = BacktestAgent(db=tmp_db, network=network, version=version, data_path=csv_path)
        result = agent.run(_NullConsole())

        proxied_ids = {
            n.id for n in version.nodes
            if n.measurability_state == MeasurabilityState.Proxied
        }
        for nid in result.node_contributions:
            assert nid in proxied_ids

    def test_invalid_csv_raises_value_error(
        self, tmp_db: Database, network: HypothesisNetwork, tmp_path: Path
    ) -> None:
        # CSV missing required columns
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("col_a,col_b\n1,2\n")
        tmp_db.create_network(network)
        version = _make_version_with_proxied(network.id)
        tmp_db.save_version(version)

        agent = BacktestAgent(db=tmp_db, network=network, version=version, data_path=bad_csv)
        with pytest.raises(ValueError, match="Data validation failed"):
            agent.run(_NullConsole())

    def test_notes_populated(
        self, tmp_db: Database, network: HypothesisNetwork, csv_path: Path
    ) -> None:
        tmp_db.create_network(network)
        version = _make_version_with_proxied(network.id)
        tmp_db.save_version(version)

        agent = BacktestAgent(db=tmp_db, network=network, version=version, data_path=csv_path)
        result = agent.run(_NullConsole())

        assert isinstance(result.notes, str)


# ---------------------------------------------------------------------------
# Parquet support
# ---------------------------------------------------------------------------


class TestParquetSupport:
    def test_run_with_parquet(
        self, tmp_db: Database, network: HypothesisNetwork, tmp_path: Path
    ) -> None:
        df = pd.read_csv(_make_claims_csv(tmp_path, n=100))
        pq_path = tmp_path / "claims.parquet"
        df.to_parquet(pq_path, index=False)

        tmp_db.create_network(network)
        version = _make_version_with_proxied(network.id)
        tmp_db.save_version(version)

        agent = BacktestAgent(db=tmp_db, network=network, version=version, data_path=pq_path)
        result = agent.run(_NullConsole())
        assert isinstance(result, BacktestResult)


# ---------------------------------------------------------------------------
# Null console stub (suppresses all Rich output during tests)
# ---------------------------------------------------------------------------


class _NullConsole:
    """Absorbs all Rich console output silently."""

    def print(self, *args, **kwargs) -> None:
        pass
