"""Tests for AdapterBase and InsuranceClaimsAdapter.

Uses small in-memory DataFrames and a temporary CSV/Parquet file on disk.
No Anthropic API calls.  sklearn is optional — tests run with the
correlation fallback if it is not installed.
"""

from __future__ import annotations

import math
import tempfile
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

from causal_hypothesis_engine.adapters.insurance import (
    CLAIM_TYPES,
    REQUIRED_COLUMNS,
    InsuranceClaimsAdapter,
)
from causal_hypothesis_engine.models.dag_version import DAGVersion
from causal_hypothesis_engine.models.network import AdapterType, HypothesisNetwork
from causal_hypothesis_engine.models.node import MeasurabilityState, Node, NodeType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate a deterministic synthetic claims DataFrame."""
    import random
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        claim_type = rng.choice(CLAIM_TYPES)
        rainfall = rng.uniform(0, 200)
        wind = rng.uniform(0, 120)
        claim_amount = rng.uniform(500, 50_000)
        is_large = int(claim_amount > 20_000 or rainfall > 150)
        rows.append(
            {
                "claim_id": f"C{i:04d}",
                "claim_type": claim_type,
                "claim_amount": claim_amount,
                "is_large_claim": is_large,
                "rainfall_mm": rainfall,
                "max_wind_speed": wind,
                "flood_zone": int(rainfall > 100),
                "fire_risk_score": rng.uniform(0, 1),
            }
        )
    return pd.DataFrame(rows)


def _make_version_with_proxied_nodes(network_id: str) -> DAGVersion:
    """DAGVersion with two Proxied nodes that have proxy_variables."""
    rain_node = Node(
        label="Rainfall",
        node_type=NodeType.Exposure,
        measurability_state=MeasurabilityState.Proxied,
        adapter_metadata={
            "proxy_variables": ["rainfall_mm", "flood_zone"],
            "claim_type": "flood",
        },
    )
    wind_node = Node(
        label="Wind Speed",
        node_type=NodeType.Exposure,
        measurability_state=MeasurabilityState.Proxied,
        adapter_metadata={
            "proxy_variables": ["max_wind_speed"],
            "claim_type": "storm",
        },
    )
    outcome_node = Node(
        label="Large Claim",
        node_type=NodeType.Outcome,
        measurability_state=MeasurabilityState.Hypothetical,
    )
    return DAGVersion(
        network_id=network_id,
        nodes=[rain_node, wind_node, outcome_node],
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter() -> InsuranceClaimsAdapter:
    return InsuranceClaimsAdapter()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return _make_df()


@pytest.fixture
def sample_version() -> DAGVersion:
    return _make_version_with_proxied_nodes("net-test")


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


class TestAdapterIdentity:
    def test_domain_label(self, adapter: InsuranceClaimsAdapter) -> None:
        assert adapter.domain_label == "Insurance Claims"

    def test_adapter_type(self, adapter: InsuranceClaimsAdapter) -> None:
        assert adapter.adapter_type == AdapterType.Insurance

    def test_node_metadata_schema_has_required_fields(self, adapter: InsuranceClaimsAdapter) -> None:
        schema = adapter.node_metadata_schema
        fields = schema.required_fields()
        assert "claim_type" in fields
        assert "proxy_variables" in fields
        assert "data_source" in fields

    def test_describe_lift(self, adapter: InsuranceClaimsAdapter) -> None:
        desc = adapter.describe_lift()
        assert isinstance(desc, str)
        assert len(desc) > 10


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


class TestDataLoading:
    def test_load_csv(self, adapter: InsuranceClaimsAdapter, tmp_path: Path) -> None:
        df = _make_df(20)
        csv_path = tmp_path / "claims.csv"
        df.to_csv(csv_path, index=False)
        loaded = adapter.load_data(csv_path)
        assert len(loaded) == 20
        assert "claim_id" in loaded.columns

    def test_load_parquet(self, adapter: InsuranceClaimsAdapter, tmp_path: Path) -> None:
        df = _make_df(20)
        pq_path = tmp_path / "claims.parquet"
        df.to_parquet(pq_path, index=False)
        loaded = adapter.load_data(pq_path)
        assert len(loaded) == 20

    def test_load_missing_file_raises(self, adapter: InsuranceClaimsAdapter) -> None:
        with pytest.raises(FileNotFoundError):
            adapter.load_data("/nonexistent/path/claims.csv")

    def test_load_unsupported_format_raises(
        self, adapter: InsuranceClaimsAdapter, tmp_path: Path
    ) -> None:
        bad = tmp_path / "data.xlsx"
        bad.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported file format"):
            adapter.load_data(bad)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidateData:
    def test_valid_df_passes(
        self, adapter: InsuranceClaimsAdapter, sample_df: pd.DataFrame
    ) -> None:
        errors = adapter.validate_data(sample_df)
        assert errors == []

    def test_missing_required_columns(self, adapter: InsuranceClaimsAdapter) -> None:
        df = pd.DataFrame({"claim_id": ["C1"], "claim_type": ["flood"]})
        errors = adapter.validate_data(df)
        assert any("Missing required columns" in e for e in errors)
        missing_in_errors = errors[0]
        assert "claim_amount" in missing_in_errors or "is_large_claim" in missing_in_errors

    def test_unknown_claim_type(self, adapter: InsuranceClaimsAdapter) -> None:
        df = _make_df(10)
        df.loc[0, "claim_type"] = "earthquake"
        errors = adapter.validate_data(df)
        assert any("earthquake" in e for e in errors)

    def test_invalid_outcome_values(self, adapter: InsuranceClaimsAdapter) -> None:
        df = _make_df(10)
        df.loc[0, "is_large_claim"] = 99  # not 0/1
        errors = adapter.validate_data(df)
        assert any("is_large_claim" in e for e in errors)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


class TestBuildProxyFeatures:
    def test_returns_columns_for_proxied_nodes(
        self,
        adapter: InsuranceClaimsAdapter,
        sample_df: pd.DataFrame,
        sample_version: DAGVersion,
    ) -> None:
        features = adapter.build_proxy_features(sample_df, sample_version)
        assert not features.empty
        # Should have columns for rainfall_mm, flood_zone, max_wind_speed
        col_names = " ".join(features.columns)
        assert "rainfall_mm" in col_names
        assert "flood_zone" in col_names
        assert "max_wind_speed" in col_names

    def test_skips_hypothetical_nodes(
        self, adapter: InsuranceClaimsAdapter, sample_df: pd.DataFrame
    ) -> None:
        """Nodes with Hypothetical state have no proxy features."""
        outcome = Node(
            label="Large Claim",
            node_type=NodeType.Outcome,
            measurability_state=MeasurabilityState.Hypothetical,
            adapter_metadata={"proxy_variables": ["rainfall_mm"]},
        )
        version = DAGVersion(network_id="net-1", nodes=[outcome])
        features = adapter.build_proxy_features(sample_df, version)
        assert features.empty

    def test_missing_proxy_column_gives_nan_column(
        self,
        adapter: InsuranceClaimsAdapter,
        sample_df: pd.DataFrame,
    ) -> None:
        node = Node(
            label="Satellite NDVI",
            node_type=NodeType.Exposure,
            measurability_state=MeasurabilityState.Proxied,
            adapter_metadata={"proxy_variables": ["ndvi_index"]},  # not in df
        )
        version = DAGVersion(network_id="net-1", nodes=[node])
        features = adapter.build_proxy_features(sample_df, version)
        assert not features.empty
        col = [c for c in features.columns if "ndvi_index" in c][0]
        assert features[col].isna().all()

    def test_no_proxied_nodes_returns_empty(
        self, adapter: InsuranceClaimsAdapter, sample_df: pd.DataFrame
    ) -> None:
        node = Node(
            label="Exposure",
            node_type=NodeType.Exposure,
            measurability_state=MeasurabilityState.Hypothetical,
        )
        version = DAGVersion(network_id="net-1", nodes=[node])
        features = adapter.build_proxy_features(sample_df, version)
        assert features.empty

    def test_feature_count_matches_data_rows(
        self,
        adapter: InsuranceClaimsAdapter,
        sample_df: pd.DataFrame,
        sample_version: DAGVersion,
    ) -> None:
        features = adapter.build_proxy_features(sample_df, sample_version)
        assert len(features) == len(sample_df)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestScoring:
    def test_baseline_score_in_valid_range(
        self, adapter: InsuranceClaimsAdapter, sample_df: pd.DataFrame
    ) -> None:
        score = adapter.compute_baseline_score(sample_df, "is_large_claim")
        assert 0.0 <= score <= 1.0

    def test_dag_score_returns_float_and_dict(
        self,
        adapter: InsuranceClaimsAdapter,
        sample_df: pd.DataFrame,
        sample_version: DAGVersion,
    ) -> None:
        proxy = adapter.build_proxy_features(sample_df, sample_version)
        dag_score, contributions = adapter.compute_dag_score(
            sample_df, proxy, "is_large_claim"
        )
        assert 0.0 <= dag_score <= 1.0
        assert isinstance(contributions, dict)

    def test_node_contributions_keyed_by_node_id(
        self,
        adapter: InsuranceClaimsAdapter,
        sample_df: pd.DataFrame,
        sample_version: DAGVersion,
    ) -> None:
        proxy = adapter.build_proxy_features(sample_df, sample_version)
        _, contributions = adapter.compute_dag_score(
            sample_df, proxy, "is_large_claim"
        )
        proxied_node_ids = {
            n.id for n in sample_version.nodes
            if n.measurability_state == MeasurabilityState.Proxied
        }
        for node_id in contributions:
            assert node_id in proxied_node_ids

    def test_dag_score_empty_proxy_equals_baseline(
        self, adapter: InsuranceClaimsAdapter, sample_df: pd.DataFrame
    ) -> None:
        """When proxy_features is empty, dag_score == baseline_score."""
        baseline = adapter.compute_baseline_score(sample_df, "is_large_claim")
        empty_proxy = pd.DataFrame(index=sample_df.index)
        dag_score, contributions = adapter.compute_dag_score(
            sample_df, empty_proxy, "is_large_claim"
        )
        assert dag_score == pytest.approx(baseline)
        assert contributions == {}

    def test_outcome_column(
        self, adapter: InsuranceClaimsAdapter, sample_df: pd.DataFrame
    ) -> None:
        col = adapter.outcome_column(sample_df)
        assert col == "is_large_claim"
        assert col in sample_df.columns


# ---------------------------------------------------------------------------
# End-to-end: load CSV → validate → build features → score
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_full_pipeline(
        self, adapter: InsuranceClaimsAdapter, tmp_path: Path
    ) -> None:
        df = _make_df(200, seed=7)
        csv_path = tmp_path / "claims.csv"
        df.to_csv(csv_path, index=False)

        loaded = adapter.load_data(csv_path)
        errors = adapter.validate_data(loaded)
        assert errors == [], f"Validation errors: {errors}"

        version = _make_version_with_proxied_nodes("net-e2e")
        proxy = adapter.build_proxy_features(loaded, version)
        assert not proxy.empty

        outcome_col = adapter.outcome_column(loaded)
        baseline = adapter.compute_baseline_score(loaded, outcome_col)
        dag_score, contributions = adapter.compute_dag_score(loaded, proxy, outcome_col)

        assert 0.0 <= baseline <= 1.0
        assert 0.0 <= dag_score <= 1.0
        # lift can be negative (proxy features may not help), just check it's a float
        lift = dag_score - baseline
        assert isinstance(lift, float)
        assert not math.isnan(lift)
