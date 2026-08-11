"""Tests for the generic TabularAdapter and DatasetResult persistence.

Covers the two gaps that made the engine unusable outside its demo:
no adapter accepted arbitrary data, and datasets were orphan files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_hypothesis_engine.adapters.tabular import TabularAdapter
from causal_hypothesis_engine.models import DAGVersion, Edge, Node
from causal_hypothesis_engine.models.dataset_result import DatasetResult
from causal_hypothesis_engine.models.node import MeasurabilityState, NodeType
from causal_hypothesis_engine.persistence.db import Database
from causal_hypothesis_engine.scoring import ScoringError


def _frame(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    driver = rng.normal(size=n)
    noise = rng.normal(size=n)
    logit = 1.6 * driver
    probability = 1 / (1 + np.exp(-logit))
    return pd.DataFrame(
        {
            "driver": driver,
            "noise": noise,
            "target": rng.binomial(1, probability),
            "category": rng.choice(["a", "b", "c"], n),
        }
    )


def _version(bind_category: bool = False) -> DAGVersion:
    driver = Node(
        label="Driver",
        node_type=NodeType.Exposure,
        measurability_state=MeasurabilityState.Proxied,
        adapter_metadata={"proxy_variables": ["driver"]},
    )
    noise = Node(
        label="Noise",
        node_type=NodeType.Exposure,
        measurability_state=MeasurabilityState.Proxied,
        adapter_metadata={"proxy_variables": ["noise"]},
    )
    outcome = Node(
        label="Target",
        node_type=NodeType.Outcome,
        measurability_state=MeasurabilityState.Identified,
        adapter_metadata={"proxy_variables": ["target"]},
    )
    nodes = [driver, noise, outcome]
    if bind_category:
        nodes.append(
            Node(
                label="Category",
                node_type=NodeType.Confounder,
                measurability_state=MeasurabilityState.Proxied,
                adapter_metadata={"proxy_variables": ["category"]},
            )
        )
    return DAGVersion(
        network_id="net-1",
        nodes=nodes,
        edges=[Edge(source_node_id=driver.id, target_node_id=outcome.id)],
    )


class TestTabularAdapterAcceptsArbitraryData:
    def test_no_schema_is_imposed(self) -> None:
        """Any table is valid — the insurance schema is not required."""
        adapter = TabularAdapter()
        df = pd.DataFrame({"anything": [1, 2], "at_all": ["x", "y"]})
        assert adapter.validate_data(df) == []

    def test_empty_frame_is_rejected(self) -> None:
        assert TabularAdapter().validate_data(pd.DataFrame()) != []

    def test_outcome_resolved_from_outcome_node(self) -> None:
        adapter = TabularAdapter()
        assert adapter.resolve_outcome(_frame(), _version()) == "target"

    def test_explicit_outcome_wins(self) -> None:
        adapter = TabularAdapter("noise")
        assert adapter.resolve_outcome(_frame(), _version()) == "noise"

    def test_unknown_explicit_outcome_is_an_actionable_error(self) -> None:
        adapter = TabularAdapter("not_a_column")
        with pytest.raises(ScoringError, match="not\n?.*in the data"):
            adapter.resolve_outcome(_frame(), _version())

    def test_unresolvable_outcome_names_the_fix(self) -> None:
        adapter = TabularAdapter()
        bare = Node(label="Lonely", node_type=NodeType.Exposure)
        version = DAGVersion(network_id="net-1", nodes=[bare])
        with pytest.raises(ScoringError, match="causal-engine bind"):
            adapter.resolve_outcome(_frame(), version)


class TestTabularAdapterScoring:
    def test_recovers_a_real_signal_and_rejects_noise(self) -> None:
        adapter = TabularAdapter("target")
        df = _frame()
        version = _version()
        features = adapter.build_proxy_features(df, version)
        detail = adapter.score_detail(df, features, "target")

        assert detail["metric_name"] == "out-of-fold ROC AUC"
        assert detail["dag_score"] > 0.7

        by_label = {n.label: n.id for n in version.nodes}
        driver_ci = detail["node_detail"][by_label["Driver"]]
        noise_ci = detail["node_detail"][by_label["Noise"]]

        # The genuine driver's interval excludes zero; noise's does not.
        assert driver_ci[0] > 0
        assert noise_ci[0] < 0 < noise_ci[1]

    def test_outcome_column_is_refused_as_a_proxy(self) -> None:
        """A proxy naming the outcome would report leakage as causal signal."""
        adapter = TabularAdapter("target")
        leaky = Node(
            label="Leak",
            node_type=NodeType.Exposure,
            measurability_state=MeasurabilityState.Proxied,
            adapter_metadata={"proxy_variables": ["target"]},
        )
        version = DAGVersion(network_id="net-1", nodes=[leaky])
        features = adapter.build_proxy_features(_frame(), version)

        assert features.empty
        assert any("Refused proxy 'target'" in w for w in adapter.feature_warnings)

    def test_categorical_proxy_is_one_hot_encoded(self) -> None:
        adapter = TabularAdapter("target")
        features = adapter.build_proxy_features(_frame(), _version(bind_category=True))
        assert any("category=" in c for c in features.columns)

    def test_validated_nodes_are_included(self) -> None:
        """Promoting to Validated must not remove a node from scoring."""
        adapter = TabularAdapter("target")
        node = Node(
            label="Driver",
            node_type=NodeType.Exposure,
            measurability_state=MeasurabilityState.Validated,
            adapter_metadata={"proxy_variables": ["driver"]},
        )
        version = DAGVersion(network_id="net-1", nodes=[node])
        assert not adapter.build_proxy_features(_frame(), version).empty

    def test_refuses_to_score_too_few_positives(self) -> None:
        adapter = TabularAdapter("target")
        df = _frame(60).copy()
        df["target"] = 0
        df.loc[df.index[:3], "target"] = 1
        features = adapter.build_proxy_features(df, _version())
        with pytest.raises(ScoringError, match="rarer outcome class"):
            adapter.score_detail(df, features, "target")


class TestDatasetResultPersistence:
    def _result(self, version_id: str = "v-1") -> DatasetResult:
        return DatasetResult(
            version_id=version_id,
            manifest_path="/tmp/m.yaml",
            columns=["A", "B"],
            start_date="2015-01-01",
            end_date="2024-12-31",
            frequency="weekly",
            adf_results={"A": {"statistic": -3.1, "pvalue": 0.02, "passed": True}},
            warnings=["[WARN] something"],
            output_path="/tmp/out.parquet",
        )

    def test_round_trip(self, tmp_path) -> None:
        db = Database(tmp_path / "t.db")
        result = self._result()
        db.save_dataset_result(result)

        loaded = db.get_dataset_results_for_version("v-1")
        assert len(loaded) == 1
        assert loaded[0].id == result.id
        assert loaded[0].columns == ["A", "B"]
        assert loaded[0].adf_results["A"]["passed"] is True
        assert loaded[0].warnings == ["[WARN] something"]
        assert loaded[0].output_path == "/tmp/out.parquet"

    def test_scoped_to_version(self, tmp_path) -> None:
        db = Database(tmp_path / "t.db")
        db.save_dataset_result(self._result("v-1"))
        db.save_dataset_result(self._result("v-2"))

        assert len(db.get_dataset_results_for_version("v-1")) == 1
        assert db.get_dataset_results_for_version("v-unknown") == []

    def test_multiple_datasets_per_version(self, tmp_path) -> None:
        db = Database(tmp_path / "t.db")
        db.save_dataset_result(self._result())
        db.save_dataset_result(self._result())
        assert len(db.get_dataset_results_for_version("v-1")) == 2
