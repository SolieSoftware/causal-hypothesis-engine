"""Tests for FinancialDataAdapter and DatasetBuilder.

All external calls (FRED, yfinance, requests) are mocked so no live APIs
are hit in CI.  The FRED_API_KEY env var is set to a dummy value for tests
that exercise the FRED fetcher path.
"""

from __future__ import annotations

import io
import os
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from causal_hypothesis_engine.adapters.financial import (
    FinancialDataAdapter,
    NodeFetchConfig,
    _apply_transform,
    _fetch_file,
    _fetch_fred,
    _fetch_url_csv,
    _fetch_yahoo,
    _resample,
)
from causal_hypothesis_engine.dataset_builder import DatasetBuilder
from causal_hypothesis_engine.models.dataset_result import DatasetResult
from causal_hypothesis_engine.models.dag_version import DAGVersion
from causal_hypothesis_engine.models.node import Node, NodeType, MeasurabilityState
from causal_hypothesis_engine.models.edge import Edge


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_series(n: int = 50, start: str = "2020-01-01", freq: str = "W") -> pd.Series:
    """Create a simple numeric time series."""
    idx = pd.date_range(start, periods=n, freq=freq)
    rng = np.random.default_rng(42)
    return pd.Series(rng.standard_normal(n).cumsum() + 100, index=idx, name="test")


def _make_version(labels: list[str] | None = None) -> DAGVersion:
    labels = labels or ["Fed Funds Rate", "S&P 500"]
    nodes = [
        Node(
            label=label,
            description=f"Test node: {label}",
            node_type=NodeType.Exposure,
            measurability_state=MeasurabilityState.Proxied,
        )
        for label in labels
    ]
    return DAGVersion(
        network_id="test-network-id",
        nodes=nodes,
        edges=[],
        modification_rationale="test",
    )


def _write_manifest(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "manifest.yaml"
    p.write_text(textwrap.dedent(content))
    return p


# ---------------------------------------------------------------------------
# 1. Manifest parsing — happy path via _load_manifest_data
# ---------------------------------------------------------------------------


def test_manifest_parsing_reads_frequency_and_dates(tmp_path: Path) -> None:
    """_load_manifest_data reads top-level manifest metadata."""
    csv_file = tmp_path / "signal.csv"
    series = _make_series(30)
    series.index.name = "date"
    series.to_frame("value").to_csv(csv_file)

    manifest = _write_manifest(
        tmp_path,
        f"""
        frequency: monthly
        resample_method: last
        start_date: "2020-01-01"
        end_date: "2022-12-31"
        nodes:
          - label: "My Signal"
            source: file
            path: "{csv_file}"
            date_column: date
            value_column: value
            transform: none
        """,
    )

    adapter = FinancialDataAdapter()
    df, transform_map, warnings = adapter._load_manifest_data(manifest)

    assert "My Signal" in df.columns
    assert transform_map["My Signal"] == "none"
    assert isinstance(warnings, list)


# ---------------------------------------------------------------------------
# 2. FRED fetch (mocked)
# ---------------------------------------------------------------------------


def test_fetch_fred_success(monkeypatch: pytest.MonkeyPatch) -> None:
    series = _make_series(50)
    mock_fred_instance = MagicMock()
    mock_fred_instance.get_series.return_value = series
    mock_fred_cls = MagicMock(return_value=mock_fred_instance)

    monkeypatch.setenv("FRED_API_KEY", "dummy-key")
    # Fred is imported locally inside _fetch_fred as `from fredapi import Fred`
    # so we patch via sys.modules
    import sys
    fake_fredapi = MagicMock()
    fake_fredapi.Fred = mock_fred_cls
    with patch.dict("sys.modules", {"fredapi": fake_fredapi}):
        result = _fetch_fred(
            NodeFetchConfig(label="DFF", source="fred", transform="diff", series_id="DFF"),
            "2020-01-01",
            "2023-12-31",
        )

    assert isinstance(result, pd.Series)
    assert not result.empty


def test_fetch_fred_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FRED API key missing"):
        _fetch_fred(
            NodeFetchConfig(label="DFF", source="fred", transform="diff", series_id="DFF"),
            "2020-01-01",
            "2023-12-31",
        )


def test_fetch_fred_invalid_series_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "dummy-key")
    mock_fred_instance = MagicMock()
    mock_fred_instance.get_series.side_effect = Exception("400 bad request - not found")
    mock_fred_cls = MagicMock(return_value=mock_fred_instance)

    fake_fredapi = MagicMock()
    fake_fredapi.Fred = mock_fred_cls
    with patch.dict("sys.modules", {"fredapi": fake_fredapi}):
        with pytest.raises(ValueError, match="not found on FRED"):
            _fetch_fred(
                NodeFetchConfig(label="BAD", source="fred", transform="none", series_id="BADID"),
                "2020-01-01",
                "2023-12-31",
            )


# ---------------------------------------------------------------------------
# 3. Yahoo fetch (mocked)
# ---------------------------------------------------------------------------


def test_fetch_yahoo_success() -> None:
    series = _make_series(50)
    mock_data = pd.DataFrame({"Close": series})
    mock_data.index = pd.to_datetime(mock_data.index)

    fake_yf = MagicMock()
    fake_yf.download.return_value = mock_data
    with patch.dict("sys.modules", {"yfinance": fake_yf}):
        result = _fetch_yahoo(
            NodeFetchConfig(label="S&P 500", source="yahoo", transform="log_return", ticker="^GSPC"),
            "2020-01-01",
            "2023-12-31",
        )

    assert isinstance(result, pd.Series)
    assert not result.empty


def test_fetch_yahoo_empty_returns_error() -> None:
    fake_yf = MagicMock()
    fake_yf.download.return_value = pd.DataFrame()
    with patch.dict("sys.modules", {"yfinance": fake_yf}):
        with pytest.raises(ValueError, match="No data returned for ticker"):
            _fetch_yahoo(
                NodeFetchConfig(label="VIX", source="yahoo", transform="diff", ticker="^VIX"),
                "2020-01-01",
                "2023-12-31",
            )


# ---------------------------------------------------------------------------
# 4. url_csv fetch (mocked)
# ---------------------------------------------------------------------------


def test_fetch_url_csv_success() -> None:
    series = _make_series(30)
    series.index.name = "DATE"
    csv_str = series.to_frame("VALUE").to_csv()

    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.text = csv_str

    fake_requests = MagicMock()
    fake_requests.get.return_value = mock_response
    fake_requests.RequestException = Exception
    with patch.dict("sys.modules", {"requests": fake_requests}):
        result = _fetch_url_csv(
            NodeFetchConfig(
                label="Inflation Exp",
                source="url_csv",
                transform="none",
                url="https://example.com/data.csv",
                date_column="DATE",
                value_column="VALUE",
            ),
            "2020-01-01",
            "2023-12-31",
        )

    assert isinstance(result, pd.Series)


def test_fetch_url_csv_http_error() -> None:
    mock_response = MagicMock()
    mock_response.ok = False
    mock_response.status_code = 404

    fake_requests = MagicMock()
    fake_requests.get.return_value = mock_response
    fake_requests.RequestException = Exception
    with patch.dict("sys.modules", {"requests": fake_requests}):
        with pytest.raises(RuntimeError, match="HTTP 404"):
            _fetch_url_csv(
                NodeFetchConfig(
                    label="X",
                    source="url_csv",
                    transform="none",
                    url="https://example.com/missing.csv",
                    date_column="date",
                    value_column="value",
                ),
                "2020-01-01",
                "2023-12-31",
            )


# ---------------------------------------------------------------------------
# 5. Local file loading
# ---------------------------------------------------------------------------


def test_fetch_file_csv(tmp_path: Path) -> None:
    series = _make_series(40)
    series.index.name = "date"
    csv_file = tmp_path / "signal.csv"
    series.to_frame("value").to_csv(csv_file)

    result = _fetch_file(
        NodeFetchConfig(
            label="Signal",
            source="file",
            transform="none",
            path=str(csv_file),
            date_column="date",
            value_column="value",
        ),
        "2020-01-01",
        "2030-12-31",
    )

    assert isinstance(result, pd.Series)
    assert not result.empty


def test_fetch_file_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="File not found"):
        _fetch_file(
            NodeFetchConfig(
                label="X",
                source="file",
                transform="none",
                path=str(tmp_path / "nonexistent.csv"),
                date_column="date",
                value_column="value",
            ),
            "2020-01-01",
            "2023-12-31",
        )


# ---------------------------------------------------------------------------
# 6. Transform application
# ---------------------------------------------------------------------------


def test_apply_transform_diff() -> None:
    s = pd.Series([100.0, 101.0, 103.0, 106.0])
    result = _apply_transform(s, "diff")
    assert list(result) == pytest.approx([1.0, 2.0, 3.0])


def test_apply_transform_log_return() -> None:
    s = pd.Series([100.0, 110.0])
    result = _apply_transform(s, "log_return")
    assert len(result) == 1
    assert result.iloc[0] == pytest.approx(np.log(110.0 / 100.0))


def test_apply_transform_none_passthrough() -> None:
    s = pd.Series([1.0, 2.0, 3.0])
    result = _apply_transform(s, "none")
    assert list(result) == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# 7. Short-series guard (< 20 observations)
# ---------------------------------------------------------------------------


def test_short_series_guard(tmp_path: Path) -> None:
    """Series with < 20 obs should get transform_applied='insufficient_data'."""
    # Build a small CSV (10 rows)
    series = _make_series(10)
    series.index.name = "date"
    csv_file = tmp_path / "short.csv"
    series.to_frame("value").to_csv(csv_file)

    manifest = _write_manifest(
        tmp_path,
        f"""
        frequency: weekly
        resample_method: last
        start_date: "2020-01-01"
        end_date: "2021-12-31"
        nodes:
          - label: "Short Signal"
            source: file
            path: "{csv_file}"
            date_column: date
            value_column: value
            transform: none
        """,
    )

    version = _make_version(["Short Signal"])
    builder = DatasetBuilder()
    result = builder.run(version=version, manifest_path=manifest)

    assert result.adf_results["Short Signal"]["transform_applied"] == "insufficient_data"
    assert result.adf_results["Short Signal"]["passed"] is False
    assert any("ADF test skipped" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# 8. none + diff → "diff" compound string
# ---------------------------------------------------------------------------


def test_none_plus_diff_compound_string(tmp_path: Path) -> None:
    """When declared transform is 'none' and extra diff is applied, record as 'diff'."""
    # Create a non-stationary random walk series (will fail first ADF)
    rng = np.random.default_rng(7)
    vals = rng.standard_normal(60).cumsum() + 1000  # random walk — likely non-stationary
    idx = pd.date_range("2020-01-05", periods=60, freq="W")
    series = pd.Series(vals, index=idx, name="rw")
    series.index.name = "date"

    csv_file = tmp_path / "rw.csv"
    series.to_frame("value").to_csv(csv_file)

    manifest = _write_manifest(
        tmp_path,
        f"""
        frequency: weekly
        resample_method: last
        start_date: "2020-01-01"
        end_date: "2022-12-31"
        nodes:
          - label: "Random Walk"
            source: file
            path: "{csv_file}"
            date_column: date
            value_column: value
            transform: none
        """,
    )

    version = _make_version(["Random Walk"])
    builder = DatasetBuilder()
    result = builder.run(version=version, manifest_path=manifest)

    transform_applied = result.adf_results["Random Walk"]["transform_applied"]
    # Either stationary on first pass (transform_applied="none") or got auto-diffed
    assert transform_applied in ("none", "diff")
    # If auto-diffed, it must NOT be "none+diff" — must be "diff"
    assert transform_applied != "none+diff"


# ---------------------------------------------------------------------------
# 9. DatasetResult schema round-trip
# ---------------------------------------------------------------------------


def test_dataset_result_round_trip() -> None:
    result = DatasetResult(
        version_id="v-001",
        manifest_path="/tmp/test.yaml",
        columns=["A", "B"],
        start_date="2020-01-01",
        end_date="2023-12-31",
        frequency="weekly",
        adf_results={
            "A": {"statistic": -3.5, "pvalue": 0.01, "passed": True, "transform_applied": "diff"},
            "B": {"statistic": -1.2, "pvalue": 0.3, "passed": False, "transform_applied": "diff+diff"},
        },
        warnings=["[WARN] something"],
        output_path="/tmp/out.parquet",
    )

    json_str = result.model_dump_json()
    restored = DatasetResult.model_validate_json(json_str)

    assert restored.version_id == result.version_id
    assert restored.columns == result.columns
    assert restored.adf_results["A"]["passed"] is True
    assert restored.adf_results["B"]["transform_applied"] == "diff+diff"


# ---------------------------------------------------------------------------
# 10. Label mismatch — warning (non-strict)
# ---------------------------------------------------------------------------


def test_label_mismatch_emits_warning(tmp_path: Path) -> None:
    """Manifest node not in DAGVersion emits warning but does not raise."""
    series = _make_series(50)
    series.index.name = "date"
    csv_file = tmp_path / "sig.csv"
    series.to_frame("value").to_csv(csv_file)

    manifest = _write_manifest(
        tmp_path,
        f"""
        frequency: weekly
        resample_method: last
        start_date: "2020-01-01"
        end_date: "2023-12-31"
        nodes:
          - label: "Extra Node"
            source: file
            path: "{csv_file}"
            date_column: date
            value_column: value
            transform: none
        """,
    )

    # DAGVersion has different label
    version = _make_version(["Unrelated Node"])
    builder = DatasetBuilder()
    result = builder.run(version=version, manifest_path=manifest, strict=False)

    warning_text = " ".join(result.warnings)
    assert "Extra Node" in warning_text or "Unrelated Node" in warning_text


# ---------------------------------------------------------------------------
# 11. --strict flag turns mismatch into error
# ---------------------------------------------------------------------------


def test_strict_flag_raises_on_label_mismatch(tmp_path: Path) -> None:
    series = _make_series(50)
    series.index.name = "date"
    csv_file = tmp_path / "sig.csv"
    series.to_frame("value").to_csv(csv_file)

    manifest = _write_manifest(
        tmp_path,
        f"""
        frequency: weekly
        resample_method: last
        start_date: "2020-01-01"
        end_date: "2023-12-31"
        nodes:
          - label: "Extra Node"
            source: file
            path: "{csv_file}"
            date_column: date
            value_column: value
            transform: none
        """,
    )

    version = _make_version(["Unrelated Node"])
    builder = DatasetBuilder()

    with pytest.raises((ValueError, RuntimeError), match="Label mismatch"):
        builder.run(version=version, manifest_path=manifest, strict=True)


# ---------------------------------------------------------------------------
# 12. Atomic write (tmp → rename)
# ---------------------------------------------------------------------------


def test_atomic_write_produces_parquet(tmp_path: Path) -> None:
    series = _make_series(50)
    series.index.name = "date"
    csv_file = tmp_path / "sig.csv"
    series.to_frame("value").to_csv(csv_file)

    manifest = _write_manifest(
        tmp_path,
        f"""
        frequency: weekly
        resample_method: last
        start_date: "2020-01-01"
        end_date: "2023-12-31"
        nodes:
          - label: "Signal"
            source: file
            path: "{csv_file}"
            date_column: date
            value_column: value
            transform: none
        """,
    )

    out_file = tmp_path / "output.parquet"
    version = _make_version(["Signal"])
    builder = DatasetBuilder()
    result = builder.run(version=version, manifest_path=manifest, out_path=out_file)

    assert Path(result.output_path).exists()
    # Tmp file must be gone
    assert not out_file.with_suffix(".parquet.tmp").exists()
    # Should be readable as Parquet
    df = pd.read_parquet(result.output_path)
    assert "Signal" in df.columns


# ---------------------------------------------------------------------------
# 13. Resample logic
# ---------------------------------------------------------------------------


def test_resample_last() -> None:
    daily = _make_series(90, freq="D")
    result = _resample(daily, "weekly", "last")
    assert len(result) < len(daily)
    assert not result.empty


def test_resample_mean() -> None:
    daily = _make_series(30, freq="D")
    result = _resample(daily, "monthly", "mean")
    assert len(result) <= 2  # ~1 month of daily data


# ---------------------------------------------------------------------------
# 14. ADF double-diff passes — transform recorded as compound
# ---------------------------------------------------------------------------


def test_adf_double_diff_compound_name(tmp_path: Path) -> None:
    """diff+diff must be recorded when diff transform + extra diff both applied."""
    # Create a random walk (non-stationary); after diff it may still be non-stationary
    # We'll force the scenario by patching adfuller to return non-stationary, then stationary
    series = _make_series(60)
    series.index.name = "date"
    csv_file = tmp_path / "sig.csv"
    series.to_frame("value").to_csv(csv_file)

    manifest = _write_manifest(
        tmp_path,
        f"""
        frequency: weekly
        resample_method: last
        start_date: "2020-01-01"
        end_date: "2023-12-31"
        nodes:
          - label: "Trend"
            source: file
            path: "{csv_file}"
            date_column: date
            value_column: value
            transform: diff
        """,
    )

    version = _make_version(["Trend"])

    # Force first ADF to fail, second to pass
    call_count = {"n": 0}
    original_adfuller_results = [
        (-1.0, 0.5, 1, 50, {}, 100.0),   # first call: non-stationary
        (-4.0, 0.01, 1, 49, {}, 90.0),   # second call: stationary
    ]

    def mock_adfuller(series, *args, **kwargs):
        r = original_adfuller_results[min(call_count["n"], 1)]
        call_count["n"] += 1
        return r

    with patch("statsmodels.tsa.stattools.adfuller", mock_adfuller):
        builder = DatasetBuilder()
        result = builder.run(version=version, manifest_path=manifest)

    assert result.adf_results["Trend"]["transform_applied"] == "diff+diff"
    assert result.adf_results["Trend"]["passed"] is True


# ---------------------------------------------------------------------------
# 15. FinancialDataAdapter adapter_type and domain_label
# ---------------------------------------------------------------------------


def test_adapter_identity() -> None:
    from causal_hypothesis_engine.models.network import AdapterType

    adapter = FinancialDataAdapter()
    assert adapter.domain_label == "Financial Markets"
    assert adapter.adapter_type == AdapterType.Financial


def test_adapter_outcome_column_raises() -> None:
    adapter = FinancialDataAdapter()
    with pytest.raises(NotImplementedError):
        adapter.outcome_column(pd.DataFrame())


def test_adapter_compute_dag_score_returns_zero() -> None:
    adapter = FinancialDataAdapter()
    score, contrib = adapter.compute_dag_score(pd.DataFrame(), pd.DataFrame(), "col")
    assert score == 0.0
    assert contrib == {}
