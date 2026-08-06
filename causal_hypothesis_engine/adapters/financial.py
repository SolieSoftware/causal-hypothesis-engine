"""FinancialDataAdapter — fetches and prepares financial time-series data.

Supported sources in v1:
  fred      — FRED (Federal Reserve Economic Data) via fredapi
  yahoo     — Yahoo Finance via yfinance
  file      — local CSV or Parquet file
  url_csv   — direct HTTP GET to a CSV URL

Each node in the manifest YAML maps to one fetcher.  All fetchers return a
raw pd.Series with a DatetimeIndex.  Resampling and declared transforms are
applied centrally by load_data / _load_manifest_data.

Error messages follow the project convention:
  [ERROR] Problem: ... Cause: ... Fix: ...
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Callable, TypedDict

if sys.version_info >= (3, 11):
    from typing import NotRequired
else:
    from typing_extensions import NotRequired

import pandas as pd

from .base import AdapterBase, NodeMetadataSchema

if TYPE_CHECKING:
    from ..models.dag_version import DAGVersion
    from ..models.network import AdapterType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed config shared by all fetchers
# ---------------------------------------------------------------------------


class NodeFetchConfig(TypedDict):
    label: str
    source: str
    transform: str
    # FRED
    series_id: NotRequired[str]
    # Yahoo
    ticker: NotRequired[str]
    # file / url_csv
    path: NotRequired[str]
    url: NotRequired[str]
    date_column: NotRequired[str]
    value_column: NotRequired[str]


_FetcherFn = Callable[[NodeFetchConfig, str, str], pd.Series]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resample(series: pd.Series, frequency: str, method: str) -> pd.Series:
    """Resample *series* to the requested *frequency* using *method*."""
    freq_map = {"daily": "D", "weekly": "W", "monthly": "ME"}
    pandas_freq = freq_map.get(frequency.lower(), "W")
    resampler = series.resample(pandas_freq)
    if method == "mean":
        return resampler.mean()
    if method == "sum":
        return resampler.sum()
    return resampler.last()  # default: "last"


def _apply_transform(series: pd.Series, transform: str) -> pd.Series:
    """Apply a manifest-declared transform to *series*."""
    if transform == "diff":
        return series.diff().dropna()
    if transform == "log_return":
        import numpy as np
        return np.log(series / series.shift(1)).dropna()
    # "none" or unknown → pass through
    return series


# ---------------------------------------------------------------------------
# Fetcher functions
# ---------------------------------------------------------------------------


def _fetch_fred(config: NodeFetchConfig, start_date: str, end_date: str) -> pd.Series:
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "[ERROR] Problem: FRED API key missing. "
            "Cause: source: fred requires FRED_API_KEY. "
            "Fix: export FRED_API_KEY=..."
        )

    series_id = config.get("series_id")
    if not series_id:
        raise ValueError(
            f"[ERROR] Problem: series_id missing for FRED node '{config['label']}'. "
            "Fix: Add series_id to the manifest node."
        )

    try:
        from fredapi import Fred  # type: ignore[import-untyped]
        fred = Fred(api_key=api_key)
        data: pd.Series = fred.get_series(
            series_id,
            observation_start=start_date,
            observation_end=end_date,
        )
    except Exception as exc:
        err_lower = str(exc).lower()
        if any(x in err_lower for x in ("not found", "400", "bad request", "does not exist")):
            raise ValueError(
                f"[ERROR] Problem: Series \"{series_id}\" not found on FRED. "
                "Cause: Invalid series_id. "
                "Fix: Check https://fred.stlouisfed.org"
            ) from exc
        raise RuntimeError(
            f"[ERROR] Problem: FRED fetch failed for \"{series_id}\". "
            f"Cause: {exc}."
        ) from exc

    if data is None or data.empty:
        raise ValueError(
            f"[ERROR] Problem: Series \"{series_id}\" returned no data from FRED. "
            "Cause: Date range may be outside availability. "
            "Fix: Check series availability on fred.stlouisfed.org."
        )

    data.index = pd.to_datetime(data.index)
    data.name = config["label"]
    return data


def _fetch_yahoo(config: NodeFetchConfig, start_date: str, end_date: str) -> pd.Series:
    ticker = config.get("ticker")
    if not ticker:
        raise ValueError(
            f"[ERROR] Problem: ticker missing for yahoo node '{config['label']}'. "
            "Fix: Add ticker to the manifest node."
        )

    try:
        import yfinance as yf  # type: ignore[import-untyped]
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"[ERROR] Problem: yfinance download failed for '{ticker}'. "
            f"Cause: {exc}."
        ) from exc

    if data is None or data.empty:
        raise ValueError(
            f"[ERROR] Problem: No data returned for ticker \"{ticker}\". "
            "Cause: yfinance rate-limit or invalid ticker. "
            "Fix: Wait 60s or check ticker."
        )

    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.squeeze()
    close.index = pd.to_datetime(close.index)
    close.name = config["label"]
    return close


def _fetch_file(config: NodeFetchConfig, start_date: str, end_date: str) -> pd.Series:
    path_str = config.get("path")
    if not path_str:
        raise ValueError(
            f"[ERROR] Problem: path missing for file node '{config['label']}'. "
            "Fix: Add path to the manifest node."
        )

    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(
            f"[ERROR] Problem: File not found. "
            f"Cause: {path_str} does not exist. "
            "Fix: Check path in manifest."
        )

    date_col = config.get("date_column", "date")
    value_col = config.get("value_column", "value")

    suffix = p.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(p, parse_dates=[date_col], index_col=date_col)
    elif suffix in (".parquet", ".pq"):
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index)
    else:
        raise ValueError(
            f"[ERROR] Problem: Unsupported file format '{suffix}'. "
            "Fix: Use .csv or .parquet."
        )

    if value_col not in df.columns:
        raise ValueError(
            f"[ERROR] Problem: Column '{value_col}' not found in {p.name}. "
            f"Cause: Available columns: {list(df.columns)}. "
            "Fix: Set value_column in manifest to a valid column name."
        )

    series = df[value_col].copy()
    series.index = pd.to_datetime(series.index)
    mask = (series.index >= pd.Timestamp(start_date)) & (
        series.index <= pd.Timestamp(end_date)
    )
    series = series[mask]
    series.name = config["label"]
    return series


# Manifests are shareable artifacts, so a manifest-supplied URL is not
# necessarily trusted input. Without these guards `source: url_csv` is an SSRF
# primitive: it would happily fetch http://169.254.169.254/latest/meta-data/
# or any service on localhost and surface the response in the output Parquet.
_MAX_URL_CSV_BYTES = 64 * 1024 * 1024


def _validate_fetch_url(url: str) -> None:
    """Reject non-HTTP schemes and hosts that resolve to private addresses."""
    import ipaddress
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"[ERROR] Problem: Unsupported URL scheme '{parsed.scheme}'. "
            "Cause: Only http and https are allowed for url_csv sources. "
            "Fix: Use an http(s) URL, or source: file for a local path."
        )
    host = parsed.hostname
    if not host:
        raise ValueError(
            "[ERROR] Problem: URL has no host. "
            "Cause: Malformed url in the manifest. "
            "Fix: Supply a full URL, e.g. https://example.com/data.csv"
        )
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise RuntimeError(
            f"[ERROR] Problem: Could not resolve host '{host}'. "
            f"Cause: {exc}. "
            "Fix: Check the URL is reachable."
        ) from exc

    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
        ):
            raise ValueError(
                f"[ERROR] Problem: Refusing to fetch from '{host}' "
                f"({address}). "
                "Cause: It resolves to a private, loopback or link-local "
                "address — fetching it could expose internal services or cloud "
                "instance metadata. "
                "Fix: Use a public URL, or source: file for local data."
            )


def _fetch_url_csv(config: NodeFetchConfig, start_date: str, end_date: str) -> pd.Series:
    url = config.get("url")
    if not url:
        raise ValueError(
            f"[ERROR] Problem: url missing for url_csv node '{config['label']}'. "
            "Fix: Add url to the manifest node."
        )

    _validate_fetch_url(url)

    date_col = config.get("date_column", "date")
    value_col = config.get("value_column", "value")

    try:
        import requests  # type: ignore[import-untyped]

        response = requests.get(
            url,
            timeout=30,
            allow_redirects=False,
            stream=True,
            headers={"Accept": "text/csv, text/plain, */*"},
        )
        # Cap the body before it is materialised. `response.text` would buffer
        # an arbitrarily large remote file straight into memory.
        body = response.raw.read(_MAX_URL_CSV_BYTES + 1, decode_content=True)
        if len(body) > _MAX_URL_CSV_BYTES:
            raise RuntimeError(
                f"Response exceeds the {_MAX_URL_CSV_BYTES // (1024 * 1024)}MB limit."
            )
        response._content = body  # noqa: SLF001 - keep `.text` consistent
    except Exception as exc:
        raise RuntimeError(
            f"[ERROR] Problem: Could not fetch CSV from URL. "
            f"Cause: {exc}. "
            "Fix: Check URL is accessible."
        ) from exc

    if not response.ok:
        raise RuntimeError(
            f"[ERROR] Problem: Could not fetch CSV from URL. "
            f"Cause: HTTP {response.status_code}. "
            "Fix: Check URL is accessible."
        )

    df = pd.read_csv(
        io.StringIO(response.text),
        parse_dates=[date_col],
        index_col=date_col,
    )

    if value_col not in df.columns:
        raise ValueError(
            f"[ERROR] Problem: Column '{value_col}' not found in CSV from URL. "
            f"Cause: Available columns: {list(df.columns)}. "
            "Fix: Set value_column in manifest to a valid column name."
        )

    series = df[value_col].copy()
    series.index = pd.to_datetime(series.index)
    mask = (series.index >= pd.Timestamp(start_date)) & (
        series.index <= pd.Timestamp(end_date)
    )
    series = series[mask]
    series.name = config["label"]
    return series


# ---------------------------------------------------------------------------
# Fetcher registry
# ---------------------------------------------------------------------------

_FETCHERS: dict[str, _FetcherFn] = {
    "fred": _fetch_fred,
    "yahoo": _fetch_yahoo,
    "file": _fetch_file,
    "url_csv": _fetch_url_csv,
}


# ---------------------------------------------------------------------------
# FinancialDataAdapter
# ---------------------------------------------------------------------------


class FinancialDataAdapter(AdapterBase):
    """Domain adapter for financial time-series data.

    Reads a YAML manifest file that maps DAGVersion node labels to real data
    sources (FRED, Yahoo Finance, local file, or direct-URL CSV).

    BacktestAgent integration is deferred to v2 — compute_dag_score returns
    0.0 and outcome_column raises NotImplementedError.  check_can_backtest()
    in BacktestAgent will block invocation before either method is reached.
    """

    @property
    def domain_label(self) -> str:
        return "Financial Markets"

    @property
    def adapter_type(self) -> "AdapterType":
        from ..models.network import AdapterType
        return AdapterType.Financial

    @property
    def node_metadata_schema(self) -> NodeMetadataSchema:
        return NodeMetadataSchema({
            "source": "Data source type (fred | yahoo | file | url_csv)",
            "series_id": "FRED series ID (for source: fred)",
            "ticker": "Yahoo Finance ticker symbol (for source: yahoo)",
            "path": "Local file path (for source: file)",
            "url": "Direct CSV URL (for source: url_csv)",
            "transform": "Transform to apply (diff | log_return | none)",
        })

    # ------------------------------------------------------------------
    # Internal — used by DatasetBuilder for full pipeline access
    # ------------------------------------------------------------------

    def _load_manifest_data(
        self,
        manifest_path: str | Path,
        strict: bool = False,
    ) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
        """Parse manifest, fetch all series, resample, apply declared transforms.

        Returns:
            (df, transform_map, warnings) where transform_map is
            {label: declared_transform_string} and warnings is a list of
            [WARN] strings accumulated during loading.
        """
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "[ERROR] Problem: PyYAML not installed. "
                "Cause: pyyaml is required for manifest parsing. "
                "Fix: pip install 'causal-hypothesis-engine[financial]'"
            ) from exc

        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"[ERROR] Problem: Manifest file not found. "
                f"Cause: {manifest_path} does not exist. "
                "Fix: Check the manifest path."
            )

        with open(manifest_path) as fh:
            manifest: dict = yaml.safe_load(fh)

        frequency: str = manifest.get("frequency", "weekly")
        resample_method: str = manifest.get("resample_method", "last")
        start_date: str = manifest["start_date"]
        end_date: str = manifest.get(
            "end_date", str(pd.Timestamp.today().date())
        )
        nodes: list[dict] = manifest.get("nodes", [])

        series_map: dict[str, pd.Series] = {}
        transform_map: dict[str, str] = {}
        warnings: list[str] = []

        for node_cfg in nodes:
            label: str = node_cfg["label"]
            source: str = node_cfg["source"]
            transform: str = node_cfg.get("transform", "none")
            transform_map[label] = transform

            fetcher = _FETCHERS.get(source)
            if fetcher is None:
                msg = (
                    f"[WARN] Unknown source '{source}' for node '{label}' — skipped."
                )
                warnings.append(msg)
                logger.warning(msg)
                continue

            try:
                raw = fetcher(node_cfg, start_date, end_date)  # type: ignore[arg-type]
            except Exception as exc:
                raise RuntimeError(str(exc)) from exc

            raw = _resample(raw, frequency, resample_method)
            raw = _apply_transform(raw, transform)
            series_map[label] = raw

        if not series_map:
            raise ValueError(
                "[ERROR] Problem: No data fetched. "
                "Cause: All nodes failed or the manifest has no nodes."
            )

        df = pd.DataFrame(series_map)
        df.index.name = "date"
        df = df.dropna(how="all")

        return df, transform_map, warnings

    # ------------------------------------------------------------------
    # AdapterBase interface
    # ------------------------------------------------------------------

    def load_data(self, path: str | Path) -> pd.DataFrame:
        """Load a financial dataset from a YAML manifest file.

        *path* is the path to a manifest YAML file (not a CSV/Parquet file).
        Overrides the base class file-based loader.
        """
        df, _, _ = self._load_manifest_data(path)
        return df

    def validate_data(self, df: pd.DataFrame) -> list[str]:
        errors: list[str] = []
        if df.empty:
            errors.append("DataFrame is empty — no data was fetched.")
        return errors

    def build_proxy_features(
        self,
        df: pd.DataFrame,
        version: "DAGVersion",
    ) -> pd.DataFrame:
        """Return df columns whose names match DAGVersion node labels.

        Columns in df that have no matching node, and nodes with no column,
        both emit [WARN] log messages.
        """
        version_labels = {n.label for n in version.nodes}
        matching = [col for col in df.columns if col in version_labels]
        for label in version_labels - set(df.columns):
            logger.warning(
                "[WARN] Node '%s' in DAGVersion has no column in the dataset.", label
            )
        return df[matching]

    def outcome_column(self, df: pd.DataFrame) -> str:  # type: ignore[override]
        raise NotImplementedError(
            "FinancialDataAdapter does not define an outcome column in v1. "
            "BacktestAgent integration is deferred. "
            "check_can_backtest() blocks invocation before this is reached."
        )

    def compute_baseline_score(
        self,
        df: pd.DataFrame,
        outcome_col: str,
    ) -> float:
        logger.warning(
            "FinancialDataAdapter.compute_baseline_score is not implemented in v1. "
            "Returning 0.0."
        )
        return 0.0

    def compute_dag_score(
        self,
        df: pd.DataFrame,
        proxy_features: pd.DataFrame,
        outcome_col: str,
    ) -> tuple[float, dict[str, float]]:
        logger.warning(
            "FinancialDataAdapter.compute_dag_score is not implemented in v1. "
            "BacktestAgent integration is deferred. Returning 0.0."
        )
        return 0.0, {}

    def describe_lift(self) -> str:
        return (
            "Predictive lift of DAG-derived financial features "
            "over a baseline model (v2, not yet implemented)."
        )
