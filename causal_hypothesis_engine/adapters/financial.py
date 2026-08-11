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

from ..models.node import NodeType
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

    Scoring is implemented: the Outcome node's series is predicted from every
    other node's series, out-of-fold, with paired bootstrap intervals. Note
    what this does and does not establish — the feature matrix is strictly
    contemporaneous, so the result is a predictive association at the same
    timestamp, not a causal effect and not a forecast. Use `causal-engine
    check` to test whether the graph's *structure* survives the data.
    """

    def __init__(self, outcome_column_name: str | None = None) -> None:
        self._outcome_column = outcome_column_name
        self._feature_warnings: list[str] = []

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
        """Load a financial dataset from a manifest *or* a built feature matrix.

        A YAML path is fetched and assembled from source; a ``.parquet`` or
        ``.csv`` path is read directly. Accepting both matters because the
        normal workflow is ``causal-engine dataset`` (which writes Parquet)
        followed by ``causal-engine backtest --data <that parquet>``.
        """
        file_path = Path(path)
        suffix = file_path.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(file_path)
        if suffix == ".csv":
            return pd.read_csv(file_path, index_col=0, parse_dates=True)
        df, _, _ = self._load_manifest_data(file_path)
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
        """Return one column per non-outcome node, keyed ``{node_id}__{column}``.

        A node is matched to a column by its bound ``proxy_variables`` first,
        then by an exact label match — the dataset pipeline names columns after
        node labels, so label matching is the common case here.
        """
        outcome = self._outcome_column or ""
        columns: dict[str, pd.Series] = {}
        warnings: list[str] = []

        for node in version.nodes:
            if node.node_type == NodeType.Outcome:
                continue
            name = self._column_for_node(node, df)
            if name is None:
                warnings.append(
                    f"Node '{node.label}' has no column in the dataset — "
                    "excluded from scoring."
                )
                continue
            if name == outcome:
                continue
            columns[f"{node.id}__{name}"] = df[name].copy()

        self._feature_warnings = warnings
        for warning in warnings:
            logger.warning(warning)

        if not columns:
            return pd.DataFrame(index=df.index)
        return pd.DataFrame(columns, index=df.index)

    @staticmethod
    def _column_for_node(node, df: pd.DataFrame) -> str | None:
        for candidate in (node.adapter_metadata or {}).get("proxy_variables", []):
            if candidate in df.columns:
                return candidate
        if node.label in df.columns:
            return node.label
        return None

    @property
    def feature_warnings(self) -> list[str]:
        return list(getattr(self, "_feature_warnings", []))

    def resolve_outcome(self, df: pd.DataFrame, version: "DAGVersion") -> str:
        """Determine which column is the outcome for *version*.

        Order: an explicit ``--outcome``; ``outcome_column`` declared on any
        node; the Outcome node's bound proxy or matching label.
        """
        from ..scoring import ScoringError

        if self._outcome_column:
            if self._outcome_column not in df.columns:
                raise ScoringError(
                    f"Problem: Outcome column '{self._outcome_column}' is not "
                    "in the dataset.\n"
                    f"  Cause: Available columns are {list(df.columns)}.\n"
                    "  Fix: Pass --outcome with a column that exists."
                )
            return self._outcome_column

        for node in version.nodes:
            declared = (node.adapter_metadata or {}).get("outcome_column")
            if declared and declared in df.columns:
                self._outcome_column = declared
                return declared

        for node in version.nodes:
            if node.node_type != NodeType.Outcome:
                continue
            name = self._column_for_node(node, df)
            if name is not None:
                self._outcome_column = name
                return name

        raise ScoringError(
            "Problem: Cannot determine which column is the outcome.\n"
            "  Cause: No --outcome given, and the DAG's Outcome node has no "
            "matching column in the dataset.\n"
            "  Fix: Pass --outcome <column>, or make sure the Outcome node's "
            "label matches a manifest node label."
        )

    def outcome_column(self, df: pd.DataFrame) -> str:
        from ..scoring import ScoringError

        if self._outcome_column and self._outcome_column in df.columns:
            return self._outcome_column
        raise ScoringError(
            "Problem: Outcome column has not been resolved.\n"
            "  Cause: The financial adapter needs an outcome before scoring.\n"
            "  Fix: Pass --outcome <column>."
        )

    def _baseline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Empty by design.

        There is no set of "free" covariates for an arbitrary macro series, so
        lift is measured against the no-information score. That makes it read
        as "how much do the DAG's series explain the outcome".
        """
        return pd.DataFrame(index=df.index)

    def compute_baseline_score(
        self,
        df: pd.DataFrame,
        outcome_col: str,
    ) -> float:
        from ..scoring import is_binary

        return 0.5 if is_binary(df[outcome_col]) else 0.0

    def compute_dag_score(
        self,
        df: pd.DataFrame,
        proxy_features: pd.DataFrame,
        outcome_col: str,
    ) -> tuple[float, dict[str, float]]:
        detail = self.score_detail(df, proxy_features, outcome_col)
        return detail["dag_score"], detail["node_contributions"]

    def score_detail(
        self,
        df: pd.DataFrame,
        proxy_features: pd.DataFrame,
        outcome_col: str,
    ) -> dict:
        """Score the DAG's series against the outcome series.

        Rows with any missing value are dropped first: the dataset pipeline
        joins series on the union of their indices, so a feature matrix can
        look complete while few rows have every column present.
        """
        from ..scoring import score_nested

        combined = pd.concat([df[[outcome_col]], proxy_features], axis=1).dropna()
        y = combined[outcome_col]
        features = combined.drop(columns=[outcome_col])
        outcome = score_nested(
            pd.DataFrame(index=combined.index), features, y
        )
        return {
            "baseline_score": outcome.baseline_score,
            "dag_score": outcome.dag_score,
            "lift": outcome.lift,
            "lift_ci": outcome.lift_ci,
            "node_contributions": outcome.node_contributions,
            "node_detail": outcome.node_detail,
            "metric_name": outcome.metric_name,
            "n_rows": outcome.n_rows,
            "n_positive": outcome.n_positive,
            "n_features_baseline": outcome.n_features_baseline,
            "n_features_dag": outcome.n_features_dag,
            "cv_scheme": outcome.cv_scheme,
        }

    def describe_lift(self) -> str:
        return (
            "Lift is how much the DAG's series improve out-of-fold prediction "
            "of the outcome series over a no-information baseline. Intervals "
            "are paired percentile bootstrap; a lift whose interval spans zero "
            "is not distinguishable from no effect. Note this measures "
            "contemporaneous predictive association, not a causal effect."
        )
