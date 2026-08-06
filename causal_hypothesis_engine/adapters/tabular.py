"""TabularAdapter — domain-agnostic adapter for any CSV or Parquet file.

The engine described itself as domain-agnostic while offering only an
insurance adapter that hard-required ``claim_id``, ``claim_type``,
``claim_amount`` and ``is_large_claim``. Any other dataset was rejected at the
door, so the claim held at the model layer and collapsed at the data layer.

This adapter imposes no schema. You declare which column is the outcome — on
the Outcome node's ``adapter_metadata``, or via ``causal-engine backtest
--outcome`` — and every measured node contributes its bound columns. Binary
outcomes are scored by AUC, continuous ones by R²; the metric that ran is
recorded on the result.

The baseline is deliberately empty: with no domain knowledge there is no
principled set of "free" covariates, so lift is measured against the
no-information score (0.5 for AUC, 0.0 for R²). That makes lift here mean
"how much do the DAG's proxies explain", which is the honest reading.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from ..models.network import AdapterType
from ..models.node import NodeType
from ..scoring import ScoringError, build_proxy_matrix, score_nested
from .base import AdapterBase, NodeMetadataSchema

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..models.dag_version import DAGVersion

logger = logging.getLogger(__name__)


class TabularAdapter(AdapterBase):
    """Schema-free adapter for arbitrary tabular data."""

    def __init__(self, outcome_column_name: str | None = None) -> None:
        self._outcome_column = outcome_column_name
        self._feature_warnings: list[str] = []

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def domain_label(self) -> str:
        return "Tabular (generic)"

    @property
    def adapter_type(self) -> AdapterType:
        return AdapterType.Tabular

    @property
    def node_metadata_schema(self) -> NodeMetadataSchema:
        return NodeMetadataSchema(
            {
                "proxy_variables": (
                    "List of column names in the data file that measure this node"
                ),
                "outcome_column": (
                    "On the Outcome node: the column to predict. Optional if "
                    "the node's proxy_variables names it."
                ),
            }
        )

    # ------------------------------------------------------------------
    # Data loading and validation
    # ------------------------------------------------------------------

    def load_data(self, path: str | Path) -> pd.DataFrame:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(
                f"[ERROR] Problem: Data file not found.\n"
                f"  Cause: {file_path} does not exist.\n"
                "  Fix: Check the --data path."
            )
        if file_path.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(file_path)
        return pd.read_csv(file_path)

    def validate_data(self, df: pd.DataFrame) -> list[str]:
        errors: list[str] = []
        if df.empty:
            errors.append("Data file contains no rows.")
        if df.shape[1] == 0:
            errors.append("Data file contains no columns.")
        return errors

    def data_warnings(self, df: pd.DataFrame) -> list[str]:
        warnings: list[str] = []
        all_null = [c for c in df.columns if df[c].isna().all()]
        if all_null:
            warnings.append(f"Column(s) entirely missing: {all_null}")
        return warnings

    # ------------------------------------------------------------------
    # Outcome resolution
    # ------------------------------------------------------------------

    def resolve_outcome(self, df: pd.DataFrame, version: "DAGVersion") -> str:
        """Determine the outcome column for *version*.

        Resolution order: an explicit ``--outcome``; ``outcome_column`` on any
        node's metadata; the Outcome node's first bound proxy column; the
        Outcome node's label if it matches a column.
        """
        if self._outcome_column:
            if self._outcome_column not in df.columns:
                raise ScoringError(
                    f"Problem: Outcome column '{self._outcome_column}' is not "
                    "in the data file.\n"
                    f"  Cause: Available columns are {list(df.columns)}.\n"
                    "  Fix: Pass --outcome with a column that exists."
                )
            return self._outcome_column

        for node in version.nodes:
            declared = (node.adapter_metadata or {}).get("outcome_column")
            if declared and declared in df.columns:
                self._outcome_column = declared
                return declared

        outcome_nodes = [n for n in version.nodes if n.node_type == NodeType.Outcome]
        for node in outcome_nodes:
            for name in (node.adapter_metadata or {}).get("proxy_variables", []):
                if name in df.columns:
                    self._outcome_column = name
                    return name
            if node.label in df.columns:
                self._outcome_column = node.label
                return node.label

        raise ScoringError(
            "Problem: Cannot determine which column is the outcome.\n"
            "  Cause: No --outcome given, and the DAG's Outcome node is not "
            "bound to a column present in the data.\n"
            "  Fix: Pass --outcome <column>, or bind the Outcome node with "
            "causal-engine bind <version> --node <label> --column <column>."
        )

    def outcome_column(self, df: pd.DataFrame) -> str:
        if self._outcome_column and self._outcome_column in df.columns:
            return self._outcome_column
        raise ScoringError(
            "Problem: Outcome column has not been resolved.\n"
            "  Cause: TabularAdapter needs an outcome before scoring.\n"
            "  Fix: Pass --outcome <column>."
        )

    # ------------------------------------------------------------------
    # Features and scoring
    # ------------------------------------------------------------------

    def build_proxy_features(
        self, df: pd.DataFrame, version: "DAGVersion"
    ) -> pd.DataFrame:
        outcome = self._outcome_column or ""
        features, warnings = build_proxy_matrix(
            df, version, forbidden_columns={outcome} if outcome else set()
        )
        self._feature_warnings = warnings
        for warning in warnings:
            logger.warning(warning)
        # Non-numeric proxies cannot enter a linear model; one-hot them.
        non_numeric = [
            c for c in features.columns
            if not pd.api.types.is_numeric_dtype(features[c])
        ]
        if non_numeric:
            features = pd.get_dummies(
                features, columns=non_numeric, dtype=float, prefix_sep="="
            )
        return features

    @property
    def feature_warnings(self) -> list[str]:
        return list(self._feature_warnings)

    def _baseline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Empty by design — see the module docstring."""
        return pd.DataFrame(index=df.index)

    def compute_baseline_score(self, df: pd.DataFrame, outcome_col: str) -> float:
        from ..scoring import is_binary

        return 0.5 if is_binary(df[outcome_col]) else 0.0

    def compute_dag_score(
        self, df: pd.DataFrame, proxy_features: pd.DataFrame, outcome_col: str
    ) -> tuple[float, dict[str, float]]:
        detail = self.score_detail(df, proxy_features, outcome_col)
        return detail["dag_score"], detail["node_contributions"]

    def score_detail(
        self, df: pd.DataFrame, proxy_features: pd.DataFrame, outcome_col: str
    ) -> dict:
        y = df[outcome_col]
        outcome = score_nested(self._baseline_features(df), proxy_features, y)
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
            "Lift is how much the DAG's bound proxy columns improve prediction "
            "of the outcome over a no-information baseline, measured "
            "out-of-fold. Intervals are paired percentile bootstrap; a lift "
            "whose interval spans zero is not distinguishable from no effect."
        )
