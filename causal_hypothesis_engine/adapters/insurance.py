"""InsuranceClaimsAdapter — domain adapter for insurance claims data.

Data source (v1): local CSV or Parquet files.
BigQuery integration is deferred to v1.1 (see TODOS.md).

Expected CSV/Parquet schema
---------------------------
Required columns:
  claim_id         TEXT       unique claim identifier
  claim_type       TEXT       e.g. fire | flood | burst_pipe | storm | other
  claim_amount     FLOAT      total incurred cost
  is_large_claim   INT/BOOL   1 if claim_amount > threshold (outcome variable)

Recommended proxy columns (any subset may be present):
  rainfall_mm        FLOAT   rainfall in mm at location (proxy for flood)
  max_wind_speed     FLOAT   maximum wind speed km/h (proxy for storm)
  property_age_years INT     age of insured property
  flood_zone         INT     1 if property is in designated flood zone
  fire_risk_score    FLOAT   0-1 fire risk rating
  burst_pipe_risk    FLOAT   0-1 burst pipe risk rating (e.g. from water hardness)
  distance_to_coast  FLOAT   km to nearest coastline

Proxy variables are mapped via Node.adapter_metadata["proxy_variables"].
Each element is a column name in the DataFrame.

Scoring
-------
Baseline:  logistic regression on [claim_type_encoded, claim_amount_log]
DAG score: baseline features + proxy features from Proxied nodes
Lift:      dag_score − baseline_score  (AUC; higher is better)

If sklearn is not installed, falls back to mean absolute Pearson correlation
with the outcome column as a proxy for predictive power.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from ..models.network import AdapterType
from ..models.node import MeasurabilityState
from .base import AdapterBase, NodeMetadataSchema

if TYPE_CHECKING:
    from ..models.dag_version import DAGVersion

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Claim types recognised by the adapter
# ---------------------------------------------------------------------------

CLAIM_TYPES = ["fire", "flood", "burst_pipe", "storm", "other"]

# Columns the adapter always expects.
REQUIRED_COLUMNS = {"claim_id", "claim_type", "claim_amount", "is_large_claim"}

# Columns the adapter can use as proxy variables.
KNOWN_PROXY_COLUMNS = {
    "rainfall_mm",
    "max_wind_speed",
    "property_age_years",
    "flood_zone",
    "fire_risk_score",
    "burst_pipe_risk",
    "distance_to_coast",
}


# ---------------------------------------------------------------------------
# InsuranceClaimsAdapter
# ---------------------------------------------------------------------------


class InsuranceClaimsAdapter(AdapterBase):
    """Adapter for insurance claims CSV / Parquet data."""

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def domain_label(self) -> str:
        return "Insurance Claims"

    @property
    def adapter_type(self) -> AdapterType:
        return AdapterType.Insurance

    @property
    def node_metadata_schema(self) -> NodeMetadataSchema:
        return NodeMetadataSchema(
            {
                "claim_type": (
                    "Claim category this node relates to "
                    "(fire | flood | burst_pipe | storm | other)"
                ),
                "proxy_variables": (
                    "List of CSV column names that serve as proxies for this node"
                ),
                "data_source": "Path to the CSV / Parquet file used for backtesting",
            }
        )

    # ------------------------------------------------------------------
    # Data loading and validation
    # ------------------------------------------------------------------

    def validate_data(self, df: pd.DataFrame) -> list[str]:
        errors: list[str] = []
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            errors.append(f"Missing required columns: {sorted(missing)}")
        if "claim_type" in df.columns:
            bad = set(df["claim_type"].dropna().unique()) - set(CLAIM_TYPES)
            if bad:
                errors.append(
                    f"Unknown claim_type values: {sorted(bad)}. "
                    f"Expected one of: {CLAIM_TYPES}"
                )
        if "is_large_claim" in df.columns:
            if not df["is_large_claim"].isin([0, 1, True, False]).all():
                errors.append("is_large_claim must contain only 0/1 or True/False values.")
        return errors

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def build_proxy_features(
        self,
        df: pd.DataFrame,
        version: "DAGVersion",
    ) -> pd.DataFrame:
        """Build proxy feature columns from all Proxied nodes in *version*.

        For each node whose measurability_state is Proxied and whose
        adapter_metadata contains proxy_variables, the corresponding columns
        from *df* are included in the returned DataFrame.

        If a proxy column is missing from *df*, a column of NaN is inserted
        and a warning is logged.
        """
        feature_cols: dict[str, pd.Series] = {}

        for node in version.nodes:
            if node.measurability_state != MeasurabilityState.Proxied:
                continue
            if not node.adapter_metadata:
                continue
            proxy_vars: list[str] = node.adapter_metadata.get("proxy_variables", [])
            for col in proxy_vars:
                if col in df.columns:
                    feature_key = f"{node.id}__{col}"
                    feature_cols[feature_key] = df[col].copy()
                else:
                    logger.warning(
                        "Proxy column '%s' for node '%s' not found in data. "
                        "Filling with NaN.",
                        col, node.label,
                    )
                    feature_cols[f"{node.id}__{col}"] = pd.Series(
                        [float("nan")] * len(df), index=df.index
                    )

        if not feature_cols:
            return pd.DataFrame(index=df.index)
        return pd.DataFrame(feature_cols, index=df.index)

    def outcome_column(self, df: pd.DataFrame) -> str:
        return "is_large_claim"

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def compute_baseline_score(
        self,
        df: pd.DataFrame,
        outcome_col: str,
    ) -> float:
        """Score on baseline features: claim_type (one-hot) + log(claim_amount)."""
        y = df[outcome_col].astype(int)
        X = self._baseline_features(df)
        return self._score(X, y)

    def compute_dag_score(
        self,
        df: pd.DataFrame,
        proxy_features: pd.DataFrame,
        outcome_col: str,
    ) -> tuple[float, dict[str, float]]:
        """Score on baseline + proxy features; also compute per-node contribution."""
        y = df[outcome_col].astype(int)
        X_base = self._baseline_features(df)
        baseline = self._score(X_base, y)

        if proxy_features.empty:
            return baseline, {}

        # Full model.
        X_full = pd.concat([X_base, proxy_features.fillna(0)], axis=1)
        dag_score = self._score(X_full, y)

        # Per-node contribution: score with each node's columns dropped.
        node_contributions: dict[str, float] = {}
        # Group proxy feature columns back to their originating node.
        node_col_groups: dict[str, list[str]] = {}
        for col in proxy_features.columns:
            node_id = col.split("__")[0]
            node_col_groups.setdefault(node_id, []).append(col)

        for node_id, cols in node_col_groups.items():
            X_without = X_full.drop(columns=cols)
            score_without = self._score(X_without, y)
            node_contributions[node_id] = round(dag_score - score_without, 6)

        return dag_score, node_contributions

    def describe_lift(self) -> str:
        return (
            "Lift is the increase in AUC (area under the ROC curve) when "
            "DAG-derived proxy features are added to a baseline logistic model "
            "predicting large claims."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _baseline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode claim_type + log1p(claim_amount)."""
        parts: list[pd.DataFrame] = []

        if "claim_type" in df.columns:
            dummies = pd.get_dummies(
                df["claim_type"].fillna("other"), prefix="ct", dtype=float
            )
            parts.append(dummies)

        if "claim_amount" in df.columns:
            log_amt = df["claim_amount"].clip(lower=0).apply(math.log1p)
            parts.append(log_amt.rename("log_claim_amount").to_frame())

        if not parts:
            return pd.DataFrame(index=df.index)
        return pd.concat(parts, axis=1).fillna(0)

    @staticmethod
    def _score(X: pd.DataFrame, y: pd.Series) -> float:
        """Return AUC if sklearn is available, else mean absolute Pearson correlation."""
        if X.empty or len(X.columns) == 0:
            return 0.5

        # Try sklearn first.
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score
            from sklearn.model_selection import cross_val_predict
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X.fillna(0))
            model = LogisticRegression(max_iter=200, random_state=42)
            # Use cross-validation to avoid overfitting on small samples.
            n_splits = min(5, max(2, y.sum()))  # need at least 1 positive per fold
            if n_splits < 2 or y.nunique() < 2:
                return 0.5
            probs = cross_val_predict(
                model, X_scaled, y, cv=int(n_splits), method="predict_proba"
            )[:, 1]
            return float(roc_auc_score(y, probs))

        except ImportError:
            pass
        except Exception as exc:
            logger.warning("sklearn scoring failed (%s); falling back to correlation.", exc)

        # Fallback: mean |Pearson r| across features.
        if y.nunique() < 2:
            return 0.5
        correlations = []
        for col in X.columns:
            series = X[col].fillna(0)
            if series.std() == 0:
                continue
            r = series.corr(y)
            if not math.isnan(r):
                correlations.append(abs(r))
        if not correlations:
            return 0.5
        # Map mean |r| to [0.5, 1.0] to stay in AUC-like range.
        mean_r = sum(correlations) / len(correlations)
        return 0.5 + mean_r * 0.5
