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
Baseline:  logistic regression on [claim_type_encoded]
DAG score: baseline features + proxy features from measured nodes
Lift:      dag_score − baseline_score  (out-of-fold AUC; higher is better)

Two deliberate choices, both corrections of earlier behaviour:

*claim_amount is NOT a baseline feature.* The outcome ``is_large_claim`` is
defined as ``claim_amount > threshold``, so including the amount hands the
model a deterministic function of the target. Baseline AUC went to ~1.0 and
lift was mechanically pinned at ~0 regardless of how good the DAG was.

*scikit-learn is a hard dependency.* There used to be a silent fallback to a
rescaled mean absolute Pearson correlation whenever sklearn was missing, while
the UI continued to call the number "AUC". The two are not comparable and not
even monotonically related — measured on identical data, AUC 0.6852 against
fallback 0.6526. Reporting the wrong metric under the right name is worse than
failing, so the fallback is gone and the metric is recorded on the result.
"""

from __future__ import annotations

import logging
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

# Columns that are the outcome, or deterministic functions of it. Using any of
# these as a proxy is leakage dressed up as a causal finding: the tool would
# report a descendant of the outcome as the strongest signal in the graph and
# then promote it. Rejected outright in build_proxy_features.
OUTCOME_DERIVED_COLUMNS = {"is_large_claim", "claim_amount"}

# Number of cross-validation folds, and the minimum number of observations in
# the rarer class required before scoring is meaningful. Previously the fold
# count was derived from the positive-class count, which silently changed the
# estimator's variance with class balance and made two scores incomparable.
CV_FOLDS = 5
MIN_MINORITY_CLASS = 10
RANDOM_SEED = 42

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
        """Return blocking validation errors (empty = usable).

        Unrecognised ``claim_type`` values are deliberately *not* an error.
        Real books contain escape_of_water, subsidence, theft, liability and
        dozens more; hard-failing on them made the adapter unusable with any
        genuine extract. Unknown types are folded into "other" at feature-build
        time and reported through :meth:`data_warnings`.
        """
        errors: list[str] = []
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            errors.append(f"Missing required columns: {sorted(missing)}")
        if "is_large_claim" in df.columns:
            if not df["is_large_claim"].isin([0, 1, True, False]).all():
                errors.append("is_large_claim must contain only 0/1 or True/False values.")
        return errors

    def data_warnings(self, df: pd.DataFrame) -> list[str]:
        """Non-blocking data-quality observations."""
        warnings: list[str] = []
        if "claim_type" in df.columns:
            unknown = sorted(set(df["claim_type"].dropna().unique()) - set(CLAIM_TYPES))
            if unknown:
                warnings.append(
                    f"Unrecognised claim_type value(s) mapped to 'other': "
                    f"{unknown}. Recognised types: {CLAIM_TYPES}"
                )
        if "claim_amount" in df.columns:
            negative = int((df["claim_amount"] < 0).sum())
            if negative:
                warnings.append(
                    f"{negative} row(s) have a negative claim_amount (recoveries, "
                    "salvage, subrogation or reserve reversals). These are "
                    "retained as-is rather than clipped to zero."
                )
        return warnings

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def build_proxy_features(
        self,
        df: pd.DataFrame,
        version: "DAGVersion",
    ) -> pd.DataFrame:
        """Build proxy feature columns from every measured node in *version*.

        "Measured" means ``Proxied`` **or** ``Validated``. Filtering on
        ``Proxied`` alone created a perverse incentive: promoting a node to
        ``Validated`` — the reward for a proxy proving useful — silently
        dropped it from the feature set, so the next backtest's lift moved for
        reasons unrelated to the graph.

        Columns that are the outcome or a deterministic function of it are
        refused: scoring them would report a descendant of the outcome as the
        strongest signal in the graph.

        A proxy column absent from *df* is skipped entirely rather than filled
        with NaN. Zero-filling a missing column made "this column isn't in your
        file" indistinguishable from "we measured this and it doesn't help",
        and in this domain 0 is a meaningful value — ``rainfall_mm = 0`` means
        no rain, ``distance_to_coast = 0`` means on the coast.
        """
        feature_cols: dict[str, pd.Series] = {}
        self._feature_warnings = []

        measured = (MeasurabilityState.Proxied, MeasurabilityState.Validated)
        for node in version.nodes:
            if node.measurability_state not in measured:
                continue
            if not node.adapter_metadata:
                continue
            proxy_vars: list[str] = node.adapter_metadata.get("proxy_variables", [])
            for col in proxy_vars:
                if col in OUTCOME_DERIVED_COLUMNS:
                    self._feature_warnings.append(
                        f"Refused proxy '{col}' for node '{node.label}': it is the "
                        "outcome or a deterministic function of it. Scoring it "
                        "would report leakage as causal signal."
                    )
                    logger.warning(
                        "Refused outcome-derived proxy '%s' for node '%s'.",
                        col, node.label,
                    )
                    continue
                if col in df.columns:
                    feature_cols[f"{node.id}__{col}"] = df[col].copy()
                else:
                    self._feature_warnings.append(
                        f"Proxy column '{col}' for node '{node.label}' is not in "
                        "the data file — node excluded from scoring. Check for a "
                        "typo before concluding it has no signal."
                    )
                    logger.warning(
                        "Proxy column '%s' for node '%s' not found in data; "
                        "excluding from feature matrix.",
                        col, node.label,
                    )

        if not feature_cols:
            return pd.DataFrame(index=df.index)
        return pd.DataFrame(feature_cols, index=df.index)

    @property
    def feature_warnings(self) -> list[str]:
        """Warnings raised during the most recent :meth:`build_proxy_features`."""
        return list(getattr(self, "_feature_warnings", []))

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
        """Score on baseline features: claim_type (one-hot) only."""
        y = df[outcome_col].astype(int)
        X = self._baseline_features(df)
        return self._score(X, y)

    def compute_dag_score(
        self,
        df: pd.DataFrame,
        proxy_features: pd.DataFrame,
        outcome_col: str,
    ) -> tuple[float, dict[str, float]]:
        """Score on baseline + proxy features; also compute per-node contribution.

        Retained for interface compatibility. :meth:`score_detail` returns the
        same numbers plus the provenance and uncertainty needed to interpret
        them, and is what :class:`BacktestAgent` calls.
        """
        detail = self.score_detail(df, proxy_features, outcome_col)
        return detail["dag_score"], detail["node_contributions"]

    def score_detail(
        self,
        df: pd.DataFrame,
        proxy_features: pd.DataFrame,
        outcome_col: str,
    ) -> dict:
        """Full scoring pass: scores, per-node contributions, and intervals.

        Bootstrap intervals are the point of this method. A leave-one-group-out
        score delta reported as a bare number and coloured by its sign is a
        hypothesis test with an implicit alpha of 0.5 — a node with a true
        contribution of exactly zero is declared "signal" half the time, so a
        five-node graph of pure noise yields at least one false discovery
        ~97% of the time. Reporting an interval makes that visible.
        """
        y = df[outcome_col].astype(int)
        X_base = self._baseline_features(df)
        baseline = self._score(X_base, y)

        if proxy_features.empty:
            return {
                "baseline_score": baseline,
                "dag_score": baseline,
                "lift": 0.0,
                "lift_ci": (None, None),
                "node_contributions": {},
                "node_detail": {},
                "metric_name": self.metric_name,
                "n_rows": int(len(df)),
                "n_positive": int(y.sum()),
                "n_features_baseline": int(X_base.shape[1]),
                "n_features_dag": int(X_base.shape[1]),
                "cv_scheme": self.cv_scheme,
            }

        X_full = pd.concat([X_base, proxy_features], axis=1)
        dag_score = self._score(X_full, y)

        # Group proxy feature columns back to their originating node.
        node_col_groups: dict[str, list[str]] = {}
        for col in proxy_features.columns:
            node_col_groups.setdefault(col.split("__")[0], []).append(col)

        node_contributions: dict[str, float] = {}
        for node_id, cols in node_col_groups.items():
            score_without = self._score(X_full.drop(columns=cols), y)
            node_contributions[node_id] = round(dag_score - score_without, 6)

        lift_ci = self._bootstrap_delta_ci(X_base, X_full, y)
        node_detail = {
            node_id: self._bootstrap_delta_ci(
                X_full.drop(columns=cols), X_full, y
            )
            for node_id, cols in node_col_groups.items()
        }

        return {
            "baseline_score": baseline,
            "dag_score": dag_score,
            "lift": round(dag_score - baseline, 6),
            "lift_ci": lift_ci,
            "node_contributions": node_contributions,
            "node_detail": node_detail,
            "metric_name": self.metric_name,
            "n_rows": int(len(df)),
            "n_positive": int(y.sum()),
            "n_features_baseline": int(X_base.shape[1]),
            "n_features_dag": int(X_full.shape[1]),
            "cv_scheme": self.cv_scheme,
        }

    @property
    def metric_name(self) -> str:
        return "out-of-fold ROC AUC"

    @property
    def cv_scheme(self) -> str:
        return f"StratifiedKFold(n_splits={CV_FOLDS}, shuffle=True, seed={RANDOM_SEED})"

    def describe_lift(self) -> str:
        return (
            f"Lift is the increase in {self.metric_name} when DAG-derived proxy "
            "features are added to a baseline logistic model predicting large "
            "claims. Intervals are percentile bootstrap over rows; a lift whose "
            "interval spans zero is not distinguishable from no effect."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # Fixed one-hot category set so the baseline feature matrix has the same
    # shape regardless of which claim types happen to appear in a given file.
    def _baseline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode claim_type over a fixed category set.

        ``claim_amount`` is deliberately excluded. The outcome is defined as
        ``claim_amount > threshold``, so including the amount gives the
        baseline model the answer — baseline AUC approaches 1.0 and lift is
        pinned near zero no matter how good the DAG is.
        """
        if "claim_type" not in df.columns:
            return pd.DataFrame(index=df.index)

        raw = df["claim_type"].fillna("other")
        # Fold unrecognised types into "other" rather than rejecting the file.
        folded = raw.where(raw.isin(CLAIM_TYPES), "other")
        categorical = pd.Categorical(folded, categories=CLAIM_TYPES)
        dummies = pd.get_dummies(categorical, prefix="ct", dtype=float)
        dummies.index = df.index
        return dummies.fillna(0)

    @staticmethod
    def _build_estimator():
        """Logistic regression with scaling *inside* the pipeline.

        Scaling before cross-validation leaks each test fold's mean and
        variance into its own training set. Wrapping it in a Pipeline makes the
        scaler re-fit per fold, which is the only correct arrangement.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
            ]
        )

    @classmethod
    def _score(cls, X: pd.DataFrame, y: pd.Series) -> float:
        """Return pooled out-of-fold ROC AUC.

        Raises ``ValueError`` rather than degrading to a different metric when
        the data cannot support a meaningful score.
        """
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict

        if X.empty or X.shape[1] == 0:
            return 0.5
        if y.nunique() < 2:
            raise ValueError(
                "Problem: Cannot score — the outcome column has only one class.\n"
                "  Cause: every row has the same is_large_claim value.\n"
                "  Fix: supply data containing both large and non-large claims."
            )

        minority = int(min(y.sum(), len(y) - y.sum()))
        if minority < MIN_MINORITY_CLASS:
            raise ValueError(
                f"Problem: Cannot score — only {minority} row(s) in the rarer "
                f"outcome class.\n"
                f"  Cause: {CV_FOLDS}-fold cross-validation needs at least "
                f"{MIN_MINORITY_CLASS} to produce a stable estimate.\n"
                "  Fix: supply more data, or lower the large-claim threshold."
            )

        cv = StratifiedKFold(
            n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED
        )
        probs = cross_val_predict(
            cls._build_estimator(),
            X.fillna(0.0),
            y,
            cv=cv,
            method="predict_proba",
        )[:, 1]
        return float(roc_auc_score(y, probs))

    @classmethod
    def _bootstrap_delta_ci(
        cls,
        X_without: pd.DataFrame,
        X_with: pd.DataFrame,
        y: pd.Series,
        n_boot: int = 200,
        alpha: float = 0.05,
    ) -> tuple[float | None, float | None]:
        """Percentile bootstrap interval for the AUC delta (with − without).

        Both models are scored on the *same* resampled rows so the two AUCs are
        paired, which is what makes the delta's interval meaningful. Returns
        ``(None, None)`` if too few resamples were scoreable.
        """
        import numpy as np
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict

        try:
            cv = StratifiedKFold(
                n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED
            )
            probs_with = cross_val_predict(
                cls._build_estimator(), X_with.fillna(0.0), y, cv=cv,
                method="predict_proba",
            )[:, 1]
            probs_without = cross_val_predict(
                cls._build_estimator(), X_without.fillna(0.0), y, cv=cv,
                method="predict_proba",
            )[:, 1]
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Bootstrap interval unavailable: %s", exc)
            return (None, None)

        y_arr = y.to_numpy()
        rng = np.random.default_rng(RANDOM_SEED)
        n = len(y_arr)
        deltas: list[float] = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            y_b = y_arr[idx]
            if len(np.unique(y_b)) < 2:
                continue
            try:
                deltas.append(
                    roc_auc_score(y_b, probs_with[idx])
                    - roc_auc_score(y_b, probs_without[idx])
                )
            except ValueError:  # pragma: no cover - degenerate resample
                continue

        if len(deltas) < n_boot // 4:
            return (None, None)
        low = float(np.percentile(deltas, 100 * alpha / 2))
        high = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
        return (round(low, 6), round(high, 6))
