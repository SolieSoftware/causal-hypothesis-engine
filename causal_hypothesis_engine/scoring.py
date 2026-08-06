"""Shared cross-validated scoring for all adapters.

Every adapter that evaluates a DAG against data uses this module, so the
guarantees hold in one place rather than being re-derived (and re-broken) per
adapter:

* the metric that is reported is the metric that was computed, and its name
  travels with the number;
* preprocessing happens inside the cross-validation folds, never before them;
* the fold count is a property of the design, not of the class balance;
* every score delta carries a paired bootstrap interval, because a delta
  rendered by its sign alone is a hypothesis test with an implicit alpha of
  0.5 — a graph of five null nodes then yields a false "signal" about 97% of
  the time.

Binary outcomes are scored by pooled out-of-fold ROC AUC; continuous outcomes
by out-of-fold R². Both are "higher is better" and both have a meaningful zero
for the *delta*, which is what the engine actually reports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

CV_FOLDS = 5
MIN_MINORITY_CLASS = 10
MIN_ROWS_CONTINUOUS = 30
RANDOM_SEED = 42
BOOTSTRAP_RESAMPLES = 200


class ScoringError(ValueError):
    """Raised when the data cannot support a meaningful score.

    Deliberately an error rather than a degraded number: silently substituting
    a different, weaker statistic is what made earlier results uninterpretable.
    """


@dataclass
class ScoreOutcome:
    """A scored comparison of two nested models."""

    baseline_score: float
    dag_score: float
    lift: float
    metric_name: str
    cv_scheme: str
    n_rows: int
    n_positive: int
    n_features_baseline: int
    n_features_dag: int
    lift_ci: tuple[float | None, float | None] = (None, None)
    node_contributions: dict[str, float] = field(default_factory=dict)
    node_detail: dict[str, tuple[float | None, float | None]] = field(
        default_factory=dict
    )


def is_binary(y: pd.Series) -> bool:
    """True when *y* has exactly two distinct non-null values."""
    return int(y.dropna().nunique()) == 2


def metric_name_for(y: pd.Series) -> str:
    return "out-of-fold ROC AUC" if is_binary(y) else "out-of-fold R²"


def cv_scheme_for(y: pd.Series) -> str:
    kind = "StratifiedKFold" if is_binary(y) else "KFold"
    return f"{kind}(n_splits={CV_FOLDS}, shuffle=True, seed={RANDOM_SEED})"


def _build_estimator(binary: bool):
    """Model with preprocessing inside the pipeline.

    Fitting a scaler before cross-validation leaks each test fold's mean and
    variance into its own training set. A Pipeline re-fits it per fold, which
    is the only correct arrangement.
    """
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = (
        LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        if binary
        else LinearRegression()
    )
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def _splitter(binary: bool):
    from sklearn.model_selection import KFold, StratifiedKFold

    cls = StratifiedKFold if binary else KFold
    return cls(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)


def _check_scoreable(y: pd.Series) -> None:
    if y.nunique() < 2:
        raise ScoringError(
            "Problem: Cannot score — the outcome column has only one distinct "
            "value.\n"
            "  Cause: There is nothing to predict.\n"
            "  Fix: Supply data with variation in the outcome."
        )
    if is_binary(y):
        minority = int(min(y.sum(), len(y) - y.sum()))
        if minority < MIN_MINORITY_CLASS:
            raise ScoringError(
                f"Problem: Cannot score — only {minority} row(s) in the rarer "
                f"outcome class.\n"
                f"  Cause: {CV_FOLDS}-fold cross-validation needs at least "
                f"{MIN_MINORITY_CLASS} for a stable estimate.\n"
                "  Fix: Supply more data, or choose a less extreme threshold."
            )
    elif len(y) < MIN_ROWS_CONTINUOUS:
        raise ScoringError(
            f"Problem: Cannot score — only {len(y)} row(s).\n"
            f"  Cause: {CV_FOLDS}-fold cross-validation on a continuous "
            f"outcome needs at least {MIN_ROWS_CONTINUOUS}.\n"
            "  Fix: Supply more data, or widen the date range."
        )


def _out_of_fold_predictions(X: pd.DataFrame, y: pd.Series):
    from sklearn.model_selection import cross_val_predict

    binary = is_binary(y)
    estimator = _build_estimator(binary)
    cv = _splitter(binary)
    if binary:
        return cross_val_predict(
            estimator, X.fillna(0.0), y, cv=cv, method="predict_proba"
        )[:, 1]
    return cross_val_predict(estimator, X.fillna(0.0), y, cv=cv)


def _metric(y, predictions) -> float:
    from sklearn.metrics import r2_score, roc_auc_score

    if is_binary(pd.Series(y)):
        return float(roc_auc_score(y, predictions))
    return float(r2_score(y, predictions))


def score(X: pd.DataFrame, y: pd.Series) -> float:
    """Return the out-of-fold score of *X* against *y*.

    An empty feature matrix scores at the no-information baseline: 0.5 for
    AUC, 0.0 for R².
    """
    if X.empty or X.shape[1] == 0:
        return 0.5 if is_binary(y) else 0.0
    _check_scoreable(y)
    return _metric(y, _out_of_fold_predictions(X, y))


def _bootstrap_delta_ci(
    X_without: pd.DataFrame,
    X_with: pd.DataFrame,
    y: pd.Series,
    alpha: float = 0.05,
) -> tuple[float | None, float | None]:
    """Percentile bootstrap interval for the score delta (with − without).

    Both models are scored on the *same* resampled rows, so the two scores are
    paired — which is what makes the interval on their difference meaningful.
    """
    import numpy as np

    try:
        preds_with = _out_of_fold_predictions(X_with, y)
        preds_without = (
            _out_of_fold_predictions(X_without, y)
            if not X_without.empty and X_without.shape[1] > 0
            else None
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Bootstrap interval unavailable: %s", exc)
        return (None, None)

    y_arr = y.to_numpy()
    binary = is_binary(y)
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(y_arr)
    deltas: list[float] = []

    for _ in range(BOOTSTRAP_RESAMPLES):
        idx = rng.integers(0, n, n)
        y_b = y_arr[idx]
        if binary and len(np.unique(y_b)) < 2:
            continue
        try:
            with_score = _metric(y_b, preds_with[idx])
            without_score = (
                _metric(y_b, preds_without[idx])
                if preds_without is not None
                else (0.5 if binary else 0.0)
            )
            deltas.append(with_score - without_score)
        except ValueError:  # pragma: no cover - degenerate resample
            continue

    if len(deltas) < BOOTSTRAP_RESAMPLES // 4:
        return (None, None)
    return (
        round(float(np.percentile(deltas, 100 * alpha / 2)), 6),
        round(float(np.percentile(deltas, 100 * (1 - alpha / 2))), 6),
    )


def score_nested(
    X_base: pd.DataFrame,
    proxy_features: pd.DataFrame,
    y: pd.Series,
) -> ScoreOutcome:
    """Score baseline vs baseline+proxies, with per-node contributions.

    Proxy feature columns are expected to be named ``{node_id}__{column}`` so
    contributions can be grouped back to the node that produced them.
    """
    _check_scoreable(y)
    baseline = score(X_base, y)
    metric = metric_name_for(y)
    scheme = cv_scheme_for(y)
    n_positive = int(y.sum()) if is_binary(y) else 0

    if proxy_features.empty or proxy_features.shape[1] == 0:
        return ScoreOutcome(
            baseline_score=baseline,
            dag_score=baseline,
            lift=0.0,
            metric_name=metric,
            cv_scheme=scheme,
            n_rows=int(len(y)),
            n_positive=n_positive,
            n_features_baseline=int(X_base.shape[1]),
            n_features_dag=int(X_base.shape[1]),
        )

    X_full = pd.concat([X_base, proxy_features], axis=1)
    dag_score = score(X_full, y)

    groups: dict[str, list[str]] = {}
    for column in proxy_features.columns:
        groups.setdefault(str(column).split("__")[0], []).append(column)

    contributions: dict[str, float] = {}
    detail: dict[str, tuple[float | None, float | None]] = {}
    for node_id, columns in groups.items():
        reduced = X_full.drop(columns=columns)
        contributions[node_id] = round(dag_score - score(reduced, y), 6)
        detail[node_id] = _bootstrap_delta_ci(reduced, X_full, y)

    return ScoreOutcome(
        baseline_score=baseline,
        dag_score=dag_score,
        lift=round(dag_score - baseline, 6),
        metric_name=metric,
        cv_scheme=scheme,
        n_rows=int(len(y)),
        n_positive=n_positive,
        n_features_baseline=int(X_base.shape[1]),
        n_features_dag=int(X_full.shape[1]),
        lift_ci=_bootstrap_delta_ci(X_base, X_full, y),
        node_contributions=contributions,
        node_detail=detail,
    )


def build_proxy_matrix(
    df: pd.DataFrame,
    version,
    forbidden_columns: set[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Collect proxy columns for every measured node in *version*.

    Returns ``(features, warnings)``. "Measured" is Proxied **or** Validated:
    filtering on Proxied alone meant promoting a node to Validated silently
    removed it from scoring.

    A column in *forbidden_columns* — the outcome or a deterministic function
    of it — is refused, and a proxy column absent from *df* is skipped rather
    than zero-filled, so "not in your file" stays distinguishable from "no
    signal".
    """
    from .models.node import MeasurabilityState

    forbidden = forbidden_columns or set()
    measured = (MeasurabilityState.Proxied, MeasurabilityState.Validated)
    columns: dict[str, pd.Series] = {}
    warnings: list[str] = []

    for node in version.nodes:
        if node.measurability_state not in measured or not node.adapter_metadata:
            continue
        for name in node.adapter_metadata.get("proxy_variables", []):
            if name in forbidden:
                warnings.append(
                    f"Refused proxy '{name}' for node '{node.label}': it is the "
                    "outcome or a deterministic function of it."
                )
                continue
            if name in df.columns:
                columns[f"{node.id}__{name}"] = df[name].copy()
            else:
                warnings.append(
                    f"Proxy column '{name}' for node '{node.label}' is not in "
                    "the data — node excluded from scoring. Check for a typo "
                    "before concluding it has no signal."
                )

    if not columns:
        return pd.DataFrame(index=df.index), warnings
    return pd.DataFrame(columns, index=df.index), warnings
