"""Data exploration and DAG falsification.

Two capabilities the engine previously lacked, both required by the goal of
"building and evaluating causal graphs via importing and exploring data":

:func:`profile_dataset`
    Ordinary exploratory statistics — shape, dtypes, missingness, distribution
    summaries, correlation structure. Previously the only data artefact the
    tool rendered was an ADF table, and the documented workflow told analysts
    to open a Python REPL and call ``df.describe()`` themselves.

:func:`test_implications`
    Tests the conditional independencies the DAG *implies* against real data.
    This is the first evaluation in the engine whose result depends on edge
    direction. The backtest does not: its score is computed from a bag of
    columns attached to nodes and never reads ``version.edges``, so reversing
    every arrow leaves the number unchanged. An implication test can actually
    falsify a graph — "your data rejects 4 of the 11 independencies this DAG
    implies" is a statement about the structure, not about predictive power.

Both are deliberately assumption-light. Partial correlation assumes linearity
and joint normality; that is stated in the output rather than hidden, and a
rejection is evidence to investigate, not proof the graph is wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from .graph import testable_implications

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .models.dag_version import DAGVersion


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    n_present: int
    n_missing: int
    pct_missing: float
    mean: float | None = None
    std: float | None = None
    minimum: float | None = None
    median: float | None = None
    maximum: float | None = None
    n_unique: int = 0

    @property
    def is_numeric(self) -> bool:
        return self.mean is not None


@dataclass
class DatasetProfile:
    n_rows: int
    n_columns: int
    n_complete_rows: int
    columns: list[ColumnProfile] = field(default_factory=list)
    correlations: pd.DataFrame | None = None
    high_correlation_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def pct_complete_rows(self) -> float:
        if self.n_rows == 0:
            return 0.0
        return 100.0 * self.n_complete_rows / self.n_rows


def profile_dataset(
    df: pd.DataFrame,
    high_correlation_threshold: float = 0.8,
) -> DatasetProfile:
    """Summarise *df*: shape, missingness, distributions, correlation structure.

    ``n_complete_rows`` matters more than it looks. The dataset pipeline joins
    each series on the union of their indices, so a feature matrix can report a
    healthy row count while very few rows are usable for a model that needs
    every column present.
    """
    columns: list[ColumnProfile] = []
    for name in df.columns:
        series = df[name]
        n_present = int(series.notna().sum())
        n_missing = int(series.isna().sum())
        profile = ColumnProfile(
            name=str(name),
            dtype=str(series.dtype),
            n_present=n_present,
            n_missing=n_missing,
            pct_missing=round(100.0 * n_missing / len(df), 2) if len(df) else 0.0,
            n_unique=int(series.nunique(dropna=True)),
        )
        if pd.api.types.is_numeric_dtype(series) and n_present > 0:
            valid = series.dropna().astype(float)
            profile.mean = float(valid.mean())
            profile.std = float(valid.std()) if n_present > 1 else 0.0
            profile.minimum = float(valid.min())
            profile.median = float(valid.median())
            profile.maximum = float(valid.max())
        columns.append(profile)

    numeric = df.select_dtypes(include="number")
    correlations: pd.DataFrame | None = None
    high_pairs: list[tuple[str, str, float]] = []
    if numeric.shape[1] >= 2:
        correlations = numeric.corr()
        seen: set[frozenset[str]] = set()
        for a in correlations.columns:
            for b in correlations.columns:
                if a == b or frozenset((a, b)) in seen:
                    continue
                seen.add(frozenset((a, b)))
                value = correlations.loc[a, b]
                if pd.notna(value) and abs(value) >= high_correlation_threshold:
                    high_pairs.append((str(a), str(b), round(float(value), 4)))
        high_pairs.sort(key=lambda item: abs(item[2]), reverse=True)

    notes: list[str] = []
    n_complete = int(df.dropna().shape[0])
    if len(df) and n_complete < len(df) * 0.5:
        notes.append(
            f"Only {n_complete} of {len(df)} rows have every column present "
            f"({100.0 * n_complete / len(df):.1f}%). Any model requiring "
            "complete cases will train on a small fraction of this file."
        )
    if high_pairs:
        notes.append(
            f"{len(high_pairs)} column pair(s) correlate at |r| >= "
            f"{high_correlation_threshold}. Leave-one-out contribution scores "
            "are unreliable under collinearity — dropping one of two correlated "
            "columns costs almost nothing, so both look unimportant."
        )
    constant = [c.name for c in columns if c.is_numeric and c.std == 0.0]
    if constant:
        notes.append(
            "Constant column(s) carry no information: " + ", ".join(constant) + "."
        )

    return DatasetProfile(
        n_rows=int(len(df)),
        n_columns=int(df.shape[1]),
        n_complete_rows=n_complete,
        columns=columns,
        correlations=correlations,
        high_correlation_pairs=high_pairs,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# DAG falsification
# ---------------------------------------------------------------------------


@dataclass
class ImplicationTest:
    """One conditional independence the DAG claims, tested against data."""

    x_label: str
    y_label: str
    given_labels: tuple[str, ...]
    partial_correlation: float | None
    p_value: float | None
    n: int
    status: str  # "consistent" | "rejected" | "untestable"
    detail: str = ""

    @property
    def claim(self) -> str:
        if self.given_labels:
            return (
                f"{self.x_label} ⊥ {self.y_label} | " + ", ".join(self.given_labels)
            )
        return f"{self.x_label} ⊥ {self.y_label}"


@dataclass
class ImplicationReport:
    tests: list[ImplicationTest] = field(default_factory=list)
    unmapped_nodes: list[str] = field(default_factory=list)

    @property
    def n_testable(self) -> int:
        return sum(1 for t in self.tests if t.status != "untestable")

    @property
    def n_rejected(self) -> int:
        return sum(1 for t in self.tests if t.status == "rejected")

    @property
    def n_consistent(self) -> int:
        return sum(1 for t in self.tests if t.status == "consistent")

    @property
    def verdict(self) -> str:
        if self.n_testable == 0:
            return "No testable implication could be evaluated against this data."
        if self.n_rejected == 0:
            return (
                f"Data is consistent with all {self.n_testable} testable "
                "implication(s). The DAG survives — this is not proof it is "
                "correct, only that this data does not contradict it."
            )
        return (
            f"Data rejects {self.n_rejected} of {self.n_testable} testable "
            "implication(s). Each rejection points at a missing edge or a "
            "missing common cause between the two variables named."
        )


def _partial_correlation(
    df: pd.DataFrame,
    x: str,
    y: str,
    given: tuple[str, ...],
) -> tuple[float | None, int]:
    """Partial correlation of *x* and *y* controlling for *given*.

    Computed by residualising both variables on the conditioning set via
    ordinary least squares and correlating the residuals.
    """
    import numpy as np

    cols = [x, y, *given]
    data = df[cols].dropna()
    n = int(len(data))
    if n < len(cols) + 3:
        return None, n

    target_x = data[x].astype(float).to_numpy()
    target_y = data[y].astype(float).to_numpy()

    if given:
        controls = data[list(given)].astype(float).to_numpy()
        design = np.column_stack([np.ones(n), controls])
        try:
            coef_x, *_ = np.linalg.lstsq(design, target_x, rcond=None)
            coef_y, *_ = np.linalg.lstsq(design, target_y, rcond=None)
        except np.linalg.LinAlgError:
            return None, n
        res_x = target_x - design @ coef_x
        res_y = target_y - design @ coef_y
    else:
        res_x = target_x - target_x.mean()
        res_y = target_y - target_y.mean()

    denom = res_x.std() * res_y.std()
    if denom == 0 or not math.isfinite(denom):
        return None, n
    correlation = float((res_x * res_y).mean() / denom)
    return max(-1.0, min(1.0, correlation)), n


def _p_value_for(correlation: float, n: int, n_controls: int) -> float | None:
    """Two-sided p-value for a partial correlation via Fisher's z."""
    import numpy as np
    from scipy import stats  # type: ignore[import-untyped]

    dof = n - n_controls - 3
    if dof <= 0 or abs(correlation) >= 1.0:
        return None
    z = 0.5 * math.log((1 + correlation) / (1 - correlation)) * math.sqrt(dof)
    return float(2 * (1 - stats.norm.cdf(abs(z))))


def test_implications(
    version: "DAGVersion",
    df: pd.DataFrame,
    column_for_node: dict[str, str] | None = None,
    alpha: float = 0.05,
    max_conditioning_set: int = 3,
) -> ImplicationReport:
    """Test every conditional independence the DAG implies against *df*.

    Args:
        version: the DAG whose structure is under test.
        df: the feature matrix.
        column_for_node: node id -> column name. Defaults to matching a node's
            label against a column of the same name, then falling back to the
            first entry of ``adapter_metadata["proxy_variables"]``.
        alpha: significance level. Reported per-test; no multiplicity
            correction is applied here because the caller decides the family.
        max_conditioning_set: bound on separating-set search.

    A rejection means the data shows an association the graph says should not
    exist — evidence of a missing edge or an unmeasured common cause.
    """
    mapping = dict(column_for_node or {})
    if not mapping:
        for node in version.nodes:
            if node.label in df.columns:
                mapping[node.id] = node.label
            elif node.adapter_metadata:
                proxies = node.adapter_metadata.get("proxy_variables") or []
                for candidate in proxies:
                    if candidate in df.columns:
                        mapping[node.id] = candidate
                        break

    labels = {node.id: node.label for node in version.nodes}
    unmapped = [labels[nid] for nid in labels if nid not in mapping]

    report = ImplicationReport(unmapped_nodes=sorted(unmapped))

    for x_id, y_id, given_ids in testable_implications(
        version.nodes, version.edges, max_conditioning_set=max_conditioning_set
    ):
        x_label = labels.get(x_id, x_id[:8])
        y_label = labels.get(y_id, y_id[:8])
        given_labels = tuple(labels.get(g, g[:8]) for g in given_ids)

        missing = [
            labels.get(nid, nid[:8])
            for nid in (x_id, y_id, *given_ids)
            if nid not in mapping
        ]
        if missing:
            report.tests.append(
                ImplicationTest(
                    x_label=x_label,
                    y_label=y_label,
                    given_labels=given_labels,
                    partial_correlation=None,
                    p_value=None,
                    n=0,
                    status="untestable",
                    detail="No data column for: " + ", ".join(missing),
                )
            )
            continue

        correlation, n = _partial_correlation(
            df, mapping[x_id], mapping[y_id], tuple(mapping[g] for g in given_ids)
        )
        if correlation is None:
            report.tests.append(
                ImplicationTest(
                    x_label=x_label,
                    y_label=y_label,
                    given_labels=given_labels,
                    partial_correlation=None,
                    p_value=None,
                    n=n,
                    status="untestable",
                    detail=f"Too few complete rows ({n}) or a degenerate column.",
                )
            )
            continue

        p_value = _p_value_for(correlation, n, len(given_ids))
        if p_value is None:
            status, detail = "untestable", "Could not compute a p-value."
        elif p_value < alpha:
            status = "rejected"
            detail = (
                f"The DAG says these are independent, but the data shows "
                f"r={correlation:.3f} (p={p_value:.4g})."
            )
        else:
            status = "consistent"
            detail = f"r={correlation:.3f}, p={p_value:.4g}"

        report.tests.append(
            ImplicationTest(
                x_label=x_label,
                y_label=y_label,
                given_labels=given_labels,
                partial_correlation=round(correlation, 4),
                p_value=round(p_value, 6) if p_value is not None else None,
                n=n,
                status=status,
                detail=detail,
            )
        )

    return report
