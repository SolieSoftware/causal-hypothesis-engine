from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """Timezone-aware UTC now (``datetime.utcnow`` is deprecated on 3.12+)."""
    return datetime.now(UTC)


class NodeContribution(BaseModel):
    """Per-node incremental contribution, with an uncertainty interval.

    A bare point estimate of a score delta is not interpretable: a node whose
    true contribution is exactly zero has a ~50% chance of a positive delta on
    any given run. Reporting an interval — and a three-way verdict rather than
    the sign — is what stops a graph of pure noise from producing confident
    "added signal" claims.
    """

    node_id: str
    node_label: str = ""
    contribution: float
    ci_low: float | None = None
    ci_high: float | None = None

    @property
    def verdict(self) -> str:
        """``"positive"``, ``"negative"``, or ``"inconclusive"``."""
        if self.ci_low is None or self.ci_high is None:
            return "inconclusive"
        if self.ci_low > 0:
            return "positive"
        if self.ci_high < 0:
            return "negative"
        return "inconclusive"


class BacktestResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    adapter_type: str  # e.g. "insurance"
    baseline_score: float
    dag_score: float
    lift: float  # dag_score - baseline_score

    # --- Provenance: what was actually computed, on what, and how -----------
    # Without these a stored result cannot be interpreted after the fact. The
    # engine previously reported a rescaled correlation under the label "AUC"
    # whenever scikit-learn was absent, and nothing recorded which had run.
    metric_name: str = "unknown"
    n_rows: int = 0
    n_positive: int = 0
    n_features_baseline: int = 0
    n_features_dag: int = 0
    cv_scheme: str = ""
    random_seed: int | None = None

    # --- Uncertainty --------------------------------------------------------
    lift_ci_low: float | None = None
    lift_ci_high: float | None = None

    # node_id -> lift contribution. Kept as a plain float map for backwards
    # compatibility with stored results; `contributions` carries the richer form.
    node_contributions: dict[str, float] = Field(default_factory=dict)
    contributions: list[NodeContribution] = Field(default_factory=list)
    notes: str = ""

    @property
    def lift_is_significant(self) -> bool:
        """True only when the lift interval excludes zero."""
        if self.lift_ci_low is None or self.lift_ci_high is None:
            return False
        return self.lift_ci_low > 0 or self.lift_ci_high < 0
