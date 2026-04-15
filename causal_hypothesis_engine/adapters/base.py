"""AdapterBase — abstract base class for domain adapters.

An adapter is a plugin that bridges the domain-agnostic causal DAG model to
a specific domain's data.  Each adapter defines:

  - domain_label: human-readable name (e.g. "Insurance Claims")
  - adapter_type: the AdapterType enum value it handles
  - node_metadata_fields: dict of field_name → description for the domain's
    node enrichment schema (used by agents to guide the user)
  - load_data: read a CSV or Parquet file into a DataFrame
  - validate_data: check that required columns are present
  - build_proxy_features: for a set of Proxied nodes, construct ML features
  - compute_baseline_score: score a model trained on raw features only
  - compute_dag_score: score a model trained on DAG-derived proxy features
  - describe_lift: human-readable explanation of what "lift" means in the domain

Adapters are stateless — they operate on data passed in rather than storing it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from ..models.dag_version import DAGVersion
    from ..models.network import AdapterType


class NodeMetadataSchema:
    """Descriptor for adapter-specific node metadata fields.

    Each adapter exposes this so agents can guide users through enrichment.
    """

    def __init__(self, fields: dict[str, str]) -> None:
        # field_name -> human-readable description
        self.fields = fields

    def required_fields(self) -> list[str]:
        return list(self.fields.keys())

    def describe(self) -> str:
        lines = ["Node metadata fields for this adapter:"]
        for name, desc in self.fields.items():
            lines.append(f"  {name}: {desc}")
        return "\n".join(lines)


class AdapterBase(ABC):
    """Abstract base for all domain adapters."""

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def domain_label(self) -> str:
        """Human-readable domain name (e.g. 'Insurance Claims')."""

    @property
    @abstractmethod
    def adapter_type(self) -> "AdapterType":
        """The AdapterType enum value this adapter handles."""

    @property
    @abstractmethod
    def node_metadata_schema(self) -> NodeMetadataSchema:
        """Fields that can be attached to nodes in this domain."""

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self, path: str | Path) -> pd.DataFrame:
        """Load a CSV or Parquet file into a DataFrame.

        Dispatches on file extension.  Subclasses may override for custom
        loading logic (e.g. BigQuery in v1.1).
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {p}")
        suffix = p.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(p)
        if suffix in (".parquet", ".pq"):
            return pd.read_parquet(p)
        raise ValueError(
            f"Unsupported file format '{suffix}'. "
            "Expected .csv, .parquet, or .pq"
        )

    @abstractmethod
    def validate_data(self, df: pd.DataFrame) -> list[str]:
        """Return a list of validation error strings (empty = valid)."""

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    @abstractmethod
    def build_proxy_features(
        self,
        df: pd.DataFrame,
        version: "DAGVersion",
    ) -> pd.DataFrame:
        """Construct proxy features for all Proxied nodes in *version*.

        Returns a DataFrame of engineered features (one column per node that
        has proxy_variables defined in its adapter_metadata).
        """

    @abstractmethod
    def outcome_column(self, df: pd.DataFrame) -> str:
        """Return the name of the outcome/target column in *df*."""

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_baseline_score(
        self,
        df: pd.DataFrame,
        outcome_col: str,
    ) -> float:
        """Score a model trained on raw columns only (no DAG features).

        Returns a scalar score (higher = better, e.g. AUC or R²).
        """

    @abstractmethod
    def compute_dag_score(
        self,
        df: pd.DataFrame,
        proxy_features: pd.DataFrame,
        outcome_col: str,
    ) -> tuple[float, dict[str, float]]:
        """Score a model that includes DAG-derived proxy features.

        Returns:
            (dag_score, node_contributions) where node_contributions maps
            node_id → incremental lift contributed by that node's proxy.
        """

    @abstractmethod
    def describe_lift(self) -> str:
        """One-sentence description of what 'lift' means in this domain."""
