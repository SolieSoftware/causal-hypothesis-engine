"""BacktestAgent — automated, optional backtesting pipeline.

Runs when:
  - The HypothesisNetwork has an adapter attached (not AdapterType.None_)
  - The DAGVersion has at least one Proxied node with proxy_variables defined

Pipeline:
  1. Resolve the correct adapter for the network's AdapterType.
  2. Load data from the path supplied by the caller (or from adapter_metadata
     of any node that has data_source set).
  3. Validate the data against the adapter's schema.
  4. Build proxy features for all Proxied nodes.
  5. Compute baseline score (no proxy features).
  6. Compute DAG score (baseline + proxy features).
  7. Record per-node contributions.
  8. Attach a BacktestResult to the DAGVersion and transition its status to Tested.
  9. Print a human-readable report via Rich.

The agent is automated (no LLM call, no conversation loop).  It is invoked
from the CLI via:

    causal-engine backtest <version-id> --data <path>
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from ..adapters.base import AdapterBase
from ..adapters.insurance import InsuranceClaimsAdapter
from ..models.backtest_result import BacktestResult, NodeContribution
from ..models.dag_version import DAGVersion, DAGVersionStatus
from ..models.network import AdapterType, HypothesisNetwork
from ..models.node import MeasurabilityState
from ..persistence.db import Database

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Adapter registry — add new adapters here as they are implemented.
# ---------------------------------------------------------------------------

_ADAPTER_REGISTRY: dict[AdapterType, type[AdapterBase]] = {
    AdapterType.Insurance: InsuranceClaimsAdapter,
}


def get_adapter(adapter_type: AdapterType) -> AdapterBase:
    """Return an instantiated adapter for *adapter_type*.

    Raises:
        ValueError: If no adapter is registered for this type.
    """
    cls = _ADAPTER_REGISTRY.get(adapter_type)
    if cls is None:
        raise ValueError(
            f"No adapter registered for AdapterType.{adapter_type.value}. "
            f"Available: {[t.value for t in _ADAPTER_REGISTRY]}"
        )
    return cls()


# ---------------------------------------------------------------------------
# Pre-flight checks (separated so CLI can call them before the run)
# ---------------------------------------------------------------------------


class BacktestPreflightError(Exception):
    """Raised when the backtest cannot proceed due to a configuration issue."""


def check_can_backtest(
    network: HypothesisNetwork,
    version: DAGVersion,
) -> None:
    """Raise BacktestPreflightError with a structured message if backtesting is blocked.

    Checks:
    - Network has a non-None_ adapter.
    - That adapter is registered.
    - The version has at least one Proxied node with proxy_variables.
    - The version is not already Tested or Archived.
    """
    if network.adapter == AdapterType.None_:
        raise BacktestPreflightError(
            "Problem: Network has no adapter attached.\n"
            "  Cause: BacktestAgent requires a domain adapter.\n"
            "  Fix: Attach an adapter when creating the network with "
            "--adapter Insurance."
        )

    if network.adapter not in _ADAPTER_REGISTRY:
        raise BacktestPreflightError(
            f"Problem: Adapter '{network.adapter.value}' is not yet implemented.\n"
            "  Cause: Only Insurance is available in v1.\n"
            "  Fix: Use --adapter Insurance."
        )

    proxied = [
        n for n in version.nodes
        if n.measurability_state == MeasurabilityState.Proxied
        and n.adapter_metadata
        and n.adapter_metadata.get("proxy_variables")
    ]
    if not proxied:
        raise BacktestPreflightError(
            "Problem: No Proxied nodes with proxy_variables found in this version.\n"
            "  Cause: BacktestAgent needs at least one node with measurability_state "
            "= Proxied and proxy_variables set in its adapter_metadata.\n"
            "  Fix: Update node measurability states and add proxy_variables."
        )

    if version.is_immutable:
        raise BacktestPreflightError(
            f"Problem: Version {version.version_id[:8]} is already "
            f"{version.status.value} and cannot be modified.\n"
            "  Cause: Tested/Archived versions are immutable.\n"
            "  Fix: Use causal-engine modify <version-id> to create a new version first."
        )


# ---------------------------------------------------------------------------
# BacktestAgent
# ---------------------------------------------------------------------------


class BacktestAgent:
    """Automated backtesting pipeline.

    Args:
        db: Open Database instance.
        network: The HypothesisNetwork being backtested.
        version: The DAGVersion to attach results to.
        data_path: Path to the CSV or Parquet file to use.
    """

    def __init__(
        self,
        db: Database,
        network: HypothesisNetwork,
        version: DAGVersion,
        data_path: str | Path,
    ) -> None:
        self.db = db
        self.network = network
        self.version = version
        self.data_path = Path(data_path)
        self.adapter = get_adapter(network.adapter)

    # ------------------------------------------------------------------
    # Public: run the automated pipeline
    # ------------------------------------------------------------------

    def run(self, console: Any) -> BacktestResult:
        """Execute the full backtest pipeline and return the result.

        Attaches the result to the DAGVersion (status → Tested) in the DB.
        Prints a Rich report to *console*.

        Raises:
            BacktestPreflightError: If pre-flight checks fail.
            FileNotFoundError: If the data file does not exist.
            ValueError: If data validation fails.
        """
        from rich.panel import Panel
        from rich.table import Table

        console.print(
            Panel(
                f"[bold cyan]Backtest Agent[/bold cyan]\n"
                f"Network:  [green]{self.network.name}[/green]\n"
                f"Version:  [dim]{self.version.version_id}[/dim]\n"
                f"Adapter:  [yellow]{self.adapter.domain_label}[/yellow]\n"
                f"Data:     {self.data_path}",
                title="causal-hypothesis-engine — backtest",
            )
        )

        # 1. Load data
        console.print("[dim]Loading data...[/dim]")
        df = self.adapter.load_data(self.data_path)
        console.print(f"[dim]  {len(df):,} rows, {len(df.columns)} columns[/dim]")

        # 2. Validate
        console.print("[dim]Validating data...[/dim]")
        errors = self.adapter.validate_data(df)
        if errors:
            error_text = "\n".join(f"  • {e}" for e in errors)
            console.print(
                f"[bold red][ERROR][/bold red] Problem: Data validation failed.\n"
                f"  Cause:\n{error_text}\n"
                "  Fix: Ensure the CSV/Parquet matches the adapter's expected schema."
            )
            raise ValueError(f"Data validation failed: {errors}")
        console.print("[dim]  Validation passed.[/dim]")

        # 3. Build proxy features
        console.print("[dim]Building proxy features...[/dim]")
        proxy_features = self.adapter.build_proxy_features(df, self.version)
        proxied_count = len([
            n for n in self.version.nodes
            if n.measurability_state == MeasurabilityState.Proxied
            and n.adapter_metadata
            and n.adapter_metadata.get("proxy_variables")
        ])
        console.print(
            f"[dim]  {proxied_count} proxied node(s), "
            f"{len(proxy_features.columns)} feature column(s)[/dim]"
        )

        # 4. Score
        outcome_col = self.adapter.outcome_column(df)
        console.print(f"[dim]Scoring (outcome: '{outcome_col}')...[/dim]")

        # `score_detail` returns the same scores plus the provenance and
        # bootstrap intervals needed to interpret them. Adapters that predate
        # it fall back to the plain interface.
        node_map = {n.id: n.label for n in self.version.nodes}
        if hasattr(self.adapter, "score_detail"):
            detail = self.adapter.score_detail(df, proxy_features, outcome_col)
            baseline_score = detail["baseline_score"]
            dag_score = detail["dag_score"]
            node_contributions = detail["node_contributions"]
            lift_ci = detail.get("lift_ci", (None, None))
            node_detail = detail.get("node_detail", {})
            metric_name = detail.get("metric_name", "unknown")
            extra = {
                "metric_name": metric_name,
                "n_rows": detail.get("n_rows", 0),
                "n_positive": detail.get("n_positive", 0),
                "n_features_baseline": detail.get("n_features_baseline", 0),
                "n_features_dag": detail.get("n_features_dag", 0),
                "cv_scheme": detail.get("cv_scheme", ""),
                "random_seed": 42,
                "lift_ci_low": lift_ci[0],
                "lift_ci_high": lift_ci[1],
            }
        else:  # pragma: no cover - legacy adapter path
            baseline_score = self.adapter.compute_baseline_score(df, outcome_col)
            dag_score, node_contributions = self.adapter.compute_dag_score(
                df, proxy_features, outcome_col
            )
            node_detail = {}
            extra = {}

        lift = dag_score - baseline_score

        contributions = [
            NodeContribution(
                node_id=node_id,
                node_label=node_map.get(node_id, node_id[:8]),
                contribution=round(value, 6),
                ci_low=node_detail.get(node_id, (None, None))[0],
                ci_high=node_detail.get(node_id, (None, None))[1],
            )
            for node_id, value in node_contributions.items()
        ]

        # 5. Build result
        result = BacktestResult(
            version_id=self.version.version_id,
            adapter_type=self.network.adapter.value,
            baseline_score=round(baseline_score, 6),
            dag_score=round(dag_score, 6),
            lift=round(lift, 6),
            node_contributions={k: round(v, 6) for k, v in node_contributions.items()},
            contributions=contributions,
            notes=self._build_notes(contributions),
            **extra,
        )

        # 6. Persist — attach result, transition to Tested
        self.db.update_version_status(
            self.version.version_id,
            DAGVersionStatus.Tested,
            result,
        )

        # 7. Report
        self._print_report(console, result)
        return result

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _print_report(self, console: Any, result: BacktestResult) -> None:
        from rich.panel import Panel
        from rich.table import Table

        # Colour by significance, not by sign. A lift whose interval spans zero
        # is not evidence of anything, and rendering it green because it
        # happened to land above zero is how a graph of pure noise produces
        # confident findings.
        if result.lift_is_significant:
            lift_color = "green" if result.lift > 0 else "red"
        else:
            lift_color = "yellow"
        lift_sign = "+" if result.lift >= 0 else ""

        if result.lift_ci_low is not None:
            interval = (
                f"  [95% CI {result.lift_ci_low:+.4f} to {result.lift_ci_high:+.4f}]"
            )
            if not result.lift_is_significant:
                interval += "  [yellow](spans zero)[/yellow]"
        else:
            interval = ""

        provenance = ""
        if result.metric_name != "unknown":
            provenance = (
                f"\nMetric:          {result.metric_name}\n"
                f"Sample:          {result.n_rows:,} rows, "
                f"{result.n_positive:,} positive\n"
                f"CV:              {result.cv_scheme}\n"
            )

        console.print(
            Panel(
                f"Baseline score:  {result.baseline_score:.4f}\n"
                f"DAG score:       {result.dag_score:.4f}\n"
                f"Lift:            [{lift_color}]{lift_sign}{result.lift:.4f}"
                f"[/{lift_color}]{interval}\n"
                f"{provenance}\n"
                f"{self.adapter.describe_lift()}",
                title="[bold]Backtest Result[/bold]",
            )
        )

        if result.contributions:
            table = Table(title="Node Contributions", show_lines=False)
            table.add_column("Node ID", style="dim")
            table.add_column("Node Label")
            table.add_column("Contribution", justify="right")
            table.add_column("95% CI", justify="center")
            table.add_column("Verdict", justify="center")

            styles = {
                "positive": "[green]adds signal[/green]",
                "negative": "[red]hurts[/red]",
                "inconclusive": "[yellow]inconclusive[/yellow]",
            }
            for item in sorted(
                result.contributions, key=lambda c: c.contribution, reverse=True
            ):
                ci = (
                    f"{item.ci_low:+.4f} … {item.ci_high:+.4f}"
                    if item.ci_low is not None
                    else "—"
                )
                table.add_row(
                    item.node_id[:8],
                    item.node_label,
                    f"{item.contribution:+.4f}",
                    ci,
                    styles[item.verdict],
                )
            console.print(table)
            console.print(
                "[dim]'inconclusive' means the interval spans zero — the data "
                "cannot distinguish this node's contribution from nothing. It is "
                "not grounds to remove the node.[/dim]"
            )

        console.print(
            f"\n[bold green]Version {self.version.version_id[:8]} → Tested[/bold green]"
        )

    def _build_notes(self, contributions: list[NodeContribution]) -> str:
        """Summarise contributions by *verdict*, not by sign.

        These notes are fed verbatim into the ModificationAgent's context, so
        a sign-based "no signal" label here becomes the LLM's justification for
        proposing that a node be deleted. Only a node whose interval lies
        wholly below zero has actually been shown to hurt.
        """
        buckets: dict[str, list[str]] = {"positive": [], "negative": [], "inconclusive": []}
        for item in contributions:
            buckets[item.verdict].append(item.node_label)

        parts: list[str] = []
        if buckets["positive"]:
            parts.append(f"Adds signal: {', '.join(buckets['positive'])}.")
        if buckets["negative"]:
            parts.append(f"Hurts predictive performance: {', '.join(buckets['negative'])}.")
        if buckets["inconclusive"]:
            parts.append(
                "Contribution indistinguishable from zero (NOT evidence of no "
                f"effect): {', '.join(buckets['inconclusive'])}."
            )
        return " ".join(parts)
