"""CLI entry point for causal-hypothesis-engine.

causal-engine new                — start new network + DAGAgent session
causal-engine resume <id>        — resume an interrupted session
causal-engine resume --last      — resume the most recent session
causal-engine list               — list all networks and versions
causal-engine modify <version-id> — start ModificationAgent session
causal-engine backtest <version-id> --data <path> — run BacktestAgent
causal-engine compare <v-id-1> <v-id-2>           — diff two DAGVersions
causal-engine doctor             — check environment
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .agents.backtest_agent import BacktestAgent, BacktestPreflightError, check_can_backtest
from .agents.dag_agent import DAGAgent, DraftState
from .agents.modification_agent import ModificationAgent
from .comparison import compute_diff, render_diff
from .models.network import AdapterType, HypothesisNetwork
from .models.session import ModificationMode, SessionMode, SessionStatus
from .persistence.checkpoint import load_checkpoint
from .persistence.db import Database

console = Console()

_DEFAULT_DB = Path.home() / ".causal_engine" / "causal_engine.db"
_DEFAULT_CHECKPOINT_DIR = Path.home() / ".causal_engine" / "checkpoints"


def _get_db() -> Database:
    _DEFAULT_DB.parent.mkdir(parents=True, exist_ok=True)
    _DEFAULT_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return Database(_DEFAULT_DB)


def _validate_env() -> None:
    """Check required environment before running agent commands."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print(
            "[bold red][ERROR][/bold red] Problem: ANTHROPIC_API_KEY is not set.\n"
            "  Cause: The agent requires an Anthropic API key to call Claude.\n"
            "  Fix: Set the environment variable: "
            "[bold]export ANTHROPIC_API_KEY=sk-...[/bold]"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group()
def cli() -> None:
    """causal-hypothesis-engine — build and experiment with causal DAGs."""


# ---------------------------------------------------------------------------
# new
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--name", "-n", prompt="Network name", help="Name for the hypothesis network.")
@click.option("--domain", "-d", default="", help="Domain label (e.g. 'insurance', 'climate').")
@click.option(
    "--adapter",
    "-a",
    type=click.Choice(["none", "Insurance", "Financial", "Clinical"], case_sensitive=False),
    default="none",
    help="Adapter type.",
)
def new(name: str, domain: str, adapter: str) -> None:
    """Start a new hypothesis network and launch a DAGAgent session."""
    _validate_env()
    db = _get_db()

    adapter_type = AdapterType(adapter.lower() if adapter.lower() == "none" else adapter)
    network = HypothesisNetwork(name=name, domain=domain, adapter=adapter_type)
    db.create_network(network)

    console.print(
        f"[green]Created network[/green] [bold]{name}[/bold] "
        f"([dim]{network.id}[/dim])"
    )

    agent = DAGAgent(db=db, network=network, checkpoint_dir=_DEFAULT_CHECKPOINT_DIR)
    agent.run(console)


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("session_id", required=False)
@click.option("--last", is_flag=True, help="Resume the most recently active session.")
def resume(session_id: str | None, last: bool) -> None:
    """Resume an interrupted DAGAgent session."""
    _validate_env()
    db = _get_db()

    if last:
        session_id = _find_last_session(db)
        if session_id is None:
            console.print("[red]No active sessions found.[/red]")
            sys.exit(1)

    if not session_id:
        console.print(
            "[bold red][ERROR][/bold red] Problem: No session ID provided.\n"
            "  Fix: Pass a session ID or use [bold]--last[/bold]."
        )
        sys.exit(1)

    session = db.get_session(session_id)
    if session is None:
        console.print(
            f"[bold red][ERROR][/bold red] Problem: Session [bold]{session_id}[/bold] not found.\n"
            "  Cause: The session ID does not exist in the database.\n"
            "  Fix: Run [bold]causal-engine list[/bold] to see available sessions."
        )
        sys.exit(1)

    if session.status == SessionStatus.Confirmed:
        console.print(
            f"[yellow]Session {session_id[:8]} is already confirmed (saved). "
            "Nothing to resume.[/yellow]"
        )
        sys.exit(0)

    network = db.get_network(session.network_id)
    if network is None:
        console.print(
            f"[bold red][ERROR][/bold red] Problem: Network for session not found.\n"
            "  Cause: network_id={session.network_id} missing from database."
        )
        sys.exit(1)

    # Load draft state from checkpoint if available.
    draft: DraftState | None = None
    if session.checkpoint_path and Path(session.checkpoint_path).exists():
        try:
            _, draft_version = load_checkpoint(session.checkpoint_path)
            draft = DraftState.from_dag_version(draft_version)
            console.print(
                f"[green]Restored checkpoint[/green] — "
                f"{len(draft.nodes)} nodes, {len(draft.edges)} edges."
            )
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[yellow]Could not load checkpoint: {exc}[/yellow]")
    else:
        console.print("[yellow]No checkpoint found — starting from empty DAG.[/yellow]")

    agent = DAGAgent(
        db=db,
        network=network,
        session=session,
        draft=draft,
        checkpoint_dir=_DEFAULT_CHECKPOINT_DIR,
    )
    agent.run(console)


def _find_last_session(db: Database) -> str | None:
    """Return the ID of the most recently touched active session."""
    networks = db.list_networks()
    candidates: list[tuple[str, str]] = []  # (last_activity, session_id)
    for net_summary in networks:
        network = db.get_network(net_summary["id"])
        if network is None:
            continue
        for session in db.list_sessions_for_network(network.id):
            if session.status == SessionStatus.Active:
                candidates.append((str(session.last_activity), session.id))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@cli.command("list")
def list_networks() -> None:
    """List all hypothesis networks and their DAG versions."""
    db = _get_db()
    summaries = db.list_networks()

    if not summaries:
        console.print("[dim]No networks found. Run [bold]causal-engine new[/bold] to start.[/dim]")
        return

    table = Table(title="Hypothesis Networks", show_lines=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="bold")
    table.add_column("Domain")
    table.add_column("Adapter")
    table.add_column("Versions", justify="right")
    table.add_column("Sessions", justify="right")
    table.add_column("Created")

    for s in summaries:
        table.add_row(
            s["id"],
            s["name"],
            s["domain"] or "—",
            s["adapter"],
            str(s["version_count"]),
            str(s["session_count"]),
            s["created_at"][:19],
        )
    console.print(table)

    # Also list versions per network if any exist.
    for s in summaries:
        network = db.get_network(s["id"])
        if network is None or not network.version_ids:
            continue
        versions = db.get_versions_for_network(network.id)
        if not versions:
            continue
        vtable = Table(
            title=f"Versions — {s['name']}", show_lines=False, box=None
        )
        vtable.add_column("Version ID", style="cyan", no_wrap=True)
        vtable.add_column("Status")
        vtable.add_column("Nodes", justify="right")
        vtable.add_column("Edges", justify="right")
        vtable.add_column("Created")
        for v in versions:
            vtable.add_row(
                v.version_id,
                v.status.value,
                str(len(v.nodes)),
                str(len(v.edges)),
                str(v.created_at)[:19],
            )
        console.print(vtable)


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("version_id")
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["theory", "backtest", "hybrid"], case_sensitive=False),
    default="theory",
    help="Modification mode (default: theory).",
)
def modify(version_id: str, mode: str) -> None:
    """Start a ModificationAgent session on an existing DAGVersion."""
    _validate_env()
    db = _get_db()

    version = db.get_version(version_id)
    if version is None:
        console.print(
            f"[bold red][ERROR][/bold red] Problem: Version [bold]{version_id}[/bold] not found.\n"
            "  Fix: Run [bold]causal-engine list[/bold] to see available version IDs."
        )
        sys.exit(1)

    network = db.get_network(version.network_id)
    if network is None:
        console.print(
            f"[bold red][ERROR][/bold red] Problem: Network for version not found.\n"
            "  Cause: network_id={version.network_id} missing from database."
        )
        sys.exit(1)

    mode_map = {
        "theory": ModificationMode.Theory,
        "backtest": ModificationMode.Backtest,
        "hybrid": ModificationMode.Hybrid,
    }
    agent = ModificationAgent(
        db=db,
        network=network,
        source_version=version,
        modification_mode=mode_map[mode.lower()],
        checkpoint_dir=_DEFAULT_CHECKPOINT_DIR,
    )
    agent.run(console)


@cli.command()
@click.argument("version_id")
@click.option(
    "--data",
    "-d",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the CSV or Parquet data file.",
)
def backtest(version_id: str, data: str) -> None:
    """Run BacktestAgent on a DAGVersion and attach the result."""
    db = _get_db()

    version = db.get_version(version_id)
    if version is None:
        console.print(
            f"[bold red][ERROR][/bold red] Problem: Version [bold]{version_id}[/bold] not found.\n"
            "  Fix: Run [bold]causal-engine list[/bold] to see available version IDs."
        )
        sys.exit(1)

    network = db.get_network(version.network_id)
    if network is None:
        console.print(
            "[bold red][ERROR][/bold red] Problem: Network for version not found."
        )
        sys.exit(1)

    try:
        check_can_backtest(network, version)
    except BacktestPreflightError as exc:
        console.print(f"[bold red][ERROR][/bold red] {exc}")
        sys.exit(1)

    try:
        agent = BacktestAgent(db=db, network=network, version=version, data_path=data)
        agent.run(console)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[bold red][ERROR][/bold red] {exc}")
        sys.exit(1)


@cli.command()
@click.argument("version_id_1")
@click.argument("version_id_2")
def compare(version_id_1: str, version_id_2: str) -> None:
    """Diff two DAGVersions and display a structured change report."""
    db = _get_db()

    v1 = db.get_version(version_id_1)
    if v1 is None:
        console.print(
            f"[bold red][ERROR][/bold red] Problem: Version [bold]{version_id_1}[/bold] not found.\n"
            "  Fix: Run [bold]causal-engine list[/bold] to see available version IDs."
        )
        sys.exit(1)

    v2 = db.get_version(version_id_2)
    if v2 is None:
        console.print(
            f"[bold red][ERROR][/bold red] Problem: Version [bold]{version_id_2}[/bold] not found.\n"
            "  Fix: Run [bold]causal-engine list[/bold] to see available version IDs."
        )
        sys.exit(1)

    diff = compute_diff(v1, v2)
    render_diff(diff, console, v1, v2)


@cli.command()
@click.option("--bigquery", is_flag=True, default=False,
              help="Also check BigQuery credentials (optional).")
def doctor(bigquery: bool) -> None:
    """Check the environment (API key, SQLite, Python version, optional BigQuery)."""
    import platform
    import sqlite3
    import sys as _sys

    from rich.table import Table

    all_ok = True
    rows: list[tuple[str, str, str]] = []  # (check, status, detail)

    def _ok(check: str, detail: str = "") -> None:
        rows.append((check, "[green]✓ ok[/green]", detail))

    def _fail(check: str, detail: str = "") -> None:
        nonlocal all_ok
        all_ok = False
        rows.append((check, "[red]✗ fail[/red]", detail))

    def _warn(check: str, detail: str = "") -> None:
        rows.append((check, "[yellow]⚠ warn[/yellow]", detail))

    # -- Python version
    py = platform.python_version()
    major, minor, *_ = py.split(".")
    if int(major) >= 3 and int(minor) >= 12:
        _ok("Python version", py)
    else:
        _fail("Python version", f"{py} — requires 3.12+")

    # -- ANTHROPIC_API_KEY
    if os.environ.get("ANTHROPIC_API_KEY"):
        _ok("ANTHROPIC_API_KEY", "set")
    else:
        _fail("ANTHROPIC_API_KEY",
              "not set — run: export ANTHROPIC_API_KEY=sk-...")

    # -- SQLite DB (create + schema_version check)
    try:
        db = _get_db()
        with db._connect() as conn:
            row = conn.execute("SELECT version FROM schema_version").fetchone()
            schema_v = row["version"] if row else "?"
        _ok("SQLite database", f"{_DEFAULT_DB}  schema_version={schema_v}")
    except Exception as exc:
        _fail("SQLite database", str(exc))

    # -- WAL mode
    try:
        db = _get_db()
        with db._connect() as conn:
            wal_row = conn.execute("PRAGMA journal_mode").fetchone()
        if wal_row and wal_row[0] == "wal":
            _ok("SQLite WAL mode", "enabled")
        else:
            _warn("SQLite WAL mode", f"expected wal, got {wal_row[0] if wal_row else '?'}")
    except Exception as exc:
        _warn("SQLite WAL mode", str(exc))

    # -- Checkpoint directory
    if _DEFAULT_CHECKPOINT_DIR.exists():
        _ok("Checkpoint directory", str(_DEFAULT_CHECKPOINT_DIR))
    else:
        _warn("Checkpoint directory",
              f"missing — will be created on first session: {_DEFAULT_CHECKPOINT_DIR}")

    # -- Required packages
    import importlib.metadata
    for pkg in ("pydantic", "anthropic", "rich", "click", "pandas", "pyarrow"):
        try:
            ver = importlib.metadata.version(pkg)
            _ok(f"package: {pkg}", ver)
        except importlib.metadata.PackageNotFoundError:
            _fail(f"package: {pkg}", "not installed — pip install causal-hypothesis-engine")

    # -- BigQuery (optional)
    if bigquery:
        try:
            import google.cloud.bigquery  # noqa: F401
            _ok("BigQuery client", "google-cloud-bigquery installed")
        except ImportError:
            _warn("BigQuery client",
                  "not installed — pip install 'causal-hypothesis-engine[bigquery]'")

    # -- Render table
    t = Table(title="Environment Check", show_lines=False, box=None)
    t.add_column("Check", style="bold")
    t.add_column("Status", justify="center")
    t.add_column("Detail", style="dim")
    for check, status, detail in rows:
        t.add_row(check, status, detail)
    console.print(t)

    if all_ok:
        console.print("\n[bold green]All checks passed.[/bold green]")
    else:
        console.print(
            "\n[bold red]Some checks failed.[/bold red] Fix issues above, then re-run "
            "[bold]causal-engine doctor[/bold]."
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------


@cli.command()
def demo() -> None:
    """Run a canned demo session — no API key required.

    Demonstrates the full causal-engine workflow with a pre-built
    insurance claims causal DAG:

      Rainfall → Flood Risk → Large Claim
                         ↑
               Flood Zone (confounder)

    The demo creates a network, populates a DAGVersion, runs a comparison
    against a modified version, and prints a summary — all without calling
    the Anthropic API or requiring real data.
    """
    from rich.panel import Panel
    from rich.table import Table

    from .comparison import compute_diff, render_diff
    from .models.dag_version import DAGVersion
    from .models.edge import Edge
    from .models.node import MeasurabilityState, Node, NodeType
    from .models.network import AdapterType, HypothesisNetwork

    console.print(
        Panel(
            "[bold cyan]causal-hypothesis-engine demo[/bold cyan]\n\n"
            "Building a sample insurance claims causal DAG.\n"
            "No API key required — this is a canned simulation.",
            title="Demo Mode",
        )
    )

    db = _get_db()

    # -- Build network
    network = HypothesisNetwork(
        name="Demo: Insurance Flood Claims",
        domain="insurance",
        adapter=AdapterType.Insurance,
    )
    db.create_network(network)
    console.print(f"[green]✓[/green] Created network: {network.name} [{network.id[:8]}]")

    # -- Version 1: initial hypothesis
    rainfall = Node(
        label="Rainfall",
        description="Monthly cumulative rainfall at property location (mm)",
        node_type=NodeType.Exposure,
        measurability_state=MeasurabilityState.Proxied,
        adapter_metadata={"proxy_variables": ["rainfall_mm"], "claim_type": "flood"},
    )
    flood_zone = Node(
        label="Flood Zone",
        description="Whether property is in a designated flood zone",
        node_type=NodeType.Confounder,
        measurability_state=MeasurabilityState.Identified,
        adapter_metadata={"proxy_variables": ["flood_zone"], "claim_type": "flood"},
    )
    flood_risk = Node(
        label="Flood Risk",
        description="Composite flood risk score for the property",
        node_type=NodeType.Mediator,
        measurability_state=MeasurabilityState.Hypothetical,
    )
    large_claim = Node(
        label="Large Claim",
        description="Claim amount exceeds threshold (outcome)",
        node_type=NodeType.Outcome,
        measurability_state=MeasurabilityState.Identified,
    )

    e1 = Edge(source_node_id=rainfall.id, target_node_id=flood_risk.id,
              label="increases", description="High rainfall raises flood risk")
    e2 = Edge(source_node_id=flood_zone.id, target_node_id=flood_risk.id,
              label="determines", description="Flood zone classification sets baseline risk")
    e3 = Edge(source_node_id=flood_risk.id, target_node_id=large_claim.id,
              label="causes", description="Elevated flood risk leads to large claims")

    v1 = DAGVersion(
        network_id=network.id,
        nodes=[rainfall, flood_zone, flood_risk, large_claim],
        edges=[e1, e2, e3],
        modification_rationale="Initial hypothesis: rainfall → flood risk → large claim",
    )
    db.save_version(v1)
    network.version_ids.append(v1.version_id)
    db.update_network_ids(network.id, network.version_ids, network.session_ids)
    console.print(
        f"[green]✓[/green] Saved Version 1 [{v1.version_id[:8]}]: "
        f"{len(v1.nodes)} nodes, {len(v1.edges)} edges"
    )

    # -- Version 2: refined hypothesis (add wind speed, update measurability)
    wind_speed = Node(
        label="Wind Speed",
        description="Maximum wind speed at location (km/h)",
        node_type=NodeType.Exposure,
        measurability_state=MeasurabilityState.Proxied,
        adapter_metadata={"proxy_variables": ["max_wind_speed"], "claim_type": "storm"},
    )
    flood_risk_v2 = flood_risk.model_copy(
        update={"measurability_state": MeasurabilityState.Proxied}
    )
    e4 = Edge(source_node_id=wind_speed.id, target_node_id=flood_risk_v2.id,
              label="amplifies", description="High winds amplify flood damage risk")

    v2 = DAGVersion(
        network_id=network.id,
        nodes=[rainfall, flood_zone, flood_risk_v2, large_claim, wind_speed],
        edges=[e1, e2, e3, e4],
        parent_version_id=v1.version_id,
        modification_rationale=(
            "Refined: added Wind Speed as a second exposure; "
            "promoted Flood Risk to Proxied measurability."
        ),
    )
    db.save_version(v2)
    network.version_ids.append(v2.version_id)
    db.update_network_ids(network.id, network.version_ids, network.session_ids)
    console.print(
        f"[green]✓[/green] Saved Version 2 [{v2.version_id[:8]}]: "
        f"{len(v2.nodes)} nodes, {len(v2.edges)} edges"
    )

    # -- Show the diff
    console.print("\n[bold]Comparing v1 → v2:[/bold]")
    diff = compute_diff(v1, v2)
    render_diff(diff, console, v1, v2)

    # -- Summary table
    t = Table(title="Demo Summary", show_lines=False)
    t.add_column("Item")
    t.add_column("Value", style="cyan")
    t.add_row("Network ID", network.id)
    t.add_row("Version 1 ID", v1.version_id)
    t.add_row("Version 2 ID", v2.version_id)
    t.add_row("Diff", diff.summary_line)
    console.print(t)

    console.print(
        "\n[dim]Next steps:[/dim]\n"
        f"  causal-engine list\n"
        f"  causal-engine compare {v1.version_id} {v2.version_id}\n"
        f"  causal-engine modify {v2.version_id}\n"
        "  causal-engine backtest <version-id> --data <claims.csv>\n"
    )


# ---------------------------------------------------------------------------
# __main__ shim
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    cli()
