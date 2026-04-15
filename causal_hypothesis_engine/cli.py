"""CLI entry point for causal-hypothesis-engine.

causal-engine new                — start new network + DAGAgent session
causal-engine resume <id>        — resume an interrupted session
causal-engine resume --last      — resume the most recent session
causal-engine list               — list all networks and versions
causal-engine modify <version-id> — start ModificationAgent session
causal-engine doctor             — check environment
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .agents.dag_agent import DAGAgent, DraftState
from .agents.modification_agent import ModificationAgent
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
def doctor() -> None:
    """Check the environment (API key, database, paths)."""
    all_ok = True

    # API key
    if os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[green]✓[/green] ANTHROPIC_API_KEY is set.")
    else:
        console.print("[red]✗[/red] ANTHROPIC_API_KEY is NOT set.")
        all_ok = False

    # DB path
    try:
        db = _get_db()
        console.print(f"[green]✓[/green] SQLite database: {_DEFAULT_DB}")
    except Exception as exc:
        console.print(f"[red]✗[/red] SQLite database error: {exc}")
        all_ok = False

    # Checkpoint dir
    if _DEFAULT_CHECKPOINT_DIR.exists():
        console.print(f"[green]✓[/green] Checkpoint directory: {_DEFAULT_CHECKPOINT_DIR}")
    else:
        console.print(f"[red]✗[/red] Checkpoint directory missing: {_DEFAULT_CHECKPOINT_DIR}")
        all_ok = False

    if all_ok:
        console.print("\n[bold green]All checks passed.[/bold green]")
    else:
        console.print("\n[bold red]Some checks failed. Fix issues above.[/bold red]")
        sys.exit(1)


# ---------------------------------------------------------------------------
# __main__ shim
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    cli()
