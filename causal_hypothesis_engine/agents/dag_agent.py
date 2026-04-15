"""DAGAgent — conversational, human-in-the-loop causal DAG builder.

Workflow:
1. Agent proposes nodes and edges in natural language.
2. User responds, shapes, adds, removes.
3. Every 10 exchanges the draft state is atomically checkpointed.
4. On "/confirm", the agent summarises the DAG and asks for final confirmation
   before writing a clean DAGVersion with status Draft.
5. On "/save dag", saves the current DAG state immediately without requiring
   confirmation summary.

Session is always resumable via causal-engine resume <session-id>.

Rolling history window: the last HISTORY_WINDOW exchanges are kept verbatim.
Older messages are replaced by a compact DAG summary injection so the context
window never grows unbounded.
"""

from __future__ import annotations

import re
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from anthropic import Anthropic

from ..models.dag_version import DAGVersion, DAGVersionStatus
from ..models.edge import Edge
from ..models.network import HypothesisNetwork
from ..models.node import MeasurabilityState, Node, NodeType
from ..models.session import ModificationMode, Session, SessionMode, SessionStatus
from ..persistence.checkpoint import write_checkpoint
from ..persistence.db import Database

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECKPOINT_EVERY = 10      # exchanges between automatic checkpoints
HISTORY_WINDOW = 20        # full exchanges kept verbatim; older ones summarised
MODEL = "claude-opus-4-6"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a causal reasoning assistant helping a researcher build a causal hypothesis \
network as a directed acyclic graph (DAG).

Your job is to:
1. Collaborate conversationally to identify nodes (variables) and directed edges \
(causal relationships).
2. Classify each node as one of: Exposure, Outcome, Confounder, Mediator, or Collider.
3. Assign each node a measurability state: Hypothetical, Identified, Proxied, or Validated.
4. Propose edges with a clear causal direction (source → target) and a brief rationale.
5. Flag potential DAG problems: cycles, backdoor paths, collider bias risks.

IMPORTANT PRINCIPLES:
- Never overstate causal claims. This is structured hypothesis generation, not causal \
identification.
- Always explain WHY you're proposing a node or edge, not just what it is.
- When the user pushes back, update your proposal rather than defending it dogmatically.
- Keep the DAG coherent: every added node should connect to the existing graph unless \
explicitly starting a new subgraph.

COMMANDS the user may type:
  /confirm       — you will summarise the full DAG and ask for final save confirmation
  /show          — display the current DAG state (nodes and edges)
  /help          — list available commands

When presenting the DAG state, use this format:
  NODES:
    [id_short] Label (NodeType, MeasurabilityState)
      - Description (if set)
  EDGES:
    [src_short] → [tgt_short]  "edge label"  — rationale

Keep responses concise. Propose one or two additions per turn; don't dump a full DAG \
in one message. Let the user steer.
"""

# ---------------------------------------------------------------------------
# DraftState — mutable in-memory DAG being built
# ---------------------------------------------------------------------------


class DraftState:
    """Mutable wrapper around the nodes and edges being assembled."""

    def __init__(self) -> None:
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []

    # -- node helpers --------------------------------------------------------

    def add_node(self, node: Node) -> None:
        self.nodes.append(node)

    def remove_node(self, node_id: str) -> bool:
        before = len(self.nodes)
        self.nodes = [n for n in self.nodes if n.id != node_id]
        # Also remove edges connected to this node.
        self.edges = [
            e for e in self.edges
            if e.source_node_id != node_id and e.target_node_id != node_id
        ]
        return len(self.nodes) < before

    def get_node(self, node_id: str) -> Node | None:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def find_node_by_label(self, label: str) -> Node | None:
        label_lower = label.lower()
        for n in self.nodes:
            if n.label.lower() == label_lower:
                return n
        return None

    # -- edge helpers --------------------------------------------------------

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

    def remove_edge(self, edge_id: str) -> bool:
        before = len(self.edges)
        self.edges = [e for e in self.edges if e.id != edge_id]
        return len(self.edges) < before

    # -- summary -------------------------------------------------------------

    def summary(self) -> str:
        if not self.nodes:
            return "No nodes yet."
        lines = ["NODES:"]
        for n in self.nodes:
            short_id = n.id[:8]
            desc_line = f"      - {n.description}" if n.description else ""
            lines.append(
                f"  [{short_id}] {n.label} ({n.node_type.value}, "
                f"{n.measurability_state.value})"
            )
            if desc_line:
                lines.append(desc_line)
        if self.edges:
            lines.append("EDGES:")
            node_map = {n.id: n.label for n in self.nodes}
            for e in self.edges:
                src = node_map.get(e.source_node_id, e.source_node_id[:8])
                tgt = node_map.get(e.target_node_id, e.target_node_id[:8])
                lines.append(f'  {src} → {tgt}  "{e.label}"')
        else:
            lines.append("EDGES: none")
        return "\n".join(lines)

    def to_dag_version(self, network_id: str) -> DAGVersion:
        return DAGVersion(
            network_id=network_id,
            nodes=deepcopy(self.nodes),
            edges=deepcopy(self.edges),
            status=DAGVersionStatus.Draft,
        )

    @classmethod
    def from_dag_version(cls, version: DAGVersion) -> "DraftState":
        state = cls()
        state.nodes = list(version.nodes)
        state.edges = list(version.edges)
        return state


# ---------------------------------------------------------------------------
# Rolling history window
# ---------------------------------------------------------------------------


def _trim_history(
    history: list[dict[str, str]],
    draft: DraftState,
    window: int = HISTORY_WINDOW,
) -> list[dict[str, str]]:
    """Keep the last *window* messages verbatim; summarise older ones."""
    if len(history) <= window:
        return history
    summary_msg = {
        "role": "user",
        "content": (
            "[CONTEXT SUMMARY — earlier messages trimmed]\n"
            "Current DAG state:\n"
            + draft.summary()
        ),
    }
    return [summary_msg] + history[-window:]


# ---------------------------------------------------------------------------
# Checkpoint path helper
# ---------------------------------------------------------------------------


def _checkpoint_path(session_id: str, base_dir: str | Path = ".") -> Path:
    checkpoints = Path(base_dir) / ".checkpoints"
    return checkpoints / f"{session_id}.checkpoint"


# ---------------------------------------------------------------------------
# DAGAgent
# ---------------------------------------------------------------------------


class DAGAgent:
    """Conversational agent for building a causal DAG collaboratively.

    Args:
        db: Open Database instance.
        network: The HypothesisNetwork this session belongs to.
        session: An existing Session (used for resume) or None to create a new one.
        draft: Existing DraftState (from a checkpoint) or None to start fresh.
        checkpoint_dir: Directory where checkpoints are written.
    """

    def __init__(
        self,
        db: Database,
        network: HypothesisNetwork,
        session: Session | None = None,
        draft: DraftState | None = None,
        checkpoint_dir: str | Path = ".",
    ) -> None:
        self.db = db
        self.network = network
        self.checkpoint_dir = Path(checkpoint_dir)
        self.client = Anthropic()

        if session is None:
            self.session = Session(
                network_id=network.id,
                mode=SessionMode.DAGCreation,
            )
            # Persist immediately so resume can find it.
            self.db.save_session(self.session)
            # Register the session ID in the network.
            network.session_ids.append(self.session.id)
            self.db.update_network_ids(network.id, network.version_ids, network.session_ids)
        else:
            self.session = session

        self.draft = draft if draft is not None else DraftState()
        # Set checkpoint path if not already stored, then persist it.
        if not self.session.checkpoint_path:
            cp = _checkpoint_path(self.session.id, self.checkpoint_dir)
            self.session.checkpoint_path = str(cp)
            self.db.update_session(self.session)

    # ------------------------------------------------------------------
    # Public: run the interactive loop (used by CLI)
    # ------------------------------------------------------------------

    def run(self, console: Any) -> DAGVersion | None:
        """Run the conversational loop.

        Args:
            console: A Rich Console instance for terminal output.

        Returns:
            The saved DAGVersion if the user confirmed, or None if the session
            was abandoned or the loop exited without saving.
        """
        from rich.markdown import Markdown
        from rich.panel import Panel
        from rich.prompt import Prompt

        console.print(
            Panel(
                f"[bold cyan]DAG Agent[/bold cyan]\n"
                f"Network: [green]{self.network.name}[/green]  "
                f"Session: [dim]{self.session.id[:8]}[/dim]\n\n"
                "Collaborate to build your causal hypothesis network.\n"
                "Type [bold]/confirm[/bold] to save, [bold]/show[/bold] to view current DAG, "
                "[bold]/help[/bold] for commands, or [bold]/quit[/bold] to exit.",
                title="causal-hypothesis-engine",
            )
        )

        # Seed first assistant message if starting fresh.
        if not self.session.conversation_history:
            intro = self._call_llm([
                {
                    "role": "user",
                    "content": (
                        f"I want to build a causal DAG for the domain: "
                        f"{self.network.domain or 'general'}. "
                        f"The network is named '{self.network.name}'. "
                        "Please introduce yourself briefly and ask me what the main "
                        "outcome or exposure I want to understand."
                    ),
                }
            ])
            self._record_exchange(
                {"role": "user", "content": f"[Starting session for network: {self.network.name}]"},
                {"role": "assistant", "content": intro},
            )
            console.print(Markdown(intro))

        while True:
            try:
                user_input = Prompt.ask("\n[bold green]You[/bold green]")
            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Session interrupted. Progress checkpointed.[/yellow]")
                self._do_checkpoint()
                break

            cmd = user_input.strip().lower()

            if cmd in ("/quit", "/exit", "/q"):
                console.print("[yellow]Exiting session. Progress checkpointed.[/yellow]")
                self._do_checkpoint()
                break

            if cmd == "/help":
                console.print(self._help_text())
                continue

            if cmd == "/show":
                console.print(Panel(self.draft.summary(), title="Current DAG"))
                continue

            if cmd in ("/confirm", "/save"):
                result = self._confirm_and_save(console)
                if result is not None:
                    return result
                continue

            # Normal conversational exchange.
            assistant_reply = self._chat(user_input)
            self._record_exchange(
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": assistant_reply},
            )
            console.print(Markdown(assistant_reply))

            # Checkpoint every N exchanges.
            if self.session.exchange_count % CHECKPOINT_EVERY == 0:
                self._do_checkpoint()
                console.print(
                    f"[dim]✓ Checkpointed at exchange {self.session.exchange_count}[/dim]"
                )

        return None

    # ------------------------------------------------------------------
    # Confirm and save
    # ------------------------------------------------------------------

    def _confirm_and_save(self, console: Any) -> DAGVersion | None:
        from rich.markdown import Markdown
        from rich.panel import Panel
        from rich.prompt import Confirm

        if not self.draft.nodes:
            console.print("[red]No nodes in the DAG yet. Nothing to save.[/red]")
            return None

        # Ask the LLM to produce a clean summary.
        summary_prompt = (
            "The user wants to confirm and save the DAG. "
            "Please produce a clear, structured summary of the current causal hypothesis network: "
            "list all nodes with their type and measurability state, all edges with direction and "
            "rationale, and any open questions or caveats. "
            "End with: 'Ready to save this as a Draft DAGVersion?'"
        )
        summary = self._chat(summary_prompt)
        console.print(Markdown(summary))
        self._record_exchange(
            {"role": "user", "content": "[/confirm triggered]"},
            {"role": "assistant", "content": summary},
        )

        confirmed = Confirm.ask("\n[bold]Save this as a Draft DAGVersion?[/bold]")
        if not confirmed:
            console.print("[yellow]Save cancelled. Continue editing.[/yellow]")
            return None

        return self._save_version(console)

    def _save_version(self, console: Any) -> DAGVersion:
        from rich.panel import Panel

        version = self.draft.to_dag_version(self.network.id)
        self.db.save_version(version)

        # Register version in network.
        self.network.version_ids.append(version.version_id)
        self.db.update_network_ids(
            self.network.id, self.network.version_ids, self.network.session_ids
        )

        # Mark session as confirmed.
        self.session.status = SessionStatus.Confirmed
        self.db.update_session(self.session)

        console.print(
            Panel(
                f"[bold green]DAGVersion saved![/bold green]\n"
                f"Version ID: [cyan]{version.version_id}[/cyan]\n"
                f"Nodes: {len(version.nodes)}   Edges: {len(version.edges)}\n"
                f"Status: [green]{version.status.value}[/green]",
                title="Saved",
            )
        )
        return version

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    def _chat(self, user_message: str) -> str:
        """Send user_message in context and return the assistant reply."""
        history = _trim_history(
            list(self.session.conversation_history), self.draft
        )
        messages = history + [{"role": "user", "content": user_message}]
        return self._call_llm(messages)

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=_SYSTEM_PROMPT,
            messages=messages,
        )
        return response.content[0].text

    # ------------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------------

    def _record_exchange(
        self, user_msg: dict[str, str], assistant_msg: dict[str, str]
    ) -> None:
        self.session.conversation_history.append(user_msg)
        self.session.conversation_history.append(assistant_msg)
        self.session.exchange_count += 1
        self.session.last_activity = datetime.utcnow()
        self.db.update_session(self.session)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _do_checkpoint(self) -> None:
        draft_version = self.draft.to_dag_version(self.network.id)
        write_checkpoint(self.session, draft_version, self.session.checkpoint_path)

    # ------------------------------------------------------------------
    # Node/edge mutation helpers (called by CLI or future tool-use)
    # ------------------------------------------------------------------

    def add_node(
        self,
        label: str,
        node_type: NodeType,
        description: str = "",
        measurability_state: MeasurabilityState = MeasurabilityState.Hypothetical,
    ) -> Node:
        node = Node(
            label=label,
            description=description,
            node_type=node_type,
            measurability_state=measurability_state,
        )
        self.draft.add_node(node)
        return node

    def add_edge(
        self,
        source_label: str,
        target_label: str,
        label: str = "",
        description: str = "",
    ) -> Edge | None:
        src = self.draft.find_node_by_label(source_label)
        tgt = self.draft.find_node_by_label(target_label)
        if src is None or tgt is None:
            return None
        edge = Edge(
            source_node_id=src.id,
            target_node_id=tgt.id,
            label=label,
            description=description,
        )
        self.draft.add_edge(edge)
        return edge

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    @staticmethod
    def _help_text() -> str:
        return (
            "\n[bold]Available commands:[/bold]\n"
            "  [cyan]/show[/cyan]     — display current DAG (nodes and edges)\n"
            "  [cyan]/confirm[/cyan]  — summarise DAG and save as Draft version\n"
            "  [cyan]/help[/cyan]     — show this help\n"
            "  [cyan]/quit[/cyan]     — exit (progress is checkpointed)\n"
        )
