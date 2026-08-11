"""ModificationAgent — human-in-the-loop DAG modification agent.

Three modes (controlled by Session.modification_mode):
  THEORY   — pure causal reasoning, no data required
  BACKTEST — data-driven proposals grounded in backtest results
  HYBRID   — combines both signals; explicitly flags agreement/disagreement

If BACKTEST or HYBRID is requested but the source DAGVersion has no
backtest_result, the agent falls back to THEORY and explains why.

Workflow:
1. Agent reads the source DAGVersion and proposes modifications one at a time.
2. Each proposal is a structured ModificationProposal (never freeform text).
3. The user approves or rejects each proposal before the next is made.
4. Every 10 exchanges the session is checkpointed.
5. On /confirm, a new DAGVersion is created with parent linkage +
   modification_rationale, status Draft.
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime

from .._time import utcnow
from pathlib import Path
from typing import Any

from anthropic import Anthropic

from ..models.backtest_result import BacktestResult
from ..models.dag_version import DAGVersion, DAGVersionStatus
from ..models.edge import Edge
from ..models.modification_proposal import (
    AddEdgeChange,
    AddNodeChange,
    ModificationProposal,
    RemoveEdgeChange,
    RemoveNodeChange,
    UpdateEdgeChange,
    UpdateNodeChange,
)
from ..models.network import HypothesisNetwork
from ..models.node import MeasurabilityState, Node, NodeType
from ..models.session import ModificationMode, Session, SessionMode, SessionStatus
from ..persistence.checkpoint import write_checkpoint
from ..persistence.db import Database

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECKPOINT_EVERY = 10
HISTORY_WINDOW = 20
MODEL = "claude-opus-4-6"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_THEORY = """\
You are a causal reasoning assistant helping a researcher refine an existing \
causal hypothesis network (DAG).

Your role is to propose targeted modifications — one at a time — to improve \
the causal structure. Each modification must be proposed as a structured JSON \
object matching the ModificationProposal schema. Never propose more than one \
change per message.

ModificationProposal schema:
{
  "source": "theory",
  "confidence": <float 0.0–1.0>,
  "rationale": "<one paragraph explaining WHY this change improves the DAG>",
  "proposed_change": {
    "change_type": "<add_node|remove_node|update_node|add_edge|remove_edge|update_edge>",
    ... (fields depend on change_type — see below)
  }
}

change_type field reference:
  add_node:    { label, node_type, measurability_state, description }
  remove_node: { node_id }
  update_node: { node_id, label?, description?, node_type?, measurability_state? }
  add_edge:    { source_node_id, target_node_id, label, description }
  remove_edge: { edge_id }
  update_edge: { edge_id, label?, description? }

node_type values: Exposure | Outcome | Confounder | Mediator | Collider
measurability_state values: Hypothetical | Identified | Proxied | Validated

IMPORTANT RULES:
- Always wrap your proposal in a ```json ... ``` fenced code block.
- After the JSON block, add a brief natural-language summary for the user.
- Never propose changes that create a cycle in the DAG.
- Never overstate causal claims — this is hypothesis generation.
- If the user rejects a proposal, acknowledge it and propose something different.
- When you have no more proposals, say so explicitly.
"""

_SYSTEM_PROMPT_BACKTEST = """\
You are a causal reasoning assistant helping a researcher refine an existing \
causal hypothesis network (DAG) using empirical backtest results.

You have been given the backtest results showing which proxy nodes added \
predictive signal and which did not. Your proposals must be grounded in \
this empirical evidence — not just causal theory.

Propose modifications — one at a time — using this schema:

{
  "source": "data",
  "confidence": <float 0.0–1.0>,
  "rationale": "<explain how the backtest evidence motivates this change>",
  "proposed_change": {
    "change_type": "<add_node|remove_node|update_node|add_edge|remove_edge|update_edge>",
    ... (same field reference as THEORY mode)
  }
}

CRITICAL — what predictive contribution does and does not tell you:

The backtest measures PREDICTIVE lift, not causal validity. These come apart, \
and in a specific direction you must guard against:

- A CONFOUNDER often adds little predictive lift yet must be retained — it is \
  what makes an effect estimate valid. Never propose removing a node because \
  its contribution is small.
- A COLLIDER, or any descendant of the outcome, typically adds a LOT of \
  predictive lift while destroying identification. High contribution is a \
  reason to check whether the node is a collider, not a reason to celebrate.
- A MEDIATOR adds lift but conditioning on it blocks the very indirect effect \
  you may be trying to measure.

Guidance for data-driven proposals:
- "inconclusive" means the confidence interval spans zero: the data cannot \
  distinguish that node's contribution from nothing. It is NOT evidence of no \
  effect, and NOT grounds for removal. Suggest more data or a better proxy.
- A node excluded because its column is missing from the file has a \
  contribution of zero for a purely mechanical reason. Check for a typo before \
  drawing any inference.
- Only propose removing a node on THEORETICAL grounds — you no longer believe \
  the causal relationship — never because its measured contribution was low.
- Never propose promoting a node to Validated on the basis of predictive \
  contribution alone. Validation requires the node's implied conditional \
  independencies to survive testing against data \
  (`causal-engine check`), not a favourable AUC delta.
- Low overall lift: suggest nodes/edges that might capture unexplained \
  variance, and consider whether the outcome is measurable at all.

node_type values: Exposure | Outcome | Confounder | Mediator | Collider
measurability_state values: Hypothetical | Identified | Proxied | Validated

IMPORTANT RULES:
- Always wrap your proposal in a ```json ... ``` fenced code block.
- After the JSON block, add a brief natural-language summary citing the data.
- Never overstate causal claims — this is structured hypothesis generation.
- Distinguish between "the data shows X" and "we hypothesise X causes Y".
- If the user rejects a proposal, acknowledge and propose something different.
- When you have no more data-driven proposals, say so explicitly.
"""

_SYSTEM_PROMPT_HYBRID = """\
You are a causal reasoning assistant helping a researcher refine an existing \
causal hypothesis network (DAG) using both causal theory and empirical backtest results.

You must evaluate each proposed change from BOTH perspectives:
  1. Does causal theory support it?
  2. Does the backtest data support it?

Then explicitly flag whether theory and data AGREE or DISAGREE.

Use this schema — note the required theory_data_agreement field:

{
  "source": "both",
  "confidence": <float 0.0–1.0>,
  "rationale": "<explain both the theoretical AND empirical reasoning>",
  "theory_data_agreement": <true if theory and data point the same direction, \
false if they conflict>,
  "proposed_change": {
    "change_type": "<add_node|remove_node|update_node|add_edge|remove_edge|update_edge>",
    ... (same field reference as THEORY mode)
  }
}

When theory and data AGREE (theory_data_agreement: true):
- State clearly that both signals point to this change.
- Express higher confidence.

When theory and data DISAGREE (theory_data_agreement: false):
- Explain the conflict explicitly: "Theory predicts X, but the data shows Y."
- Suggest a way to resolve the tension (better proxy, confounding variable, etc.).
- Express lower confidence and let the user decide.

node_type values: Exposure | Outcome | Confounder | Mediator | Collider
measurability_state values: Hypothetical | Identified | Proxied | Validated

IMPORTANT RULES:
- Always wrap your proposal in a ```json ... ``` fenced code block.
- theory_data_agreement is REQUIRED in every proposal in HYBRID mode.
- Never suppress a disagreement — surface it, explain it, let the human decide.
- Never overstate causal claims — this is structured hypothesis generation.
- If the user rejects a proposal, acknowledge and propose something different.
"""

# ---------------------------------------------------------------------------
# Backtest context formatter
# ---------------------------------------------------------------------------


def _format_backtest_context(
    backtest_result: BacktestResult,
    version: DAGVersion,
) -> str:
    """Produce a human-readable + LLM-readable summary of a BacktestResult."""
    node_map = {n.id: n.label for n in version.nodes}
    lines = [
        "BACKTEST RESULTS:",
        f"  Baseline score (no proxy features): {backtest_result.baseline_score:.4f}",
        f"  DAG score (with proxy features):    {backtest_result.dag_score:.4f}",
        f"  Lift:                               {backtest_result.lift:+.4f}",
        "",
        "Per-node contributions (positive = added signal, negative = hurt):",
    ]
    if backtest_result.node_contributions:
        sorted_contribs = sorted(
            backtest_result.node_contributions.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for node_id, contrib in sorted_contribs:
            label = node_map.get(node_id, node_id[:8])
            signal = "✓ added signal" if contrib > 0 else "✗ no signal"
            lines.append(f"  [{node_id[:8]}] {label}: {contrib:+.4f}  ({signal})")
    else:
        lines.append("  (no per-node data available)")

    if backtest_result.notes:
        lines += ["", f"Notes: {backtest_result.notes}"]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers: extract JSON proposal from LLM response
# ---------------------------------------------------------------------------

_JSON_FENCE = "```json"


def _extract_proposal_json(text: str) -> ModificationProposal | None:
    """Parse the first ```json ... ``` block in *text* as a ModificationProposal."""
    start = text.find(_JSON_FENCE)
    if start == -1:
        return None
    inner_start = start + len(_JSON_FENCE)
    end = text.find("```", inner_start)
    if end == -1:
        return None
    raw = text[inner_start:end].strip()
    try:
        data = json.loads(raw)
        return ModificationProposal.model_validate(data)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# MutableDAG — working copy of the DAG being modified
# ---------------------------------------------------------------------------


class MutableDAG:
    """A working copy of a DAGVersion's nodes and edges."""

    def __init__(self, source: DAGVersion) -> None:
        self.nodes: list[Node] = deepcopy(source.nodes)
        self.edges: list[Edge] = deepcopy(source.edges)
        self.source_version_id = source.version_id

    # -- lookups -------------------------------------------------------------

    def get_node(self, node_id: str) -> Node | None:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def get_edge(self, edge_id: str) -> Edge | None:
        for e in self.edges:
            if e.id == edge_id:
                return e
        return None

    # -- mutations -----------------------------------------------------------

    def apply(self, proposal: ModificationProposal) -> tuple[bool, str]:
        """Apply the proposed_change to the working copy.

        Returns:
            (success, message) — message explains what changed or why it failed.
        """
        change = proposal.proposed_change

        if isinstance(change, AddNodeChange):
            node = Node(
                label=change.label,
                description=change.description,
                node_type=NodeType(change.node_type),
                measurability_state=MeasurabilityState(change.measurability_state),
            )
            self.nodes.append(node)
            return True, f"Added node '{change.label}' ({change.node_type})"

        if isinstance(change, RemoveNodeChange):
            before = len(self.nodes)
            self.nodes = [n for n in self.nodes if n.id != change.node_id]
            self.edges = [
                e for e in self.edges
                if e.source_node_id != change.node_id
                and e.target_node_id != change.node_id
            ]
            if len(self.nodes) == before:
                return False, f"Node {change.node_id} not found."
            return True, f"Removed node {change.node_id} and its connected edges."

        if isinstance(change, UpdateNodeChange):
            node = self.get_node(change.node_id)
            if node is None:
                return False, f"Node {change.node_id} not found."
            if change.label is not None:
                node.label = change.label
            if change.description is not None:
                node.description = change.description
            if change.node_type is not None:
                node.node_type = NodeType(change.node_type)
            if change.measurability_state is not None:
                node.measurability_state = MeasurabilityState(
                    change.measurability_state
                )
            return True, f"Updated node {change.node_id}."

        if isinstance(change, AddEdgeChange):
            src = self.get_node(change.source_node_id)
            tgt = self.get_node(change.target_node_id)
            if src is None:
                return False, f"Source node {change.source_node_id} not found."
            if tgt is None:
                return False, f"Target node {change.target_node_id} not found."
            edge = Edge(
                source_node_id=change.source_node_id,
                target_node_id=change.target_node_id,
                label=change.label,
                description=change.description,
            )
            self.edges.append(edge)
            return True, f"Added edge {src.label} → {tgt.label}."

        if isinstance(change, RemoveEdgeChange):
            before = len(self.edges)
            self.edges = [e for e in self.edges if e.id != change.edge_id]
            if len(self.edges) == before:
                return False, f"Edge {change.edge_id} not found."
            return True, f"Removed edge {change.edge_id}."

        if isinstance(change, UpdateEdgeChange):
            edge = self.get_edge(change.edge_id)
            if edge is None:
                return False, f"Edge {change.edge_id} not found."
            if change.label is not None:
                edge.label = change.label
            if change.description is not None:
                edge.description = change.description
            return True, f"Updated edge {change.edge_id}."

        return False, "Unknown change type."

    def summary(self) -> str:
        if not self.nodes:
            return "No nodes."
        lines = ["NODES:"]
        for n in self.nodes:
            lines.append(
                f"  [{n.id[:8]}] {n.label} ({n.node_type.value}, "
                f"{n.measurability_state.value})"
            )
            if n.description:
                lines.append(f"      {n.description}")
        if self.edges:
            lines.append("EDGES:")
            node_map = {n.id: n.label for n in self.nodes}
            for e in self.edges:
                src = node_map.get(e.source_node_id, e.source_node_id[:8])
                tgt = node_map.get(e.target_node_id, e.target_node_id[:8])
                lines.append(f'  [{e.id[:8]}] {src} → {tgt}  "{e.label}"')
        else:
            lines.append("EDGES: none")
        return "\n".join(lines)

    def to_dag_version(self, network_id: str, parent_version_id: str) -> DAGVersion:
        return DAGVersion(
            network_id=network_id,
            nodes=deepcopy(self.nodes),
            edges=deepcopy(self.edges),
            parent_version_id=parent_version_id,
            status=DAGVersionStatus.Draft,
        )


# ---------------------------------------------------------------------------
# History trimming (same pattern as DAGAgent)
# ---------------------------------------------------------------------------


def _trim_history(
    history: list[dict[str, str]],
    dag: MutableDAG,
    window: int = HISTORY_WINDOW,
) -> list[dict[str, str]]:
    if len(history) <= window:
        return history
    summary_msg = {
        "role": "user",
        "content": (
            "[CONTEXT SUMMARY — earlier messages trimmed]\n"
            "Current DAG state:\n" + dag.summary()
        ),
    }
    return [summary_msg] + history[-window:]


# ---------------------------------------------------------------------------
# ModificationAgent
# ---------------------------------------------------------------------------


class ModificationAgent:
    """Human-in-the-loop agent for refining an existing DAGVersion.

    Supports THEORY, BACKTEST, and HYBRID modes. BACKTEST and HYBRID require
    the source DAGVersion to have a backtest_result attached; if not present
    both modes fall back to THEORY automatically.

    Args:
        db: Open Database instance.
        network: The HypothesisNetwork this session belongs to.
        source_version: The DAGVersion to modify.
        modification_mode: THEORY, BACKTEST, or HYBRID. If BACKTEST/HYBRID
            is requested but the version has no backtest_result, falls back
            to THEORY.
        session: Existing Session for resume, or None to create a new one.
        checkpoint_dir: Directory for checkpoints.
    """

    def __init__(
        self,
        db: Database,
        network: HypothesisNetwork,
        source_version: DAGVersion,
        modification_mode: ModificationMode = ModificationMode.Theory,
        session: Session | None = None,
        checkpoint_dir: str | Path = ".",
    ) -> None:
        self.db = db
        self.network = network
        self.source_version = source_version
        self.checkpoint_dir = Path(checkpoint_dir)
        self.client = Anthropic()

        # Fall back to THEORY if BACKTEST/HYBRID requested but no backtest result.
        self.modification_mode = modification_mode
        self._fell_back = False
        if modification_mode in (ModificationMode.Backtest, ModificationMode.Hybrid):
            if source_version.backtest_result is None:
                self.modification_mode = ModificationMode.Theory
                self._fell_back = True

        if session is None:
            self.session = Session(
                network_id=network.id,
                mode=SessionMode.Modification,
                modification_mode=self.modification_mode,
            )
            self.db.save_session(self.session)
            network.session_ids.append(self.session.id)
            self.db.update_network_ids(network.id, network.version_ids, network.session_ids)
        else:
            self.session = session

        self.dag = MutableDAG(source_version)
        self.accepted_proposals: list[ModificationProposal] = []

        if not self.session.checkpoint_path:
            from .dag_agent import _checkpoint_path
            cp = _checkpoint_path(self.session.id, self.checkpoint_dir)
            self.session.checkpoint_path = str(cp)
            self.db.update_session(self.session)

    # ------------------------------------------------------------------
    # Public: run the interactive loop
    # ------------------------------------------------------------------

    def run(self, console: Any) -> DAGVersion | None:
        from rich.markdown import Markdown
        from rich.panel import Panel
        from rich.prompt import Prompt

        mode_label = self.modification_mode.value.upper()
        console.print(
            Panel(
                f"[bold cyan]Modification Agent[/bold cyan]  mode=[bold]{mode_label}[/bold]\n"
                f"Network: [green]{self.network.name}[/green]  "
                f"Source version: [dim]{self.source_version.version_id[:8]}[/dim]\n\n"
                "The agent will propose changes one at a time.\n"
                "Type [bold]yes[/bold]/[bold]no[/bold] to accept/reject each proposal.\n"
                "Type [bold]/confirm[/bold] to save the modified DAG, "
                "[bold]/show[/bold] to view current state, [bold]/quit[/bold] to exit.",
                title="causal-hypothesis-engine — modify",
            )
        )

        if self._fell_back:
            console.print(
                "[yellow]Note:[/yellow] BACKTEST/HYBRID requested but the source version "
                "has no backtest result. Falling back to THEORY mode.\n"
            )

        # Seed with opening analysis.
        if not self.session.conversation_history:
            opening = self._call_llm(self._build_opening_messages())
            self._record_exchange(
                {"role": "user", "content": "[session start]"},
                {"role": "assistant", "content": opening},
            )
            console.print(Markdown(opening))

        while True:
            try:
                user_input = Prompt.ask("\n[bold green]You[/bold green]")
            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Session interrupted. Progress checkpointed.[/yellow]")
                self._do_checkpoint()
                break

            cmd = user_input.strip().lower()

            if cmd in ("/quit", "/exit", "/q"):
                console.print("[yellow]Exiting. Progress checkpointed.[/yellow]")
                self._do_checkpoint()
                break

            if cmd == "/help":
                console.print(self._help_text())
                continue

            if cmd == "/show":
                console.print(Panel(self.dag.summary(), title="Current DAG"))
                continue

            if cmd in ("/confirm", "/save"):
                result = self._confirm_and_save(console)
                if result is not None:
                    return result
                continue

            # Normal exchange — send to LLM, attempt to parse proposal.
            reply = self._chat(user_input)
            self._record_exchange(
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": reply},
            )
            console.print(Markdown(reply))

            # If the reply contains a proposal, handle accept/reject.
            proposal = _extract_proposal_json(reply)
            if proposal is not None:
                self._handle_proposal(proposal, console)

            if self.session.exchange_count % CHECKPOINT_EVERY == 0:
                self._do_checkpoint()
                console.print(
                    f"[dim]✓ Checkpointed at exchange {self.session.exchange_count}[/dim]"
                )

        return None

    # ------------------------------------------------------------------
    # Proposal handling
    # ------------------------------------------------------------------

    def _handle_proposal(self, proposal: ModificationProposal, console: Any) -> None:
        from rich.prompt import Confirm

        console.print(
            f"\n[bold]Proposal[/bold] (confidence {proposal.confidence:.0%}, "
            f"source: {proposal.source})"
        )
        accepted = Confirm.ask("Accept this change?")
        if accepted:
            ok, msg = self.dag.apply(proposal)
            if ok:
                self.accepted_proposals.append(proposal)
                console.print(f"[green]✓[/green] Applied: {msg}")
            else:
                console.print(f"[red]✗[/red] Could not apply: {msg}")
            # Feed result back to LLM.
            ack = "accepted" if ok else f"rejected — {msg}"
            self._record_exchange({"role": "user", "content": f"[Proposal {ack}]"})
        else:
            console.print("[yellow]Proposal rejected.[/yellow]")
            self._record_exchange(
                {"role": "user", "content": "[Proposal rejected by user]"}
            )

    # ------------------------------------------------------------------
    # Confirm and save
    # ------------------------------------------------------------------

    def _confirm_and_save(self, console: Any) -> DAGVersion | None:
        from rich.markdown import Markdown
        from rich.panel import Panel
        from rich.prompt import Confirm, Prompt

        if not self.accepted_proposals:
            console.print("[yellow]No changes accepted yet. Nothing new to save.[/yellow]")
            return None

        # Ask LLM for a rationale summary.
        rationale_prompt = (
            "The user wants to save this modified DAG. "
            "Write a concise modification rationale (2-4 sentences) summarising what changed "
            "and why, suitable for a version history record."
        )
        rationale_text = self._chat(rationale_prompt)
        console.print(Markdown(rationale_text))
        self._record_exchange(
            {"role": "user", "content": "[/confirm triggered]"},
            {"role": "assistant", "content": rationale_text},
        )

        # Let user edit the rationale if they want.
        final_rationale = Prompt.ask(
            "\n[bold]Rationale[/bold] (press Enter to use above, or type your own)",
            default=rationale_text,
        )

        confirmed = Confirm.ask("Save as new Draft DAGVersion?")
        if not confirmed:
            console.print("[yellow]Save cancelled.[/yellow]")
            return None

        return self._save_version(final_rationale, console)

    def _save_version(self, rationale: str, console: Any) -> DAGVersion:
        from rich.panel import Panel

        version = self.dag.to_dag_version(self.network.id, self.source_version.version_id)
        version = version.model_copy(update={"modification_rationale": rationale})
        self.db.save_version(version)

        self.network.version_ids.append(version.version_id)
        self.db.update_network_ids(
            self.network.id, self.network.version_ids, self.network.session_ids
        )

        self.session.status = SessionStatus.Confirmed
        self.db.update_session(self.session)

        console.print(
            Panel(
                f"[bold green]Modified DAGVersion saved![/bold green]\n"
                f"Version ID:  [cyan]{version.version_id}[/cyan]\n"
                f"Parent ID:   [dim]{version.parent_version_id}[/dim]\n"
                f"Nodes: {len(version.nodes)}   Edges: {len(version.edges)}\n"
                f"Changes accepted: {len(self.accepted_proposals)}",
                title="Saved",
            )
        )
        return version

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    def _build_opening_messages(self) -> list[dict[str, str]]:
        dag_section = f"Here is the causal DAG you will help refine:\n\n{self.dag.summary()}"

        if self.modification_mode in (
            ModificationMode.Backtest, ModificationMode.Hybrid
        ) and self.source_version.backtest_result is not None:
            backtest_section = _format_backtest_context(
                self.source_version.backtest_result, self.source_version
            )
            if self.modification_mode == ModificationMode.Backtest:
                instruction = (
                    "Using the backtest results above, propose one data-driven "
                    "improvement to the DAG. Wrap your proposal in a ```json ... ``` block."
                )
            else:  # Hybrid
                instruction = (
                    "Using both the DAG structure and the backtest results, propose "
                    "one improvement. Explicitly state whether theory and data agree. "
                    "Wrap your proposal in a ```json ... ``` block. "
                    "Remember: theory_data_agreement is required."
                )
            content = f"{dag_section}\n\n{backtest_section}\n\n{instruction}"
        else:
            content = (
                f"{dag_section}\n\n"
                "Analyse the structure and propose one improvement. "
                "Wrap your proposal in a ```json ... ``` block as instructed."
            )

        return [{"role": "user", "content": content}]

    def _chat(self, user_message: str) -> str:
        history = _trim_history(list(self.session.conversation_history), self.dag)
        messages = history + [{"role": "user", "content": user_message}]
        return self._call_llm(messages)

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        system_prompt = {
            ModificationMode.Theory: _SYSTEM_PROMPT_THEORY,
            ModificationMode.Backtest: _SYSTEM_PROMPT_BACKTEST,
            ModificationMode.Hybrid: _SYSTEM_PROMPT_HYBRID,
        }.get(self.modification_mode, _SYSTEM_PROMPT_THEORY)

        response = self.client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text

    # ------------------------------------------------------------------
    # Session bookkeeping
    # ------------------------------------------------------------------

    def _record_exchange(
        self, user_msg: dict[str, str], assistant_msg: dict[str, str] | None = None
    ) -> None:
        """Append a turn to the persisted conversation history.

        Messages with empty content are dropped. The Anthropic Messages API
        rejects empty text content blocks, and this history is replayed
        verbatim on every subsequent turn — so a single empty placeholder
        turn would break the rest of the session *and*, because the history
        is persisted to SQLite and the checkpoint, break `resume` permanently.
        """
        for message in (user_msg, assistant_msg):
            if message and str(message.get("content", "")).strip():
                self.session.conversation_history.append(message)
        self.session.exchange_count += 1
        self.session.last_activity = utcnow()
        self.db.update_session(self.session)

    def _do_checkpoint(self) -> None:
        draft_version = self.dag.to_dag_version(
            self.network.id, self.source_version.version_id
        )
        write_checkpoint(self.session, draft_version, self.session.checkpoint_path)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    @staticmethod
    def _help_text() -> str:
        return (
            "\n[bold]Available commands:[/bold]\n"
            "  [cyan]yes / no[/cyan]   — accept or reject the current proposal\n"
            "  [cyan]/show[/cyan]      — display current DAG state\n"
            "  [cyan]/confirm[/cyan]   — save modified DAG as new Draft version\n"
            "  [cyan]/help[/cyan]      — show this help\n"
            "  [cyan]/quit[/cyan]      — exit (progress is checkpointed)\n"
        )
