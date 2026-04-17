# causal-hypothesis-engine

A domain-agnostic, human-in-the-loop system for constructing, versioning, and experimenting with causal hypothesis networks.

Build causal DAGs collaboratively with an AI agent. Save versioned states, backtest against real data, and iteratively refine hypotheses. Works with no data at all (pure theory mode) or with a domain adapter attached.

---

## Installation

Requires Python 3.12+.

```bash
git clone https://github.com/yourorg/causal-hypothesis-engine
cd causal-hypothesis-engine
python3.12 -m pip install -e .
```

Set your API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Verify everything is working:

```bash
causal-engine doctor
```

---

## Quick Start

### 1. Try the demo (no API key needed)

```bash
causal-engine demo
```

Builds a 4-node insurance flood claims DAG across two versions and renders a live diff. Good first check that persistence and rendering are working.

### 2. Start a new causal hypothesis session

```bash
causal-engine new --name "Revenue churn hypothesis" --domain "SaaS"
```

This launches an interactive DAGAgent session. The agent will ask you about your outcome of interest and help you build a causal graph node by node.

**In-session commands:**

| Command    | What it does                                     |
|------------|--------------------------------------------------|
| `/show`    | Print the current DAG state                      |
| `/confirm` | Summarise the DAG and save it as a DAGVersion    |
| `/help`    | List all commands                                |
| `/quit`    | Exit without saving                              |

Sessions checkpoint automatically every 10 exchanges, so nothing is lost if interrupted.

### 3. Resume an interrupted session

```bash
causal-engine resume --last       # most recent session
causal-engine resume <session-id> # specific session
```

### 4. List all networks and versions

```bash
causal-engine list
```

Shows every network with its version count and IDs you can copy-paste into other commands.

---

## All Commands

```
causal-engine new                            Start a new network + DAGAgent session
causal-engine resume [SESSION_ID] [--last]   Resume an interrupted session
causal-engine list                           List all networks and versions
causal-engine modify <version-id>            Start a ModificationAgent session
causal-engine backtest <version-id>          Run BacktestAgent on a version
causal-engine compare <v-id-1> <v-id-2>      Diff two DAGVersions
causal-engine doctor [--bigquery]            Check environment
causal-engine demo                           Canned demo (no API key needed)
```

---

## Modifying an Existing DAG

```bash
causal-engine modify <version-id> --mode theory
causal-engine modify <version-id> --mode backtest
causal-engine modify <version-id> --mode hybrid
```

Three modes:

- **theory** — pure causal reasoning, no data required
- **backtest** — proposals driven by backtest results (requires a Tested version)
- **hybrid** — combines both signals, explicitly flags where theory and data agree or disagree

---

## Backtesting Against Data

The insurance adapter reads local CSV or Parquet files. Prepare a file with at minimum these columns:

```
claim_id, claim_type, claim_amount, is_large_claim
```

Run the backtest:

```bash
causal-engine backtest <version-id> --data claims.csv
```

This scores the DAG's proxy features against a baseline and attaches a `BacktestResult` to the version. The version status transitions to `Tested`. Once tested, the version is immutable — further modifications produce a new child version.

---

## Comparing Versions

```bash
causal-engine compare <v-id-1> <v-id-2>
```

Outputs a colour-coded diff showing nodes added/removed/modified, edges added/removed/modified, and metadata changes. Matching is identity-based: nodes are tracked by their `id` field, so a re-created node with the same label but a new id appears as a remove + add.

---

## Node Model

Every node has a universal layer:

| Field                | Type                                                  |
|----------------------|-------------------------------------------------------|
| `label`              | str                                                   |
| `node_type`          | Exposure / Outcome / Confounder / Mediator / Collider |
| `measurability_state`| Hypothetical → Identified → Proxied → Validated       |
| `description`        | str (optional)                                        |

The measurability lifecycle:

1. **Hypothetical** — exists in theory only
2. **Identified** — a real data column or proxy is known
3. **Proxied** — a proxy variable is mapped and ready to feature-engineer
4. **Validated** — proxy has been backtested and shown to add signal

---

## Data Storage

All state is stored locally in SQLite at `~/.causal_engine/causal_engine.db`. WAL mode is enabled. Nothing is sent anywhere except Anthropic API calls for agent interactions.

Checkpoints are written atomically (tmp file → fsync → rename) to `~/.causal_engine/checkpoints/`.

---

## Project Structure

```
causal_hypothesis_engine/
  agents/
    dag_agent.py          # Interactive DAG builder (human-in-the-loop)
    modification_agent.py # Theory / Backtest / Hybrid modifier
    backtest_agent.py     # Automated scoring pipeline
  adapters/
    base.py               # AdapterBase abstract class
    insurance.py          # InsuranceClaimsAdapter (CSV/Parquet)
  models/
    node.py               # Node with typed adapter mixin
    edge.py
    dag_version.py
    session.py
    network.py
    backtest_result.py
    modification_proposal.py
  persistence/
    db.py                 # SQLite layer, migrations, schema_version
    checkpoint.py         # Atomic checkpoint writer
  comparison.py           # DAGVersion diff engine
  cli.py                  # Click CLI entry points
```

---

## What's Next (TODOS)

- Export to DOT / Mermaid / JSON (`causal-engine export`)
- BigQuery integration for the Insurance adapter
- MCP server mode for Claude Code integration
- FinancialEventsAdapter, ClinicalAdapter
- REST API and web visualisation
