# TODOS — causal-hypothesis-engine

Items deferred from /autoplan review (2026-04-13):

## Post-v1 Features
- [ ] `causal-engine export <version-id>` — export to DOT/Mermaid/JSON
- [ ] MCP server mode — expose DAGAgent as MCP tool for Claude Code integration
- [ ] BigQuery integration — replace CSV adapter with BigQuery (InsuranceClaimsAdapter)
- [ ] FinancialEventsAdapter (v2)
- [ ] ClinicalAdapter (v2)

## Distribution & Ecosystem
- [ ] Publish to PyPI
- [ ] Team/multi-user collaboration (SQLite → cloud persistence)
- [ ] REST API for embedding in analyst workflows
- [ ] Web visualization (export DAG to Mermaid/D3)

## DX Improvements (post-v1)
- [ ] Interactive playground (no install, browser-based)
- [ ] `causal-engine version` command
- [ ] Shared hypothesis network library

## Architecture Improvements (post-v1)
- [ ] Graph isomorphism checking in compare (beyond simple node/edge diff)
- [ ] SQLite → PostgreSQL migration path
- [ ] Conversation history export
