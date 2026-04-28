# TODOS — causal-hypothesis-engine

Items deferred from /autoplan review (2026-04-13). Updated 2026-04-28.

## Shipped
- [x] `causal-engine export <version-id> --format mermaid|dot|json` — export DAG to text formats
- [x] `causal-engine view <version-id>` — 3D interactive graph viewer (opens in browser)
- [x] `FinancialDataAdapter` — FRED, Yahoo Finance, local file, url_csv sources
- [x] `DatasetBuilder` — fetch → resample → transform → ADF → Parquet pipeline
- [x] `causal-engine dataset <version-id> --manifest` — full financial data pipeline CLI command
- [x] `causal-engine doctor --financial` — check FRED key + financial package installs

## Next Up
- [ ] BacktestAgent integration for FinancialDataAdapter (v2) — `compute_dag_score` currently returns 0.0
- [ ] MCP server mode — expose DAGAgent as MCP tool for Claude Code integration
- [ ] BigQuery integration — replace CSV adapter with BigQuery (InsuranceClaimsAdapter)

## Adapters (v2+)
- [ ] ClinicalAdapter
- [ ] HTML table scraping source (`web`) for FinancialDataAdapter

## Distribution & Ecosystem
- [ ] Publish to PyPI
- [ ] Team/multi-user collaboration (SQLite → cloud persistence)
- [ ] REST API for embedding in analyst workflows

## DX Improvements
- [ ] Interactive playground (no install, browser-based)
- [ ] `causal-engine version` command
- [ ] Shared hypothesis network library
- [ ] Import manifest from existing DAGVersion (round-trip)

## Architecture Improvements
- [ ] Graph isomorphism checking in compare (beyond simple node/edge diff)
- [ ] SQLite → PostgreSQL migration path
- [ ] Conversation history export
- [ ] DatasetResult persistence to SQLite (currently JSON-only)
