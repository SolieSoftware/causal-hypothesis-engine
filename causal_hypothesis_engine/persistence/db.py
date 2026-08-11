"""SQLite persistence layer for causal-hypothesis-engine.

Design decisions:
- WAL mode + foreign keys enabled on every connection.
- DAG structures (nodes, edges, backtest_result) stored as JSON blobs.
- HypothesisNetwork stores only version_ids / session_ids as JSON arrays.
- Each method opens and closes its own connection — no persistent state.
- schema_version table tracks applied migrations from day one.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from causal_hypothesis_engine.models.backtest_result import BacktestResult
from causal_hypothesis_engine.models.dag_version import DAGVersion, DAGVersionStatus
from causal_hypothesis_engine.models.dataset_result import DatasetResult
from causal_hypothesis_engine.models.edge import Edge
from causal_hypothesis_engine.models.network import AdapterType, HypothesisNetwork
from causal_hypothesis_engine.models.node import Node
from causal_hypothesis_engine.models.session import (
    ModificationMode,
    Session,
    SessionMode,
    SessionStatus,
)

# ---------------------------------------------------------------------------
# Schema migrations
# Each entry is (target_version: int, sql: str).  They are applied in order
# whenever the stored schema_version is less than the target_version.
# Version 0 → 1 is the initial schema creation.
# ---------------------------------------------------------------------------

_MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS networks (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            domain      TEXT NOT NULL DEFAULT '',
            adapter     TEXT NOT NULL DEFAULT 'none',
            created_at  TEXT NOT NULL,
            version_ids TEXT NOT NULL DEFAULT '[]',
            session_ids TEXT NOT NULL DEFAULT '[]'
        );

        CREATE TABLE IF NOT EXISTS dag_versions (
            version_id               TEXT PRIMARY KEY,
            network_id               TEXT NOT NULL,
            created_at               TEXT NOT NULL,
            nodes_json               TEXT NOT NULL DEFAULT '[]',
            edges_json               TEXT NOT NULL DEFAULT '[]',
            parent_version_id        TEXT,
            modification_rationale   TEXT NOT NULL DEFAULT '',
            backtest_result_json     TEXT,
            status                   TEXT NOT NULL DEFAULT 'Draft',
            FOREIGN KEY (network_id) REFERENCES networks(id)
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id                       TEXT PRIMARY KEY,
            network_id               TEXT NOT NULL,
            mode                     TEXT NOT NULL,
            modification_mode        TEXT,
            checkpoint_path          TEXT NOT NULL DEFAULT '',
            conversation_history_json TEXT NOT NULL DEFAULT '[]',
            status                   TEXT NOT NULL DEFAULT 'Active',
            created_at               TEXT NOT NULL,
            last_activity            TEXT NOT NULL,
            exchange_count           INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (network_id) REFERENCES networks(id)
        );
        """,
    ),
    (
        2,
        # DatasetResult persistence. Datasets were previously written to disk
        # as orphan Parquet files with no record in the database, so there was
        # no way to answer "which data belongs to this hypothesis?".
        """
        CREATE TABLE IF NOT EXISTS dataset_results (
            id            TEXT PRIMARY KEY,
            version_id    TEXT NOT NULL,
            manifest_path TEXT NOT NULL,
            output_path   TEXT NOT NULL,
            columns_json  TEXT NOT NULL,
            start_date    TEXT NOT NULL,
            end_date      TEXT NOT NULL,
            frequency     TEXT NOT NULL,
            adf_json      TEXT NOT NULL,
            warnings_json TEXT NOT NULL,
            created_at    TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_dataset_results_version
            ON dataset_results(version_id);

        CREATE INDEX IF NOT EXISTS idx_dag_versions_network
            ON dag_versions(network_id);

        CREATE INDEX IF NOT EXISTS idx_sessions_network
            ON sessions(network_id);
        """,
    ),
]

CURRENT_SCHEMA_VERSION = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dt_to_str(dt: datetime) -> str:
    return dt.isoformat()


def _str_to_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _nodes_to_json(nodes: list[Node]) -> str:
    return json.dumps([n.model_dump(mode="json") for n in nodes])


def _json_to_nodes(raw: str) -> list[Node]:
    return [Node.model_validate(d) for d in json.loads(raw)]


def _edges_to_json(edges: list[Edge]) -> str:
    return json.dumps([e.model_dump(mode="json") for e in edges])


def _json_to_edges(raw: str) -> list[Edge]:
    return [Edge.model_validate(d) for d in json.loads(raw)]


def _backtest_to_json(br: BacktestResult | None) -> str | None:
    if br is None:
        return None
    return br.model_dump_json()


def _json_to_backtest(raw: str | None) -> BacktestResult | None:
    if raw is None:
        return None
    return BacktestResult.model_validate_json(raw)


def _conv_history_to_json(history: list[dict]) -> str:
    return json.dumps(history)


def _json_to_conv_history(raw: str) -> list[dict]:
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------


class Database:
    """SQLite-backed persistence layer.

    Each public method opens its own connection via :meth:`_connect` — a
    context manager that commits on success, rolls back on error, and
    **always closes**. It previously returned a bare ``sqlite3.Connection``
    used as ``with self._connect() as conn:``, which commits the transaction
    but does not close the handle, so every repository call leaked a file
    descriptor and a WAL reader.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # Connection factory
    # ------------------------------------------------------------------

    def _open(self) -> sqlite3.Connection:
        """Open a raw connection with per-connection pragmas applied.

        ``journal_mode=WAL`` is a persistent database property and is set once
        in :meth:`_init_schema`; ``foreign_keys`` is per-connection and must be
        set every time.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection, commit on success, roll back on error, always close."""
        conn = self._open()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def health(self) -> dict[str, str]:
        """Return diagnostic facts about the database, for ``causal-engine doctor``.

        Exists so the CLI does not have to reach into the connection factory.
        """
        info: dict[str, str] = {"db_path": str(self.db_path)}
        with self._connect() as conn:
            row = conn.execute("PRAGMA journal_mode").fetchone()
            info["journal_mode"] = str(row[0]) if row else "unknown"
            try:
                row = conn.execute(
                    "SELECT version FROM schema_version LIMIT 1"
                ).fetchone()
                info["schema_version"] = str(row[0]) if row else "unset"
            except sqlite3.Error as exc:
                info["schema_version"] = f"error: {exc}"
        return info

    # ------------------------------------------------------------------
    # Schema initialisation & migrations
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._connect() as conn:
            # WAL is a persistent database property — set once here rather than
            # incurring a disk write on every repository call.
            conn.execute("PRAGMA journal_mode=WAL")
            self._run_migrations(conn)

    def _run_migrations(self, conn: sqlite3.Connection) -> None:
        """Apply any pending migrations in version order."""
        # Determine the current version stored in the DB.
        # schema_version may not exist yet on a brand-new file.
        try:
            row = conn.execute("SELECT version FROM schema_version").fetchone()
            current_version: int = row["version"] if row else 0
        except sqlite3.OperationalError:
            # Table doesn't exist yet — we're starting from scratch.
            current_version = 0

        for target_version, sql in _MIGRATIONS:
            if current_version < target_version:
                # Execute each statement in the migration block individually
                # so that SQLite's implicit-transaction behaviour doesn't
                # cause issues with multi-statement strings.
                for statement in sql.strip().split(";"):
                    statement = statement.strip()
                    if statement:
                        conn.execute(statement)

                # Update or insert the schema_version record.
                existing = conn.execute(
                    "SELECT COUNT(*) as cnt FROM schema_version"
                ).fetchone()
                if existing["cnt"] == 0:
                    conn.execute(
                        "INSERT INTO schema_version (version) VALUES (?)",
                        (target_version,),
                    )
                else:
                    conn.execute(
                        "UPDATE schema_version SET version = ?",
                        (target_version,),
                    )
                conn.commit()
                current_version = target_version

    # ------------------------------------------------------------------
    # Networks
    # ------------------------------------------------------------------

    def create_network(self, network: HypothesisNetwork) -> None:
        """Insert a new HypothesisNetwork row."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO networks
                    (id, name, domain, adapter, created_at, version_ids, session_ids)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    network.id,
                    network.name,
                    network.domain,
                    network.adapter.value,
                    _dt_to_str(network.created_at),
                    json.dumps(network.version_ids),
                    json.dumps(network.session_ids),
                ),
            )
            conn.commit()

    def get_network(self, network_id: str) -> HypothesisNetwork | None:
        """Return a fully-hydrated HypothesisNetwork or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM networks WHERE id = ?", (network_id,)
            ).fetchone()
            if row is None:
                return None
            return HypothesisNetwork(
                id=row["id"],
                name=row["name"],
                domain=row["domain"],
                adapter=AdapterType(row["adapter"]),
                created_at=_str_to_dt(row["created_at"]),
                version_ids=json.loads(row["version_ids"]),
                session_ids=json.loads(row["session_ids"]),
            )

    def list_networks(self) -> list[dict[str, Any]]:
        """Return lean summary dicts — no full child collections loaded."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, name, domain, adapter, created_at, version_ids, session_ids FROM networks"
            ).fetchall()
            result = []
            for row in rows:
                version_ids: list[str] = json.loads(row["version_ids"])
                session_ids: list[str] = json.loads(row["session_ids"])
                result.append(
                    {
                        "id": row["id"],
                        "name": row["name"],
                        "domain": row["domain"],
                        "adapter": row["adapter"],
                        "created_at": row["created_at"],
                        "version_count": len(version_ids),
                        "session_count": len(session_ids),
                    }
                )
            return result

    def update_network_ids(
        self,
        network_id: str,
        version_ids: list[str],
        session_ids: list[str],
    ) -> None:
        """Overwrite the version_ids and session_ids arrays for a network."""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE networks
                SET version_ids = ?, session_ids = ?
                WHERE id = ?
                """,
                (json.dumps(version_ids), json.dumps(session_ids), network_id),
            )
            conn.commit()

    def delete_network(self, network_id: str) -> None:
        """Delete a network row (does not cascade to child tables)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM networks WHERE id = ?", (network_id,))
            conn.commit()

    # ------------------------------------------------------------------
    # DAGVersions
    # ------------------------------------------------------------------

    def save_version(self, version: DAGVersion) -> None:
        """Insert or replace a DAGVersion row."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO dag_versions
                    (version_id, network_id, created_at, nodes_json, edges_json,
                     parent_version_id, modification_rationale,
                     backtest_result_json, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version.version_id,
                    version.network_id,
                    _dt_to_str(version.created_at),
                    _nodes_to_json(version.nodes),
                    _edges_to_json(version.edges),
                    version.parent_version_id,
                    version.modification_rationale,
                    _backtest_to_json(version.backtest_result),
                    version.status.value,
                ),
            )
            conn.commit()

    def get_version(self, version_id: str) -> DAGVersion | None:
        """Return a fully-hydrated DAGVersion or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM dag_versions WHERE version_id = ?", (version_id,)
            ).fetchone()
            if row is None:
                return None
            return self._row_to_dag_version(row)

    def get_versions_for_network(self, network_id: str) -> list[DAGVersion]:
        """Return all DAGVersions belonging to a network, ordered by creation time."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM dag_versions WHERE network_id = ? ORDER BY created_at",
                (network_id,),
            ).fetchall()
            return [self._row_to_dag_version(r) for r in rows]

    def update_version_status(
        self,
        version_id: str,
        status: DAGVersionStatus,
        backtest_result: BacktestResult | None = None,
    ) -> None:
        """Update only the status (and optionally the backtest_result) of a version."""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE dag_versions
                SET status = ?, backtest_result_json = ?
                WHERE version_id = ?
                """,
                (
                    status.value,
                    _backtest_to_json(backtest_result),
                    version_id,
                ),
            )
            conn.commit()

    @staticmethod
    def _row_to_dag_version(row: sqlite3.Row) -> DAGVersion:
        return DAGVersion(
            version_id=row["version_id"],
            network_id=row["network_id"],
            created_at=_str_to_dt(row["created_at"]),
            nodes=_json_to_nodes(row["nodes_json"]),
            edges=_json_to_edges(row["edges_json"]),
            parent_version_id=row["parent_version_id"],
            modification_rationale=row["modification_rationale"] or "",
            backtest_result=_json_to_backtest(row["backtest_result_json"]),
            status=DAGVersionStatus(row["status"]),
        )

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def save_session(self, session: Session) -> None:
        """Insert a new Session row."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions
                    (id, network_id, mode, modification_mode, checkpoint_path,
                     conversation_history_json, status, created_at,
                     last_activity, exchange_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.id,
                    session.network_id,
                    session.mode.value,
                    session.modification_mode.value
                    if session.modification_mode
                    else None,
                    session.checkpoint_path,
                    _conv_history_to_json(session.conversation_history),
                    session.status.value,
                    _dt_to_str(session.created_at),
                    _dt_to_str(session.last_activity),
                    session.exchange_count,
                ),
            )
            conn.commit()

    def get_session(self, session_id: str) -> Session | None:
        """Return a fully-hydrated Session or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None:
                return None
            return self._row_to_session(row)

    def update_session(self, session: Session) -> None:
        """Overwrite all mutable fields of a Session row."""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET mode                      = ?,
                    modification_mode         = ?,
                    checkpoint_path           = ?,
                    conversation_history_json = ?,
                    status                    = ?,
                    last_activity             = ?,
                    exchange_count            = ?
                WHERE id = ?
                """,
                (
                    session.mode.value,
                    session.modification_mode.value
                    if session.modification_mode
                    else None,
                    session.checkpoint_path,
                    _conv_history_to_json(session.conversation_history),
                    session.status.value,
                    _dt_to_str(session.last_activity),
                    session.exchange_count,
                    session.id,
                ),
            )
            conn.commit()

    def list_sessions_for_network(self, network_id: str) -> list[Session]:
        """Return all Sessions for a network, ordered by creation time."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE network_id = ? ORDER BY created_at",
                (network_id,),
            ).fetchall()
            return [self._row_to_session(r) for r in rows]

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> Session:
        mod_mode_raw = row["modification_mode"]
        return Session(
            id=row["id"],
            network_id=row["network_id"],
            mode=SessionMode(row["mode"]),
            modification_mode=ModificationMode(mod_mode_raw)
            if mod_mode_raw
            else None,
            checkpoint_path=row["checkpoint_path"] or "",
            conversation_history=_json_to_conv_history(
                row["conversation_history_json"]
            ),
            status=SessionStatus(row["status"]),
            created_at=_str_to_dt(row["created_at"]),
            last_activity=_str_to_dt(row["last_activity"]),
            exchange_count=row["exchange_count"],
        )

    # ------------------------------------------------------------------
    # Dataset results
    # ------------------------------------------------------------------

    def save_dataset_result(self, result: DatasetResult) -> None:
        """Insert or replace a DatasetResult row."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO dataset_results
                    (id, version_id, manifest_path, output_path, columns_json,
                     start_date, end_date, frequency, adf_json, warnings_json,
                     created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.id,
                    result.version_id,
                    result.manifest_path,
                    result.output_path,
                    json.dumps(result.columns),
                    result.start_date,
                    result.end_date,
                    result.frequency,
                    json.dumps(result.adf_results),
                    json.dumps(result.warnings),
                    _dt_to_str(result.created_at),
                ),
            )

    def get_dataset_results_for_version(self, version_id: str) -> list[DatasetResult]:
        """Return every dataset built for *version_id*, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM dataset_results
                WHERE version_id = ?
                ORDER BY created_at DESC
                """,
                (version_id,),
            ).fetchall()
            return [self._row_to_dataset_result(r) for r in rows]

    @staticmethod
    def _row_to_dataset_result(row: sqlite3.Row) -> DatasetResult:
        return DatasetResult(
            id=row["id"],
            version_id=row["version_id"],
            manifest_path=row["manifest_path"],
            output_path=row["output_path"],
            columns=json.loads(row["columns_json"]),
            start_date=row["start_date"],
            end_date=row["end_date"],
            frequency=row["frequency"],
            adf_results=json.loads(row["adf_json"]),
            warnings=json.loads(row["warnings_json"]),
            created_at=_str_to_dt(row["created_at"]),
        )
