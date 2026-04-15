"""Atomic checkpoint writer for causal-hypothesis-engine.

Writes draft session state to disk safely:
  1. Serialise to JSON.
  2. Write to <path>.tmp.
  3. fsync the temp file.
  4. os.rename() atomically replaces the target (POSIX guarantee).

On resume, the engine reads the checkpoint path stored in Session.checkpoint_path.
If the file is missing (e.g. interrupted before first checkpoint), the caller
should surface a clear error and fall back to listing available sessions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..models.dag_version import DAGVersion
from ..models.session import Session


def write_checkpoint(session: Session, draft_version: DAGVersion, path: str | Path) -> None:
    """Atomically write session + draft DAG state to *path*.

    The file at *path* is replaced in a single atomic rename so a crash
    mid-write can never leave a partial file.

    Args:
        session: The active Session object (conversation history, status, etc.).
        draft_version: The current in-progress DAGVersion (Draft status).
        path: Destination file path.  Parent directory must already exist.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "session": session.model_dump(mode="json"),
        "draft_version": draft_version.model_dump(mode="json"),
    }
    data = json.dumps(payload, indent=2)

    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.rename(tmp_path, dest)
    except Exception:
        # Clean up the temp file if rename failed.
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def load_checkpoint(path: str | Path) -> tuple[Session, DAGVersion]:
    """Load a checkpoint file written by :func:`write_checkpoint`.

    Returns:
        A ``(Session, DAGVersion)`` tuple.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file cannot be parsed.
    """
    dest = Path(path)
    if not dest.exists():
        raise FileNotFoundError(f"Checkpoint not found: {dest}")

    try:
        with open(dest, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        session = Session.model_validate(payload["session"])
        draft_version = DAGVersion.model_validate(payload["draft_version"])
    except (KeyError, json.JSONDecodeError, Exception) as exc:
        raise ValueError(f"Failed to parse checkpoint at {dest}: {exc}") from exc

    return session, draft_version
