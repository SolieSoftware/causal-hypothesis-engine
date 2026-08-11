"""Atomic checkpoint writer for causal-hypothesis-engine.

Writes draft session state to disk safely:
  1. Serialise to JSON.
  2. Write to a unique temp file in the destination directory.
  3. fsync the temp file.
  4. os.replace() atomically replaces the target (POSIX and Windows).
  5. fsync the containing directory so the rename itself is durable.

On resume, the engine reads the checkpoint path stored in Session.checkpoint_path.
If the file is missing (e.g. interrupted before first checkpoint), the caller
should surface a clear error and fall back to listing available sessions.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from pydantic import ValidationError

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

    # A unique temp file per writer. A deterministic ".tmp" name is shared by
    # every writer of the same checkpoint, so two processes resuming the same
    # session interleave into one file — and a failing writer would unlink the
    # temp file the other is mid-rename on.
    fd, tmp_name = tempfile.mkstemp(
        dir=str(dest.parent), prefix=f".{dest.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        # os.replace is atomic-overwrite on POSIX *and* Windows; os.rename is not.
        os.replace(tmp_path, dest)
        _fsync_directory(dest.parent)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _fsync_directory(directory: Path) -> None:
    """Flush a directory entry so a completed rename survives a crash.

    Without this the file's *contents* are durable but the rename that makes
    them visible under the final name is not, which defeats the point of the
    write-temp-then-rename dance.
    """
    try:
        dir_fd = os.open(str(directory), os.O_DIRECTORY)
    except (AttributeError, OSError):
        return  # Not supported on this platform (e.g. Windows).
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


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
    except (KeyError, json.JSONDecodeError, ValidationError, OSError) as exc:
        # Deliberately narrow: the previous `(KeyError, JSONDecodeError, Exception)`
        # tuple swallowed genuine programming errors into a misleading
        # "failed to parse checkpoint" message.
        raise ValueError(f"Failed to parse checkpoint at {dest}: {exc}") from exc

    return session, draft_version
