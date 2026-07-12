"""Session managers: map a session to its conversation threads.

A :class:`~opendatasci.session.threads.SessionThread` identifies one conversation in
the graph checkpointer.  Clearing the conversation creates a new thread,
abandoning the old one, so no checkpointed state survives.  Keeping the
session → threads mapping out of the :class:`~opendatasci.agents.agents.Agent`
keeps the agent itself stateless about thread identity: a cloud deployment
can supply a :class:`BaseSessionManager` backed by shared storage and run the
agent in a stateless microservice.
"""

import json
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from opendatasci.context.local import OPENDATASCI_DIRNAME
from opendatasci.session.threads import SessionThread

logger = logging.getLogger(__name__)

_SESSION_FILE = "session.json"


class BaseSessionManager(ABC):
    """Tracks the conversation threads of a single session."""

    @abstractmethod
    def get_or_create_thread(self) -> uuid.UUID:
        """Return the session's current thread, creating the first one if needed."""
        ...

    @abstractmethod
    def create_thread(self) -> uuid.UUID:
        """Create a new thread for the session and make it current."""
        ...

    @abstractmethod
    def get_current_thread(self) -> uuid.UUID:
        """Return the session's current thread.

        Raises:
            LookupError: if the session has no threads yet.
        """
        ...


class LocalSessionManager(BaseSessionManager):
    """File-backed session manager for single-process local runs.

    Persists every session's state to ``.opendatasci/session.json`` in the
    workspace, as a mapping of session id to::

        {
            "created_at": "<iso8601>",
            "last_updated_at": "<iso8601>",
            "threads": [{"thread_id": "<uuid>", "created_at": "<iso8601>"}, ...]
        }

    ``created_at`` is stamped when the session entry is first written and
    ``last_updated_at`` on every write.  The file is read on every lookup and
    rewritten on every thread creation, so no thread state is held in memory.
    Not safe for concurrent writers; the TUI runs it on a single event loop
    in a single process.

    Args:
        workspace_path: Root directory of the active workspace.
        session_id: Identifier of the session whose threads are managed.
    """

    def __init__(self, workspace_path: Path, session_id: str) -> None:
        self._session_id = session_id
        self._session_file = workspace_path / OPENDATASCI_DIRNAME / _SESSION_FILE

    def get_or_create_thread(self) -> uuid.UUID:
        try:
            return self.get_current_thread()
        except LookupError:
            return self.create_thread()

    def create_thread(self) -> uuid.UUID:
        now = datetime.now(timezone.utc)
        thread = SessionThread(thread_id=uuid.uuid4(), created_at=now)
        sessions = self._load()
        session = sessions.setdefault(self._session_id, {})
        session.setdefault("created_at", now.isoformat())
        session["last_updated_at"] = now.isoformat()
        session.setdefault("threads", []).append(thread.model_dump(mode="json"))
        self._save(sessions)
        return thread.thread_id

    def get_current_thread(self) -> uuid.UUID:
        threads = self._load().get(self._session_id, {}).get("threads", [])
        if not threads:
            raise LookupError(f"Session {self._session_id!r} has no threads yet")
        return SessionThread.model_validate(threads[-1]).thread_id

    def _load(self) -> dict[str, Any]:
        if not self._session_file.exists():
            return {}
        try:
            data = json.loads(self._session_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("Could not read session file: %s", self._session_file, exc_info=True)
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self, sessions: dict[str, Any]) -> None:
        self._session_file.parent.mkdir(parents=True, exist_ok=True)
        self._session_file.write_text(json.dumps(sessions, indent=2), encoding="utf-8")
